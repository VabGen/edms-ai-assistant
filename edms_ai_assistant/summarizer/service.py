"""
SummarizationService — главная точка входа модуля суммаризации.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from edms_ai_assistant.config import settings
from edms_ai_assistant.summarizer.cache.cache import CacheEntry, TwoLevelCache
from edms_ai_assistant.summarizer.errors import PipelineError, TextExtractionError
from edms_ai_assistant.summarizer.chunking.structural import SmartChunker
from edms_ai_assistant.summarizer.chunking.token_aware import count_tokens
from edms_ai_assistant.summarizer.observability.logging_ctx import request_id_var
from edms_ai_assistant.summarizer.observability.tracing import (
    RequestCostAccumulator,
    Stopwatch,
    get_cost_usd,
    set_current_accumulator,
    trace_stage,
)
from edms_ai_assistant.summarizer.pipeline.direct import (
    DirectSummarizationPipeline,
    LLMClient,
    PipelineResult,
    StreamEvent,
)
from edms_ai_assistant.summarizer.pipeline.map_reduce import MapReducePipeline
from edms_ai_assistant.summarizer.prompts.registry import (
    PromptRegistry,
    get_prompt_registry,
)
from edms_ai_assistant.summarizer.structured.models import (
    QualityScore,
    SummaryMode,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

_DIRECT_CONTEXT_WINDOW_TOKENS = 4096


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class SummarizationRequest(BaseModel):
    model_config = {"frozen": True}

    file_content: bytes = Field(description="Raw file bytes")
    file_name: str = Field(
        description="Имя файла для определения расширения", max_length=255
    )
    mode: SummaryMode = Field(default=SummaryMode.ABSTRACTIVE)
    language: str = Field(
        default="ru",
        description="BCP-47 код языка вывода. 'ru' по умолчанию.",
        max_length=10,
    )
    request_id: str = Field(description="Уникальный ID запроса для трейсинга")
    force_refresh: bool = Field(default=False)
    enable_quality_score: bool = Field(
        default=False,
        description="Если True — после генерации запускается LLM-as-judge "
        "и заполняет поле SummarizationResponse.quality.",
    )


class SummarizationResponse(BaseModel):
    model_config = {"frozen": True}

    request_id: str
    mode: SummaryMode
    language: str
    output: dict
    quality: QualityScore | None = None
    cache_hit: bool = False
    cache_source: str = "miss"

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    chunking_strategy: str
    chunk_count: int

    file_hash: str
    prompt_version: str


# ---------------------------------------------------------------------------
# Text Extractor
# ---------------------------------------------------------------------------


async def extract_text_from_bytes(file_content: bytes, file_name: str) -> str:
    """Async text extraction from file bytes (offloads heavy parsers to a worker thread)."""
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "txt"

    def _sync_extract() -> str:
        if ext == "pdf":
            return _extract_pdf(file_content)
        if ext in ("docx", "doc"):
            return _extract_docx(file_content)
        if ext in ("txt", "md", "rst", "csv"):
            return _decode_text(file_content)
        # Уже извлечённый UTF-8 текст или произвольная кодировка
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            return _decode_text(file_content)

    return await asyncio.to_thread(_sync_extract)


def _extract_pdf(data: bytes) -> str:
    try:
        import fitz  # type: ignore[import]

        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        pass
    try:
        import io

        import pdfplumber  # type: ignore[import]

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        raise TextExtractionError(f"PDF extraction failed: {exc}") from exc


def _extract_docx(data: bytes) -> str:
    import io

    try:
        import docx  # type: ignore[import]

        doc = docx.Document(io.BytesIO(data))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except ImportError:
        pass
    try:
        import docx2txt  # type: ignore[import]

        return docx2txt.process(io.BytesIO(data))
    except Exception as exc:
        raise TextExtractionError(f"DOCX extraction failed: {exc}") from exc


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Output → Text formatter
# ---------------------------------------------------------------------------


def format_output_as_markdown(resp: SummarizationResponse) -> str:
    """
    Конвертирует структурированный вывод LLM в читаемый Markdown.
    Все тексты на русском (если LLM ответил правильно).
    """
    output = resp.output

    if resp.mode == SummaryMode.EXECUTIVE:
        headline = output.get("headline", "")
        bullets = output.get("bullets", [])
        recommendation = output.get("recommendation")

        lines = [f"**{headline}**", ""] if headline else []
        for bullet in bullets:
            if bullet.strip():
                lines.append(f"• {bullet}")
        if recommendation:
            lines.extend(["", f"**Рекомендация:** {recommendation}"])
        return "\n".join(lines) if lines else "Анализ завершён."

    elif resp.mode == SummaryMode.DETAILED_NOTES:
        doc_type = output.get("document_type", "Документ")
        sections = output.get("sections", [])
        entities = output.get("key_entities", [])
        date_range = output.get("date_range")

        lines = [f"**Тип документа:** {doc_type}", ""]
        for sec in sections:
            title = sec.get("title", "")
            content = sec.get("content", "")
            subsections = sec.get("subsections", [])
            if title:
                lines.append(f"## {title}")
            if content:
                lines.append(content)
            for sub in subsections:
                lines.append(f"  - {sub}")
            lines.append("")
        if entities:
            lines.append(f"**Ключевые участники:** {', '.join(entities)}")
        if date_range:
            lines.append(f"**Период:** {date_range}")
        return "\n".join(lines) if lines else "Анализ завершён."

    elif resp.mode == SummaryMode.ACTION_ITEMS:
        items = output.get("action_items", [])
        context = output.get("document_context", "")

        if not items:
            ctx = f"\n\n*Контекст: {context}*" if context else ""
            return f"Задачи и поручения не найдены.{ctx}"

        lines = [f"**Найдено задач: {len(items)}**", ""]
        for i, item in enumerate(items, 1):
            priority = item.get("priority", "medium")
            priority_label = {"high": "[!]", "medium": "[*]", "low": "[-]"}.get(priority, "•")
            task = item.get("task", "")
            owner = item.get("owner")
            deadline = item.get("deadline")

            lines.append(f"{i}. {priority_label} {task}")
            if owner:
                lines.append(f"   Ответственный: {owner}")
            if deadline:
                lines.append(f"   Срок: {deadline}")
            lines.append("")
        return "\n".join(lines)

    elif resp.mode == SummaryMode.THESIS:
        main_arg = output.get("main_argument", "")
        sections = output.get("sections", [])
        conclusion = output.get("conclusion", "")

        lines = []
        if main_arg:
            lines.extend([f"**Главный тезис:** {main_arg}", ""])
        for sec in sections:
            title = sec.get("title", "")
            thesis = sec.get("thesis", "")
            points = sec.get("points", [])
            if title:
                lines.append(f"## {title}")
            if thesis:
                lines.append(f"*{thesis}*")
            for pt in points:
                claim = pt.get("claim", "")
                evidence = pt.get("evidence")
                if claim:
                    lines.append(f"- {claim}")
                if evidence:
                    lines.append(f"  _{evidence}_")
            lines.append("")
        if conclusion:
            lines.append(f"**Вывод:** {conclusion}")
        return "\n".join(lines) if lines else "Анализ завершён."

    elif resp.mode == SummaryMode.EXTRACTIVE:
        facts = output.get("facts", [])
        doc_summary = output.get("document_summary", "")

        lines = []
        if doc_summary:
            lines.extend([f"*{doc_summary}*", ""])
        for fact in facts:
            label = fact.get("label", "")
            value = fact.get("value", "")
            if label and value:
                lines.append(f"• **{label}**: {value}")
        return "\n".join(lines) if lines else "Факты не извлечены."

    elif resp.mode in (SummaryMode.ABSTRACTIVE, SummaryMode.MULTILINGUAL):
        summary = output.get("summary", "")
        themes = output.get("key_themes", [])

        lines = [summary] if summary else []
        if themes:
            lines.extend(["", f"**Ключевые темы:** {', '.join(themes)}"])
        return "\n".join(lines) if lines else "Анализ завершён."

    else:
        # Fallback для неизвестных режимов
        return (
            output.get("summary", "")
            or output.get("content", "")
            or json.dumps(output, ensure_ascii=False, indent=2)
        )


# ---------------------------------------------------------------------------
# SummarizationService
# ---------------------------------------------------------------------------


class SummarizationService:
    """
    Facade над всем пайплайном суммаризации.

    Использование:
        service = await build_summarization_service(settings)
        response = await service.summarize(request)
        text = format_output_as_markdown(response)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cache: TwoLevelCache,
        *,
        model: str = settings.LLM_GENERATIVE_MODEL,
        direct_context_window: int = _DIRECT_CONTEXT_WINDOW_TOKENS,
        max_concurrent_map: int = 6,
        max_output_tokens: int = 4096,
    ) -> None:
        self._llm = llm_client
        self._cache = cache
        self._model = model
        self._context_window = direct_context_window
        self._max_output_tokens = max_output_tokens
        self._prompts: PromptRegistry = get_prompt_registry()
        self._chunker = SmartChunker()
        self._direct_pipeline = DirectSummarizationPipeline(
            llm_client,
            self._prompts,
            model,
            max_output_tokens=max_output_tokens,
        )
        self._map_reduce_pipeline = MapReducePipeline(
            llm_client,
            self._prompts,
            model,
            max_concurrent_map=max_concurrent_map,
            max_output_tokens=max_output_tokens,
            direct_pipeline=self._direct_pipeline,
        )
        self._bg_tasks: set[asyncio.Task[None]] = set()
        self._inflight: dict[str, asyncio.Future[SummarizationResponse]] = {}
        self._inflight_lock = asyncio.Lock()

    async def summarize(self, request: SummarizationRequest) -> SummarizationResponse:
        """Выполняет суммаризацию с кэшированием и in-flight дедупликацией."""
        request_id_var.set(request.request_id)
        file_hash = await asyncio.to_thread(
            lambda: hashlib.sha256(request.file_content).hexdigest()
        )
        language = request.language or "ru"
        cache_key = CacheEntry.build_key(
            file_hash=file_hash,
            mode=request.mode.value,
            language=language,
            prompt_version=self._prompts.cache_version_tag(),
        )

        if not request.force_refresh:
            async with self._inflight_lock:
                pending = self._inflight.get(cache_key)
                if pending is not None:
                    logger.info(
                        "In-flight dedup: request_id=%s waits for key=%s",
                        request.request_id,
                        cache_key[:12],
                    )
                    pending_request_id = request.request_id
                    result = await asyncio.shield(pending)
                    return result.model_copy(update={"request_id": pending_request_id})

                future: asyncio.Future[SummarizationResponse] = (
                    asyncio.get_running_loop().create_future()
                )
                self._inflight[cache_key] = future
        else:
            future = None  # type: ignore[assignment]

        try:
            response = await self._summarize_internal(
                request, file_hash, language, cache_key
            )
            if future is not None and not future.done():
                future.set_result(response)
            return response
        except BaseException as exc:
            if future is not None and not future.done():
                future.set_exception(
                    exc if isinstance(exc, Exception) else RuntimeError(str(exc))
                )
            raise
        finally:
            if future is not None:
                async with self._inflight_lock:
                    self._inflight.pop(cache_key, None)

    async def _summarize_internal(
        self,
        request: SummarizationRequest,
        file_hash: str,
        language: str,
        cache_key: str,
    ) -> SummarizationResponse:
        """Основной поток суммаризации (без дедупликации)."""
        sw_total = Stopwatch()

        acc = RequestCostAccumulator(
            request_id=request.request_id,
            model=self._model,
        )
        set_current_accumulator(acc)

        # --- Cache lookup ---
        if not request.force_refresh:
            cached_entry, cache_source = await self._cache.get(cache_key)
            if cached_entry is not None:
                logger.info(
                    "Cache %s hit: request_id=%s key=%s",
                    cache_source.upper(),
                    request.request_id,
                    cache_key[:12],
                )
                return self._response_from_cache(
                    cached_entry, cache_source, request.request_id
                )

        # --- Text extraction ---
        async with trace_stage("extract_text", {"file": request.file_name}):
            text = await extract_text_from_bytes(
                request.file_content, request.file_name
            )

        if not text.strip():
            raise TextExtractionError(f"Не удалось извлечь текст из '{request.file_name}'")

        doc_tokens = count_tokens(text)
        logger.info(
            "Суммаризация: mode=%s lang=%s tokens=%d request_id=%s",
            request.mode.value,
            language,
            doc_tokens,
            request.request_id,
        )

        # --- Pipeline selection ---
        async with trace_stage(
            "pipeline",
            {
                "mode": request.mode.value,
                "doc_tokens": doc_tokens,
                "model": self._model,
            },
        ) as pipeline_span:
            needs_map_reduce = self._chunker.needs_map_reduce(
                text, context_window=self._context_window
            )

            if needs_map_reduce:
                logger.info(
                    "MapReducePipeline: %d токенов > порог %d",
                    doc_tokens,
                    int(self._context_window * 0.70),
                )
                pipeline_result = await self._map_reduce_pipeline.run(
                    text,
                    request.mode,
                    language=language,
                    span=pipeline_span,
                )
            else:
                logger.info(
                    "DirectPipeline: %d токенов укладываются в контекст",
                    doc_tokens,
                )
                pipeline_result = await self._direct_pipeline.run(
                    text,
                    request.mode,
                    language=language,
                    span=pipeline_span,
                )

        # Optional: LLM-as-judge quality scoring
        quality: QualityScore | None = None
        if request.enable_quality_score:
            async with trace_stage("quality.judge", {"mode": request.mode.value}):
                summary_text = pipeline_result.raw_json
                quality = await self.score_quality(text, summary_text)

        total_latency = sw_total.elapsed_ms()

        response = SummarizationResponse(
            request_id=request.request_id,
            mode=request.mode,
            language=language,
            output=pipeline_result.output.model_dump(),
            quality=quality,
            cache_hit=False,
            cache_source="miss",
            model=self._model,
            input_tokens=pipeline_result.input_tokens,
            output_tokens=pipeline_result.output_tokens,
            cost_usd=round(
                get_cost_usd(
                    self._model,
                    pipeline_result.input_tokens,
                    pipeline_result.output_tokens,
                ),
                6,
            ),
            latency_ms=round(total_latency, 1),
            chunking_strategy=pipeline_result.chunking_strategy,
            chunk_count=pipeline_result.chunk_count,
            file_hash=file_hash,
            prompt_version=self._prompts.cache_version_tag(),
        )

        task = asyncio.create_task(
            self._write_cache(response, pipeline_result, cache_key, file_hash)
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

        logger.info(
            "Суммаризация завершена: request_id=%s mode=%s tokens=%d/%d "
            "cost=$%.4f latency=%.0fms strategy=%s",
            request.request_id,
            request.mode.value,
            pipeline_result.input_tokens,
            pipeline_result.output_tokens,
            response.cost_usd,
            total_latency,
            pipeline_result.chunking_strategy,
        )

        return response

    async def _write_cache(
        self,
        response: SummarizationResponse,
        result: PipelineResult,
        cache_key: str,
        file_hash: str,
    ) -> None:
        try:
            entry = CacheEntry(
                cache_key=cache_key,
                mode=response.mode.value,
                language=response.language,
                prompt_version=response.prompt_version,
                result_json=result.raw_json,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                model_name=response.model,
                chunking_strategy=response.chunking_strategy,
                created_at_ms=int(time.time() * 1000),
                file_hash=file_hash,
            )
            await self._cache.set(entry)
            logger.debug("Cache write: key=%s", cache_key[:12])
        except Exception as exc:
            logger.error("Cache write failed (non-critical): %s", exc)

    @staticmethod
    def _response_from_cache(
        entry: CacheEntry,
        source: str,
        request_id: str,
    ) -> SummarizationResponse:
        try:
            output_dict = json.loads(entry.result_json)
        except Exception:
            output_dict = {"raw": entry.result_json}

        return SummarizationResponse(
            request_id=request_id,
            mode=SummaryMode(entry.mode),
            language=entry.language,
            output=output_dict,
            cache_hit=True,
            cache_source=source,
            model=entry.model_name,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            cost_usd=entry.cost_usd,
            latency_ms=0.0,
            chunking_strategy=entry.chunking_strategy,
            chunk_count=0,
            file_hash=entry.file_hash,
            prompt_version=entry.prompt_version,
        )

    async def summarize_stream(
        self, request: SummarizationRequest
    ) -> AsyncIterator[StreamEvent | SummarizationResponse]:
        """
        Стримит вывод LLM по токенам и завершается финальным SummarizationResponse.

        Поведение:
          - Cache hit: сразу yield финальный response (без дельт).
          - Малый документ: DirectPipeline.run_stream → дельты + финальный результат.
          - Большой документ: фолбэк на не-стримящий map-reduce, в конце один response.

        Финальный SummarizationResponse кладётся в кэш как обычно.
        """
        request_id_var.set(request.request_id)
        sw_total = Stopwatch()

        acc = RequestCostAccumulator(request_id=request.request_id, model=self._model)
        set_current_accumulator(acc)

        file_hash = await asyncio.to_thread(
            lambda: hashlib.sha256(request.file_content).hexdigest()
        )
        language = request.language or "ru"
        cache_key = CacheEntry.build_key(
            file_hash=file_hash,
            mode=request.mode.value,
            language=language,
            prompt_version=self._prompts.cache_version_tag(),
        )

        # Cache hit → выдаём готовый response без дельт
        if not request.force_refresh:
            cached_entry, cache_source = await self._cache.get(cache_key)
            if cached_entry is not None:
                yield self._response_from_cache(
                    cached_entry, cache_source, request.request_id
                )
                return

        async with trace_stage("extract_text", {"file": request.file_name}):
            text = await extract_text_from_bytes(
                request.file_content, request.file_name
            )
        if not text.strip():
            raise TextExtractionError(f"Не удалось извлечь текст из '{request.file_name}'")

        needs_map_reduce = self._chunker.needs_map_reduce(
            text, context_window=self._context_window
        )

        async with trace_stage(
            "pipeline.stream",
            {
                "mode": request.mode.value,
                "model": self._model,
                "map_reduce": needs_map_reduce,
            },
        ) as span:
            pipeline = (
                self._map_reduce_pipeline if needs_map_reduce else self._direct_pipeline
            )
            pipeline_result = None
            async for event in pipeline.run_stream(
                text,
                request.mode,
                language=language,
                span=span,
            ):
                if isinstance(event, PipelineResult):
                    pipeline_result = event
                else:
                    yield event
            if pipeline_result is None:
                raise PipelineError("Pipeline.run_stream did not yield a PipelineResult")

        total_latency = sw_total.elapsed_ms()
        response = SummarizationResponse(
            request_id=request.request_id,
            mode=request.mode,
            language=language,
            output=pipeline_result.output.model_dump(),
            cache_hit=False,
            cache_source="miss",
            model=self._model,
            input_tokens=pipeline_result.input_tokens,
            output_tokens=pipeline_result.output_tokens,
            cost_usd=round(
                get_cost_usd(
                    self._model,
                    pipeline_result.input_tokens,
                    pipeline_result.output_tokens,
                ),
                6,
            ),
            latency_ms=round(total_latency, 1),
            chunking_strategy=pipeline_result.chunking_strategy,
            chunk_count=pipeline_result.chunk_count,
            file_hash=file_hash,
            prompt_version=self._prompts.cache_version_tag(),
        )

        task = asyncio.create_task(
            self._write_cache(response, pipeline_result, cache_key, file_hash)
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

        yield response

    async def score_quality(
        self,
        source_text: str,
        summary_text: str,
        *,
        max_source_chars: int = 8000,
    ) -> QualityScore | None:
        """LLM-as-judge: оценивает качество суммаризации (0.0–1.0).

        Если LLM возвращает невалидный JSON — возвращаем None (best-effort).
        Не бросает исключений: ошибка judge не должна ронять основной запрос.
        """
        if not summary_text.strip():
            return None

        template = self._prompts.get_judge()
        system = template.system
        user = template.user_template.format(
            text=source_text[:max_source_chars],
            summary=summary_text,
        )

        try:
            raw, _, _ = await self._llm.complete(
                system,
                user,
                model=self._model,
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as exc:
            logger.warning("Quality judge LLM call failed: %s", exc)
            return None

        try:
            parsed = json.loads(raw.strip())
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            critique = parsed.get("critique")
            return QualityScore.from_score(score, critique=critique)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.debug("Quality judge returned non-JSON or malformed: %s", exc)
            return None

    async def invalidate_cache(self, file_hash: str, mode: str | None = None) -> None:
        await self._cache.invalidate_by_file(file_hash, mode=mode)

    def cache_stats(self) -> dict:
        return self._cache.stats()

    async def health(self) -> dict[str, Any]:
        """Public health probe: cache layers + cache stats."""
        return {
            "cache": await self._cache.health(),
            "cache_stats": self._cache.stats(),
        }

    async def aclose(self) -> None:
        """Корректное завершение: дожидаемся background-задач и закрываем LLM-клиент."""
        if self._bg_tasks:
            pending = list(self._bg_tasks)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning(
                    "Background cache writes did not finish within 5s (%d pending)",
                    len(pending),
                )
        try:
            await self._llm.aclose()
        except Exception as exc:
            logger.warning("LLM client close failed: %s", exc)
