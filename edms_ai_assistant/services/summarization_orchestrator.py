# edms_ai_assistant/services/summarization_orchestrator.py
"""Summarization orchestrator: cache, map-reduce, streaming, self-critique.

Public API:
    get_orchestrator() -> SummarizationOrchestrator
    stream_summarize(text, fmt, file_identifier) -> AsyncGenerator[str, None]
    _make_cache_key(file_identifier, summary_type) -> str  # used by cache router
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from typing import Final

import json_repair
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select

from edms_ai_assistant.db.database import AsyncSessionLocal
from edms_ai_assistant.db.generated.models.summarization_cache import SummarizationCache
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.schemas.summarization import SummarizationResult, SummaryFormat
from edms_ai_assistant.utils.async_utils import spawn_background_task
from edms_ai_assistant.utils.token_counter import token_counter

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

PROMPT_VERSION: Final[str] = "v1"
_MAP_REDUCE_THRESHOLD_TOKENS: Final[int] = 3_000
_MAP_CHUNK_MAX_TOKENS: Final[int] = 1_000
_MAP_CONCURRENCY: Final[int] = 4
_SELF_CRITIQUE_MIN_LEN: Final[int] = 50
_TEXT_MIN_LEN: Final[int] = 30

# ── Cache key ──────────────────────────────────────────────────────────────────


def _make_cache_key(file_identifier: str, summary_type: str) -> str:
    """Deterministic 48-char SHA-256 key for (file, type, prompt_version)."""
    raw = f"{file_identifier}::{summary_type}::{PROMPT_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:48]


# ── Prompt templates ───────────────────────────────────────────────────────────

_DIRECT_PROMPTS: Final[dict[SummaryFormat, str]] = {
    SummaryFormat.EXTRACTIVE: (
        "Ты — старший аналитик СЭД. Извлеки ключевые факты.\n"
        "ПРАВИЛА: 1) Формат: Markdown-список. 2) Только цифры, даты, имена. 3) Максимум 12 пунктов.\n"
        "Текст:\n<text>\n{text}\n</text>\nКлючевые факты:"
    ),
    SummaryFormat.ABSTRACTIVE: (
        "Ты — эксперт по деловой коммуникации. Напиши Executive Summary.\n"
        "ПРАВИЛА: 1) Формат: Markdown (**Суть**, **Детали**, **Рекомендация**). "
        "2) Максимум 150 слов. Без канцеляризмов. 3) Только факты из текста.\n"
        "Текст:\n<text>\n{text}\n</text>\nExecutive Summary:"
    ),
    SummaryFormat.THESIS: (
        "Ты — методолог. Построй иерархический тезисный план.\n"
        "ПРАВИЛА: 1) Формат: Markdown (## Раздел, - Подтезис). "
        "2) Максимум 5 разделов, до 3 тезисов. 3) Сохраняй логику документа.\n"
        "Текст:\n<text>\n{text}\n</text>\nТезисный план:"
    ),
}

_MAP_PROMPTS: Final[dict[SummaryFormat, str]] = {
    SummaryFormat.EXTRACTIVE: (
        "Ты — аналитик. Извлеки ключевые факты ИЗ ЧАСТИ документа.\n"
        "ПРАВИЛА: Выдай только список фактов. Без вводных слов.\n"
        "Часть текста:\n<chunk>\n{chunk}\n</chunk>\nФакты:"
    ),
    SummaryFormat.ABSTRACTIVE: (
        "Ты — эксперт. Напиши краткое саммари ЧАСТИ документа.\n"
        "ПРАВИЛА: Максимум 50 слов. Только факты.\n"
        "Часть текста:\n<chunk>\n{chunk}\n</chunk>\nСаммари:"
    ),
    SummaryFormat.THESIS: (
        "Ты — методолог. Выдели основные тезисы ЧАСТИ документа.\n"
        "ПРАВИЛА: Формат: Markdown-список.\n"
        "Часть текста:\n<chunk>\n{chunk}\n</chunk>\nТезисы:"
    ),
}

_REDUCE_PROMPTS: Final[dict[SummaryFormat, str]] = {
    SummaryFormat.EXTRACTIVE: (
        "Ты — главный аналитик. Тебе предоставлены факты из разных частей документа.\n"
        "ПРАВИЛА: Объедини дубликаты, оставь самые важные. Максимум 12 пунктов. Формат: Markdown-список.\n"
        "Сборные факты:\n<summaries>\n{summaries}\n</summaries>\nИтоговые факты:"
    ),
    SummaryFormat.ABSTRACTIVE: (
        "Ты — главный эксперт. На основе саммари частей документа напиши итоговое Executive Summary.\n"
        "ПРАВИЛА: Формат: Markdown. Максимум 150 слов.\n"
        "Части саммари:\n<summaries>\n{summaries}\n</summaries>\nExecutive Summary:"
    ),
    SummaryFormat.THESIS: (
        "Ты — главный методолог. На основе тезисов частей документа построй итоговый иерархический план.\n"
        "ПРАВИЛА: Формат: Markdown (## Раздел, - Подтезис). Максимум 5 разделов.\n"
        "Тезисы частей:\n<summaries>\n{summaries}\n</summaries>\nТезисный план:"
    ),
}


# ── Text splitting ─────────────────────────────────────────────────────────────


def _token_chunks(text: str, max_tokens: int = _MAP_CHUNK_MAX_TOKENS) -> list[str]:
    """Split text into token-bounded chunks with 10% overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=int(max_tokens * 0.1),
        length_function=token_counter.count,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


# ── Cache I/O ──────────────────────────────────────────────────────────────────


async def _load_from_cache(
    file_identifier: str, summary_type: str
) -> SummarizationResult | None:
    """Load cached result — opens its own session, safe to call anywhere."""
    cache_key = _make_cache_key(file_identifier, summary_type)
    async with AsyncSessionLocal() as db:
        row = await db.scalar(
            select(SummarizationCache).where(
                SummarizationCache.file_identifier == cache_key,
                SummarizationCache.summary_type == summary_type,
            )
        )
        if row is not None:
            return SummarizationResult.model_validate_json(row.content)
    return None


async def _save_to_cache(
    file_identifier: str, summary_type: str, data: SummarizationResult
) -> None:
    """Upsert result — atomic via db.begin(), runs in background after stream."""
    cache_key = _make_cache_key(file_identifier, summary_type)
    async with AsyncSessionLocal() as db, db.begin():
        existing = await db.scalar(
            select(SummarizationCache).where(
                SummarizationCache.file_identifier == cache_key,
                SummarizationCache.summary_type == summary_type,
            )
        )
        if existing is not None:
            existing.content = data.model_dump_json()
        else:
            db.add(
                SummarizationCache(
                    file_identifier=cache_key,
                    summary_type=summary_type,
                    content=data.model_dump_json(),
                )
            )


# ── Self-critique (background) ────────────────────────────────────────────────


async def _self_critique_async(
    summary: str,           # ← CORRECT: built summary, not the prompt
    file_identifier: str,
    summary_type: str,
) -> None:
    """Score summary quality and patch DB row in background. Never raises."""
    if len(summary) < _SELF_CRITIQUE_MIN_LEN:
        return

    critique_prompt = (
        "Оцени качество саммари ниже по шкале от 0 до 1 (где 1 — идеально). "
        'Ответь ТОЛЬКО в формате JSON: {"score": 0.85, "confidence": "high"}\n'
        f"Саммари:\n{summary[:2000]}"
    )

    try:
        llm = get_chat_model()
        raw = await llm.ainvoke([HumanMessage(content=critique_prompt)])
        data = json_repair.loads(raw.content)
        score = round(min(max(float(data.get("score", 0.7)), 0.0), 1.0), 2)
        confidence: str = data.get("confidence", "medium")

        cache_key = _make_cache_key(file_identifier, summary_type)
        async with AsyncSessionLocal() as db:
            row = await db.scalar(
                select(SummarizationCache).where(
                    SummarizationCache.file_identifier == cache_key,
                    SummarizationCache.summary_type == summary_type,
                )
            )
            if row is not None:
                payload = SummarizationResult.model_validate_json(row.content)
                payload.quality_score = score
                payload.confidence = confidence
                row.content = payload.model_dump_json()
                await db.commit()

    except Exception as exc:
        logger.debug("Async self-critique failed (non-fatal): %s", exc)


# ── Core streaming logic ───────────────────────────────────────────────────────


async def stream_summarize(
    text: str,
    fmt: SummaryFormat | str,
    file_identifier: str | None = None,
    map_reduce_threshold_tokens: int = _MAP_REDUCE_THRESHOLD_TOKENS,
) -> AsyncGenerator[str, None]:
    """Stream LLM summary tokens with progress events for the caller.

    Routing:
        text_tokens ≤ threshold  → direct single LLM call, stream tokens immediately
        text_tokens > threshold  → map phase (parallel, non-streaming per chunk)
                                   followed by streaming reduce

    SSE progress protocol (yielded as raw strings starting with "__progress__:"):
        __progress__:{"stage":"map","done":2,"total":5}
        __progress__:{"stage":"reduce","total":5}

    The SSE route layer MUST intercept lines starting with ``__progress__:`` and
    emit them as ``data: {"type":"progress", ...}`` events instead of raw tokens.

    Self-critique is scheduled after the last token yield so it never delays
    the stream.

    Args:
        text: Source document text (extracted, stripped).
        fmt: Summary format — accepts SummaryFormat enum or string alias.
        file_identifier: Stable file ID for self-critique background update.
        map_reduce_threshold_tokens: Token threshold for map-reduce routing.

    Yields:
        Raw LLM token strings, or ``__progress__:…`` progress markers.
    """
    # Normalise fmt to enum once
    if not isinstance(fmt, SummaryFormat):
        try:
            fmt = SummaryFormat(str(fmt).strip().lower())
        except ValueError:
            fmt = SummaryFormat.EXTRACTIVE

    llm = get_chat_model()
    text_tokens = token_counter.count(text)
    collected_tokens: list[str] = []

    if text_tokens <= map_reduce_threshold_tokens:
        # ── Direct path: stream tokens immediately ────────────────────────
        final_prompt = _DIRECT_PROMPTS[fmt].format(text=text)
        async for chunk in llm.astream([HumanMessage(content=final_prompt)]):
            t = chunk.content
            collected_tokens.append(t)
            yield t

    else:
        # ── Map-reduce path ───────────────────────────────────────────────
        chunks = _token_chunks(text)
        total = len(chunks)
        sem = asyncio.Semaphore(_MAP_CONCURRENCY)
        done_count = 0

        # --- Map phase: parallel non-streaming (chunks are small, ~1k tokens)
        map_parts: list[str | BaseException] = [None] * total  # type: ignore[list-item]

        async def _map_one(chunk: str, idx: int) -> None:
            nonlocal done_count
            async with sem:
                prompt = _MAP_PROMPTS[fmt].format(chunk=chunk)
                res = await llm.ainvoke([HumanMessage(content=prompt)])
                map_parts[idx] = f"Фрагмент {idx + 1}:\n{res.content.strip()}"
            done_count += 1

        # Launch all map tasks; yield progress events as each finishes
        tasks = [asyncio.create_task(_map_one(c, i)) for i, c in enumerate(chunks)]

        # Progress ticker: polls done_count while tasks are running
        async def _progress_ticker() -> None:
            last = -1
            while any(not t.done() for t in tasks):
                if done_count != last:
                    last = done_count
                    yield f"__progress__:{{'\"stage\":\"map\",\"done\":{done_count},\"total\":{total}}}"
                await asyncio.sleep(0.25)
            yield f"__progress__:{{\"stage\":\"map\",\"done\":{total},\"total\":{total}}}"

        # Interleave progress ticks with awaiting completion
        ticker = _progress_ticker()
        gather_task = asyncio.gather(*tasks, return_exceptions=True)
        async for progress_event in ticker:
            yield progress_event
        await gather_task

        valid_parts = [p for p in map_parts if isinstance(p, str)]
        if not valid_parts:
            yield "Ошибка: не удалось обработать фрагменты документа."
            return

        # --- Reduce phase: stream tokens so user sees output immediately
        yield f"__progress__:{{\"stage\":\"reduce\",\"total\":{total}}}"
        reduce_prompt = _REDUCE_PROMPTS[fmt].format(
            summaries="\n\n---\n\n".join(valid_parts)
        )
        async for chunk in llm.astream([HumanMessage(content=reduce_prompt)]):
            t = chunk.content
            collected_tokens.append(t)
            yield t

    # ── Fire-and-forget self-critique ─────────────────────────────────────
    # Passes the built summary — NOT the prompt — to avoid the previous bug.
    if file_identifier and collected_tokens:
        summary_text = "".join(collected_tokens)
        spawn_background_task(
            _self_critique_async(summary_text, file_identifier, fmt.value)
        )


# ── Orchestrator class ─────────────────────────────────────────────────────────


class SummarizationOrchestrator:
    """High-level facade: cache check, LLM health probe, full summarize().

    Args:
        enable_cache: Read/write the summarization cache.
    """

    def __init__(self, enable_cache: bool = True) -> None:
        self.enable_cache = enable_cache

    async def check_cache(
        self, file_identifier: str, summary_type: str
    ) -> SummarizationResult | None:
        """Return cached result or None — never raises."""
        return await _load_from_cache(file_identifier, summary_type)

    async def check_llm_health(self) -> bool:
        """True if LLM reachable, False on any error."""
        try:
            llm = get_chat_model()
            await llm.ainvoke([HumanMessage(content="test")], max_tokens=1)
            return True
        except Exception:
            return False

    async def summarize(
        self,
        text: str,
        summary_type: str | SummaryFormat,
        file_identifier: str | None = None,
    ) -> SummarizationResult:
        """Collect full stream and return structured result.

        Filters out __progress__ markers that stream_summarize yields.
        Saves to cache in background after collecting tokens.

        Args:
            text: Source document text.
            summary_type: Desired format.
            file_identifier: Optional stable file ID for cache write.

        Returns:
            SummarizationResult with status "success" or "error".
        """
        start = time.monotonic()

        try:
            fmt = SummaryFormat(str(summary_type).strip().lower())
        except ValueError:
            fmt = SummaryFormat.EXTRACTIVE

        text = (text or "").strip()
        if len(text) < _TEXT_MIN_LEN:
            return SummarizationResult(
                status="error",
                content="Текст слишком короткий.",
                format_used=fmt,
                text_length=len(text),
            )

        tokens: list[str] = []
        try:
            async for token in stream_summarize(text, fmt, file_identifier):
                # Skip internal progress markers — they are SSE-only
                if not token.startswith("__progress__:"):
                    tokens.append(token)
        except Exception as exc:
            return SummarizationResult(
                status="error",
                content=f"Ошибка генерации: {exc}",
                format_used=fmt,
                text_length=len(text),
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        result = SummarizationResult(
            status="success",
            content="".join(tokens),
            format_used=fmt,
            processing_time_ms=elapsed_ms,
            chunks_processed=1,
            text_length=len(text),
            pipeline="stream",
        )

        if self.enable_cache and file_identifier:
            await _save_to_cache(file_identifier, fmt.value, result)

        return result


# ── Factory ────────────────────────────────────────────────────────────────────

def get_orchestrator() -> SummarizationOrchestrator:
    """Stateless factory — safe to call on every request."""
    return SummarizationOrchestrator(enable_cache=True)