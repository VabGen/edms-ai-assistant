"""
SummarizationService — main entry point for the summarization subsystem.

This is the ONLY public API for the summarizer module. All internal pipeline
complexity is hidden behind this facade.

Responsibilities:
  1. Cache lookup (L1 Redis → L2 Postgres)
  2. Pipeline selection (Direct vs Map-Reduce based on token count)
  3. Tracing + cost accumulation
  4. Cache write (async, non-blocking)
  5. Quality scoring (async background task)

Thread/Concurrency model:
  - Fully async, no blocking calls
  - Singleton via dependency injection (not class-level state)
  - Safe to use from multiple concurrent FastAPI requests

2025 Design:
  - Single responsibility per method
  - All inputs/outputs typed via Pydantic v2
  - File content hashed for content-addressed caching
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from edms_ai_assistant.config import settings
from edms_ai_assistant.summarizer.cache.cache import CacheEntry, TwoLevelCache
from edms_ai_assistant.summarizer.chunking.structural import SmartChunker
from edms_ai_assistant.summarizer.chunking.token_aware import count_tokens, estimate_cost
from edms_ai_assistant.summarizer.observability.tracing import (
    RequestCostAccumulator,
    Stopwatch,
    set_current_accumulator,
    trace_stage,
)
from edms_ai_assistant.summarizer.pipeline.direct import (
    DirectSummarizationPipeline,
    LLMClient,
    PipelineResult,
)
from edms_ai_assistant.summarizer.pipeline.map_reduce import MapReducePipeline
from edms_ai_assistant.summarizer.prompts.registry import PromptRegistry, get_prompt_registry
from edms_ai_assistant.summarizer.structured.models import (
    LLMBaseModel,
    QualityScore,
    SummaryMode,
)

logger = logging.getLogger(__name__)

# Context window threshold: above this → Map-Reduce
_DIRECT_CONTEXT_WINDOW_TOKENS = 4096


# ---------------------------------------------------------------------------
# Request / Response Models (Public API)
# ---------------------------------------------------------------------------


class SummarizationRequest(BaseModel):
    """Validated input for a summarization request."""

    model_config = {"frozen": True}

    file_content: bytes = Field(description="Raw file bytes (PDF, DOCX, TXT, etc.)")
    file_name: str = Field(description="Original filename for extension detection", max_length=255)
    mode: SummaryMode = Field(default=SummaryMode.ABSTRACTIVE)
    language: str = Field(
        default="ru",
        description="BCP-47 output language tag. Use 'auto' to match document language.",
        max_length=10,
    )
    request_id: str = Field(description="Unique request ID for tracing/idempotency")
    force_refresh: bool = Field(
        default=False,
        description="If True, bypass cache and re-generate",
    )


class SummarizationResponse(BaseModel):
    """Typed summarization result returned to API consumers."""

    model_config = {"frozen": True}

    request_id: str
    mode: SummaryMode
    language: str
    output: dict                  # Serialized structured output (mode-dependent)
    quality: QualityScore | None = None
    cache_hit: bool = False
    cache_source: str = "miss"    # "l1", "l2", or "miss"

    # Observability
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    chunking_strategy: str
    chunk_count: int

    # File metadata
    file_hash: str
    prompt_version: str


# ---------------------------------------------------------------------------
# Quality Scorer (runs in background, enriches cached entry)
# ---------------------------------------------------------------------------


async def _background_quality_score(
    result: PipelineResult,
    llm_client: LLMClient,
    model: str,
) -> QualityScore | None:
    """Score summarization quality via lightweight LLM self-critique.

    Runs as a background task — does NOT block the response.
    Score is stored back into cache if available.
    Uses a simple 0.0-1.0 scoring prompt (cheaper model recommended).
    """
    try:
        system = (
            "You are a quality evaluator. Rate the given summary on a scale from 0.0 to 1.0 "
            "based on: accuracy, completeness, and clarity. "
            "Respond ONLY with a JSON object: {\"score\": float, \"critique\": string}"
        )
        user = (
            f"Summary to evaluate:\n\n{result.raw_json[:1500]}\n\n"
            "Provide score (0.0-1.0) and 1-sentence critique."
        )
        async with asyncio.timeout(15.0):
            import json
            raw, _, _ = await llm_client.complete(
                system, user,
                model=model,
                temperature=0.0,
                max_tokens=150,
            )
            data = json.loads(raw)
            score = float(data.get("score", 0.5))
            critique = str(data.get("critique", ""))
            return QualityScore.from_score(min(1.0, max(0.0, score)), critique)
    except Exception as exc:
        logger.debug("Background quality scoring failed (non-critical): %s", exc)
        return None


# ---------------------------------------------------------------------------
# Text Extractor (async wrapper around sync extraction)
# ---------------------------------------------------------------------------


async def extract_text_from_bytes(
    file_content: bytes,
    file_name: str,
) -> str:
    """Async text extraction from file bytes.

    Dispatches to appropriate sync extractor based on extension,
    runs in a thread executor to avoid blocking the event loop.
    """
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    def _sync_extract() -> str:
        if ext == "pdf":
            return _extract_pdf(file_content)
        elif ext in ("docx", "doc"):
            return _extract_docx(file_content)
        elif ext in ("txt", "md", "rst", "csv"):
            return _decode_text(file_content)
        else:
            # Best-effort: try text decode
            return _decode_text(file_content)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_extract)


def _extract_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore[import]
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        logger.warning("PyMuPDF not installed — trying pdfplumber")
    try:
        import io
        import pdfplumber  # type: ignore[import]
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


def _extract_docx(data: bytes) -> str:
    """Extract text from DOCX bytes."""
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
        raise RuntimeError(f"DOCX extraction failed: {exc}") from exc


def _decode_text(data: bytes) -> str:
    """Decode raw bytes as UTF-8 with fallback encodings."""
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# SummarizationService
# ---------------------------------------------------------------------------


class SummarizationService:
    """Facade over the entire summarization pipeline.

    Instantiate once (e.g., via FastAPI lifespan) and inject via Depends().

    Args:
        llm_client: Async LLM client.
        cache: Two-level cache (Redis L1 + Postgres L2).
        model: Primary LLM model identifier.
        quality_model: Model for quality scoring (can be smaller/cheaper).
        direct_context_window: Token threshold for switching to Map-Reduce.
        max_concurrent_map: Global semaphore limit for Map-stage parallel calls.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cache: TwoLevelCache,
        *,
        model: str = settings.LLM_GENERATIVE_MODEL,
        quality_model: str | None = None,
        direct_context_window: int = _DIRECT_CONTEXT_WINDOW_TOKENS,
        max_concurrent_map: int = 6,
    ) -> None:
        self._llm = llm_client
        self._cache = cache
        self._model = model
        self._quality_model = quality_model or model
        self._context_window = direct_context_window
        self._prompts: PromptRegistry = get_prompt_registry()
        self._chunker = SmartChunker()
        self._direct_pipeline = DirectSummarizationPipeline(
            llm_client, self._prompts, model
        )
        self._map_reduce_pipeline = MapReducePipeline(
            llm_client,
            self._prompts,
            model,
            max_concurrent_map=max_concurrent_map,
        )

    async def summarize(self, request: SummarizationRequest) -> SummarizationResponse:
        """Execute full summarization pipeline with caching.

        This is the primary public method. Everything else is internal.
        """
        sw_total = Stopwatch()

        # Set up request-scoped cost accumulator (for OTel)
        acc = RequestCostAccumulator(
            request_id=request.request_id,
            model=self._model,
        )
        set_current_accumulator(acc)

        # Content-address the file
        file_hash = hashlib.sha256(request.file_content).hexdigest()

        # Resolve effective language
        language = request.language
        if language == "auto":
            # Will be resolved by multilingual mode, use 'auto' as-is
            language = "auto"

        # Build cache key
        cache_key = CacheEntry.build_key(
            file_hash=file_hash,
            mode=request.mode.value,
            language=language,
            prompt_version=self._prompts.cache_version_tag(),
        )

        # --- Cache lookup (skip if force_refresh) ---
        if not request.force_refresh:
            cached_entry, cache_source = await self._cache.get(cache_key)
            if cached_entry is not None:
                logger.info(
                    "Cache %s hit for request_id=%s key=%s",
                    cache_source.upper(), request.request_id, cache_key[:12],
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
            raise ValueError(f"No text could be extracted from '{request.file_name}'")

        doc_tokens = count_tokens(text)
        logger.info(
            "Summarizing: mode=%s lang=%s tokens=%d request_id=%s",
            request.mode.value, language, doc_tokens, request.request_id,
        )

        # --- Pipeline selection ---
        async with trace_stage("pipeline", {
            "mode": request.mode.value,
            "doc_tokens": doc_tokens,
            "model": self._model,
        }) as pipeline_span:
            needs_map_reduce = self._chunker.needs_map_reduce(
                text, context_window=self._context_window
            )

            if needs_map_reduce:
                logger.info(
                    "Routing to MapReducePipeline: %d tokens > %d threshold",
                    doc_tokens, int(self._context_window * 0.70),
                )
                pipeline_result = await self._map_reduce_pipeline.run(
                    text,
                    request.mode,
                    language=language,
                    span=pipeline_span,
                )
            else:
                logger.info(
                    "Routing to DirectPipeline: %d tokens fits context window",
                    doc_tokens,
                )
                pipeline_result = await self._direct_pipeline.run(
                    text,
                    request.mode,
                    language=language,
                    span=pipeline_span,
                )

        total_latency = sw_total.elapsed_ms()

        # --- Build response ---
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
                estimate_cost(pipeline_result.input_tokens, pipeline_result.output_tokens),
                6,
            ),
            latency_ms=round(total_latency, 1),
            chunking_strategy=pipeline_result.chunking_strategy,
            chunk_count=pipeline_result.chunk_count,
            file_hash=file_hash,
            prompt_version=self._prompts.cache_version_tag(),
        )

        # --- Cache write (fire-and-forget, non-blocking) ---
        asyncio.create_task(
            self._write_cache(response, pipeline_result, cache_key, file_hash)
        )

        # --- Background quality scoring ---
        asyncio.create_task(
            self._enrich_with_quality_score(response, pipeline_result)
        )

        logger.info(
            "Summarization complete: request_id=%s mode=%s tokens=%d/%d "
            "cost=$%.4f latency=%.0fms strategy=%s",
            request.request_id, request.mode.value,
            pipeline_result.input_tokens, pipeline_result.output_tokens,
            response.cost_usd, total_latency,
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
        """Write result to cache. Called as background task."""
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
            logger.debug("Cache write complete for key=%s", cache_key[:12])
        except Exception as exc:
            logger.error("Background cache write failed (non-critical): %s", exc)

    async def _enrich_with_quality_score(
        self,
        response: SummarizationResponse,
        result: PipelineResult,
    ) -> None:
        """Run quality scoring in background. Non-critical path."""
        score = await _background_quality_score(result, self._llm, self._quality_model)
        if score:
            logger.debug(
                "Quality score for request_id=%s: %.2f (%s)",
                response.request_id, score.score, score.confidence.value,
            )

    @staticmethod
    def _response_from_cache(
        entry: CacheEntry,
        source: str,
        request_id: str,
    ) -> SummarizationResponse:
        """Reconstruct SummarizationResponse from a cached CacheEntry."""
        import json
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
            latency_ms=0.0,  # Cache hit — no LLM latency
            chunking_strategy=entry.chunking_strategy,
            chunk_count=0,
            file_hash=entry.file_hash,
            prompt_version=entry.prompt_version,
        )

    # async def invalidate_cache(self, file_hash: str) -> None:
    #     """Invalidate all cached summaries for a given file (all modes/languages)."""
    #     await self._cache.invalidate_by_file(file_hash)

    async def invalidate_cache(self, file_hash: str, mode: str | None = None) -> None:
        """Invalidate cached summaries for a given file (optionally for a specific mode)."""
        await self._cache.invalidate_by_file(file_hash, mode=mode)

    def cache_stats(self) -> dict:
        """Return cache hit rate statistics."""
        return self._cache.stats()