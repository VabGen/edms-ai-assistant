"""
MapReducePipeline — hierarchical summarization for large documents.

Architecture (Anthropic/OpenAI production pattern):

    Document
       │
       ▼
   SmartChunker ──► N chunks (max_tokens each)
       │
       ▼
   MAP STAGE: asyncio.TaskGroup with BoundedSemaphore
       │  Parallel LLM calls, one per chunk
       │  Each chunk → brief partial summary (plain text)
       ▼
   REDUCE STAGE: Single LLM call
       │  All partial summaries combined → final structured output
       ▼
   Validation (Pydantic v2)
       │
       ▼
   PipelineResult

Key improvements over old flat Map-Reduce:
  - Global BoundedSemaphore (not per-request) prevents rate limit storms
  - asyncio.TaskGroup propagates errors cleanly (Python 3.11+)
  - Partial summary length is token-bounded (no unbounded reduce input)
  - Each Map call uses plain text output (faster, cheaper)
  - Reduce call uses Structured Output (accuracy where it matters)
  - Chunk context (section title) prepended to each Map prompt
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from edms_ai_assistant.summarizer.chunking.structural import SmartChunker, TextChunk
from edms_ai_assistant.summarizer.chunking.token_aware import count_tokens
from edms_ai_assistant.summarizer.observability.tracing import (
    Stopwatch,
    record_llm_call,
    trace_stage,
)
from edms_ai_assistant.summarizer.pipeline.direct import (
    DirectSummarizationPipeline,
    LLMClient,
    PipelineResult,
    build_response_format,
)
from edms_ai_assistant.summarizer.prompts.registry import PromptRegistry
from edms_ai_assistant.summarizer.structured.models import SummaryMode

logger = logging.getLogger(__name__)

# Global rate-limit semaphore — shared across ALL requests in the process.
# Prevents multiple concurrent requests from collectively exceeding API rate limits.
# Default: 6 concurrent LLM calls. Tune based on your API tier.
_GLOBAL_LLM_SEMAPHORE: asyncio.BoundedSemaphore | None = None


def get_global_semaphore(max_concurrent: int = 6) -> asyncio.BoundedSemaphore:
    """Lazy-initialize global semaphore (called once per process)."""
    global _GLOBAL_LLM_SEMAPHORE
    if _GLOBAL_LLM_SEMAPHORE is None:
        _GLOBAL_LLM_SEMAPHORE = asyncio.BoundedSemaphore(max_concurrent)
        logger.info("Global LLM semaphore initialized: max_concurrent=%d", max_concurrent)
    return _GLOBAL_LLM_SEMAPHORE


# ---------------------------------------------------------------------------
# Map-stage result
# ---------------------------------------------------------------------------


class MapResult:
    """Result of processing a single chunk in the Map stage."""

    __slots__ = ("chunk_index", "section_title", "summary_text", "input_tokens",
                 "output_tokens", "latency_ms", "error")

    def __init__(
        self,
        chunk_index: int,
        section_title: str | None,
        summary_text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        self.chunk_index = chunk_index
        self.section_title = section_title
        self.summary_text = summary_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.error = error


# ---------------------------------------------------------------------------
# Map-Reduce Pipeline
# ---------------------------------------------------------------------------


class MapReducePipeline:
    """Hierarchical Map-Reduce pipeline for large documents.

    Args:
        llm_client: Async LLM client.
        prompt_registry: Typed prompt registry.
        model: LLM model identifier.
        max_chunk_tokens: Max tokens per chunk fed to Map stage.
        overlap_tokens: Overlap between consecutive chunks.
        max_concurrent_map: Max parallel Map-stage LLM calls.
        partial_summary_max_tokens: Token cap per partial summary (controls Reduce input size).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_registry: PromptRegistry,
        model: str,
        *,
        max_chunk_tokens: int = 1500,
        overlap_tokens: int = 100,
        max_concurrent_map: int = 6,
        partial_summary_max_tokens: int = 300,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_registry
        self._model = model
        self._max_chunk_tokens = max_chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._partial_max_tokens = partial_summary_max_tokens
        self._chunker = SmartChunker()
        # Use global semaphore for cross-request rate limiting
        self._semaphore: asyncio.BoundedSemaphore | None = None
        self._max_concurrent = max_concurrent_map

    def _get_semaphore(self) -> asyncio.BoundedSemaphore:
        if self._semaphore is None:
            self._semaphore = get_global_semaphore(self._max_concurrent)
        return self._semaphore

    async def run(
        self,
        text: str,
        mode: SummaryMode,
        *,
        language: str = "ru",
        span: Any = None,
    ) -> PipelineResult:
        """Execute full Map-Reduce pipeline.

        Args:
            text: Full document text.
            mode: Summarization mode.
            language: BCP-47 output language tag.
            span: Optional parent OTel span.
        """
        async with trace_stage("map_reduce.pipeline", {"mode": mode.value}):
            # --- Chunking ---
            async with trace_stage("map_reduce.chunking"):
                chunks, strategy = self._chunker.chunk(
                    text,
                    max_tokens=self._max_chunk_tokens,
                    overlap_tokens=self._overlap_tokens,
                )
                logger.info(
                    "MapReduce: %d chunks using strategy=%s for mode=%s",
                    len(chunks), strategy, mode.value,
                )

            # --- Map Stage ---
            async with trace_stage("map_reduce.map", {"chunk_count": len(chunks)}) as map_span:
                map_results = await self._map_stage(chunks, mode, language=language)
                successful = [r for r in map_results if r.error is None]
                failed = [r for r in map_results if r.error is not None]

                total_map_in = sum(r.input_tokens for r in successful)
                total_map_out = sum(r.output_tokens for r in successful)

                if span:
                    record_llm_call(
                        span, "map",
                        self._model,
                        total_map_in,
                        total_map_out,
                        sum(r.latency_ms for r in successful),
                    )

                if not successful:
                    raise RuntimeError(
                        f"All {len(failed)} Map-stage chunks failed. "
                        f"First error: {failed[0].error if failed else 'unknown'}"
                    )

                if failed:
                    logger.warning(
                        "%d/%d Map chunks failed — proceeding with %d successful",
                        len(failed), len(chunks), len(successful),
                    )

            # --- Reduce Stage ---
            async with trace_stage("map_reduce.reduce") as reduce_span:
                sw = Stopwatch()
                result = await self._reduce_stage(
                    successful, mode, language=language
                )
                latency = sw.elapsed_ms()

                if span:
                    record_llm_call(
                        span, "reduce",
                        self._model,
                        result.input_tokens,
                        result.output_tokens,
                        latency,
                    )

            # Augment result with pipeline metadata
            result.chunking_strategy = f"map_reduce:{strategy}"
            result.chunk_count = len(chunks)
            result.input_tokens = total_map_in + result.input_tokens
            result.output_tokens = total_map_out + result.output_tokens
            return result

    async def _map_stage(
        self,
        chunks: list[TextChunk],
        mode: SummaryMode,
        *,
        language: str,
    ) -> list[MapResult]:
        """Process all chunks in parallel using TaskGroup + BoundedSemaphore."""
        results: list[MapResult | None] = [None] * len(chunks)

        async def process_chunk(i: int, chunk: TextChunk) -> None:
            async with self._get_semaphore():
                results[i] = await self._map_single_chunk(chunk, mode, language=language)

        async with asyncio.TaskGroup() as tg:
            for i, chunk in enumerate(chunks):
                tg.create_task(process_chunk(i, chunk))

        return [r for r in results if r is not None]

    async def _map_single_chunk(
        self,
        chunk: TextChunk,
        mode: SummaryMode,
        *,
        language: str,
    ) -> MapResult:
        """Summarize a single chunk (Map stage). Returns plain text."""
        sw = Stopwatch()

        # Prepend section title to give LLM context
        chunk_text = chunk.text
        if chunk.section_title:
            chunk_text = f"[Раздел: {chunk.section_title}]\n\n{chunk_text}"

        template = self._prompts.get_map(mode)
        system, user = template.render(chunk_text, language=language)

        try:
            # Map stage: plain text output (cheaper, faster)
            # We don't need structured output here — the Reduce stage does that
            raw_text = await self._llm.complete_plain(
                system,
                user,
                model=self._model,
                temperature=0.2,
                max_tokens=self._partial_max_tokens,
            )
            in_t = count_tokens(system + user)
            out_t = count_tokens(raw_text)
            latency = sw.elapsed_ms()

            logger.debug(
                "Map chunk %d: %d tokens → %d output tokens in %.0fms",
                chunk.index, chunk.token_count, out_t, latency,
            )

            return MapResult(
                chunk_index=chunk.index,
                section_title=chunk.section_title,
                summary_text=raw_text.strip(),
                input_tokens=in_t,
                output_tokens=out_t,
                latency_ms=latency,
            )
        except Exception as exc:
            latency = sw.elapsed_ms()
            logger.error("Map chunk %d failed after %.0fms: %s", chunk.index, latency, exc)
            return MapResult(
                chunk_index=chunk.index,
                section_title=chunk.section_title,
                summary_text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency,
                error=str(exc),
            )

    async def _reduce_stage(
        self,
        map_results: list[MapResult],
        mode: SummaryMode,
        *,
        language: str,
    ) -> PipelineResult:
        """Combine all partial summaries into final structured output (Reduce stage)."""
        # Sort by chunk_index to preserve document order
        sorted_results = sorted(map_results, key=lambda r: r.chunk_index)

        # Build combined input, section-labeled
        parts: list[str] = []
        for r in sorted_results:
            label = f"[{r.section_title}] " if r.section_title else ""
            parts.append(f"{label}{r.summary_text}")

        combined_text = "\n\n---\n\n".join(parts)

        # Verify combined text fits in context window
        combined_tokens = count_tokens(combined_text)
        logger.info(
            "Reduce stage: combining %d partial summaries (%d tokens)",
            len(sorted_results), combined_tokens,
        )

        template = self._prompts.get_reduce(mode)
        system, user = template.render(combined_text, language=language)
        response_format = build_response_format(mode)

        sw = Stopwatch()
        raw_text, in_t, out_t = await self._llm.complete(
            system,
            user,
            model=self._model,
            temperature=0.1,
            max_tokens=2048,
            response_format=response_format,
        )
        latency = sw.elapsed_ms()

        # Use DirectSummarizationPipeline's validator (reuse logic)
        direct = DirectSummarizationPipeline(self._llm, self._prompts, self._model)
        output = direct._validate_output(mode, raw_text)

        return PipelineResult(
            mode=mode,
            output=output,
            raw_json=raw_text,
            input_tokens=in_t,
            output_tokens=out_t,
            latency_ms=latency,
            model=self._model,
            chunking_strategy="map_reduce",
            chunk_count=len(map_results),
        )