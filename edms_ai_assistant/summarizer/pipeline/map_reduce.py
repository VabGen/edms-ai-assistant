"""
MapReducePipeline — иерархическая суммаризация для больших документов.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from edms_ai_assistant.summarizer.chunking.structural import SmartChunker, TextChunk
from edms_ai_assistant.summarizer.chunking.token_aware import count_tokens
from edms_ai_assistant.summarizer.errors import LLMTransportError, MapStageError
from edms_ai_assistant.summarizer.observability.tracing import (
    Stopwatch,
    record_llm_call,
    trace_stage,
)
from edms_ai_assistant.summarizer.pipeline.direct import (
    DirectSummarizationPipeline,
    LLMClient,
    PipelineResult,
    StreamEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from edms_ai_assistant.summarizer.prompts.registry import PromptRegistry
    from edms_ai_assistant.summarizer.structured.models import SummaryMode

logger = logging.getLogger(__name__)


@dataclass
class MapResult:
    chunk_index: int
    section_title: str | None
    summary_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str | None = None


class MapReducePipeline:
    """Hierarchical Map-Reduce pipeline for large documents."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_registry: PromptRegistry,
        model: str,
        *,
        max_chunk_tokens: int = 1500,
        overlap_tokens: int = 100,
        max_concurrent_map: int = 6,
        partial_summary_max_tokens: int = 512,
        max_output_tokens: int = 4096,
        direct_pipeline: DirectSummarizationPipeline | None = None,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_registry
        self._model = model
        self._max_chunk_tokens = max_chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._partial_max_tokens = partial_summary_max_tokens
        self._max_output_tokens = max_output_tokens
        self._chunker = SmartChunker()
        self._semaphore = asyncio.BoundedSemaphore(max_concurrent_map)
        self._max_concurrent = max_concurrent_map
        self._direct = direct_pipeline or DirectSummarizationPipeline(
            llm_client,
            prompt_registry,
            model,
            max_output_tokens=max_output_tokens,
        )

    def _get_semaphore(self) -> asyncio.BoundedSemaphore:
        return self._semaphore

    async def run(
        self,
        text: str,
        mode: SummaryMode,
        *,
        language: str = "ru",
        span: Any = None,
    ) -> PipelineResult:
        async with trace_stage("map_reduce.pipeline", {"mode": mode.value}):
            # --- Chunking ---
            async with trace_stage("map_reduce.chunking"):
                chunks, strategy = self._chunker.chunk(
                    text,
                    max_tokens=self._max_chunk_tokens,
                    overlap_tokens=self._overlap_tokens,
                )
                logger.info(
                    "MapReduce: %d chunks, strategy=%s, mode=%s",
                    len(chunks),
                    strategy,
                    mode.value,
                )

            async with trace_stage("map_reduce.map", {"chunk_count": len(chunks)}):
                map_tasks = [
                    self._map_single_chunk(chunk, mode, language=language)
                    for chunk in chunks
                ]
                raw_results = await asyncio.gather(*map_tasks, return_exceptions=True)

                map_results: list[MapResult] = []
                for i, result in enumerate(raw_results):
                    if isinstance(result, Exception):
                        logger.error("Map chunk %d failed: %s", i, result)
                        map_results.append(
                            MapResult(
                                chunk_index=i,
                                section_title=(
                                    chunks[i].section_title if i < len(chunks) else None
                                ),
                                summary_text="",
                                input_tokens=0,
                                output_tokens=0,
                                latency_ms=0,
                                error=str(result),
                            )
                        )
                    else:
                        map_results.append(result)

                successful = [r for r in map_results if not r.error]
                failed = [r for r in map_results if r.error]

                total_map_in = sum(r.input_tokens for r in successful)
                total_map_out = sum(r.output_tokens for r in successful)

                if span:
                    record_llm_call(
                        span,
                        "map",
                        self._model,
                        total_map_in,
                        total_map_out,
                        sum(r.latency_ms for r in successful),
                    )

                if not successful:
                    raise MapStageError(
                        f"Все {len(failed)} чанков Map-стадии завершились с ошибкой. "
                        f"Первая ошибка: {failed[0].error if failed else 'unknown'}"
                    )

                if failed:
                    logger.warning(
                        "%d/%d Map чанков с ошибкой — продолжаем с %d успешными",
                        len(failed),
                        len(chunks),
                        len(successful),
                    )

            # --- Reduce Stage ---
            async with trace_stage("map_reduce.reduce"):
                sw = Stopwatch()
                result = await self._reduce_stage(successful, mode, language=language)
                latency = sw.elapsed_ms()

                if span:
                    record_llm_call(
                        span,
                        "reduce",
                        self._model,
                        result.input_tokens,
                        result.output_tokens,
                        latency,
                    )

            result.chunking_strategy = f"map_reduce:{strategy}"
            result.chunk_count = len(chunks)
            result.input_tokens = total_map_in + result.input_tokens
            result.output_tokens = total_map_out + result.output_tokens
            return result

    async def _map_single_chunk(
        self,
        chunk: TextChunk,
        mode: SummaryMode,
        *,
        language: str,
    ) -> MapResult:
        sw = Stopwatch()

        chunk_text = chunk.text
        if chunk.section_title:
            chunk_text = f"[Раздел: {chunk.section_title}]\n\n{chunk_text}"

        template = self._prompts.get_map(mode)
        system, user = template.render(chunk_text, language=language)

        async with self._get_semaphore():
            try:
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
                    "Map chunk %d: %d tokens -> %d output in %.0fms",
                    chunk.index,
                    chunk.token_count,
                    out_t,
                    latency,
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
                logger.error(
                    "Map chunk %d failed after %.0fms: %s", chunk.index, latency, exc
                )
                return MapResult(
                    chunk_index=chunk.index,
                    section_title=chunk.section_title,
                    summary_text="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency,
                    error=str(exc),
                )

    async def run_stream(
        self,
        text: str,
        mode: SummaryMode,
        *,
        language: str = "ru",
        span: Any = None,
    ) -> AsyncIterator[StreamEvent | PipelineResult]:
        """
        Стрим-версия map-reduce: map выполняется параллельно (нестримящий),
        а reduce — стримится по токенам. Это компромисс: нет смысла стримить
        N параллельных map-вызовов, зато финальный reduce можно показать
        пользователю по мере генерации.
        """
        async with trace_stage("map_reduce.stream", {"mode": mode.value}):
            chunks, strategy = self._chunker.chunk(
                text,
                max_tokens=self._max_chunk_tokens,
                overlap_tokens=self._overlap_tokens,
            )
            logger.info(
                "MapReduce stream: %d chunks, strategy=%s, mode=%s",
                len(chunks),
                strategy,
                mode.value,
            )

            # --- Map (параллельно, без стрима) ---
            async with trace_stage("map_reduce.map", {"chunk_count": len(chunks)}):
                raw_results = await asyncio.gather(
                    *[
                        self._map_single_chunk(c, mode, language=language)
                        for c in chunks
                    ],
                    return_exceptions=True,
                )
                map_results: list[MapResult] = []
                for i, result in enumerate(raw_results):
                    if isinstance(result, Exception):
                        logger.error("Map chunk %d failed: %s", i, result)
                        continue
                    map_results.append(result)
                successful = [r for r in map_results if not r.error]
                if not successful:
                    raise MapStageError("Все Map-чанки упали — стриминг невозможен")

                total_map_in = sum(r.input_tokens for r in successful)
                total_map_out = sum(r.output_tokens for r in successful)

            # --- Reduce (стримится) ---
            combined_text = self._build_combined_text(successful)

            template = self._prompts.get_reduce(mode)
            system, user = template.render(combined_text, language=language)
            if language == "ru":
                system = (
                    system
                    + "\n\nВАЖНО: Все значения в JSON должны быть на русском языке."
                )

            sw = Stopwatch()
            accumulated = ""
            in_t = 0
            out_t = 0

            async for event in self._llm.complete_stream(
                system,
                user,
                model=self._model,
                temperature=0.1,
                max_tokens=self._max_output_tokens,
            ):
                if event.kind == "delta":
                    accumulated += event.text
                    yield event
                elif event.kind == "done":
                    in_t = event.input_tokens
                    out_t = event.output_tokens
                elif event.kind == "error":
                    raise LLMTransportError(f"Reduce stream failed: {event.error}")

            latency = sw.elapsed_ms()
            if span:
                record_llm_call(
                    span, "map_reduce.reduce.stream", self._model, in_t, out_t, latency
                )

            output = self._direct.validate_output(mode, accumulated)

            yield PipelineResult(
                mode=mode,
                output=output,
                raw_json=accumulated,
                input_tokens=total_map_in + in_t,
                output_tokens=total_map_out + out_t,
                latency_ms=latency,
                model=self._model,
                chunking_strategy=f"map_reduce.stream:{strategy}",
                chunk_count=len(chunks),
            )

    @staticmethod
    def _build_combined_text(map_results: list[MapResult]) -> str:
        """Сортирует частичные результаты и объединяет их в один текст."""
        sorted_results = sorted(map_results, key=lambda r: r.chunk_index)
        parts: list[str] = []
        for r in sorted_results:
            if not r.summary_text:
                continue
            label = f"[{r.section_title}]\n" if r.section_title else ""
            parts.append(f"{label}{r.summary_text}")
        return "\n\n---\n\n".join(parts)

    async def _reduce_stage(
        self,
        map_results: list[MapResult],
        mode: SummaryMode,
        *,
        language: str,
    ) -> PipelineResult:
        """Объединяем частичные изложения в финальный структурированный вывод."""
        combined_text = self._build_combined_text(map_results)
        combined_tokens = count_tokens(combined_text)
        logger.info(
            "Reduce: объединяем %d частичных изложений (%d токенов)",
            len(map_results),
            combined_tokens,
        )

        template = self._prompts.get_reduce(mode)
        system, user = template.render(combined_text, language=language)

        sw = Stopwatch()
        raw_text, in_t, out_t = await self._llm.complete(
            system,
            user,
            model=self._model,
            temperature=0.1,
            max_tokens=self._max_output_tokens,
        )
        latency = sw.elapsed_ms()

        output = self._direct.validate_output(mode, raw_text)

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
