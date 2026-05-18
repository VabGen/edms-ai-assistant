"""
Pipeline protocol & shared types.

Определяет минимальный интерфейс пайплайна суммаризации, общий для
DirectSummarizationPipeline и MapReducePipeline.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from edms_ai_assistant.summarizer.structured.models import SummaryMode


@runtime_checkable
class SummarizationPipeline(Protocol):
    """Базовый протокол пайплайна суммаризации."""

    async def run(
        self,
        text: str,
        mode: SummaryMode,
        *,
        language: str = "ru",
        span: Any = None,
    ) -> Any:
        """Выполняет пайплайн и возвращает PipelineResult."""
        ...


__all__ = ["SummarizationPipeline"]
