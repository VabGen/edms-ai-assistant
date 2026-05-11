"""
EDMS Document Summarization package.

Public API:
    SummarizationService          — фасад поверх пайплайна.
    SummarizationRequest/Response — типизированные DTO.
    SummaryMode                   — режимы суммаризации.
    format_output_as_markdown     — каноничный форматтер вывода в markdown.
    build_summarization_service   — фабрика сервиса из объекта settings.

Использование:
    from edms_ai_assistant.summarizer import (
        build_summarization_service,
        SummarizationRequest,
        SummaryMode,
        format_output_as_markdown,
    )
    service = await build_summarization_service(settings)
    resp = await service.summarize(SummarizationRequest(...))
    text = format_output_as_markdown(resp)
"""

from __future__ import annotations

from edms_ai_assistant.summarizer.container import build_summarization_service
from edms_ai_assistant.summarizer.service import (
    SummarizationRequest,
    SummarizationResponse,
    SummarizationService,
    format_output_as_markdown,
)
from edms_ai_assistant.summarizer.structured.models import SummaryMode

__all__ = [
    "SummarizationRequest",
    "SummarizationResponse",
    "SummarizationService",
    "SummaryMode",
    "build_summarization_service",
    "format_output_as_markdown",
]
