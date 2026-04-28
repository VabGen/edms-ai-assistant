# edms_ai_assistant/schemas/summarization.py
"""Pydantic v2 schemas for summarization — shared between orchestrator and API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SummaryFormat(StrEnum):
    """Supported summarization output formats."""

    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


class SummarizationResult(BaseModel):
    """Result returned by SummarizationOrchestrator.summarize().

    Used both internally (orchestrator → tool → agent) and as a cache payload.
    """

    model_config = ConfigDict(populate_by_name=True)

    status: str = "success"
    content: str = ""
    format_used: SummaryFormat = SummaryFormat.EXTRACTIVE
    quality_score: float | None = Field(None, ge=0.0, le=1.0)
    confidence: str | None = None
    processing_time_ms: int = 0
    chunks_processed: int = 0
    was_truncated: bool = False
    text_length: int = 0
    pipeline: str = "direct"
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    from_cache: bool = False
    cache_key: str | None = None


class SummarizeRequest(BaseModel):
    """Request body for the streaming summarization endpoint."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    text: str = Field(..., min_length=30, max_length=50_000)
    summary_type: SummaryFormat = SummaryFormat.EXTRACTIVE
    file_identifier: str | None = Field(
        None,
        description="Stable file ID for cache keying (UUID or SHA-256).",
    )