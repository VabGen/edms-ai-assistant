# edms_ai_assistant/summarizer/api/schemas.py
"""
API request/response schemas for the Summarization v2 router.

These models are specific to the HTTP API layer. Core domain models
(SummarizationRequest, SummarizationResponse) live in summarizer/service.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SummarizeModeInfo(BaseModel):
    """Information about a single summarization mode."""
    mode: str
    description: str
    output_schema: dict
    use_case: str


class SummarizeModesResponse(BaseModel):
    """Response for GET /modes endpoint."""
    modes: list[SummarizeModeInfo]
    prompt_registry_version: str


class CacheInvalidationResponse(BaseModel):
    """Response for DELETE /cache/{file_hash} endpoint."""
    invalidated: bool
    file_hash: str
    message: str