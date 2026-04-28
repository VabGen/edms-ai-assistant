# edms_ai_assistant/schemas/cache.py
"""Pydantic v2 response schemas for cache API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CacheEntryBrief(BaseModel):
    """Lightweight cache entry for list responses."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    file_identifier: str
    summary_type: str


class CacheEntryDetail(BaseModel):
    """Full cache entry with existence flag."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    exists: bool
    file_identifier_hashed: str
    summary_type: str


class DeleteResult(BaseModel):
    """Result of a single-entry delete operation."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    deleted: bool
    message: str | None = None
    file_identifier: str | None = None
    summary_type: str | None = None


class BulkDeleteResult(BaseModel):
    """Result of a bulk delete operation."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    deleted: bool
    count: int
    keys: list[str] = []


class ClearResult(BaseModel):
    """Result of clearing the entire cache."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    deleted: bool
    count: int