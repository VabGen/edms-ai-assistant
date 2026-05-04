from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from edms_ai_assistant.db.database import get_db
from edms_ai_assistant.db.generated.models.summarization_cache import SummarizationCache
from edms_ai_assistant.services.summarization_orchestrator import _make_cache_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cache", tags=["Cache"])

_VALID_SUMMARY_TYPES = {"extractive", "abstractive", "thesis"}


def _validate_summary_type(summary_type: str) -> str:
    normalized = summary_type.strip().lower()
    if normalized not in _VALID_SUMMARY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid summary_type '{summary_type}'. Allowed: {', '.join(sorted(_VALID_SUMMARY_TYPES))}",
        )
    return normalized


@router.get("/summarization", summary="List all cached summarization entries")
async def list_cache(db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    try:
        rows = (await db.scalars(select(SummarizationCache))).all()
        return [
            {"file_identifier": r.file_identifier, "summary_type": r.summary_type}
            for r in rows
        ]
    except Exception as exc:
        logger.error("Cache list error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/summarization/{file_identifier}/{summary_type}",
    summary="Check if a cache entry exists",
)
async def get_cache_entry(
    file_identifier: str, summary_type: str, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    stype = _validate_summary_type(summary_type)
    cache_key = _make_cache_key(file_identifier, stype)
    try:
        row = await db.scalar(
            select(SummarizationCache).where(
                SummarizationCache.file_identifier == cache_key,
                SummarizationCache.summary_type == stype,
            )
        )
        if row:
            return {
                "exists": True,
                "file_identifier_hashed": cache_key,
                "summary_type": row.summary_type,
            }
        return {
            "exists": False,
            "file_identifier_hashed": cache_key,
            "summary_type": stype,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete(
    "/summarization/{file_identifier}/{summary_type}",
    summary="Invalidate one cache entry",
)
async def delete_cache_entry(
    file_identifier: str, summary_type: str, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    stype = _validate_summary_type(summary_type)
    cache_key = _make_cache_key(file_identifier, stype)
    try:
        row = await db.scalar(
            select(SummarizationCache).where(
                SummarizationCache.file_identifier == cache_key,
                SummarizationCache.summary_type == stype,
            )
        )
        if not row:
            return {"deleted": False, "message": "No cache entry found."}

        await db.delete(row)
        await db.commit()
        return {
            "deleted": True,
            "file_identifier": row.file_identifier,
            "summary_type": row.summary_type,
        }
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete(
    "/summarization/{file_identifier}",
    summary="Invalidate ALL cache entries for a given file",
)
async def delete_all_cache_for_file(
    file_identifier: str, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    deleted_keys = []
    try:
        for stype in _VALID_SUMMARY_TYPES:
            cache_key = _make_cache_key(file_identifier, stype)
            rows = (
                await db.scalars(
                    select(SummarizationCache).where(
                        SummarizationCache.file_identifier == cache_key,
                        SummarizationCache.summary_type == stype,
                    )
                )
            ).all()
            for row in rows:
                deleted_keys.append(f"{row.file_identifier}:{row.summary_type}")
                await db.delete(row)
        await db.commit()
        return {
            "deleted": len(deleted_keys) > 0,
            "count": len(deleted_keys),
            "keys": deleted_keys,
        }
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/summarization", summary="Clear the entire summarization cache")
async def clear_all_cache(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    try:
        result = await db.execute(delete(SummarizationCache))
        await db.commit()
        return {"deleted": True, "count": result.rowcount}
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
