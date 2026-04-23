# edms_ai_assistant/api/routes/cache.py
"""
EDMS AI Assistant — Cache Management Router.

Endpoints for listing, invalidating, and inspecting the summarization cache.

KEY DESIGN:
  The DB stores file_identifier as sha256(original_uuid::type::PROMPT_VERSION)[:48]
  All DELETE/GET endpoints receive the *original* UUID from the client and must
  compute the same hash before querying — they never store/query the raw UUID.

  _make_cache_key() is re-exported from summarization_orchestrator so both
  modules always use identical hashing logic.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import delete, select

from edms_ai_assistant.db.database import AsyncSessionLocal, SummarizationCache
from edms_ai_assistant.services.summarization_orchestrator import _make_cache_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cache", tags=["Cache"])

_VALID_SUMMARY_TYPES = {"extractive", "abstractive", "thesis"}


# ─── helpers ──────────────────────────────────────────────────────────────────


def _validate_summary_type(summary_type: str) -> str:
    """Normalize and validate summary_type path parameter.

    Args:
        summary_type: Raw path segment from the URL.

    Returns:
        Lowercase validated summary type string.

    Raises:
        HTTPException 400: If value is not in the allowed set.
    """
    normalized = summary_type.strip().lower()
    if normalized not in _VALID_SUMMARY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid summary_type '{summary_type}'. "
                f"Allowed: {', '.join(sorted(_VALID_SUMMARY_TYPES))}"
            ),
        )
    return normalized


# ─── endpoints ────────────────────────────────────────────────────────────────


@router.get(
    "/summarization",
    summary="List all cached summarization entries",
    response_model=list[dict[str, Any]],
)
async def list_cache() -> list[dict[str, Any]]:
    """Return all rows from summarization_cache (id, file_identifier, summary_type).

    Content field is excluded to keep response size manageable.
    """
    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.scalars(select(SummarizationCache))).all()
        return [
            {
                "id": str(r.id),
                "file_identifier": r.file_identifier,
                "summary_type": r.summary_type,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("Cache list error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/summarization/{file_identifier}/{summary_type}",
    summary="Check if a cache entry exists for given file + type",
)
async def get_cache_entry(
    file_identifier: str,
    summary_type: str,
) -> dict[str, Any]:
    """Return metadata for a cache entry identified by the original file UUID.

    The endpoint computes the internal hash key from the provided UUID so
    callers never need to know about the internal hashing scheme.

    Args:
        file_identifier: Original attachment UUID or local file SHA-256.
        summary_type: extractive | abstractive | thesis.

    Returns:
        Dict with exists=True/False and entry metadata.
    """
    stype = _validate_summary_type(summary_type)
    # Compute the same hash the orchestrator uses when saving
    cache_key = _make_cache_key(file_identifier, stype)

    try:
        async with AsyncSessionLocal() as db:
            row = await db.scalar(
                select(SummarizationCache).where(
                    SummarizationCache.file_identifier == cache_key,
                    SummarizationCache.summary_type == stype,
                )
            )
        if row:
            return {
                "exists": True,
                "id": str(row.id),
                "file_identifier_original": file_identifier,
                "file_identifier_hashed": cache_key,
                "summary_type": row.summary_type,
            }
        return {
            "exists": False,
            "file_identifier_original": file_identifier,
            "file_identifier_hashed": cache_key,
            "summary_type": stype,
        }
    except Exception as exc:
        logger.error("Cache GET error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete(
    "/summarization/{file_identifier}/{summary_type}",
    summary="Invalidate one cache entry for given file + type",
    status_code=200,
)
async def delete_cache_entry(
    file_identifier: str,
    summary_type: str,
) -> dict[str, Any]:
    """Delete one cache entry by original file UUID and summary type.

    ROOT FIX: The client sends the raw UUID; this endpoint hashes it
    exactly as _make_cache_key() does before issuing the DELETE, so the
    key always matches what was stored by the orchestrator.

    Previously the router queried by raw UUID → row never found → 404.

    Args:
        file_identifier: Original attachment UUID or file hash.
        summary_type: extractive | abstractive | thesis.

    Returns:
        Dict with deleted=True and matched row id, or deleted=False if not found.
    """
    stype = _validate_summary_type(summary_type)
    cache_key = _make_cache_key(file_identifier, stype)

    logger.info(
        "Cache DELETE: original=%s... hashed=%s... type=%s",
        file_identifier[:8],
        cache_key[:8],
        stype,
    )

    try:
        async with AsyncSessionLocal() as db, db.begin():
            # First check the row exists so we can return a meaningful response
            row = await db.scalar(
                select(SummarizationCache).where(
                    SummarizationCache.file_identifier == cache_key,
                    SummarizationCache.summary_type == stype,
                )
            )
            if not row:
                logger.warning(
                    "Cache DELETE: no entry found for key=%s... type=%s",
                    cache_key[:8],
                    stype,
                )
                # Return 200 (idempotent delete) — not 404
                return {
                    "deleted": False,
                    "message": "No cache entry found (already deleted or never existed).",
                    "file_identifier_original": file_identifier,
                    "file_identifier_hashed": cache_key,
                    "summary_type": stype,
                }

            row_id = str(row.id)
            await db.execute(
                delete(SummarizationCache).where(
                    SummarizationCache.file_identifier == cache_key,
                    SummarizationCache.summary_type == stype,
                )
            )

        logger.info(
            "Cache DELETE: removed id=%s key=%s... type=%s",
            row_id,
            cache_key[:8],
            stype,
        )
        return {
            "deleted": True,
            "id": row_id,
            "file_identifier_original": file_identifier,
            "file_identifier_hashed": cache_key,
            "summary_type": stype,
        }
    except Exception as exc:
        logger.error("Cache DELETE error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete(
    "/summarization/{file_identifier}",
    summary="Invalidate ALL cache entries for a given file (all summary types)",
    status_code=200,
)
async def delete_all_cache_for_file(file_identifier: str) -> dict[str, Any]:
    """Delete all cache entries (all summary types) for one file.

    Useful when a file is updated in EDMS and all cached summaries become stale.

    Args:
        file_identifier: Original attachment UUID.

    Returns:
        Dict with count of deleted rows.
    """
    deleted_ids: list[str] = []

    try:
        async with AsyncSessionLocal() as db, db.begin():
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
                    deleted_ids.append(str(row.id))
                    await db.delete(row)

        logger.info(
            "Cache DELETE ALL: file=%s... removed %d entries",
            file_identifier[:8],
            len(deleted_ids),
        )
        return {
            "deleted": len(deleted_ids) > 0,
            "count": len(deleted_ids),
            "ids": deleted_ids,
            "file_identifier_original": file_identifier,
        }
    except Exception as exc:
        logger.error("Cache DELETE ALL error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete(
    "/summarization",
    summary="Clear the entire summarization cache",
    status_code=200,
)
async def clear_all_cache() -> dict[str, Any]:
    """Delete every row in summarization_cache. Use with caution.

    Returns:
        Dict with count of deleted rows.
    """
    try:
        async with AsyncSessionLocal() as db, db.begin():
            result = await db.execute(delete(SummarizationCache))
            count = result.rowcount

        logger.warning("Cache CLEAR ALL: removed %d entries", count)
        return {"deleted": True, "count": count}
    except Exception as exc:
        logger.error("Cache CLEAR ALL error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
