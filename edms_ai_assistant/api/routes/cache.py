# edms_ai_assistant/api/routes/cache.py
"""Cache management API router.

Endpoints:
    GET    /api/cache/summarization               — list all entries
    GET    /api/cache/summarization/{fid}/{type}  — check entry
    DELETE /api/cache/summarization/{fid}/{type}  — invalidate one
    DELETE /api/cache/summarization/{fid}         — invalidate all for file
    DELETE /api/cache/summarization               — clear entire cache
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from edms_ai_assistant.db.database import get_db
from edms_ai_assistant.db.repositories.cache_repository import CacheRepository
from edms_ai_assistant.schemas.cache import (
    BulkDeleteResult,
    CacheEntryBrief,
    CacheEntryDetail,
    ClearResult,
    DeleteResult,
)
from edms_ai_assistant.services.summarization_orchestrator import _make_cache_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cache", tags=["Cache"])

_VALID_SUMMARY_TYPES: frozenset[str] = frozenset({"extractive", "abstractive", "thesis"})


# ── Dependency ────────────────────────────────────────────────────────────────


def _get_repo(session: AsyncSession = Depends(get_db)) -> CacheRepository:
    return CacheRepository(session)


# ── Validation helper ─────────────────────────────────────────────────────────


def _validate_summary_type(summary_type: str) -> str:
    """Normalize and validate a summary_type path parameter.

    Args:
        summary_type: Raw path segment.

    Returns:
        Lowercase validated summary type.

    Raises:
        HTTPException 400: If value is not in the allowed set.
    """
    normalized = summary_type.strip().lower()
    if normalized not in _VALID_SUMMARY_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid summary_type '{summary_type}'. "
                f"Allowed: {', '.join(sorted(_VALID_SUMMARY_TYPES))}"
            ),
        )
    return normalized


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/summarization",
    response_model=list[CacheEntryBrief],
    summary="List all cached summarization entries",
)
async def list_cache(
    repo: CacheRepository = Depends(_get_repo),
) -> list[CacheEntryBrief]:
    """Return id-less brief list of all cache rows.

    Content field is excluded to keep response size manageable.
    """
    try:
        rows = await repo.list_all()
        return [
            CacheEntryBrief(
                file_identifier=row.file_identifier,
                summary_type=row.summary_type,
            )
            for row in rows
        ]
    except Exception as exc:
        logger.error("Cache list error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get(
    "/summarization/{file_identifier}/{summary_type}",
    response_model=CacheEntryDetail,
    summary="Check if a cache entry exists for given file + type",
)
async def get_cache_entry(
    file_identifier: str,
    summary_type: str,
    repo: CacheRepository = Depends(_get_repo),
) -> CacheEntryDetail:
    """Return metadata for a cache entry identified by the original file UUID.

    Args:
        file_identifier: Original attachment UUID or local file SHA-256.
        summary_type: extractive | abstractive | thesis.
    """
    stype = _validate_summary_type(summary_type)
    cache_key = _make_cache_key(file_identifier, stype)

    try:
        row = await repo.find_one(cache_key, stype)
        return CacheEntryDetail(
            exists=row is not None,
            file_identifier_hashed=cache_key,
            summary_type=stype,
        )
    except Exception as exc:
        logger.error("Cache GET error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.delete(
    "/summarization/{file_identifier}/{summary_type}",
    response_model=DeleteResult,
    summary="Invalidate one cache entry for given file + type",
)
async def delete_cache_entry(
    file_identifier: str,
    summary_type: str,
    session: AsyncSession = Depends(get_db),
    repo: CacheRepository = Depends(_get_repo),
) -> DeleteResult:
    """Delete one cache entry by original file UUID and summary type.

    Idempotent — returns 200 even when entry does not exist.

    Args:
        file_identifier: Original attachment UUID.
        summary_type: extractive | abstractive | thesis.
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
        row = await repo.find_one(cache_key, stype)
        if row is None:
            return DeleteResult(
                deleted=False,
                message="No cache entry found (already deleted or never existed).",
            )

        file_id_snapshot = row.file_identifier
        stype_snapshot = row.summary_type

        await repo.delete_one(row)
        await session.commit()

        return DeleteResult(
            deleted=True,
            file_identifier=file_id_snapshot,
            summary_type=stype_snapshot,
        )
    except Exception as exc:
        await session.rollback()
        logger.error("Cache DELETE error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.delete(
    "/summarization/{file_identifier}",
    response_model=BulkDeleteResult,
    summary="Invalidate ALL cache entries for a given file (all summary types)",
)
async def delete_all_cache_for_file(
    file_identifier: str,
    session: AsyncSession = Depends(get_db),
    repo: CacheRepository = Depends(_get_repo),
) -> BulkDeleteResult:
    """Delete all cache entries (all summary types) for one file.

    Args:
        file_identifier: Original attachment UUID.
    """
    cache_keys: dict[str, str] = {
        stype: _make_cache_key(file_identifier, stype)
        for stype in _VALID_SUMMARY_TYPES
    }

    try:
        rows = await repo.find_all_for_file(cache_keys)
        deleted_keys: list[str] = []

        for row in rows:
            deleted_keys.append(f"{row.file_identifier}:{row.summary_type}")
            await repo.delete_one(row)

        await session.commit()

        logger.info(
            "Cache DELETE ALL: file=%s... removed %d entries",
            file_identifier[:8],
            len(deleted_keys),
        )
        return BulkDeleteResult(
            deleted=bool(deleted_keys),
            count=len(deleted_keys),
            keys=deleted_keys,
        )
    except Exception as exc:
        await session.rollback()
        logger.error("Cache DELETE ALL error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.delete(
    "/summarization",
    response_model=ClearResult,
    summary="Clear the entire summarization cache",
)
async def clear_all_cache(
    session: AsyncSession = Depends(get_db),
    repo: CacheRepository = Depends(_get_repo),
) -> ClearResult:
    """Delete every row in summarization_cache. Use with caution."""
    try:
        count = await repo.delete_all()
        await session.commit()
        logger.warning("Cache CLEAR ALL: removed %d entries", count)
        return ClearResult(deleted=True, count=count)
    except Exception as exc:
        await session.rollback()
        logger.error("Cache CLEAR ALL error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc