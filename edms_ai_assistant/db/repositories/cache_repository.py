# edms_ai_assistant/db/repositories/cache_repository.py
"""Repository for SummarizationCache DB operations.

Encapsulates all SQLAlchemy queries; routers receive only domain objects.
Handles the case where SummarizationCache may live in either:
  - edms_ai_assistant.db.database  (original monolith layout)
  - edms_ai_assistant.db.models.summarization_cache  (split layout)
Import the one that matches your project — only ONE import block needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import and_, delete as sa_delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

# ── Import the ORM model ──────────────────────────────────────────────────────
from edms_ai_assistant.db.generated.models.summarization_cache import SummarizationCache
# ─────────────────────────────────────────────────────────────────────────────

if TYPE_CHECKING:
    from sqlalchemy.orm import InstrumentedAttribute

logger = logging.getLogger(__name__)


class CacheRepository:
    """Async repository for SummarizationCache table.

    All SQLAlchemy query logic lives here.
    Routers inject this class via FastAPI Depends and never touch the session
    directly for cache operations.

    Args:
        session: SQLAlchemy async session (injected via FastAPI Depends).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Read ──────────────────────────────────────────────────────────────────

    async def list_all(self) -> list[SummarizationCache]:
        """Return all cache rows ordered by creation date descending.

        Returns:
            List of ORM objects; empty list when table is empty.
        """
        result = await self._session.scalars(
            select(SummarizationCache).order_by(
                SummarizationCache.created_at.desc()
            )
        )
        return list(result.all())

    async def find_one(
        self,
        cache_key: str,
        summary_type: str,
    ) -> SummarizationCache | None:
        """Look up a single cache entry by hashed key and summary type.

        Args:
            cache_key: SHA-256 hashed file identifier (as stored in DB).
            summary_type: One of extractive | abstractive | thesis.

        Returns:
            ORM row, or None if not found.
        """
        return await self._session.scalar(
            select(SummarizationCache).where(
                SummarizationCache.file_identifier == cache_key,
                SummarizationCache.summary_type == summary_type,
            )
        )

    async def find_all_for_file(
        self,
        cache_keys: dict[str, str],
    ) -> list[SummarizationCache]:
        """Return all cache entries matching any (summary_type, hashed_key) pair.

        Args:
            cache_keys: Mapping of ``summary_type → hashed_cache_key``.
                        Built by the caller for each valid summary type.

        Returns:
            All matching ORM rows; empty list when no matches found.
        """
        if not cache_keys:
            return []

        conditions = [
            and_(
                SummarizationCache.file_identifier == key,
                SummarizationCache.summary_type == stype,
            )
            for stype, key in cache_keys.items()
        ]

        result = await self._session.scalars(
            select(SummarizationCache).where(or_(*conditions))
        )
        return list(result.all())

    # ── Write ─────────────────────────────────────────────────────────────────

    async def delete_one(self, row: SummarizationCache) -> None:
        """Mark a single ORM row for deletion in the current session.

        The caller is responsible for calling ``session.commit()``
        to persist the deletion.

        Args:
            row: ORM instance that belongs to the current session.
        """
        await self._session.delete(row)

    async def delete_all(self) -> int:
        """Issue a bulk DELETE against the entire cache table.

        More efficient than loading rows and deleting one-by-one.
        The caller is responsible for committing.

        Returns:
            Number of rows deleted (may be 0 if table was already empty).
        """
        result = await self._session.execute(sa_delete(SummarizationCache))
        # rowcount is always an int for DELETE statements; ignore spurious
        # type-checker complaint about Optional[int].
        return result.rowcount  # type: ignore[return-value]