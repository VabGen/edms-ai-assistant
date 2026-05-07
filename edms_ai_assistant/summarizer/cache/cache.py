# edms_ai_assistant/summarizer/cache/cache.py
"""
Two-level cache for summarization results.

L1: Redis (in-memory, sub-millisecond, TTL 1h)
L2: PostgreSQL (persistent, async via asyncpg, TTL 30 days)

Cache Key Strategy:
    SHA256(file_content_hash + "::" + mode + "::" + language + "::" + prompt_version)

This means:
  - Different modes produce different cache entries ✓
  - Prompt version bump invalidates ALL cache entries ✓
  - Same file re-uploaded gets cache hit ✓
  - Mutated file (different content) produces new cache entry ✓

2025 Best Practices:
  - All I/O is async (no sync blocking)
  - Cache writes do NOT block the response (fire-and-forget with timeout guard)
  - Graceful degradation: cache miss on any error, never raise to caller
  - Pydantic v2 (de)serialization for stored payloads
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache entry model
# ---------------------------------------------------------------------------


class CacheEntry(BaseModel):
    """Serializable cache entry stored in both Redis and Postgres."""

    model_config = {"frozen": True}

    cache_key: str
    mode: str
    language: str
    prompt_version: str
    result_json: str           # JSON-serialized SummarizationResult
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model_name: str
    chunking_strategy: str
    created_at_ms: int
    file_hash: str

    @classmethod
    def build_key(
        cls,
        file_hash: str,
        mode: str,
        language: str,
        prompt_version: str,
    ) -> str:
        """Build deterministic cache key from content + parameters."""
        raw = f"{file_hash}::{mode}::{language}::{prompt_version}"
        return "smz:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Abstract Cache Interface
# ---------------------------------------------------------------------------


class SummarizationCache(ABC):
    """Abstract cache interface for summarization results."""

    @abstractmethod
    async def get(self, cache_key: str) -> CacheEntry | None: ...

    @abstractmethod
    async def set(self, entry: CacheEntry, ttl_seconds: int = 3600) -> None: ...

    @abstractmethod
    async def delete(self, cache_key: str) -> None: ...

    @abstractmethod
    async def health_check(self) -> bool: ...


# ---------------------------------------------------------------------------
# Redis L1 Cache
# ---------------------------------------------------------------------------


class RedisL1Cache(SummarizationCache):
    """L1 cache backed by Redis. Sub-millisecond hot path.

    Uses the central RedisClient singleton from clients/redis_client.
    Falls back gracefully if Redis unavailable.
    """

    def __init__(self) -> None:
        pass

    async def get(self, cache_key: str) -> CacheEntry | None:
        try:
            from edms_ai_assistant.clients.redis_client import redis_client
            data = await redis_client.get_client().get(cache_key)
            if data is None:
                return None
            return CacheEntry.model_validate_json(data)
        except Exception as exc:
            logger.debug("Redis GET failed for key %s: %s", cache_key, exc)
            return None

    async def set(self, entry: CacheEntry, ttl_seconds: int = 3600) -> None:
        try:
            from edms_ai_assistant.clients.redis_client import redis_client
            await redis_client.get_client().setex(
                entry.cache_key,
                ttl_seconds,
                entry.model_dump_json(),
            )
        except Exception as exc:
            logger.debug("Redis SET failed: %s", exc)

    async def delete(self, cache_key: str) -> None:
        try:
            from edms_ai_assistant.clients.redis_client import redis_client
            await redis_client.get_client().delete(cache_key)
        except Exception as exc:
            logger.debug("Redis DELETE failed: %s", exc)

    async def health_check(self) -> bool:
        try:
            from edms_ai_assistant.clients.redis_client import redis_client
            await redis_client.get_client().ping()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# PostgreSQL L2 Cache
# ---------------------------------------------------------------------------


class PostgresL2Cache(SummarizationCache):
    """L2 persistent cache backed by PostgreSQL via SQLAlchemy async.

    Table: edms.summarization_cache
    Schema defined in alembic migration 001_create_summarization_cache.py
    """

    def __init__(self, session_factory: Any) -> None:
        """
        Args:
            session_factory: SQLAlchemy async_sessionmaker instance.
        """
        self._session_factory = session_factory

    async def get(self, cache_key: str) -> CacheEntry | None:
        from sqlalchemy import text
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    text(
                        "SELECT cache_entry_json FROM edms.summarization_cache "
                        "WHERE cache_key = :key AND expires_at > NOW()"
                    ),
                    {"key": cache_key},
                )
                row = result.fetchone()
                if row is None:
                    return None
                return CacheEntry.model_validate_json(row[0])
        except Exception as exc:
            logger.warning("Postgres L2 GET failed: %s", exc)
            return None

    async def set(self, entry: CacheEntry, ttl_seconds: int = 2_592_000) -> None:
        """Store entry with TTL (default: 30 days)."""
        from sqlalchemy import text
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    await session.execute(
                        text("""
                            INSERT INTO edms.summarization_cache
                                (cache_key, file_hash, mode, language, prompt_version,
                                 cache_entry_json, input_tokens, output_tokens,
                                 cost_usd, model_name, expires_at, created_at)
                            VALUES
                                (:key, :file_hash, :mode, :language, :pv,
                                 :entry_json, :input_t, :output_t,
                                 :cost, :model,
                                 NOW() + INTERVAL '1 second' * :ttl,
                                 NOW())
                            ON CONFLICT (cache_key) DO UPDATE SET
                                cache_entry_json = EXCLUDED.cache_entry_json,
                                expires_at = EXCLUDED.expires_at,
                                updated_at = NOW()
                        """),
                        {
                            "key": entry.cache_key,
                            "file_hash": entry.file_hash,
                            "mode": entry.mode,
                            "language": entry.language,
                            "pv": entry.prompt_version,
                            "entry_json": entry.model_dump_json(),
                            "input_t": entry.input_tokens,
                            "output_t": entry.output_tokens,
                            "cost": entry.cost_usd,
                            "model": entry.model_name,
                            "ttl": ttl_seconds,
                        },
                    )
        except Exception as exc:
            logger.error("Postgres L2 SET failed — cache write dropped: %s", exc)

    async def delete(self, cache_key: str) -> None:
        from sqlalchemy import text
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    await session.execute(
                        text("DELETE FROM edms.summarization_cache WHERE cache_key = :key"),
                        {"key": cache_key},
                    )
        except Exception as exc:
            logger.warning("Postgres L2 DELETE failed: %s", exc)

    async def health_check(self) -> bool:
        from sqlalchemy import text
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Two-Level Cache Facade
# ---------------------------------------------------------------------------


class TwoLevelCache:
    """Transparent two-level cache (Redis L1 + Postgres L2).

    Read path:  L1 → L2 → None
    Write path: L1 + L2 (concurrent)
    L1 miss + L2 hit: backfills L1 automatically
    """

    def __init__(
        self,
        l1: RedisL1Cache,
        l2: PostgresL2Cache,
        *,
        l1_ttl: int = 3600,
        l2_ttl: int = 2_592_000,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._l1_ttl = l1_ttl
        self._l2_ttl = l2_ttl
        self._hits_l1 = 0
        self._hits_l2 = 0
        self._misses = 0

    async def get(self, cache_key: str) -> tuple[CacheEntry | None, str]:
        """Get cached entry.

        Returns:
            (entry, source) where source is "l1", "l2", or "miss".
        """
        # Try L1
        entry = await self._l1.get(cache_key)
        if entry is not None:
            self._hits_l1 += 1
            return entry, "l1"

        # Try L2
        entry = await self._l2.get(cache_key)
        if entry is not None:
            self._hits_l2 += 1
            # Backfill L1
            await self._l1.set(entry, self._l1_ttl)
            return entry, "l2"

        self._misses += 1
        return None, "miss"

    async def set(self, entry: CacheEntry) -> None:
        """Write to both caches concurrently (best-effort — never raises)."""
        import asyncio
        try:
            async with asyncio.timeout(5.0):
                await asyncio.gather(
                    self._l1.set(entry, self._l1_ttl),
                    self._l2.set(entry, self._l2_ttl),
                    return_exceptions=True,
                )
        except TimeoutError:
            logger.warning("Cache write timed out after 5s")

    # async def invalidate_by_file(self, file_hash: str) -> None:
    #     """Invalidate all cache entries for a given file hash (across modes/languages)."""
    #     from sqlalchemy import text
    #     # Delete from Postgres by file_hash
    #     try:
    #         async with self._l2._session_factory() as session:
    #             async with session.begin():
    #                 result = await session.execute(
    #                     text(
    #                         "DELETE FROM edms.summarization_cache "
    #                         "WHERE file_hash = :fh RETURNING cache_key"
    #                     ),
    #                     {"fh": file_hash},
    #                 )
    #                 keys = [row[0] for row in result.fetchall()]
    #         # Delete matching keys from Redis
    #         import asyncio
    #         await asyncio.gather(
    #             *[self._l1.delete(k) for k in keys],
    #             return_exceptions=True,
    #         )
    #         logger.info("Invalidated %d cache entries for file_hash=%s", len(keys), file_hash[:8])
    #     except Exception as exc:
    #         logger.error("Cache invalidation failed: %s", exc)

    async def invalidate_by_file(self, file_hash: str, mode: str | None = None) -> None:
        """Invalidate cache entries for a given file hash, optionally filtered by mode."""
        from sqlalchemy import text
        try:
            async with self._l2._session_factory() as session:
                async with session.begin():
                    if mode:
                        result = await session.execute(
                            text(
                                "DELETE FROM edms.summarization_cache "
                                "WHERE file_hash = :fh AND mode = :mode RETURNING cache_key"
                            ),
                            {"fh": file_hash, "mode": mode},
                        )
                    else:
                        result = await session.execute(
                            text(
                                "DELETE FROM edms.summarization_cache "
                                "WHERE file_hash = :fh RETURNING cache_key"
                            ),
                            {"fh": file_hash},
                        )
                    keys = [row[0] for row in result.fetchall()]
            # Delete matching keys from Redis
            import asyncio
            await asyncio.gather(
                *[self._l1.delete(k) for k in keys],
                return_exceptions=True,
            )
            logger.info(
                "Invalidated %d cache entries for file_hash=%s mode=%s",
                len(keys), file_hash[:8], mode or "ALL"
            )
        except Exception as exc:
            logger.error("Cache invalidation failed: %s", exc)

    def stats(self) -> dict:
        total = self._hits_l1 + self._hits_l2 + self._misses
        return {
            "l1_hits": self._hits_l1,
            "l2_hits": self._hits_l2,
            "misses": self._misses,
            "hit_rate": round((self._hits_l1 + self._hits_l2) / max(total, 1), 3),
        }