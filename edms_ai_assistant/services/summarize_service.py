# edms_ai_assistant/services/summarize_service.py
"""
SummarizeService — handles summarization caching via Redis.

Responsibilities:
  1. Check Redis for cached result (document_id + attachment_id + summary_type)
  2. On cache HIT → return cached result + metadata to show «Обновить» button
  3. On forced_refresh=True → skip cache, re-run agent, save new result
  4. save_result() / invalidate() for cache lifecycle management

Frontend contract:
  Response includes metadata:
    {
      "cached": true/false,
      "summary_type": "thesis" | "extractive" | "abstractive",
      "summary_type_label": "Тезисы" | ...,
      "can_refresh": true,
      "attachment_id": "<uuid>" (optional)
    }
  Frontend shows «Обновить «Тезисы»» button when cached=True.
  Clicking it sends the same request with forced_refresh=True.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Redis TTL for summarization cache (24 hours)
_SUMMARY_CACHE_TTL: int = 86_400


class SummarizeService:
    """
    Two-tier summarization cache service.

    Tier 1: Redis (fast, in-memory, auto-expiry).
    Tier 2: PostgreSQL via AsyncSessionLocal (persistent, managed by main.py).

    This class handles only the Redis tier.
    PostgreSQL operations are handled directly in main.py to avoid circular imports.

    Works with any async Redis client (aioredis.Redis) or sync client
    (redis.Redis) that implements get / setex / delete.
    Also works with None (cache disabled).
    """

    # Display labels for summary types — used in frontend button labels
    _SUMMARY_TYPE_LABELS: dict[str, str] = {
        "thesis": "Тезисы",
        "extractive": "Краткое изложение",
        "abstractive": "Аннотация",
    }

    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis = redis_client

    @property
    def cache_available(self) -> bool:
        """True when a Redis client is configured."""
        return self._redis is not None

    # ── Cache key ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(
        document_id: str | None,
        attachment_id: str | None,
        file_name: str | None,
        summary_type: str,
    ) -> str:
        """
        Stable cache key: summary:{doc_id}:{att_id_or_name}:{type}

        Examples:
          summary:uuid-doc:uuid-att:thesis
          summary:no_doc:hash-of-local-file:extractive
        """
        doc_part = (document_id or "no_doc").strip()
        att_part = (attachment_id or file_name or "no_att").strip()
        return f"summary:{doc_part}:{att_part}:{summary_type.strip()}"

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_cached(
        self,
        document_id: str | None,
        attachment_id: str | None,
        file_name: str | None,
        summary_type: str,
    ) -> dict[str, Any] | None:
        """
        Return cached summary result dict or None.

        Returns the full result dict as stored by save_result().
        """
        if not self.cache_available or not summary_type:
            return None
        key = self._make_key(document_id, attachment_id, file_name, summary_type)
        try:
            raw = await self._safe_get(key)
            if raw:
                data = json.loads(raw)
                logger.info("Redis summary cache HIT: key=%s", key[:60])
                return data
        except Exception as exc:
            logger.warning("Redis GET error for key=%s: %s", key[:60], exc)
        return None

    async def save_result(
        self,
        document_id: str | None,
        attachment_id: str | None,
        file_name: str | None,
        summary_type: str,
        result: dict[str, Any],
    ) -> None:
        """
        Save agent result dict to Redis with TTL.

        Only caches successful results — errors are never cached.
        """
        if not self.cache_available or not summary_type:
            return
        if result.get("status") != "success":
            return
        key = self._make_key(document_id, attachment_id, file_name, summary_type)
        try:
            payload = json.dumps(result, ensure_ascii=False, default=str)
            await self._safe_setex(key, _SUMMARY_CACHE_TTL, payload)
            logger.info(
                "Redis summary cached: key=%s ttl=%ds", key[:60], _SUMMARY_CACHE_TTL
            )
        except Exception as exc:
            logger.warning("Redis SET error for key=%s: %s", key[:60], exc)

    async def invalidate(
        self,
        document_id: str | None,
        attachment_id: str | None,
        file_name: str | None,
        summary_type: str,
    ) -> None:
        """Remove a cached result (called before forced refresh)."""
        if not self.cache_available or not summary_type:
            return
        key = self._make_key(document_id, attachment_id, file_name, summary_type)
        try:
            await self._safe_delete(key)
            logger.info("Redis summary invalidated: key=%s", key[:60])
        except Exception as exc:
            logger.warning("Redis DELETE error for key=%s: %s", key[:60], exc)

    # ── Response builders ─────────────────────────────────────────────────────

    @classmethod
    def build_cached_response(
        cls,
        cached_data: dict[str, Any],
        summary_type: str,
        attachment_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a full response dict from cached data.

        Adds metadata so frontend renders the «Обновить» button.
        """
        metadata = dict(cached_data.get("metadata") or {})
        metadata.update({
            "cached": True,
            "summary_type": summary_type,
            "summary_type_label": cls._SUMMARY_TYPE_LABELS.get(summary_type, summary_type),
            "can_refresh": True,
        })
        if attachment_id:
            metadata["attachment_id"] = attachment_id

        return {
            "status": "success",
            "content": cached_data.get("content") or cached_data.get("response"),
            "message": cached_data.get("message"),
            "requires_reload": False,
            "navigate_url": None,
            "metadata": metadata,
        }

    @classmethod
    def enrich_fresh_response(
        cls,
        result: dict[str, Any],
        summary_type: str,
        attachment_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Add cache metadata to a fresh (non-cached) summarization result.

        Frontend will show «Обновить» button on subsequent requests.
        """
        metadata = dict(result.get("metadata") or {})
        metadata.update({
            "cached": False,
            "summary_type": summary_type,
            "summary_type_label": cls._SUMMARY_TYPE_LABELS.get(summary_type, summary_type),
            "can_refresh": True,
        })
        if attachment_id:
            metadata["attachment_id"] = attachment_id
        result["metadata"] = metadata
        return result

    # ── Redis helpers (handle both sync and async clients) ────────────────────

    async def _safe_get(self, key: str) -> bytes | str | None:
        """Transparently handle async (aioredis) and sync (redis) clients."""
        if not self._redis:
            return None
        try:
            result = self._redis.get(key)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[misc]
            return result
        except Exception as exc:
            logger.debug("Redis get() error: %s", exc)
            return None

    async def _safe_setex(self, key: str, ttl: int, value: str) -> None:
        try:
            result = self._redis.setex(key, ttl, value)
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]
        except Exception as exc:
            logger.debug("Redis setex() error: %s", exc)

    async def _safe_delete(self, key: str) -> None:
        try:
            result = self._redis.delete(key)
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]
        except Exception as exc:
            logger.debug("Redis delete() error: %s", exc)