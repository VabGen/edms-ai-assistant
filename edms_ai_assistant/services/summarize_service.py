# edms_ai_assistant/services/summarize_service.py
"""
SummarizeService — handles /actions/summarize endpoint logic.

Responsibilities:
  1. Check Redis for cached result (document_id + attachment_id + summary_type)
  2. On cache HIT → return cached result + metadata to show "Обновить" button
  3. On cache MISS → trigger agent.chat() with the summarize intent
  4. On forced_refresh=True → skip cache, re-run agent, save new result

Frontend contract:
  Response includes metadata:
    {
      "cached": true/false,
      "summary_type": "thesis" / "extractive" / "abstractive",
      "can_refresh": true,
      "attachment_id": "<uuid>" (optional)
    }
  Frontend shows "Обновить «Тезисы»" button when cached=True.
  Clicking it sends the same request with forced_refresh=True.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Redis TTL for summarization cache (24 hours)
_SUMMARY_CACHE_TTL = 86400

# Summary type display labels
_SUMMARY_TYPE_LABELS: dict[str, str] = {
    "thesis": "Тезисы",
    "extractive": "Краткое изложение",
    "abstractive": "Аннотация",
}


def _make_cache_key(
    document_id: str | None,
    attachment_id: str | None,
    file_name: str | None,
    summary_type: str,
) -> str:
    """
    Cache key: summary:{document_id}:{attachment_id or file_name}:{summary_type}
    """
    doc_part = document_id or "no_doc"
    att_part = attachment_id or file_name or "no_att"
    return f"summary:{doc_part}:{att_part}:{summary_type}"


class SummarizeService:
    """
    Handles caching and retrieval of summarization results.

    Works with any Redis client that implements:
      - get(key) -> bytes | None
      - setex(key, ttl, value)
    """

    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis = redis_client

    @property
    def cache_available(self) -> bool:
        return self._redis is not None

    async def get_cached(
        self,
        document_id: str | None,
        attachment_id: str | None,
        file_name: str | None,
        summary_type: str,
    ) -> dict[str, Any] | None:
        """Returns cached summary result or None."""
        if not self.cache_available:
            return None
        key = _make_cache_key(document_id, attachment_id, file_name, summary_type)
        try:
            raw = self._redis.get(key)
            if raw:
                data = json.loads(raw)
                logger.info("Summary cache HIT: key=%s", key)
                return data
        except Exception as exc:
            logger.warning("Summary cache GET failed: %s", exc)
        return None

    async def save_result(
        self,
        document_id: str | None,
        attachment_id: str | None,
        file_name: str | None,
        summary_type: str,
        result: dict[str, Any],
    ) -> None:
        """Saves summary result to Redis."""
        if not self.cache_available:
            return
        key = _make_cache_key(document_id, attachment_id, file_name, summary_type)
        try:
            self._redis.setex(key, _SUMMARY_CACHE_TTL, json.dumps(result, ensure_ascii=False))
            logger.info("Summary cached: key=%s ttl=%ds", key, _SUMMARY_CACHE_TTL)
        except Exception as exc:
            logger.warning("Summary cache SET failed: %s", exc)

    async def invalidate(
        self,
        document_id: str | None,
        attachment_id: str | None,
        file_name: str | None,
        summary_type: str,
    ) -> None:
        """Removes a cached result (called before forced refresh)."""
        if not self.cache_available:
            return
        key = _make_cache_key(document_id, attachment_id, file_name, summary_type)
        try:
            self._redis.delete(key)
            logger.info("Summary cache invalidated: key=%s", key)
        except Exception as exc:
            logger.warning("Summary cache DELETE failed: %s", exc)

    @staticmethod
    def build_cached_response(
        cached_data: dict[str, Any],
        summary_type: str,
        attachment_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Builds a full AgentResponse dict from cached data,
        adding metadata so frontend renders the «Обновить» button.
        """
        metadata = cached_data.get("metadata", {})
        metadata.update({
            "cached": True,
            "summary_type": summary_type,
            "summary_type_label": _SUMMARY_TYPE_LABELS.get(summary_type, summary_type),
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

    @staticmethod
    def enrich_fresh_response(
        result: dict[str, Any],
        summary_type: str,
        attachment_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Adds metadata to a fresh (non-cached) summarization result.
        Frontend will show «Обновить» button on next request for same doc.
        """
        metadata = result.get("metadata", {})
        metadata.update({
            "cached": False,
            "summary_type": summary_type,
            "summary_type_label": _SUMMARY_TYPE_LABELS.get(summary_type, summary_type),
            "can_refresh": True,
        })
        if attachment_id:
            metadata["attachment_id"] = attachment_id
        result["metadata"] = metadata
        return result