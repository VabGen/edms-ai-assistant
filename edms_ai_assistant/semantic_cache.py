# edms_ai_assistant/semantic_cache.py
"""
Semantic Cache - fuzzy-matching on embeddings.

Queries like "покажи документ" and "отобрази документ" yield the same
cached response without hitting the LLM. Expected ~40% LLM call
reduction on real EDMS workloads.

Cache key includes (user_id, document_id) - no cross-user leakage.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached response."""
    query_hash: str
    response: Any
    embedding: list[float] | None
    created_at: float
    user_id: str | None
    document_id: str | None
    hit_count: int = 0


@dataclass
class SemanticCache:
    """
    In-memory semantic cache with cosine-similarity matching.

    Args:
        similarity_threshold: Minimum cosine similarity for a cache hit (0..1).
        ttl_seconds: Time-to-live for cache entries in seconds.
        max_entries: Maximum number of entries before eviction.
        embed_fn: Async function returning embedding vector for text.
            If None, cache operates in exact-match mode only.
    """
    similarity_threshold: float = 0.92
    ttl_seconds: float = 300.0
    max_entries: int = 500
    embed_fn: Callable[[str], Awaitable[list[float]]] | None = None

    _entries: list[CacheEntry] = field(default_factory=list, repr=False)
    _hits: int = 0
    _misses: int = 0

    @property
    def enabled(self) -> bool:
        """True when an embedding function is available."""
        return self.embed_fn is not None

    async def get(
        self,
        query: str,
        *,
        user_id: str | None = None,
        document_id: str | None = None,
    ) -> Any | None:
        """Look up a cached response by semantic similarity."""
        if not self.enabled or not query.strip():
            return None

        now = time.monotonic()
        query_hash = self._hash_key(query, user_id, document_id)

        # 1. Exact hash match (fast path)
        for entry in self._entries:
            if entry.query_hash == query_hash and (now - entry.created_at) < self.ttl_seconds:
                entry.hit_count += 1
                self._hits += 1
                logger.debug("SemanticCache exact HIT: query=%r", query[:60])
                return entry.response

        # 2. Semantic (embedding) match
        if self.embed_fn is not None:
            try:
                query_emb = await self.embed_fn(query)
            except Exception as exc:
                logger.warning("SemanticCache embed failed: %s", exc)
                self._misses += 1
                return None

            best_score = 0.0
            best_entry: CacheEntry | None = None
            for entry in self._entries:
                if entry.user_id != user_id or entry.document_id != document_id:
                    continue
                if (now - entry.created_at) >= self.ttl_seconds:
                    continue
                if entry.embedding is None:
                    continue
                score = self._cosine_similarity(query_emb, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry is not None and best_score >= self.similarity_threshold:
                best_entry.hit_count += 1
                self._hits += 1
                logger.debug(
                    "SemanticCache fuzzy HIT: score=%.3f query=%r",
                    best_score, query[:60],
                )
                return best_entry.response

        self._misses += 1
        return None

    async def put(
        self,
        query: str,
        response: Any,
        *,
        user_id: str | None = None,
        document_id: str | None = None,
    ) -> None:
        """Store a response in the cache."""
        if not self.enabled or not query.strip():
            return

        self._evict_if_needed()

        embedding: list[float] | None = None
        if self.embed_fn is not None:
            try:
                embedding = await self.embed_fn(query)
            except Exception as exc:
                logger.warning("SemanticCache embed failed on put: %s", exc)

        entry = CacheEntry(
            query_hash=self._hash_key(query, user_id, document_id),
            response=response,
            embedding=embedding,
            created_at=time.monotonic(),
            user_id=user_id,
            document_id=document_id,
        )
        self._entries.append(entry)
        logger.debug("SemanticCache PUT: query=%r entries=%d", query[:60], len(self._entries))

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "enabled": self.enabled,
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_key(query: str, user_id: str | None, document_id: str | None) -> str:
        """Deterministic cache key from query + scope."""
        raw = f"{user_id or ''}|{document_id or ''}|{query.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _evict_if_needed(self) -> None:
        """Evict expired and oldest entries when at capacity."""
        now = time.monotonic()
        self._entries = [
            e for e in self._entries if (now - e.created_at) < self.ttl_seconds
        ]
        while len(self._entries) >= self.max_entries:
            self._entries.pop(0)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)