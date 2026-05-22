# edms_ai_assistant/clients/redis_client.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
from edms_ai_assistant.config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class RedisClient:
    """Управление соединением с Redis."""

    def __init__(self, url: str):
        self._url = url
        self._pool: aioredis.Redis | None = None

    async def connect(self) -> None:
        if self._pool is None:
            try:
                self._pool = aioredis.from_url(
                    self._url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._pool.ping()
                logger.info("Successfully connected to Redis")
            except Exception as e:
                logger.warning("Redis unavailable at startup — caching disabled: %s", e)
                self._pool = None

    async def close(self) -> None:
        if self._pool:
            await self._pool.aclose()
            self._pool = None
            logger.info("Redis connection closed")

    def get_client(self) -> aioredis.Redis:
        if not self._pool:
            # Fallback для тестов или среды без Redis
            return aioredis.Redis()
        return self._pool


# ── Global instances (Legacy Support / Lifecycle) ──────────────────────────

_global_client = RedisClient(str(settings.REDIS_URL))

async def init_redis() -> None:
    await _global_client.connect()

async def close_redis() -> None:
    await _global_client.close()

def get_redis_client() -> aioredis.Redis:
    return _global_client.get_client()

# ── FastAPI Dependencies ───────────────────────────────────────────────────

async def get_redis() -> AsyncGenerator[aioredis.Redis]:
    """FastAPI dependency: предоставляет активное соединение Redis."""
    # Используем глобальный пул для FastAPI
    yield _global_client.get_client()
