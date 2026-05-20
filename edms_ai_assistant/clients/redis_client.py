# edms_ai_assistant/clients/redis_client.py
import logging
from typing import AsyncGenerator

import redis.asyncio as aioredis

from edms_ai_assistant.config import settings

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
            raise RuntimeError("Redis client is not initialized. Call connect() first.")
        return self._pool


# Фабрика для FastAPI Depends
async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """FastAPI dependency: предоставляет активное соединение Redis."""
    client = RedisClient(str(settings.REDIS_URL))
    await client.connect()
    try:
        yield client.get_client()
    finally:
        await client.close()
