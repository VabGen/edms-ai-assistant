# edms_ai_assistant/clients/redis_client.py
"""
RedisClient — клиент для работы с Redis.
"""
import logging

import redis.asyncio as aioredis

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self):
        self.redis: aioredis.Redis | None = None

    async def connect(self):
        try:
            self.redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            await self.redis.ping()
            logger.info("Successfully connected to Redis at %s", settings.REDIS_URL)
        except Exception as e:
            logger.warning("Redis unavailable at startup — caching disabled: %s", e)
            self.redis = None

    async def close(self):
        if self.redis:
            await self.redis.aclose()
            self.redis = None
            logger.info("Redis connection closed.")

    def get_client(self) -> aioredis.Redis:
        if not self.redis:
            raise RuntimeError("Redis client is not initialized. Call connect() first.")
        return self.redis


redis_client = RedisClient()


async def init_redis() -> aioredis.Redis:
    """Initialize the shared Redis client. Call from lifespan startup."""
    await redis_client.connect()
    return redis_client.get_client()


async def close_redis() -> None:
    """Close the shared Redis client. Call from lifespan shutdown."""
    await redis_client.close()


def get_redis() -> aioredis.Redis:
    """FastAPI dependency / helper: provides the shared Redis client."""
    return redis_client.get_client()
