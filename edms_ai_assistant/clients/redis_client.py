import logging

import redis.asyncio as redis

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self):
        self.redis: redis.Redis | None = None

    async def connect(self):
        try:
            self.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
            )
            await self.redis.ping()
            logger.info(
                "Successfully connected to Redis at %s:%s",
                settings.REDIS_HOST,
                settings.REDIS_PORT,
            )
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)
            self.redis = None

    async def close(self):
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed.")

    def get_client(self) -> redis.Redis:
        if not self.redis:
            raise RuntimeError("Redis client is not initialized. Call connect() first.")
        return self.redis


redis_client = RedisClient()
