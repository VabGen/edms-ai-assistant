from edms_ai_assistant.clients.redis_client import redis_client


class RateLimiter:
    def __init__(self, redis_client, limit: int = 20, window: int = 60):
        self.redis = redis_client
        self.limit = limit
        self.window = window

    async def is_rate_limited(self, key: str) -> bool:
        client = self.redis.get_client()
        current = await client.get(key)
        if current and int(current) >= self.limit:
            return True

        pipe = client.pipeline()
        pipe.incr(key, 1)
        pipe.expire(key, self.window)
        await pipe.execute()
        return False
