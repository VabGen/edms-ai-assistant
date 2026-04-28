# edms_ai_assistant/services/rate_limiter.py

class RateLimiter:
    def __init__(self, redis_client, limit: int = 20, window: int = 60):
        self.redis = redis_client
        self.limit = limit
        self.window = window

    async def is_rate_limited(self, key: str) -> bool:
        """Check if the key has exceeded the rate limit.

        Returns False (not limited) if Redis is unavailable.
        """
        try:
            client = self.redis.get_client()
        except RuntimeError:
            # Redis not connected — skip rate limiting
            return False

        try:
            current = await client.get(key)
            if current is None:
                await client.setex(key, self.window, 1)
                return False

            count = int(current)
            if count >= self.limit:
                return True

            await client.incr(key)
            return False
        except Exception:
            # Any Redis error — fail open (don't block requests)
            return False