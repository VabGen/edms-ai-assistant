from edms_ai_assistant.clients.redis_client import redis_client
from edms_ai_assistant.services.rate_limiter import RateLimiter


def get_rate_limiter() -> RateLimiter:
    return RateLimiter(redis_client=redis_client, limit=20, window=60)
