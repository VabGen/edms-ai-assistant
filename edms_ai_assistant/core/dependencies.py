from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from edms_ai_assistant.clients.redis_client import redis_client
from edms_ai_assistant.db.database import get_db
from edms_ai_assistant.services.rate_limiter import RateLimiter
from edms_ai_assistant.services.summarization_orchestrator import (
    SummarizationOrchestrator,
)


def get_orchestrator(db: AsyncSession = Depends(get_db)) -> SummarizationOrchestrator:
    return SummarizationOrchestrator(enable_cache=True)


def get_rate_limiter() -> RateLimiter:
    return RateLimiter(redis_client=redis_client, limit=20, window=60)
