"""
Dependency wiring — builds SummarizationService from application config.

Provides:
  - build_summarization_service(): constructs full service with all dependencies
  - FastAPI lifespan integration helpers

Usage in main.py lifespan:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = await build_summarization_service(settings)
        app.state.summarization_service = service
        yield
        await service._llm.aclose()
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def build_summarization_service(settings: object) -> "SummarizationService":  # type: ignore[name-defined]
    """
    Construct a fully-wired SummarizationService from application settings.

    Args:
        settings: Application settings object (pydantic-settings).
                  Expected attributes:
                    - REDIS_URL: str
                    - DATABASE_URL: str  (asyncpg)
                    - LLM_GENERATIVE_URL: str
                    - LLM_GENERATIVE_MODEL: str
                    - LLM_API_KEY: SecretStr | None
                    - AGENT_MAX_ITERATIONS: int  (used as max_concurrent_map)
                    - SUMMARIZER_CONTEXT_WINDOW: int  (default: 4096)
                    - SUMMARIZER_QUALITY_MODEL: str | None

    Returns:
        Configured SummarizationService instance.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from edms_ai_assistant.summarizer.cache.cache import (
        PostgresL2Cache,
        RedisL1Cache,
        TwoLevelCache,
    )
    from edms_ai_assistant.summarizer.observability.tracing import setup_tracing
    from edms_ai_assistant.summarizer.pipeline.direct import OpenAICompatibleClient
    from edms_ai_assistant.summarizer.service import SummarizationService

    # ── OTel Tracing ──────────────────────────────────────────────────────
    telemetry_endpoint = getattr(settings, "TELEMETRY_ENDPOINT", None)
    setup_tracing(
        service_name="edms-summarizer",
        otlp_endpoint=telemetry_endpoint,
        enable_in_memory=False,
    )

    # ── LLM Client ────────────────────────────────────────────────────────
    base_url = str(getattr(settings, "LLM_GENERATIVE_URL", "http://localhost:11434/v1"))

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    model = str(getattr(settings, "LLM_GENERATIVE_MODEL"))
    api_key_secret = getattr(settings, "LLM_API_KEY", None) or getattr(settings, "OPENAI_API_KEY", None)

    if api_key_secret:
        if hasattr(api_key_secret, "get_secret_value"):
            api_key = api_key_secret.get_secret_value()
        else:
            api_key = str(api_key_secret)
    else:
        api_key = None

    if not api_key or not api_key.strip():
        api_key = "placeholder"

    timeout = float(getattr(settings, "LLM_TIMEOUT", 120))

    llm_client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    logger.info("LLM client configured: model=%s base_url=%s", model, base_url)

    # ── Cache ─────────────────────────────────────────────────────────────
    db_url = str(getattr(settings, "DATABASE_URL", ""))

    # Ensure asyncpg driver
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(
        db_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    l1 = RedisL1Cache()
    l2 = PostgresL2Cache(session_factory=session_factory)
    cache = TwoLevelCache(l1=l1, l2=l2)

    # ── Service ───────────────────────────────────────────────────────────
    context_window = int(getattr(settings, "SUMMARIZER_CONTEXT_WINDOW", 4096))
    quality_model = getattr(settings, "SUMMARIZER_QUALITY_MODEL", None)
    max_concurrent = int(getattr(settings, "AGENT_MAX_ITERATIONS", 6))

    service = SummarizationService(
        llm_client=llm_client,
        cache=cache,
        model=model,
        quality_model=quality_model,
        direct_context_window=context_window,
        max_concurrent_map=max_concurrent,
    )

    logger.info(
        "SummarizationService v2 ready: model=%s context_window=%d max_concurrent=%d",
        model, context_window, max_concurrent,
    )
    return service