"""
Dependency wiring — сборка SummarizationService из конфига приложения.

Поведение:
  - LLM-клиент конструируется из типизированного `SummarizerConfig`.
  - Cache (Redis L1 + Postgres L2) переиспользует глобальный engine из db.database.
  - Идемпотентный setup трейсинга.
  - Регистрация сервиса в tool-обёртке (без app.state).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from edms_ai_assistant.summarizer.config import SummarizerConfig

if TYPE_CHECKING:
    from edms_ai_assistant.summarizer.service import SummarizationService

logger = logging.getLogger(__name__)


async def build_summarization_service(settings: object) -> SummarizationService:
    from edms_ai_assistant.db.database import AsyncSessionLocal
    from edms_ai_assistant.summarizer.cache.cache import (
        PostgresL2Cache,
        RedisL1Cache,
        TwoLevelCache,
    )
    from edms_ai_assistant.summarizer.observability.tracing import setup_tracing
    from edms_ai_assistant.summarizer.pipeline.direct import OpenAICompatibleClient
    from edms_ai_assistant.summarizer.service import SummarizationService

    config = SummarizerConfig.from_app_settings(settings)

    # ── OpenTelemetry (idempotent) ──────────────────────────────────────────
    setup_tracing(
        service_name="edms-summarizer",
        otlp_endpoint=config.otlp_endpoint,
        enable_in_memory=False,
    )

    # ── LLM client ─────────────────────────────────────────────────────────
    base_url = config.normalized_base_url()
    llm_client = OpenAICompatibleClient(
        api_key=config.resolved_api_key(),
        base_url=base_url,
        timeout=config.llm_timeout_s,
    )
    logger.info("LLM client: model=%s base_url=%s", config.llm_model, base_url)

    # ── Cache: переиспользуем глобальный async engine из db/database.py ────
    cache = TwoLevelCache(
        l1=RedisL1Cache(),
        l2=PostgresL2Cache(session_factory=AsyncSessionLocal),
    )

    service = SummarizationService(
        llm_client=llm_client,
        cache=cache,
        model=config.llm_model,
        direct_context_window=config.context_window_tokens,
        max_concurrent_map=config.max_concurrent_map,
        max_output_tokens=config.max_output_tokens,
    )

    try:
        from edms_ai_assistant.tools.summarization import set_summarization_service

        set_summarization_service(service)
    except ImportError:
        logger.warning("Could not set summarization service in tools (ImportError)")

    logger.info(
        "SummarizationService готов: model=%s context_window=%d max_output_tokens=%d",
        config.llm_model,
        config.context_window_tokens,
        config.max_output_tokens,
    )
    return service
