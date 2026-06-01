# edms_ai_assistant/api/routes/system.py
"""System API routes — Cloud-Native Health Probes (2024–2026 Standard).

Endpoints:
    GET /health/live  — K8s Liveness (is event loop alive?)
    GET /health/ready — K8s Readiness (can agent handle traffic? LLM available?)
    GET /health       — Legacy (deprecated, proxies to ready)
    GET /health/ocr   — Secured OCR diagnostics (Admin only)
"""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from opentelemetry import trace
from pydantic import BaseModel

from edms_ai_assistant.api.deps import AgentDep, DepsDep, get_admin_user
from edms_ai_assistant.config import settings
from edms_ai_assistant.core.di_container import get_container
from edms_ai_assistant.db.database import engine

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter(tags=["System"])


# ---------------------------------------------------------------------------
# OpenAPI Contract (Pydantic Models)
# ---------------------------------------------------------------------------


class ComponentStatus(BaseModel):
    """Статус отдельных компонентов агента."""

    llm: bool = False
    graph: bool = False
    tools: bool = False
    database: bool = False


class HealthResponse(BaseModel):
    """Стандартизированный ответ для health-эндпоинтов."""

    status: Literal["alive", "ready", "degraded", "not_ready"]
    version: str
    build: str | None = None
    components: ComponentStatus | None = None


# ---------------------------------------------------------------------------
# Liveness Probe
# ---------------------------------------------------------------------------


@router.get(
    "/health/live",
    summary="Liveness probe (process is alive)",
    response_model=HealthResponse,
)
async def liveness_probe() -> HealthResponse:
    """Kubernetes liveness probe.

    Проверяет только то, что процесс жив и event loop не заблокирован.
    Не проверяет внешние зависимости (LLM, БД), чтобы K8s не убивал под
    при временной недоступности провайдера.
    """
    return HealthResponse(
        status="alive",
        version=settings.APP_VERSION,
        build=settings.BUILD_COMMIT,
    )


# ---------------------------------------------------------------------------
# Readiness Probe (with TTL cache & Concurrency Lock)
# ---------------------------------------------------------------------------

_readiness_cache: dict[str, Any] = {"result": None, "ts": 0.0}
_readiness_lock: asyncio.Lock | None = None
READINESS_CACHE_TTL: float = 15.0


def _get_readiness_lock() -> asyncio.Lock:
    """Инициализирует Lock лениво внутри работающего event loop."""
    global _readiness_lock
    if _readiness_lock is None:
        _readiness_lock = asyncio.Lock()
    return _readiness_lock


async def _check_database_health() -> bool:
    """Проверяет доступность базы данных и schema version.
    
    Returns:
        True if database is accessible and schema is current, False otherwise.
    """
    try:
        from sqlalchemy import text
        
        async with engine.connect() as conn:
            # Check basic connectivity
            await conn.execute(text("SELECT 1"))
            
            # Check alembic version table exists and has data
            version_check = await conn.execute(
                text("SELECT version_num FROM alembic_version LIMIT 1")
            )
            version = version_check.scalar()
            
            if version:
                logger.debug(f"Database schema version: {version}")
                return True
            else:
                logger.warning("Database alembic_version table is empty")
                return False
                
    except Exception as exc:
        logger.error("Database health check failed: %s", exc, exc_info=True)
        return False


@router.get(
    "/health/ready",
    summary="Readiness probe (agent can handle traffic)",
    response_model=HealthResponse,
)
async def readiness_probe(
    agent: AgentDep,
) -> HealthResponse:
    """Kubernetes readiness probe.

    Проверяет критичные зависимости (LLM, Graph). Если они недоступны,
    под должен быть исключён из балансировщика (HTTP 503),
    но K8s НЕ должен его рестартить (для этого есть /health/live).

    Результат кэшируется на `READINESS_CACHE_TTL` секунд во избежание
    спама запросами к LLM-API при частых проверках K8s.
    """
    with tracer.start_as_current_span("health.readiness") as span:
        now = monotonic()
        health_data = _readiness_cache["result"]

        if health_data is None or (now - _readiness_cache["ts"] > READINESS_CACHE_TTL):
            async with _get_readiness_lock():
                now = monotonic()
                if _readiness_cache["result"] is None or (
                    now - _readiness_cache["ts"] > READINESS_CACHE_TTL
                ):
                    health_data = await agent.health_check()
                    # Add database health check
                    health_data["database"] = await _check_database_health()
                    _readiness_cache["result"] = health_data
                    _readiness_cache["ts"] = now

        components = ComponentStatus(**health_data)

        span.set_attribute("health.llm", components.llm)
        span.set_attribute("health.graph", components.graph)
        span.set_attribute("health.tools", components.tools)
        span.set_attribute("health.database", components.database)

        is_ready = components.llm and components.graph and components.database

        if not is_ready:
            logger.warning("Readiness check failed: %s", components.model_dump_json())
            raise HTTPException(
                status_code=503,
                detail=HealthResponse(
                    status="not_ready",
                    version=settings.APP_VERSION,
                    build=settings.BUILD_COMMIT,
                    components=components,
                ).model_dump(),
            )

        status: Literal["degraded", "ready"] = (
            "degraded" if not components.tools else "ready"
        )

        return HealthResponse(
            status=status,
            version=settings.APP_VERSION,
            build=settings.BUILD_COMMIT,
            components=components,
        )


# ---------------------------------------------------------------------------
# Legacy Health Check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    summary="Health check (legacy)",
    response_model=HealthResponse,
    deprecated=True,
)
async def health_check(
    agent: AgentDep,
) -> HealthResponse:
    """Deprecated. Use /health/live for liveness and /health/ready for readiness."""
    return await readiness_probe(agent)


# ---------------------------------------------------------------------------
# Secured Diagnostics
# ---------------------------------------------------------------------------


@router.get(
    "/health/ocr",
    tags=["diagnostics"],
    summary="Tesseract OCR diagnostics (Admin only)",
    dependencies=[Depends(get_admin_user)],
)
async def ocr_diagnostics() -> dict[str, Any]:
    """Диагностика Tesseract OCR.

    Раскрывает пути файловой системы и конфигурацию,
    поэтому требует прав администратора.
    """
    with tracer.start_as_current_span("diagnostics.ocr"):
        from edms_ai_assistant.services.file_processor import FileProcessorService

        return FileProcessorService.get_ocr_diagnostic()


@router.get(
    "/health/cache",
    tags=["diagnostics"],
    summary="Redis cache circuit breaker diagnostics (Admin only)",
    dependencies=[Depends(get_admin_user)],
)
async def cache_diagnostics(deps: DepsDep) -> dict[str, Any]:
    """Диагностика Redis cache circuit breaker.

    Показывает метрики circuit breaker и здоровье кэша.
    """
    with tracer.start_as_current_span("diagnostics.cache"):
        try:
            cache_health = await deps.document_service._cache.get_cache_health()
            circuit_metrics = (
                await deps.document_service._cache._cache.get_circuit_breaker_metrics()
                if deps.document_service._cache._cache
                else {}
            )
            
            return {
                "cache_health": cache_health,
                "circuit_breaker_metrics": circuit_metrics,
                "settings": {
                    "circuit_breaker_enabled": settings.REDIS_CIRCUIT_BREAKER_ENABLED,
                    "failure_threshold": settings.REDIS_CIRCUIT_FAILURE_THRESHOLD,
                    "recovery_timeout": settings.REDIS_CIRCUIT_RECOVERY_TIMEOUT,
                    "operation_timeout": settings.REDIS_CIRCUIT_OPERATION_TIMEOUT,
                },
            }
        except Exception as exc:
            logger.error("Cache diagnostics failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get cache diagnostics: {exc}"
            )


@router.get(
    "/health/di",
    tags=["diagnostics"],
    summary="DI Container diagnostics (Admin only)",
    dependencies=[Depends(get_admin_user)],
)
async def di_container_diagnostics() -> dict[str, Any]:
    """Диагностика DI контейнера.

    Показывает состояние DI контейнера и его зависимостей.
    """
    with tracer.start_as_current_span("diagnostics.container"):
        try:
            container = get_container()
            
            return {
                "container_status": "initialized",
                "providers": {
                    "redis": "singleton" if hasattr(container, "redis") else "not_configured",
                    "transport": "singleton" if hasattr(container, "transport") else "not_configured",
                    "chat_model": "singleton" if hasattr(container, "chat_model") else "not_configured",
                    "app_deps": "singleton" if hasattr(container, "app_deps") else "not_configured",
                    "agent": "singleton" if hasattr(container, "agent") else "not_configured",
                },
                "configuration": {
                    "redis_url": str(settings.REDIS_URL)[:20] + "***",  # Partially masked
                    "edms_base_url": str(settings.EDMS_BASE_URL),
                    "llm_generative_url": str(settings.LLM_GENERATIVE_URL),
                    "llm_model": settings.LLM_GENERATIVE_MODEL,
                },
            }
        except RuntimeError as exc:
            logger.error("DI container not initialized: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"DI container not initialized: {exc}"
            )
        except Exception as exc:
            logger.error("DI diagnostics failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get DI diagnostics: {exc}"
            )
