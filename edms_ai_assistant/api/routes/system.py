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
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from opentelemetry import trace
from pydantic import BaseModel

from edms_ai_assistant.api.deps import get_admin_user, get_agent
from edms_ai_assistant.config import settings

if TYPE_CHECKING:
    from edms_ai_assistant.agent.agent import EdmsDocumentAgent

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


@router.get(
    "/health/ready",
    summary="Readiness probe (agent can handle traffic)",
    response_model=HealthResponse,
)
async def readiness_probe(
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
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
                    _readiness_cache["result"] = health_data
                    _readiness_cache["ts"] = now

        components = ComponentStatus(**health_data)

        span.set_attribute("health.llm", components.llm)
        span.set_attribute("health.graph", components.graph)
        span.set_attribute("health.tools", components.tools)

        is_ready = components.llm and components.graph

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
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
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
