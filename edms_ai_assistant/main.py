# edms_ai_assistant/main.py

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from edms_ai_assistant.agent.agent import EdmsDocumentAgent
from edms_ai_assistant.api import (
    actions_router,
    chat_router,
    files_router,
    settings_router,
    system_router,
)
from edms_ai_assistant.api.deps import UPLOAD_DIR
from edms_ai_assistant.clients.transport import HttpxTransport
from edms_ai_assistant.core.deps import init_deps
from edms_ai_assistant.config import settings, edms_settings
from edms_ai_assistant.db.database import init_db
from edms_ai_assistant.summarizer.api.router import router as summarizer_router
from edms_ai_assistant.summarizer.container import build_summarization_service
from edms_ai_assistant.summarizer.observability.logging_ctx import install_request_id_filter
from edms_ai_assistant.llm import get_chat_model
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
install_request_id_filter()
logger = logging.getLogger(__name__)


def _setup_telemetry(app: FastAPI) -> None:
    """Настройка OpenTelemetry инструментации."""
    if not settings.OTEL_ENABLED:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()

        logger.info("OpenTelemetry instrumentation enabled for FastAPI and HTTPX.")
    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed. "
            "Install opentelemetry-instrumentation-fastapi and "
            "opentelemetry-instrumentation-httpx to enable tracing."
        )
    except Exception as exc:
        logger.error(f"Failed to initialize OpenTelemetry: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan — startup and shutdown."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()

    import redis.asyncio as aioredis
    redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    transport = HttpxTransport(base_url=str(edms_settings.base_url), default_timeout=edms_settings.timeout)
    llm = get_chat_model()

    deps = init_deps(transport, redis, llm)
    _app.state.deps = deps

    try:
        summarization_service = await build_summarization_service(settings)
        _app.state.summarization_service = summarization_service
        deps.summarization_service = summarization_service
        logger.info("SummarizationService ready")
    except Exception as exc:
        logger.error("SummarizationService initialization failed: %s", exc, exc_info=True)
        _app.state.summarization_service = None

    try:
        agent = EdmsDocumentAgent(deps=deps, llm=llm)
        _app.state.agent = agent
        logger.info("EDMS AI Assistant started")
    except Exception as exc:
        logger.critical("Agent initialization failed. Exiting.", exc_info=True)
        raise SystemExit(1) from exc

    yield

    await redis.close()
    await transport.close()

    service = getattr(_app.state, "summarization_service", None)
    if service is not None:
        try:
            await service.aclose()
        except Exception as exc:
            logger.warning("Error closing SummarizationService: %s", exc)


if settings.OTEL_ENABLED:
    from edms_ai_assistant.observability.tracing import setup_tracing
    setup_tracing(
        service_name="edms-ai-assistant",
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )

def create_app() -> FastAPI:
    """Application factory."""

    application = FastAPI(
        title="EDMS AI Assistant API",
        version=settings.APP_VERSION,
        description="AI-powered assistant for EDMS document management workflows.",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(chat_router)
    application.include_router(files_router)
    application.include_router(actions_router)
    application.include_router(settings_router)
    application.include_router(system_router)
    application.include_router(summarizer_router)

    _setup_telemetry(application)

    return application

app = create_app()

@app.get("/download/extension", tags=["Plugin"])
async def download_extension():
    """
    Скачивание браузерного расширения EDMS AI Assistant (.zip).
    Плагин собирается на этапе сборки Docker-образа и лежит в static/plugin/extension.zip
    """
    plugin_zip = "static/plugin/extension.zip"

    if not os.path.exists(plugin_zip):
        return {"error": "Extension is not built in this environment. Check Dockerfile."}

    return FileResponse(
        plugin_zip,
        media_type="application/zip",
        filename="edms-ai-assistant-extension.zip"
    )

if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        reload_excludes=[".venv", "*.pyc", "__pycache__"],
        log_level=settings.LOGGING_LEVEL.lower(),
    )
