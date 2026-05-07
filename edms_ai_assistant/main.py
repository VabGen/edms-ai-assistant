# edms_ai_assistant/main.py
"""
Application entry point.

Responsibilities:
  - lifespan: startup/shutdown (DB, Redis, Agent, SummarizationService)
  - FastAPI app creation and middleware
  - Router registration

Route handlers live in api/routes/:
  chat.py     — /chat, /chat/history, /chat/new
  files.py    — /upload-file
  actions.py  — /actions/summarize
  settings.py — /api/settings
  system.py   — /health
"""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
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
from edms_ai_assistant.config import settings
from edms_ai_assistant.db.database import init_db
from edms_ai_assistant.clients.redis_client import close_redis, init_redis
from edms_ai_assistant.summarizer.api.router import router as summarizer_router
from edms_ai_assistant.summarizer.container import build_summarization_service

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan — startup and shutdown."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    await init_redis()

    try:
        agent = EdmsDocumentAgent()
        health_status = await agent.health_check()
        _app.state.agent = agent  # type: ignore[attr-defined]
        logger.info("EDMS AI Assistant started", extra={"health": health_status})
    except Exception:
        logger.critical("Agent initialization failed", exc_info=True)

    try:
        summarization_service = await build_summarization_service(settings)
        _app.state.summarization_service = summarization_service  # type: ignore[attr-defined]
        logger.info("SummarizationService ready")
    except Exception as exc:
        logger.critical(
            "SummarizationService initialization failed: %s", exc, exc_info=True
        )

    yield

    await close_redis()
    service = getattr(_app.state, "summarization_service", None)
    if service is not None:
        try:
            await service._llm.aclose()
        except Exception as exc:
            logger.warning("Error closing LLM client: %s", exc)

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.2.0",
    description="AI-powered assistant for EDMS document management workflows.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        settings.ALLOWED_ORIGINS
        if isinstance(settings.ALLOWED_ORIGINS, list)
        else ["*"]
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(files_router)
app.include_router(actions_router)
app.include_router(settings_router)
app.include_router(system_router)
app.include_router(summarizer_router)


if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        reload_excludes=[".venv", "*.pyc", "__pycache__"],
        log_level=settings.LOGGING_LEVEL.lower(),
    )