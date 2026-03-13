# edms_ai_assistant/main.py
"""
EDMS AI Assistant — FastAPI Application Entry Point.

Слой: Interface (Transport).
Содержит только HTTP-маршруты, валидацию входных данных,
маппинг запросов/ответов и фоновые задачи очистки файлов.
Вся бизнес-логика делегируется EdmsDocumentAgent (Service Layer).
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import aiofiles
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from langchain_core.messages import AIMessage, HumanMessage
from starlette.middleware.cors import CORSMiddleware

from edms_ai_assistant.agent import EdmsDocumentAgent
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.config import settings
from edms_ai_assistant.model import (
    AssistantResponse,
    FileUploadResponse,
    NewChatRequest,
    UserInput,
)
from edms_ai_assistant.security import extract_user_id_from_token
from edms_ai_assistant.services.document_service import close_redis, init_redis
from edms_ai_assistant.utils.regex_utils import UUID_RE

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

# ─────────────────────────────────────────────────────────────────────────────
# Application state (singleton agent)
# ─────────────────────────────────────────────────────────────────────────────

_agent: Optional[EdmsDocumentAgent] = None


def get_agent() -> EdmsDocumentAgent:
    """
    FastAPI dependency that returns the singleton EdmsDocumentAgent.

    Raises:
        HTTPException 503: If the agent failed to initialize at startup.
    """
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="ИИ-Агент не инициализирован. Повторите попытку позже.",
        )
    return _agent


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Application lifespan: initializes Redis, agent and upload directory
    on startup; shuts them down cleanly on stop.

    Порядок запуска:
        1. Создаём директорию для загрузок
        2. Подключаемся к Redis (async, с PING-проверкой)
        3. Инициализируем агента
    Порядок остановки (обратный):
        1. Закрываем Redis-клиент
        2. Удаляем временные файлы
    """
    global _agent

    # ── Startup ───────────────────────────────────────────────────────────────
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Redis — запускается автоматически вместе с приложением.
    # Если Redis недоступен — init_redis логирует WARNING, но не падает.
    # Кэш DocumentService просто будет пропускаться до восстановления соединения.
    await init_redis()

    try:
        _agent = EdmsDocumentAgent()
        logger.info(
            "EDMS AI Assistant started",
            extra={"health": _agent.health_check()},
        )
    except Exception:
        logger.critical(
            "Agent initialization failed — all /chat requests will return 503",
            exc_info=True,
        )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await close_redis()

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("Temporary upload directory removed")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.2.0",
    description="AI-powered assistant for EDMS document management workflows.",
    lifespan=lifespan,
)

# CORS — https://fastapi.tiangolo.com/advanced/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, "CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────


def _is_system_attachment(file_path: Optional[str]) -> bool:
    """Returns True if *file_path* is an EDMS attachment UUID (not a local file)."""
    return bool(file_path and UUID_RE.match(str(file_path)))


def _cleanup_file(file_path: str) -> None:
    """
    Safely removes a temporary uploaded file.

    Designed to be called as a FastAPI BackgroundTask — errors are logged
    but never raised.

    Args:
        file_path: Absolute path to the file to remove.
    """
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            logger.debug("Temporary file removed", extra={"path": file_path})
    except Exception as exc:
        logger.warning(
            "Failed to remove temporary file",
            extra={"path": file_path, "error": str(exc)},
        )


async def _resolve_user_context(
    user_input: UserInput,
    user_id: str,
) -> dict:
    """
    Resolves user context from request or EDMS employee API.

    Priority:
    1. UserContext provided directly in the request body
    2. Fetched from EmployeeClient using user_id from token
    3. Fallback: {"firstName": "Коллега"}

    Args:
        user_input: Validated HTTP request body.
        user_id: User ID extracted from JWT token.

    Returns:
        Dict with at least {"firstName": str}.
    """
    if user_input.context:
        return user_input.context.model_dump(exclude_none=True)

    try:
        async with EmployeeClient() as emp_client:
            ctx = await emp_client.get_employee(user_input.user_token, user_id)
            if ctx:
                return ctx
    except Exception as exc:
        logger.warning(
            "Failed to fetch employee context",
            extra={"user_id": user_id, "error": str(exc)},
        )

    return {"firstName": "Коллега"}


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@app.post(
    "/chat",
    response_model=AssistantResponse,
    summary="Send a message to the EDMS AI assistant",
    tags=["Chat"],
)
async def chat_endpoint(
    user_input: UserInput,
    background_tasks: BackgroundTasks,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    """
    Main chat endpoint. Handles both fresh messages and Human-in-the-Loop
    resumptions (disambiguation, summarization type selection).

    The thread_id is persisted by LangGraph MemorySaver and enables
    multi-turn conversations with full history.
    """
    user_id = extract_user_id_from_token(user_input.user_token)
    thread_id = (
        user_input.thread_id
        or f"user_{user_id}_doc_{user_input.context_ui_id or 'general'}"
    )

    user_context = await _resolve_user_context(user_input, user_id)

    result = await agent.chat(
        message=user_input.message,
        user_token=user_input.user_token,
        context_ui_id=user_input.context_ui_id,
        thread_id=thread_id,
        user_context=user_context,
        file_path=user_input.file_path,
        file_name=user_input.file_name,
        human_choice=user_input.human_choice,
    )

    _FILE_OPERATION_KEYWORDS = (
        "сравни",
        "сравнение",
        "сравн",
        "compare",
        "отличи",
        "анализ",
        "проанализируй",
        "суммаризир",
        "прочит",
        "содержим",
        "прочти",
        "что в файл",
        "читай",
        "изучи",
    )
    _is_file_operation = any(
        kw in (user_input.message or "").lower() for kw in _FILE_OPERATION_KEYWORDS
    )
    _is_continuation = bool(user_input.human_choice)

    if user_input.file_path and not _is_system_attachment(user_input.file_path):
        result_status = result.get("status", "success")
        _should_cleanup = (
            result_status not in ("requires_action",)
            and not _is_file_operation
            and not _is_continuation
            and result.get("requires_reload", False)
        )
        if _should_cleanup:
            background_tasks.add_task(_cleanup_file, user_input.file_path)
            logger.debug(
                "Scheduled file cleanup",
                extra={"file_path": user_input.file_path},
            )

    return AssistantResponse(
        status=result.get("status") or "success",
        response=result.get("content"),
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
        requires_reload=result.get("requires_reload", False),
        metadata=result.get("metadata", {}),
    )


@app.post(
    "/actions/summarize",
    response_model=AssistantResponse,
    summary="Trigger direct file summarization from the EDMS attachment star button",
    tags=["Actions"],
)
async def api_direct_summarize(
    user_input: UserInput,
    background_tasks: BackgroundTasks,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    """
    Direct summarization action triggered from the attachment star-button
    in the EDMS UI (via the Chrome extension).
    """
    try:
        user_id = extract_user_id_from_token(user_input.user_token)
        new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:8]}"

        current_path = user_input.file_path
        is_uuid = _is_system_attachment(current_path)

        if current_path and not is_uuid:
            if not Path(current_path).exists():
                logger.warning(
                    "Local file not found — resetting file_path",
                    extra={"path": current_path},
                )
                current_path = None

        summary_type = user_input.human_choice or "extractive"
        _type_labels: dict[str, str] = {
            "extractive": "ключевые факты, даты, суммы",
            "abstractive": "краткое изложение своими словами",
            "thesis": "структурированный тезисный план",
        }
        type_label = _type_labels.get(summary_type, summary_type)

        user_context = await _resolve_user_context(user_input, user_id)

        agent_result = await agent.chat(
            message=f"Проанализируй вложение «{user_input.message}» в формате: {type_label}.",
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
            thread_id=new_thread_id,
            user_context=user_context,
            file_path=current_path,
            human_choice=summary_type,
        )

        if current_path and not is_uuid:
            background_tasks.add_task(_cleanup_file, current_path)

        return AssistantResponse(
            status="success",
            response=agent_result.get("content") or "Анализ завершён.",
            thread_id=new_thread_id,
        )

    except Exception as exc:
        logger.error("Direct summarize failed", exc_info=True)
        if user_input.file_path and not _is_system_attachment(user_input.file_path):
            _cleanup_file(user_input.file_path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/chat/history/{thread_id}",
    summary="Get conversation history for a thread",
    tags=["Chat"],
)
async def get_history(
    thread_id: str,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    """
    Returns filtered conversation history for *thread_id*.
    Filters out ToolMessages and empty AIMessages.
    """
    try:
        state = await agent.state_manager.get_state(thread_id)
        messages = state.values.get("messages", [])

        filtered = []
        for m in messages:
            if not isinstance(m, (HumanMessage, AIMessage)):
                continue
            if isinstance(m, AIMessage) and not m.content:
                continue
            filtered.append(
                {
                    "type": "human" if isinstance(m, HumanMessage) else "ai",
                    "content": m.content,
                }
            )

        return {"messages": filtered}

    except Exception as exc:
        logger.error(
            "History retrieval failed",
            extra={"thread_id": thread_id, "error": str(exc)},
        )
        return {"messages": []}


@app.post(
    "/chat/new",
    summary="Create a new conversation thread",
    tags=["Chat"],
)
async def create_new_thread(request: NewChatRequest) -> dict:
    """
    Generates a fresh thread_id for a new conversation.
    The thread is initialized lazily on the first /chat request.
    """
    try:
        user_id = extract_user_id_from_token(request.user_token)
        new_thread_id = f"chat_{user_id}_{uuid.uuid4().hex[:8]}"
        return {"status": "success", "thread_id": new_thread_id}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


@app.post(
    "/upload-file",
    response_model=FileUploadResponse,
    summary="Upload a file for in-chat analysis",
    tags=["Files"],
)
async def upload_file(
    user_token: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    """
    Receives a file upload and stores it in a temporary directory.
    The returned file_path is passed back to /chat as file_path.
    """
    try:
        extract_user_id_from_token(user_token)

        if not file.filename:
            raise HTTPException(status_code=400, detail="Имя файла не указано")

        original_path = Path(file.filename or "file")

        suffix = original_path.suffix.lower()
        if not suffix:
            ct = file.content_type or ""
            suffix = {
                "application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "application/msword": ".doc",
                "text/plain": ".txt",
            }.get(ct, "")

        safe_stem = re.sub(r"[^\w\-.]", "_", original_path.stem[:80])
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")

        dest_path = UPLOAD_DIR / f"{safe_stem}{suffix}"

        async with aiofiles.open(dest_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)

        logger.info(
            "File uploaded",
            extra={"orig_filename": file.filename, "dest": str(dest_path)},
        )
        return FileUploadResponse(
            file_path=str(dest_path),
            file_name=file.filename,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File upload failed", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Ошибка при сохранении файла"
        ) from exc


@app.get(
    "/health",
    summary="Agent and service health check",
    tags=["System"],
)
async def health_check(
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    """Returns component-level health status of the agent."""
    return {
        "status": "ok",
        "version": app.version,
        "components": agent.health_check(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        reload_excludes=[".venv", "*.pyc", "__pycache__"],
        log_level=settings.LOGGING_LEVEL.lower(),
    )
