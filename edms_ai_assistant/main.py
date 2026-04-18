"""
FastAPI application — production-ready with streaming & caching support.

Architecture (v3 merge):
- Standard /chat with smart file cleanup
- /chat/stream (SSE) for real-time responses (Part 5: Streaming)
- /actions/summarize with smart caching & EDMS fallback resolution
- confirmed=True support for destructive operations (Part 5: HITL)
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

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
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select
from starlette.middleware.cors import CORSMiddleware

from edms_ai_assistant.agent import EdmsDocumentAgent
from edms_ai_assistant.api.routes.cache import router as cache_router
from edms_ai_assistant.api.routes.settings import router as settings_router
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.config import settings
from edms_ai_assistant.db.database import AsyncSessionLocal, SummarizationCache, init_db
from edms_ai_assistant.model import (
    AssistantResponse,
    FileUploadResponse,
    NewChatRequest,
    UserInput,
)
from edms_ai_assistant.security import extract_user_id_from_token
from edms_ai_assistant.services.document_service import close_redis, init_redis
from edms_ai_assistant.utils.hash_utils import get_file_hash
from edms_ai_assistant.utils.regex_utils import UUID_RE

# ─── Logging & Config ────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

# ─── Module-level Constants (DRY) ───────────────────────────────────────────

_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/plain": ".txt",
}

# Слова, при наличии которых локальный файл НЕ удаляется после ответа
_FILE_OPERATION_KEYWORDS = (
    "сравни", "сравнение", "сравн", "compare", "отличи",
    "анализ", "проанализируй", "суммаризир", "прочит",
    "содержим", "прочти", "что в файл", "читай", "изучи",
)

_SUMMARY_TYPE_LABELS = {
    "extractive": "ключевые факты, даты, суммы",
    "abstractive": "краткое изложение своими словами",
    "thesis": "структурированный тезисный план",
}

# ─── Application Setup ───────────────────────────────────────────────────────

_agent: EdmsDocumentAgent | None = None


def get_agent() -> EdmsDocumentAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="ИИ-Агент не инициализирован.")
    return _agent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing services...")
    await init_db()
    await init_redis()

    try:
        _agent = EdmsDocumentAgent()
        logger.info("EDMS AI Assistant started (v3)", extra={"health": _agent.health_check()})
    except Exception:
        logger.critical("Agent initialization failed — /chat will return 503", exc_info=True)

    yield

    await close_redis()
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("Temporary upload directory removed")


app = FastAPI(
    title="EDMS AI Assistant API",
    version="3.0.0",
    description="AI-powered EDMS assistant — Supervisor architecture with streaming.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, "CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings_router)
app.include_router(cache_router)


# ─── Core Helpers ────────────────────────────────────────────────────────────

def _is_system_attachment(file_path: str | None) -> bool:
    return bool(file_path and UUID_RE.match(str(file_path)))


def _cleanup_file(file_path: str) -> None:
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            logger.debug("Temp file removed", extra={"path": file_path})
    except Exception as exc:
        logger.warning("Failed to remove temp file", extra={"path": file_path, "error": str(exc)})


async def _resolve_user_context(user_input: UserInput, user_id: str) -> dict:
    if user_input.context:
        return user_input.context.model_dump(exclude_none=True)
    try:
        async with EmployeeClient() as emp_client:
            ctx = await emp_client.get_employee(user_input.user_token, user_id)
            if ctx:
                return ctx
    except Exception as exc:
        logger.warning("Failed to fetch employee context", extra={"user_id": user_id, "error": str(exc)})
    return {"firstName": "Коллега"}


def _should_defer_file_cleanup(message: str | None, result: dict) -> bool:
    """Determines if a temp file should be kept for the next turn."""
    is_file_kw = any(kw in (message or "").lower() for kw in _FILE_OPERATION_KEYWORDS)
    is_continuation = bool(result.get("human_choice"))
    is_disambiguation = result.get("action_type") in ("requires_disambiguation", "summarize_selection")
    is_requires_action = result.get("status") == "requires_action"

    return is_file_kw or is_continuation or is_disambiguation or is_requires_action


# ─── Summarize Helpers (Extracted SRP) ──────────────────────────────────────

def _strict_normalize(s: str) -> str:
    return re.sub(r"[^a-zA-Zа-яА-Я0-9]", "", s.lower()) if s else ""


async def _resolve_file_identifier(
        current_path: str, context_ui_id: str | None, user_token: str
) -> tuple[str | None, bool]:
    """Resolves local file or EDMS UUID into a stable cache identifier."""
    is_uuid = _is_system_attachment(current_path)

    if is_uuid:
        return current_path, True

    if current_path and Path(current_path).exists():
        return get_file_hash(current_path), False

    # Fallback: search in EDMS document attachments by name
    if context_ui_id:
        try:
            async with DocumentClient() as doc_client:
                doc_dto = await doc_client.get_document_metadata(user_token, context_ui_id)
                attachments = getattr(doc_dto, "attachmentDocument", None) or []

                clean_input = _strict_normalize(current_path)
                if clean_input and attachments:
                    for att in attachments:
                        att_name = (getattr(att, "name", "") or "").strip()
                        att_id = str(getattr(att, "id", "") or "")
                        if clean_input in _strict_normalize(att_name):
                            return att_id, True

                # Ultimate fallback: first attachment
                if attachments:
                    first_id = str(getattr(attachments[0], "id", "") or "")
                    return first_id, True
        except Exception as exc:
            logger.error("EDMS attachment resolution failed", extra={"error": str(exc)})

    return None, False


async def _get_cached_summary(identifier: str, summary_type: str) -> str | None:
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(SummarizationCache).where(
                SummarizationCache.file_identifier == str(identifier),
                SummarizationCache.summary_type == summary_type,
            )
            result = await db.execute(stmt)
            cached_row = result.scalar_one_or_none()
            if cached_row:
                logger.info("Summarization cache HIT", extra={"identifier": identifier})
                return cached_row.content
    except Exception as db_err:
        logger.error("Cache read error", extra={"error": str(db_err)})
    return None


async def _save_summary_to_cache(identifier: str, summary_type: str, content: str) -> None:
    if not content or not content.strip():
        return
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                db.add(SummarizationCache(
                    id=str(uuid.uuid4()),
                    file_identifier=str(identifier),
                    summary_type=summary_type,
                    content=content,
                ))
        logger.info("Summarization cache SAVED", extra={"identifier": identifier})
    except Exception as db_exc:
        logger.error("Cache save error", extra={"error": str(db_exc)})


# ─── Endpoints: Chat ────────────────────────────────────────────────────────

@app.post("/chat", response_model=AssistantResponse, tags=["Chat"])
async def chat_endpoint(
        user_input: UserInput,
        background_tasks: BackgroundTasks,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    user_id = extract_user_id_from_token(user_input.user_token)
    thread_id = user_input.thread_id or f"user_{user_id}_doc_{user_input.context_ui_id or 'general'}"

    user_context = await _resolve_user_context(user_input, user_id)
    if user_input.preferred_summary_format and user_input.preferred_summary_format != "ask":
        user_context["preferred_summary_format"] = user_input.preferred_summary_format

    # Part 5: HITL confirmation detection
    confirmed = (
        user_input.confirmed
        or (
            user_input.human_choice is not None
            and user_input.human_choice.lower().strip()
            in ("confirm", "да", "yes", "подтверждаю")
        )
    )

    result = await agent.chat(
        message=user_input.message,
        user_token=user_input.user_token,
        context_ui_id=user_input.context_ui_id,
        thread_id=thread_id,
        user_context=user_context,
        file_path=user_input.file_path,
        file_name=user_input.file_name,
        human_choice=user_input.human_choice,
        confirmed=confirmed,
    )

    # Smart file cleanup logic
    if user_input.file_path and not _is_system_attachment(user_input.file_path):
        if result.get("requires_reload") and not _should_defer_file_cleanup(user_input.message, result):
            background_tasks.add_task(_cleanup_file, user_input.file_path)

    return AssistantResponse(
        status=result.get("status") or "success",
        response=result.get("content") or result.get("message"),
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
        requires_reload=result.get("requires_reload", False),
        navigate_url=result.get("navigate_url"),
        metadata=result.get("metadata", {}),
    )


@app.get("/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(
    message: str,
    user_token: str,
    context_ui_id: str | None = None,
    thread_id: str | None = None,
    agent: EdmsDocumentAgent = Depends(get_agent),
) -> StreamingResponse:
    """Part 5: Server-Sent Events stream for real-time token delivery."""
    user_id = extract_user_id_from_token(user_token)
    resolved_thread_id = thread_id or f"user_{user_id}_doc_{context_ui_id or 'general'}"

    async def event_generator() -> AsyncIterator[str]:
        async for chunk in agent.chat_stream(
            message=message,
            user_token=user_token,
            context_ui_id=context_ui_id,
            thread_id=resolved_thread_id,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/chat/history/{thread_id}", tags=["Chat"])
async def get_history(
        thread_id: str,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    try:
        state = await agent.state_manager.get_state(thread_id)
        messages = state.values.get("messages", [])
        filtered = [
            {"type": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in messages
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        ]
        return {"messages": filtered}
    except Exception as exc:
        logger.error("History retrieval failed", extra={"thread_id": thread_id, "error": str(exc)})
        return {"messages": []}


@app.post("/chat/new", tags=["Chat"])
async def create_new_thread(request: NewChatRequest) -> dict:
    try:
        user_id = extract_user_id_from_token(request.user_token)
        return {"status": "success", "thread_id": f"chat_{user_id}_{uuid.uuid4().hex[:8]}"}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


# ─── Endpoints: Actions ─────────────────────────────────────────────────────

@app.post("/actions/summarize", response_model=AssistantResponse, tags=["Actions"])
async def api_direct_summarize(
        user_input: UserInput,
        background_tasks: BackgroundTasks,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    """Smart summarization with DB caching and EDMS attachment fallback."""
    user_id = extract_user_id_from_token(user_input.user_token)
    new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:8]}"
    summary_type = user_input.human_choice or "extractive"
    current_path = (user_input.file_path or "").strip()

    try:
        file_identifier, is_uuid = await _resolve_file_identifier(
            current_path, user_input.context_ui_id, user_input.user_token
        )

        # Check Cache
        if file_identifier:
            cached_text = await _get_cached_summary(file_identifier, summary_type)
            if cached_text:
                return AssistantResponse(
                    status="success",
                    response=cached_text,
                    thread_id=new_thread_id,
                    metadata={"cache_file_identifier": file_identifier, "from_cache": True},
                )

        # Fallback to Agent if no cache
        user_context = await _resolve_user_context(user_input, user_id)
        type_label = _SUMMARY_TYPE_LABELS.get(summary_type, summary_type)
        instructions = f"Работай с вложением {current_path}. " if is_uuid else ""
        agent_msg = f"{instructions}Проанализируй этот файл и выдели {type_label}."

        agent_result = await agent.chat(
            message=agent_msg,
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
            thread_id=new_thread_id,
            user_context=user_context,
            file_path=current_path if is_uuid else current_path,
            human_choice=summary_type,
        )

        response_text = agent_result.get("content") or agent_result.get("response")

        # Save to Cache on success
        if file_identifier and agent_result.get("status") == "success":
            await _save_summary_to_cache(file_identifier, summary_type, response_text or "")

        # Cleanup local file if used
        if current_path and not is_uuid:
            background_tasks.add_task(_cleanup_file, current_path)

        return AssistantResponse(
            status=agent_result.get("status", "success"),
            response=response_text or "Анализ завершён.",
            thread_id=new_thread_id,
            metadata={"cache_file_identifier": file_identifier, "from_cache": False},
        )

    except Exception as exc:
        logger.error("Summarize endpoint error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Endpoints: Files & System ──────────────────────────────────────────────

@app.post("/upload-file", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
        user_token: Annotated[str, Form(...)],
        file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    try:
        extract_user_id_from_token(user_token)
        if not file.filename:
            raise HTTPException(status_code=400, detail="Имя файла не указано")

        original_path = Path(file.filename or "file")
        suffix = original_path.suffix.lower()
        if not suffix:
            suffix = _MIME_TO_EXT.get(file.content_type or "", "")

        safe_stem = re.sub(r"[^\w\-.]", "_", original_path.stem[:80])
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")
        dest_path = UPLOAD_DIR / f"{safe_stem}{suffix}"

        async with aiofiles.open(dest_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)

        logger.info("File uploaded", extra={"orig": file.filename, "dest": str(dest_path)})
        return FileUploadResponse(file_path=str(dest_path), file_name=file.filename)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File upload failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла") from exc


@app.get("/health", tags=["System"])
async def health_check(agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]) -> dict:
    return {"status": "ok", "version": app.version, "components": agent.health_check()}


if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        reload_excludes=[".venv", "*.pyc", "__pycache__"],
        log_level=settings.LOGGING_LEVEL.lower(),
    )