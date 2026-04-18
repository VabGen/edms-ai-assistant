# edms_ai_assistant/main.py
"""
FastAPI application — production-ready with streaming & caching support.

Architecture (v3.1):
- Standard /chat with smart file cleanup
- /chat/stream (SSE) for real-time responses (Part 5: Streaming)
- /actions/summarize with DB caching, Redis caching & EDMS fallback resolution
- confirmed=True support for destructive operations (Part 5: HITL)
- /appeal/autofill endpoint
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

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
from pydantic import BaseModel, Field
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
from edms_ai_assistant.services.summarize_service import SummarizeService
from edms_ai_assistant.utils.hash_utils import get_file_hash
from edms_ai_assistant.utils.regex_utils import UUID_RE

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

# ─── Constants ───────────────────────────────────────────────────────────────

_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/plain": ".txt",
}

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

# ─── Application state ────────────────────────────────────────────────────────

_agent: EdmsDocumentAgent | None = None
_summarize_service: SummarizeService | None = None


def get_agent() -> EdmsDocumentAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="ИИ-Агент не инициализирован.")
    return _agent


def get_summarize_service() -> SummarizeService:
    if _summarize_service is None:
        raise HTTPException(status_code=503, detail="SummarizeService не инициализирован.")
    return _summarize_service


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent, _summarize_service

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing services...")

    await init_db()
    redis_client = await init_redis()

    # Init SummarizeService with Redis
    try:
        _summarize_service = SummarizeService(redis_client)
        logger.info("SummarizeService initialized with Redis")
    except Exception as exc:
        logger.warning("SummarizeService fallback to no-cache: %s", exc)
        _summarize_service = SummarizeService(None)

    # Init Agent
    try:
        _agent = EdmsDocumentAgent()
        logger.info("EDMS AI Assistant started (v3.1)", extra={"health": _agent.health_check()})
    except Exception:
        logger.critical("Agent initialization failed — /chat will return 503", exc_info=True)

    yield

    await close_redis()
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("Temporary upload directory removed")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EDMS AI Assistant API",
    version="3.1.0",
    description="AI-powered EDMS assistant — Supervisor architecture with streaming.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings_router)
app.include_router(cache_router)


# ─── Core helpers ─────────────────────────────────────────────────────────────

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


# ─── Summarize helpers ────────────────────────────────────────────────────────

def _strict_normalize(s: str) -> str:
    return re.sub(r"[^a-zA-Zа-яА-Я0-9]", "", s.lower()) if s else ""


async def _resolve_file_identifier(
    current_path: str, context_ui_id: str | None, user_token: str
) -> tuple[str | None, bool]:
    """Resolves local file path or EDMS attachment name into a stable cache identifier."""
    is_uuid = _is_system_attachment(current_path)

    if is_uuid:
        return current_path, True

    if current_path and Path(current_path).exists():
        return get_file_hash(current_path), False

    # Fallback: fuzzy-search in EDMS document attachments by name
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
    """Check PostgreSQL summarization cache."""
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(SummarizationCache).where(
                SummarizationCache.file_identifier == str(identifier),
                SummarizationCache.summary_type == summary_type,
            )
            result = await db.execute(stmt)
            cached_row = result.scalar_one_or_none()
            if cached_row:
                logger.info("Summarization DB cache HIT", extra={"identifier": identifier[:16]})
                return cached_row.content
    except Exception as db_err:
        logger.error("Cache read error", extra={"error": str(db_err)})
    return None


async def _save_summary_to_cache(identifier: str, summary_type: str, content: str) -> None:
    """Persist summarization result to PostgreSQL."""
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
        logger.info("Summarization DB cache SAVED", extra={"identifier": identifier[:16]})
    except Exception as db_exc:
        logger.error("Cache save error", extra={"error": str(db_exc)})


# ─── Pydantic models for new endpoints ───────────────────────────────────────


class SummarizeRequest(BaseModel):
    """Request body for POST /actions/summarize (v2 with Redis + forced_refresh)."""
    user_token: str = Field(..., min_length=10)
    context_ui_id: str | None = None
    attachment_id: str | None = None
    file_path: str | None = None
    file_name: str | None = None
    thread_id: str | None = None
    summary_type: str | None = Field(
        None,
        description="'thesis' | 'extractive' | 'abstractive' | None (asks user)"
    )
    forced_refresh: bool = Field(
        default=False,
        description="If True: bypass cache and re-run analysis"
    )
    preferred_summary_format: str | None = None


class AppealAutofillRequest(BaseModel):
    """Request body for POST /appeal/autofill."""
    user_token: str = Field(..., min_length=10)
    context_ui_id: str = Field(..., description="UUID документа (APPEAL)")
    attachment_id: str | None = None
    message: str | None = None
    file_path: str | None = None


# ─── Endpoints: Chat ──────────────────────────────────────────────────────────


@app.post("/chat", response_model=AssistantResponse, tags=["Chat"])
async def chat_endpoint(
    user_input: UserInput,
    background_tasks: BackgroundTasks,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    """
    Main chat endpoint.

    Part 5 features:
    - HITL: confirmed=True for destructive operations
    - Smart file cleanup: keeps file alive while disambiguation is pending
    - preferred_summary_format: injected from user preferences
    """
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

    # Part 5: Smart file cleanup
    if user_input.file_path and not _is_system_attachment(user_input.file_path):
        if result.get("requires_reload") and not _should_defer_file_cleanup(user_input.message, result):
            background_tasks.add_task(_cleanup_file, user_input.file_path)

    # Pass cache metadata from result to frontend
    metadata = result.get("metadata", {})
    if user_input.file_path:
        metadata.setdefault("cache_context_ui_id", user_input.context_ui_id)

    return AssistantResponse(
        status=result.get("status") or "success",
        response=result.get("content") or result.get("message"),
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
        requires_reload=result.get("requires_reload", False),
        navigate_url=result.get("navigate_url"),
        metadata=metadata,
    )


@app.get("/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(
    message: str,
    user_token: str,
    context_ui_id: str | None = None,
    thread_id: str | None = None,
    agent: EdmsDocumentAgent = Depends(get_agent),
) -> StreamingResponse:
    """
    Part 5: Server-Sent Events stream for real-time token delivery.

    Frontend connects with EventSource:
      const es = new EventSource('/chat/stream?message=...&user_token=...')
      es.onmessage = (e) => { if (e.data === '[DONE]') es.close(); else appendToken(JSON.parse(e.data).content) }
    """
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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chat/history/{thread_id}", tags=["Chat"])
async def get_history(
    thread_id: str,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    """Return chat history for a given thread."""
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
    """Create a fresh conversation thread."""
    try:
        user_id = extract_user_id_from_token(request.user_token)
        return {"status": "success", "thread_id": f"chat_{user_id}_{uuid.uuid4().hex[:8]}"}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


# ─── Endpoints: Actions ───────────────────────────────────────────────────────


@app.post("/actions/summarize", response_model=AssistantResponse, tags=["Actions"])
async def summarize_endpoint(
    request: SummarizeRequest,
    background_tasks: BackgroundTasks,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
    svc: Annotated[SummarizeService, Depends(get_summarize_service)],
) -> AssistantResponse:
    """
    Smart summarization with two-tier caching (Redis + PostgreSQL) and EDMS fallback.

    Flow:
    1. forced_refresh=False AND Redis cache hit → instant cached response
    2. forced_refresh=False AND DB cache hit → instant cached response
    3. forced_refresh=True → invalidate caches → re-run agent
    4. Cache miss → run agent → save to both caches
    5. summary_type=None → agent asks user to select type (requires_choice)

    Frontend «Обновить» button logic:
      metadata.can_refresh=True + metadata.summary_type → render button
      Click → POST { ..., forced_refresh: true }
    """
    user_id = extract_user_id_from_token(request.user_token)
    thread_id = request.thread_id or f"sum_{user_id}_{request.context_ui_id or 'local'}"
    summary_type = request.summary_type or ""

    # Determine file reference
    file_ref = (
        request.attachment_id
        or request.file_path
        or request.file_name
        or ""
    ).strip()

    # ── Step 1: Redis cache check ──────────────────────────────────────────
    if summary_type and not request.forced_refresh:
        cached = await svc.get_cached(
            document_id=request.context_ui_id,
            attachment_id=request.attachment_id,
            file_name=request.file_name,
            summary_type=summary_type,
        )
        if cached:
            response_data = SummarizeService.build_cached_response(
                cached_data=cached,
                summary_type=summary_type,
                attachment_id=request.attachment_id,
            )
            return AssistantResponse(
                status=response_data["status"],
                response=response_data.get("content") or response_data.get("message"),
                thread_id=thread_id,
                requires_reload=False,
                metadata={
                    **response_data["metadata"],
                    "cache_file_identifier": request.attachment_id or request.file_name,
                    "cache_summary_type": summary_type,
                    "cache_context_ui_id": request.context_ui_id,
                },
            )

    # ── Step 2: PostgreSQL DB cache check ──────────────────────────────────
    if summary_type and not request.forced_refresh and file_ref:
        db_identifier, is_uuid = await _resolve_file_identifier(
            file_ref, request.context_ui_id, request.user_token
        )
        if db_identifier:
            cached_text = await _get_cached_summary(db_identifier, summary_type)
            if cached_text:
                return AssistantResponse(
                    status="success",
                    response=cached_text,
                    thread_id=thread_id,
                    metadata={
                        "cached": True,
                        "can_refresh": True,
                        "summary_type": summary_type,
                        "summary_type_label": SummarizeService._SUMMARY_TYPE_LABELS.get(summary_type, summary_type),
                        "cache_file_identifier": db_identifier,
                        "cache_summary_type": summary_type,
                        "cache_context_ui_id": request.context_ui_id,
                    },
                )

    # ── Step 3: Invalidate on forced refresh ───────────────────────────────
    if request.forced_refresh and summary_type:
        await svc.invalidate(
            document_id=request.context_ui_id,
            attachment_id=request.attachment_id,
            file_name=request.file_name,
            summary_type=summary_type,
        )

    # ── Step 4: Build and run agent ────────────────────────────────────────
    if request.attachment_id:
        target = f"вложение {request.attachment_id}"
    elif request.file_name:
        target = f"файл «{request.file_name}»"
    elif request.context_ui_id:
        target = "документ"
    else:
        target = "файл"

    type_hint = f" в формате «{summary_type}»" if summary_type else ""
    agent_msg = f"Сделай анализ {target}{type_hint}"

    user_context: dict[str, Any] = {
        "preferred_summary_format": summary_type or "ask",
    }

    result = await agent.chat(
        message=agent_msg,
        user_token=request.user_token,
        context_ui_id=request.context_ui_id,
        thread_id=thread_id,
        user_context=user_context,
        file_path=request.attachment_id or request.file_path or request.file_name,
        file_name=request.file_name,
    )

    # ── Step 5: Cache successful result ───────────────────────────────────
    result_status = result.get("status", "")
    response_text = result.get("content") or result.get("message") or ""

    if summary_type and result_status == "success" and response_text:
        # Save to Redis
        await svc.save_result(
            document_id=request.context_ui_id,
            attachment_id=request.attachment_id,
            file_name=request.file_name,
            summary_type=summary_type,
            result=result,
        )
        # Save to PostgreSQL
        if file_ref:
            db_identifier, _ = await _resolve_file_identifier(
                file_ref, request.context_ui_id, request.user_token
            )
            if db_identifier:
                await _save_summary_to_cache(db_identifier, summary_type, response_text)

    # ── Step 6: Cleanup local file if not a UUID ───────────────────────────
    if request.file_path and not _is_system_attachment(request.file_path):
        if result_status == "success":
            background_tasks.add_task(_cleanup_file, request.file_path)

    # ── Step 7: Enrich metadata ────────────────────────────────────────────
    if summary_type and result_status == "success":
        result = SummarizeService.enrich_fresh_response(
            result=result,
            summary_type=summary_type,
            attachment_id=request.attachment_id,
        )

    metadata = result.get("metadata", {})
    metadata.update({
        "cache_file_identifier": request.attachment_id or request.file_name,
        "cache_summary_type": summary_type or None,
        "cache_context_ui_id": request.context_ui_id,
    })

    return AssistantResponse(
        status=result.get("status") or "success",
        response=response_text or "Анализ завершён.",
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
        requires_reload=result.get("requires_reload", False),
        navigate_url=result.get("navigate_url"),
        metadata=metadata,
    )


@app.post("/appeal/autofill", response_model=AssistantResponse, tags=["Actions"])
async def appeal_autofill_endpoint(
    request: AppealAutofillRequest,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    """
    Auto-fill appeal (APPEAL) document card from attachment content.

    Calls autofill_appeal_document tool directly via agent.
    """
    user_id = extract_user_id_from_token(request.user_token)
    thread_id = f"autofill_{user_id}_{request.context_ui_id}"

    msg = request.message or "Автоматически заполни карточку обращения из вложения"

    result = await agent.chat(
        message=msg,
        user_token=request.user_token,
        context_ui_id=request.context_ui_id,
        thread_id=thread_id,
        user_context={},
        file_path=request.attachment_id or request.file_path,
    )

    return AssistantResponse(
        status=result.get("status") or "success",
        response=result.get("content") or result.get("message"),
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
        requires_reload=result.get("requires_reload", True),
        navigate_url=result.get("navigate_url"),
        metadata=result.get("metadata", {}),
    )


# ─── Endpoints: Documents ─────────────────────────────────────────────────────


@app.get("/document/{document_id}", tags=["Documents"])
async def get_document_endpoint(
    document_id: str,
    token: str,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    """Refresh document data — used by frontend after field updates."""
    try:
        async with DocumentClient() as client:
            raw = await client.get_document_metadata(token, document_id)
            return {"status": "success", "document": raw}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── Endpoints: Files ─────────────────────────────────────────────────────────


@app.post("/upload-file", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    user_token: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    """
    Upload a local file for agent processing.

    Saves to temp directory with sanitized filename.
    Returns file_path and file_name for use in subsequent /chat calls.
    """
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


# ─── Endpoints: System ────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health_check(
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "components": agent.health_check(),
    }


@app.get("/", tags=["System"])
async def root() -> dict:
    return {"name": "EDMS AI Assistant", "version": app.version, "status": "running"}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        reload_excludes=[".venv", "*.pyc", "__pycache__"],
        log_level=settings.LOGGING_LEVEL.lower(),
    )