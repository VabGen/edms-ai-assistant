# edms_ai_assistant/main.py
"""
FastAPI application — production-ready with streaming & caching support.

Architecture (v3.2):
- /chat — standard chat with smart file cleanup
- /chat/stream — SSE streaming (Part 5)
- /actions/summarize — ВАЖНО: принимает UserInput (как старый main.py),
  чтобы не ломать фронтенд. Внутри — двухуровневый кэш (Redis + PostgreSQL)
  и EDMS-резолвинг имени вложения в UUID.
- /appeal/autofill — автозаполнение обращения
- /document/{id} — обновление данных документа
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

# Keywords that indicate file operation — defer cleanup
_FILE_OPERATION_KEYWORDS: tuple[str, ...] = (
    "сравни", "сравнение", "сравн", "compare", "отличи",
    "анализ", "проанализируй", "суммаризир", "прочит",
    "содержим", "прочти", "что в файл", "читай", "изучи",
)

_SUMMARY_TYPE_LABELS: dict[str, str] = {
    "extractive": "ключевые факты, даты, суммы",
    "abstractive": "краткое изложение своими словами",
    "thesis": "структурированный тезисный план",
}

# ─── Application state ────────────────────────────────────────────────────────

_agent: EdmsDocumentAgent | None = None
_summarize_service: SummarizeService | None = None


def get_agent() -> EdmsDocumentAgent:
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="ИИ-Агент не инициализирован. Повторите попытку позже.",
        )
    return _agent


def get_summarize_service() -> SummarizeService:
    # Returns None-wrapped service if not initialized — never raises 503
    # because summarize can degrade gracefully without Redis
    return _summarize_service or SummarizeService(None)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent, _summarize_service

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing services...")

    await init_db()
    redis_client = await init_redis()

    # SummarizeService with Redis — graceful fallback if Redis unavailable
    try:
        _summarize_service = SummarizeService(redis_client)
        logger.info("SummarizeService initialized (Redis available)")
    except Exception as exc:
        logger.warning("SummarizeService fallback to no-Redis mode: %s", exc)
        _summarize_service = SummarizeService(None)

    # Agent
    try:
        _agent = EdmsDocumentAgent()
        logger.info(
            "EDMS AI Assistant started (v3.2)",
            extra={"health": _agent.health_check()},
        )
    except Exception:
        logger.critical(
            "Agent initialization failed — /chat will return 503",
            exc_info=True,
        )

    yield

    await close_redis()
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("Temporary upload directory removed")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EDMS AI Assistant API",
    version="3.2.0",
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
    """True if file_path is an EDMS attachment UUID."""
    return bool(file_path and UUID_RE.match(str(file_path).strip()))


def _cleanup_file(file_path: str) -> None:
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            logger.debug("Temp file removed: %s", file_path)
    except Exception as exc:
        logger.warning("Failed to remove temp file %s: %s", file_path, exc)


async def _resolve_user_context(user_input: UserInput, user_id: str) -> dict:
    """Fetch employee context from EDMS; fall back to minimal dict."""
    if user_input.context:
        return user_input.context.model_dump(exclude_none=True)
    try:
        async with EmployeeClient() as emp_client:
            ctx = await emp_client.get_employee(user_input.user_token, user_id)
            if ctx:
                return ctx
    except Exception as exc:
        logger.warning(
            "Failed to fetch employee context: %s", exc,
            extra={"user_id": user_id},
        )
    return {"firstName": "Коллега"}


def _should_defer_file_cleanup(message: str | None, result: dict) -> bool:
    """True when temp file must be kept alive for the next turn."""
    is_file_kw = any(kw in (message or "").lower() for kw in _FILE_OPERATION_KEYWORDS)
    is_continuation = bool(result.get("human_choice"))
    is_disambiguation = result.get("action_type") in (
        "requires_disambiguation", "summarize_selection"
    )
    is_requires_action = result.get("status") == "requires_action"
    return is_file_kw or is_continuation or is_disambiguation or is_requires_action


# ─── Summarize helpers ────────────────────────────────────────────────────────

def _strict_normalize(s: str) -> str:
    """Strip all non-alphanumeric chars for fuzzy name matching."""
    return re.sub(r"[^a-zA-Zа-яА-Я0-9]", "", s.lower()) if s else ""


async def _resolve_file_identifier(
    current_path: str,
    context_ui_id: str | None,
    user_token: str,
) -> tuple[str | None, bool]:
    """
    Resolve local file path OR attachment name into a stable cache key.

    Returns (identifier, is_uuid):
      - UUID attachment  → (uuid_str, True)
      - Local file       → (sha256_hash, False)
      - EDMS name match  → (att_uuid, True)
      - Fallback first   → (first_att_uuid, True)
      - Nothing          → (None, False)
    """
    # Already a UUID — use directly
    if _is_system_attachment(current_path):
        return current_path.strip(), True

    # Local file exists — hash it
    if current_path and Path(current_path).exists():
        file_hash = get_file_hash(current_path)
        logger.info("CACHE: local file → hash %s", file_hash[:16])
        return file_hash, False

    # Fallback: fuzzy-match attachment name in EDMS document
    if context_ui_id:
        try:
            async with DocumentClient() as doc_client:
                doc_dto = await doc_client.get_document_metadata(
                    user_token, context_ui_id
                )
                # Support both Pydantic model and raw dict
                if hasattr(doc_dto, "attachmentDocument"):
                    attachments = doc_dto.attachmentDocument or []
                elif isinstance(doc_dto, dict):
                    attachments = doc_dto.get("attachmentDocument") or []
                else:
                    attachments = []

                if not attachments:
                    logger.warning("CACHE: No attachments found in document %s", context_ui_id[:8])
                    return None, False

                clean_input = _strict_normalize(current_path)

                # Fuzzy name match
                if clean_input:
                    for att in attachments:
                        if isinstance(att, dict):
                            att_name = (att.get("name", "") or "").strip()
                            att_id = str(att.get("id", "") or "")
                        else:
                            att_name = (getattr(att, "name", "") or "").strip()
                            att_id = str(getattr(att, "id", "") or "")

                        if clean_input in _strict_normalize(att_name):
                            logger.info(
                                "CACHE: name match '%s' → %s", att_name, att_id[:8]
                            )
                            return att_id, True

                # Fallback: first attachment
                first = attachments[0]
                first_id = str(
                    (first.get("id") if isinstance(first, dict) else getattr(first, "id", ""))
                    or ""
                )
                if first_id:
                    logger.info("CACHE: using first attachment %s", first_id[:8])
                    return first_id, True

        except Exception as exc:
            logger.error("CACHE: EDMS resolution failed: %s", exc)

    return None, False


async def _get_db_cached_summary(identifier: str, summary_type: str) -> str | None:
    """Check PostgreSQL summarization cache. Returns content string or None."""
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(SummarizationCache).where(
                SummarizationCache.file_identifier == str(identifier),
                SummarizationCache.summary_type == summary_type,
            )
            result = await db.execute(stmt)
            cached_row = result.scalar_one_or_none()
            if cached_row:
                logger.info(
                    "DB cache HIT: id=%s type=%s", identifier[:16], summary_type
                )
                return cached_row.content
    except Exception as exc:
        logger.error("DB cache read error: %s", exc)
    return None


async def _save_db_cache(
    identifier: str, summary_type: str, content: str
) -> None:
    """Persist summarization result to PostgreSQL cache."""
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
        logger.info("DB cache SAVED: id=%s type=%s", identifier[:16], summary_type)
    except Exception as exc:
        # UniqueConstraint violation on duplicate — not an error, just skip
        logger.warning("DB cache save skipped (probably duplicate): %s", exc)


# ─── Pydantic models ──────────────────────────────────────────────────────────


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

    Accepts UserInput with optional file_path, human_choice, preferred_summary_format.
    Handles HITL confirmation via confirmed=True or human_choice='да'/'confirm'.
    """
    user_id = extract_user_id_from_token(user_input.user_token)
    thread_id = (
        user_input.thread_id
        or f"user_{user_id}_doc_{user_input.context_ui_id or 'general'}"
    )

    user_context = await _resolve_user_context(user_input, user_id)
    if user_input.preferred_summary_format and user_input.preferred_summary_format != "ask":
        user_context["preferred_summary_format"] = user_input.preferred_summary_format

    # HITL confirmation detection
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

    # Smart file cleanup
    if user_input.file_path and not _is_system_attachment(user_input.file_path):
        if result.get("requires_reload") and not _should_defer_file_cleanup(
            user_input.message, result
        ):
            background_tasks.add_task(_cleanup_file, user_input.file_path)

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
    """SSE stream for real-time token delivery."""
    user_id = extract_user_id_from_token(user_token)
    resolved_thread_id = (
        thread_id or f"user_{user_id}_doc_{context_ui_id or 'general'}"
    )

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
    """Return filtered chat history for a thread."""
    try:
        state = await agent.state_manager.get_state(thread_id)
        messages = state.values.get("messages", [])
        filtered = [
            {
                "type": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content,
            }
            for m in messages
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        ]
        return {"messages": filtered}
    except Exception as exc:
        logger.error(
            "History retrieval failed",
            extra={"thread_id": thread_id, "error": str(exc)},
        )
        return {"messages": []}


@app.post("/chat/new", tags=["Chat"])
async def create_new_thread(request: NewChatRequest) -> dict:
    """Create a fresh conversation thread."""
    try:
        user_id = extract_user_id_from_token(request.user_token)
        return {
            "status": "success",
            "thread_id": f"chat_{user_id}_{uuid.uuid4().hex[:8]}",
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


# ─── Endpoints: Actions ───────────────────────────────────────────────────────


@app.post("/actions/summarize", response_model=AssistantResponse, tags=["Actions"])
async def api_direct_summarize(
    user_input: UserInput,  # ← сохраняем UserInput как в старом main.py
    background_tasks: BackgroundTasks,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    """
    Smart summarization with two-tier caching (Redis + PostgreSQL).

    ВАЖНО: принимает UserInput (не SummarizeRequest) — фронтенд шлёт именно его.
    Поля фронтенда:
      file_path    = имя файла ИЛИ UUID вложения ИЛИ локальный путь
      human_choice = тип анализа ('extractive' | 'abstractive' | 'thesis')
      context_ui_id = UUID документа (для EDMS резолвинга)

    Порядок работы:
    1. Резолвим file_path → стабильный file_identifier (UUID или хэш)
    2. Redis cache hit → мгновенный ответ
    3. DB (PostgreSQL) cache hit → мгновенный ответ
    4. Cache miss → запускаем агента → сохраняем в оба кэша
    """
    user_id = extract_user_id_from_token(user_input.user_token)
    new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:8]}"
    summary_type = (user_input.human_choice or "extractive").strip()

    # ── Шаг 1: резолвим file_identifier ──────────────────────────────────────
    current_path = (user_input.file_path or "").strip()

    logger.info(
        "SUMMARIZE START: path='%s' context=%s type=%s",
        current_path, user_input.context_ui_id, summary_type,
    )

    file_identifier, is_uuid = await _resolve_file_identifier(
        current_path=current_path,
        context_ui_id=user_input.context_ui_id,
        user_token=user_input.user_token,
    )

    # После резолвинга: если получили UUID вложения — использовать его как file_path
    effective_path = file_identifier if is_uuid else current_path

    logger.info(
        "SUMMARIZE: file_identifier=%s is_uuid=%s effective_path=%s",
        (file_identifier or "None")[:16], is_uuid, (effective_path or "None")[:32],
    )

    svc = get_summarize_service()

    # ── Шаг 2: Redis cache check ──────────────────────────────────────────────
    redis_cached = await svc.get_cached(
        document_id=user_input.context_ui_id,
        attachment_id=file_identifier if is_uuid else None,
        file_name=user_input.file_path if not is_uuid else None,
        summary_type=summary_type,
    )
    if redis_cached:
        logger.info("SUMMARIZE: Redis cache HIT")
        return AssistantResponse(
            status="success",
            response=redis_cached.get("content") or redis_cached.get("response"),
            thread_id=new_thread_id,
            metadata={
                **redis_cached.get("metadata", {}),
                "cache_file_identifier": file_identifier,
                "cache_summary_type": summary_type,
                "cache_context_ui_id": user_input.context_ui_id,
                "from_cache": True,
            },
        )

    # ── Шаг 3: PostgreSQL DB cache check ──────────────────────────────────────
    if file_identifier:
        db_cached_text = await _get_db_cached_summary(file_identifier, summary_type)
        if db_cached_text:
            logger.info("SUMMARIZE: DB cache HIT")
            return AssistantResponse(
                status="success",
                response=db_cached_text,
                thread_id=new_thread_id,
                metadata={
                    "cached": True,
                    "can_refresh": True,
                    "summary_type": summary_type,
                    "summary_type_label": SummarizeService._SUMMARY_TYPE_LABELS.get(
                        summary_type, summary_type
                    ),
                    "cache_file_identifier": file_identifier,
                    "cache_summary_type": summary_type,
                    "cache_context_ui_id": user_input.context_ui_id,
                    "from_cache": True,
                },
            )

    # ── Шаг 4: запускаем агента ───────────────────────────────────────────────
    type_label = _SUMMARY_TYPE_LABELS.get(summary_type, summary_type)
    user_context = await _resolve_user_context(user_input, user_id)
    user_context["preferred_summary_format"] = summary_type

    if is_uuid and effective_path:
        instructions = f"Работай с вложением {effective_path}. "
    else:
        instructions = ""

    agent_msg = f"{instructions}Проанализируй этот файл и выдели {type_label}."

    logger.info("SUMMARIZE: calling agent, path='%s'", effective_path or "None")

    agent_result = await agent.chat(
        message=agent_msg,
        user_token=user_input.user_token,
        context_ui_id=user_input.context_ui_id,
        thread_id=new_thread_id,
        user_context=user_context,
        file_path=effective_path or current_path or None,
        file_name=user_input.file_name,
        human_choice=summary_type,
    )

    response_text = agent_result.get("content") or agent_result.get("response") or ""
    result_status = agent_result.get("status", "")

    # ── Шаг 5: сохраняем в кэши при успехе ───────────────────────────────────
    if result_status == "success" and response_text.strip() and file_identifier:
        # Redis
        await svc.save_result(
            document_id=user_input.context_ui_id,
            attachment_id=file_identifier if is_uuid else None,
            file_name=user_input.file_path if not is_uuid else None,
            summary_type=summary_type,
            result={
                "status": "success",
                "content": response_text,
                "metadata": agent_result.get("metadata", {}),
            },
        )
        # PostgreSQL
        await _save_db_cache(file_identifier, summary_type, response_text)
        logger.info(
            "SUMMARIZE: result cached (Redis + DB) for id=%s", file_identifier[:16]
        )

    # ── Шаг 6: cleanup локального файла ──────────────────────────────────────
    if current_path and not is_uuid and result_status == "success":
        background_tasks.add_task(_cleanup_file, current_path)

    # ── Шаг 7: обогащаем метаданные ответа ───────────────────────────────────
    if summary_type and result_status == "success":
        agent_result = SummarizeService.enrich_fresh_response(
            result=agent_result,
            summary_type=summary_type,
            attachment_id=file_identifier if is_uuid else None,
        )

    metadata = agent_result.get("metadata", {})
    metadata.update({
        "cache_file_identifier": file_identifier,
        "cache_summary_type": summary_type,
        "cache_context_ui_id": user_input.context_ui_id,
        "from_cache": False,
    })

    return AssistantResponse(
        status=result_status or "success",
        response=response_text or "Анализ завершён.",
        action_type=agent_result.get("action_type"),
        message=agent_result.get("message"),
        thread_id=new_thread_id,
        requires_reload=agent_result.get("requires_reload", False),
        navigate_url=agent_result.get("navigate_url"),
        metadata=metadata,
    )


@app.post("/appeal/autofill", response_model=AssistantResponse, tags=["Actions"])
async def appeal_autofill_endpoint(
    request: AppealAutofillRequest,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    """Auto-fill appeal document card from attachment content."""
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
async def get_document_endpoint(document_id: str, token: str) -> dict:
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
    """Upload a local file for in-chat analysis. Returns file_path and file_name."""
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

        logger.info(
            "File uploaded",
            extra={"orig": file.filename, "dest": str(dest_path)},
        )
        return FileUploadResponse(file_path=str(dest_path), file_name=file.filename)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File upload failed", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Ошибка при сохранении файла"
        ) from exc


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
    return {
        "name": "EDMS AI Assistant",
        "version": app.version,
        "status": "running",
    }


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