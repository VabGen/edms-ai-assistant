# edms_ai_assistant/main.py
from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

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
from edms_ai_assistant.api.routes.cache import router as cache_router
from edms_ai_assistant.api.routes.settings import router as settings_router
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.config import settings
from edms_ai_assistant.db.database import init_db
from edms_ai_assistant.model import (
    AssistantResponse,
    FileUploadResponse,
    NewChatRequest,
    UserInput,
)
from edms_ai_assistant.security import extract_user_id_from_token
from edms_ai_assistant.services.document_service import close_redis, init_redis
from edms_ai_assistant.services.summarization_orchestrator import (
    get_orchestrator,
)
from edms_ai_assistant.utils.hash_utils import get_file_hash
from edms_ai_assistant.utils.regex_utils import UUID_RE

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

_agent: EdmsDocumentAgent | None = None


def get_agent() -> EdmsDocumentAgent:
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="ИИ-Агент не инициализирован. Повторите попытку позже.",
        )
    return _agent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing database...")
    await init_db()

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

    get_orchestrator()
    logger.info("SummarizationOrchestrator ready")

    yield

    await close_redis()

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("Temporary upload directory removed")


app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.2.0",
    description="AI-powered assistant for EDMS document management workflows.",
    lifespan=lifespan,
)

if isinstance(settings.ALLOWED_ORIGINS, list):
    allowed_origins = settings.ALLOWED_ORIGINS
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings_router)
app.include_router(cache_router)


def _is_system_attachment(file_path: str | None) -> bool:
    return bool(file_path and UUID_RE.match(str(file_path)))


def _cleanup_file(file_path: str) -> None:
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


def _unwrap_text_from_agent_result(content: str) -> str:
    """Extract plain text from agent result that may be a JSON envelope."""
    if not content or not content.strip().startswith("{"):
        return content
    try:
        payload = json.loads(content)
        for key in ("content", "text", "document_info"):
            val = payload.get(key)
            if val and isinstance(val, str) and len(val) > 50:
                return val
    except (json.JSONDecodeError, AttributeError):
        pass
    return content


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
    user_id = extract_user_id_from_token(user_input.user_token)
    thread_id = (
        user_input.thread_id
        or f"user_{user_id}_doc_{user_input.context_ui_id or 'general'}"
    )

    user_context = await _resolve_user_context(user_input, user_id)

    if (
        user_input.preferred_summary_format
        and user_input.preferred_summary_format != "ask"
    ):
        user_context["preferred_summary_format"] = user_input.preferred_summary_format

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
        _is_disambiguation = result.get("action_type") in (
            "requires_disambiguation",
            "summarize_selection",
        )
        _should_cleanup = (
            result_status not in ("requires_action",)
            and not _is_file_operation
            and not _is_continuation
            and not _is_disambiguation
            and result.get("requires_reload", False)
        )
        if _should_cleanup:
            background_tasks.add_task(_cleanup_file, user_input.file_path)

    final_response_text = result.get("content") or result.get("message")

    return AssistantResponse(
        status=result.get("status") or "success",
        response=final_response_text,
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
        requires_reload=result.get("requires_reload", False),
        navigate_url=result.get("navigate_url"),
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
    """Direct summarization endpoint."""
    current_path = (user_input.file_path or "").strip()
    is_uuid = _is_system_attachment(current_path)
    summary_type = user_input.human_choice or "extractive"

    logger.info(
        "START SUMMARIZE",
        extra={
            "path_prefix": current_path[:40],
            "is_uuid": is_uuid,
            "context": user_input.context_ui_id,
            "summary_type": summary_type,
        },
    )

    user_id = extract_user_id_from_token(user_input.user_token)
    new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:8]}"
    file_identifier: str | None = None

    # ── 1. Resolve stable file_identifier ─────────────────────────────────
    if is_uuid:
        file_identifier = current_path

    elif current_path and Path(current_path).exists():
        try:
            file_identifier = get_file_hash(current_path)
        except Exception as exc:
            logger.warning("Could not hash local file: %s", exc)

    elif user_input.context_ui_id:
        try:
            async with DocumentClient() as doc_client:
                doc_dto = await doc_client.get_document_metadata(
                    user_input.user_token, user_input.context_ui_id
                )
                attachments: list = []
                if hasattr(doc_dto, "attachmentDocument"):
                    attachments = doc_dto.attachmentDocument or []
                elif isinstance(doc_dto, dict):
                    attachments = doc_dto.get("attachmentDocument") or []

                if attachments:

                    def _norm(s: str) -> str:
                        return re.sub(r"[^a-zA-Zа-яА-Я0-9]", "", s.lower()) if s else ""

                    clean_input = _norm(current_path)
                    if clean_input:
                        for att in attachments:
                            att_name = (
                                (
                                    att.get("name", "")
                                    if isinstance(att, dict)
                                    else getattr(att, "name", "")
                                )
                                or ""
                            ).strip()
                            att_id = str(
                                (
                                    att.get("id", "")
                                    if isinstance(att, dict)
                                    else getattr(att, "id", "")
                                )
                                or ""
                            )
                            if clean_input in _norm(att_name):
                                file_identifier = att_id
                                break

                    if not file_identifier and attachments:
                        first = attachments[0]
                        file_identifier = (
                            str(
                                (
                                    first.get("id", "")
                                    if isinstance(first, dict)
                                    else getattr(first, "id", "")
                                )
                                or ""
                            )
                            or None
                        )

                    if file_identifier:
                        current_path = file_identifier
                        is_uuid = True

        except Exception as exc:
            logger.error("Error resolving EDMS attachments: %s", exc)

    # ── 2. Cache lookup via public orchestrator method ─────────────────────
    orchestrator = get_orchestrator()

    if file_identifier:
        cached = await orchestrator.check_cache(file_identifier, summary_type)
        if cached is not None:
            logger.info("Cache HIT — returning without agent call")
            return AssistantResponse(
                status="success",
                response=cached.content,
                thread_id=new_thread_id,
                metadata={
                    "cache_file_identifier": file_identifier,
                    "cache_summary_type": summary_type,
                    "cache_context_ui_id": user_input.context_ui_id,
                    "from_cache": True,
                    "pipeline": cached.pipeline,
                    "quality_score": cached.quality_score,
                },
            )

    # ── 3. Extract raw text via agent ──────────────────────────────────────
    user_context = await _resolve_user_context(user_input, user_id)
    instructions = f"Работай с вложением {current_path}. " if is_uuid else ""
    extract_msg = (
        f"{instructions}Прочитай файл и верни его полное текстовое содержимое. "
        "Не суммаризируй — верни исходный текст."
    )

    extract_result = await agent.chat(
        message=extract_msg,
        user_token=user_input.user_token,
        context_ui_id=user_input.context_ui_id,
        thread_id=new_thread_id,
        user_context=user_context,
        file_path=current_path,
        human_choice=None,
    )

    raw_text = _unwrap_text_from_agent_result(
        extract_result.get("content") or extract_result.get("response") or ""
    )

    # ── 4a. Fallback: insufficient text → let agent do full summarization ──
    if not raw_text or len(raw_text.strip()) < 30:
        logger.warning(
            "Text extraction returned < 30 chars — falling back to agent summarize"
        )

        if not await orchestrator.check_llm_health():
            return AssistantResponse(
                status="error",
                response=(
                    "Не удалось подключиться к сервису анализа (LLM недоступен). "
                    "Пожалуйста, повторите попытку позже или обратитесь к администратору."
                ),
                thread_id=new_thread_id,
                metadata={
                    "error": "llm_unreachable",
                    "cache_file_identifier": file_identifier,
                    "from_cache": False,
                },
            )

        _type_labels = {
            "extractive": "ключевые факты, даты, суммы",
            "abstractive": "краткое изложение своими словами",
            "thesis": "структурированный тезисный план",
        }
        type_label = _type_labels.get(summary_type, summary_type)
        fallback_result = await agent.chat(
            message=f"{instructions}Проанализируй этот файл и выдели {type_label}.",
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
            thread_id=new_thread_id,
            user_context=user_context,
            file_path=current_path,
            human_choice=summary_type,
        )
        response_text = (
            fallback_result.get("content") or fallback_result.get("response") or ""
        )
        return AssistantResponse(
            status=fallback_result.get("status", "success"),
            response=response_text or "Анализ завершён.",
            thread_id=new_thread_id,
            metadata={
                **fallback_result.get("metadata", {}),
                "cache_file_identifier": file_identifier,
                "cache_summary_type": summary_type,
                "cache_context_ui_id": user_input.context_ui_id,
                "from_cache": False,
                "pipeline": "agent_fallback",
            },
        )

    # ── 4b. Run orchestrator (routing + map-reduce + self-critique + cache) ─
    try:
        orch_result = await orchestrator.summarize(
            text=raw_text,
            summary_type=summary_type,
            file_identifier=file_identifier,
        )

        # Schedule local file cleanup (EDMS attachments are never on disk)
        if current_path and not is_uuid and Path(current_path).exists():
            background_tasks.add_task(_cleanup_file, current_path)

        return AssistantResponse(
            status=(
                "success" if orch_result.status in ("success", "partial") else "error"
            ),
            response=orch_result.content or "Анализ завершён.",
            thread_id=new_thread_id,
            metadata={
                "cache_file_identifier": file_identifier,
                "cache_summary_type": summary_type,
                "cache_context_ui_id": user_input.context_ui_id,
                "from_cache": False,
                "pipeline": orch_result.pipeline,
                "chunks_processed": orch_result.chunks_processed,
                "processing_time_ms": orch_result.processing_time_ms,
                "quality_score": orch_result.quality_score,
                "confidence": orch_result.confidence,
                "degraded": orch_result.degraded,
            },
        )

    except Exception as exc:
        logger.error("Orchestrator error in /actions/summarize: %s", exc, exc_info=True)
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

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    return {
        "status": "ok",
        "version": app.version,
        "components": agent.health_check(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        reload_excludes=[".venv", "*.pyc", "__pycache__"],
        log_level=settings.LOGGING_LEVEL.lower(),
    )