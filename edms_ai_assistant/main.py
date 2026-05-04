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
    Request,
    UploadFile,
)
from langchain_core.messages import AIMessage, HumanMessage
from starlette.middleware.cors import CORSMiddleware

from edms_ai_assistant.agent.agent import EdmsDocumentAgent
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
from edms_ai_assistant.utils.hash_utils import get_file_hash
from edms_ai_assistant.utils.regex_utils import UUID_RE

from edms_ai_assistant.summarizer.container import build_summarization_service
from edms_ai_assistant.summarizer.api.router import router as summarizer_router

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

_agent: EdmsDocumentAgent | None = None


def get_agent() -> EdmsDocumentAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="ИИ-Агент не инициализирован.")
    return _agent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan — startup and shutdown."""
    global _agent

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    await init_redis()

    try:
        _agent = EdmsDocumentAgent()
        health_status = await _agent.health_check()
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

app.include_router(settings_router)
app.include_router(summarizer_router)


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


def _format_output_as_text(resp) -> str:
    """Format structured output as human-readable text for legacy clients."""
    from edms_ai_assistant.summarizer.structured.models import SummaryMode

    output = resp.output

    if resp.mode == SummaryMode.EXECUTIVE:
        lines = [f"**{output.get('headline', '')}**", ""]
        for bullet in output.get("bullets", []):
            lines.append(f"• {bullet}")
        rec = output.get("recommendation")
        if rec:
            lines.extend(["", f"💡 **Рекомендация:** {rec}"])
        return "\n".join(lines)

    elif resp.mode == SummaryMode.ACTION_ITEMS:
        items = output.get("action_items", [])
        if not items:
            return "Задачи и поручения не найдены."
        lines = [f"**Найдено задач: {len(items)}**", ""]
        for i, item in enumerate(items, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                item.get("priority", "medium"), "⚪"
            )
            lines.append(f"{i}. {priority_emoji} {item.get('task', '')}")
            if item.get("owner"):
                lines.append(f"   Ответственный: {item['owner']}")
            if item.get("deadline"):
                lines.append(f"   Срок: {item['deadline']}")
        return "\n".join(lines)

    elif resp.mode == SummaryMode.THESIS:
        sections = output.get("sections", [])
        lines = [f"**{output.get('main_argument', 'Анализ документа')}**", ""]
        for sec in sections:
            lines.append(f"## {sec.get('title', '')}")
            lines.append(sec.get("thesis", ""))
            for pt in sec.get("points", []):
                lines.append(f"- {pt.get('claim', '')}")
            lines.append("")
        return "\n".join(lines)

    elif resp.mode == SummaryMode.EXTRACTIVE:
        facts = output.get("facts", [])
        lines = [output.get("document_summary", ""), ""]
        for fact in facts:
            lines.append(f"- **{fact.get('label', '')}**: {fact.get('value', '')}")
        return "\n".join(lines)

    else:
        return (
                output.get("summary", "")
                or output.get("content", "")
                or json.dumps(output, ensure_ascii=False, indent=2)
        )


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
        "сравни", "сравнение", "сравн", "compare", "отличи", "анализ",
        "проанализируй", "суммаризир", "прочит", "содержим", "прочти",
        "что в файл", "читай", "изучи",
    )
    _is_file_operation = any(
        kw in (user_input.message or "").lower() for kw in _FILE_OPERATION_KEYWORDS
    )
    _is_continuation = bool(user_input.human_choice)

    if user_input.file_path and not _is_system_attachment(user_input.file_path):
        result_status = result.get("status", "success")
        _is_disambiguation = result.get("action_type") in (
            "requires_disambiguation", "summarize_selection",
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
    summary="Trigger file summarization via v2 pipeline",
    tags=["Actions"],
)
async def api_direct_summarize(
        request: Request,
        user_input: UserInput,
        background_tasks: BackgroundTasks,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    from edms_ai_assistant.summarizer.service import SummarizationRequest
    from edms_ai_assistant.summarizer.structured.models import SummaryMode

    current_path = (user_input.file_path or "").strip()
    is_uuid = _is_system_attachment(current_path)
    summary_type = user_input.human_choice or "extractive"

    user_id = extract_user_id_from_token(user_input.user_token)
    new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:8]}"
    file_identifier: str | None = None

    logger.info(
        "START SUMMARIZE",
        extra={
            "path_prefix": current_path[:40],
            "is_uuid": is_uuid,
            "context": user_input.context_ui_id,
            "summary_type": summary_type,
        },
    )

    # ── 1. Resolve stable file_identifier ─────────────────────────────
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
                                    (att.get("name", "") if isinstance(att, dict) else getattr(att, "name", "")) or ""
                            ).strip()
                            att_id = str(
                                (att.get("id", "") if isinstance(att, dict) else getattr(att, "id", "")) or ""
                            )
                            if clean_input in _norm(att_name):
                                file_identifier = att_id
                                break

                    if not file_identifier and attachments:
                        first = attachments[0]
                        file_identifier = (
                                str((first.get("id", "") if isinstance(first, dict) else getattr(first, "id",
                                                                                                 "")) or "")
                                or None
                        )

                    if file_identifier:
                        current_path = file_identifier
                        is_uuid = True
        except Exception as exc:
            logger.error("Error resolving EDMS attachments: %s", exc)

    # ── 2. Try v2 pipeline ────────────────────────────────────────────
    service = getattr(request.app.state, "summarization_service", None)

    if service is not None:
        try:
            _mode_map = {
                "extractive": SummaryMode.EXTRACTIVE,
                "abstractive": SummaryMode.ABSTRACTIVE,
                "thesis": SummaryMode.THESIS,
            }
            mode = _mode_map.get(
                (user_input.human_choice or "abstractive").lower(),
                SummaryMode.ABSTRACTIVE,
            )

            file_bytes: bytes | None = None
            file_name = "document"

            # Try reading from local path
            if current_path and not is_uuid and Path(current_path).exists():
                async with aiofiles.open(current_path, "rb") as f:
                    file_bytes = await f.read()
                file_name = Path(current_path).name

            # Try reading from EDMS attachment
            elif is_uuid and user_input.context_ui_id:
                try:
                    from edms_ai_assistant.tools.attachment import doc_get_file_content

                    tool_input = {
                        "token": user_input.user_token,
                        "document_id": user_input.context_ui_id,
                        "attachment_id": current_path,
                    }
                    raw_result = await doc_get_file_content.ainvoke(tool_input)

                    # Надежная конвертация результата LangChain инструмента в байты
                    if isinstance(raw_result, bytes):
                        file_bytes = raw_result
                    elif isinstance(raw_result, str):
                        file_bytes = raw_result.encode("utf-8")
                    else:
                        file_bytes = str(raw_result).encode("utf-8")

                    if file_bytes:
                        file_name = "extracted_text.txt"

                    logger.info(
                        f"Prepared {len(file_bytes) if file_bytes else 0} bytes for pipeline (file_name={file_name})"
                    )
                except Exception as exc:
                    logger.warning("Failed to read EDMS attachment for v2: %s", exc)

            if file_bytes:
                req = SummarizationRequest(
                    file_content=file_bytes,
                    file_name=file_name,
                    mode=mode,
                    language="ru",
                    request_id=str(uuid.uuid4()),
                    force_refresh=False,
                )

                resp = await service.summarize(req)
                output_text = _format_output_as_text(resp)

                if current_path and not is_uuid and Path(current_path).exists():
                    background_tasks.add_task(_cleanup_file, current_path)

                return AssistantResponse(
                    status="success",
                    response=output_text,
                    thread_id=f"v2_{req.request_id[:8]}",
                    metadata={
                        "cache_file_identifier": resp.file_hash,
                        "cache_summary_type": mode.value,
                        "from_cache": resp.cache_hit,
                        "pipeline": resp.chunking_strategy,
                        "cost_usd": resp.cost_usd,
                        "v2": True,
                    },
                )
            else:
                logger.warning("file_bytes is empty for v2 pipeline, falling back to agent.")

        except Exception as exc:
            logger.warning(
                "v2 pipeline failed (%s) — falling back to agent", exc, exc_info=True
            )

    # ── 3. Agent fallback ─────────────────────────────────────────────
    logger.info("Using agent fallback for summarization")

    raw_text = ""
    try:
        if is_uuid and user_input.context_ui_id:
            from edms_ai_assistant.tools.attachment import doc_get_file_content
            tool_input = {
                "token": user_input.user_token,
                "document_id": user_input.context_ui_id,
                "attachment_id": current_path,
            }
            raw_text = await doc_get_file_content.ainvoke(tool_input)
            raw_text = str(raw_text)
        elif current_path and Path(current_path).exists():
            from edms_ai_assistant.services.file_processor import FileProcessorService
            raw_text = await FileProcessorService.extract_text_async(current_path)
    except Exception as exc:
        logger.warning("Direct text extraction failed (%s), falling back to Agent...", exc)

    if not raw_text or len(raw_text.strip()) < 30:
        logger.info("Using Agent fallback for text extraction...")
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

    if not raw_text or len(raw_text.strip()) < 30:
        logger.warning("Text extraction returned < 30 chars — cannot summarize")
        return AssistantResponse(
            status="error",
            response="Не удалось извлечь текст из файла. Пожалуйста, убедитесь, что файл содержит текстовое содержимое.",
            thread_id=new_thread_id,
            metadata={"error": "text_extraction_failed", "cache_file_identifier": file_identifier, "from_cache": False,
                      "pipeline": "agent_fallback"},
        )

    _type_labels = {
        "extractive": "ключевые факты, даты, суммы",
        "abstractive": "краткое изложение своими словами",
        "thesis": "структурированный тезисный план",
    }
    type_label = _type_labels.get(summary_type, summary_type)
    instructions = f"Работай с вложением {current_path}. " if is_uuid else ""
    user_context = await _resolve_user_context(user_input, user_id)

    fallback_result = await agent.chat(
        message=f"{instructions}Проанализируй этот файл и выдели {type_label}.",
        user_token=user_input.user_token,
        context_ui_id=user_input.context_ui_id,
        thread_id=new_thread_id,
        user_context=user_context,
        file_path=current_path,
        human_choice=summary_type,
    )
    response_text = fallback_result.get("content") or fallback_result.get("response") or ""

    if current_path and not is_uuid and Path(current_path).exists():
        background_tasks.add_task(_cleanup_file, current_path)

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
        state = await agent._state_manager.get_state(thread_id)
        messages = state.values.get("messages", [])
        filtered = []
        for m in messages:
            if not isinstance(m, (HumanMessage, AIMessage)):
                continue
            if isinstance(m, AIMessage) and not m.content:
                continue
            filtered.append({"type": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content})
        return {"messages": filtered}
    except Exception as exc:
        logger.error("History retrieval failed", extra={"thread_id": thread_id, "error": str(exc)})
        return {"messages": []}


@app.post("/chat/new", summary="Create a new conversation thread", tags=["Chat"])
async def create_new_thread(request: NewChatRequest) -> dict:
    try:
        user_id = extract_user_id_from_token(request.user_token)
        new_thread_id = f"chat_{user_id}_{uuid.uuid4().hex[:8]}"
        return {"status": "success", "thread_id": new_thread_id}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


@app.post("/upload-file", response_model=FileUploadResponse, summary="Upload a file for in-chat analysis",
          tags=["Files"])
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

        logger.info("File uploaded", extra={"orig_filename": file.filename, "dest": str(dest_path)})
        return FileUploadResponse(file_path=str(dest_path), file_name=file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File upload failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла") from exc


@app.get("/health", summary="Agent and service health check", tags=["System"])
async def health_check(agent: Annotated[EdmsDocumentAgent, Depends(get_agent)], request: Request,) -> dict:
    health_data = await agent.health_check()
    return {"status": "ok", "version": request.app.version, "components": health_data}


if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        reload_excludes=[".venv", "*.pyc", "__pycache__"],
        log_level=settings.LOGGING_LEVEL.lower(),
    )