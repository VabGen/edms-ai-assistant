# edms_ai_assistant/api/routes/actions.py
"""
Actions API routes.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Annotated, Any, TYPE_CHECKING

import json

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from edms_ai_assistant.api.deps import DepsDep, get_agent, get_deps
from edms_ai_assistant.api.helpers import (
    cleanup_file,
    is_system_attachment,
    resolve_user_context,
    unwrap_text_from_agent_result,
)
from edms_ai_assistant.model import AssistantResponse, UserInput
from edms_ai_assistant.security import extract_user_id_from_token
from edms_ai_assistant.summarizer.errors import SummarizerError
from edms_ai_assistant.summarizer.pipeline.direct import StreamEvent
from edms_ai_assistant.summarizer.service import (
    SummarizationRequest,
    SummarizationResponse,
)
from edms_ai_assistant.summarizer.structured.models import SummaryMode
from edms_ai_assistant.tools.attachment import create_attachment_fetch_tool
from edms_ai_assistant.utils.hash_utils import get_file_hash

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from edms_ai_assistant.core.deps import AppDeps
    from edms_ai_assistant.agent.agent import EdmsDocumentAgent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Actions"])

_MODE_MAP: dict[str, SummaryMode] = {
    "extractive": SummaryMode.EXTRACTIVE,
    "abstractive": SummaryMode.ABSTRACTIVE,
    "thesis": SummaryMode.THESIS,
}


# ── Helper: build RunnableConfig for direct tool invocation ────────────────

def _make_tool_config(
    user_token: str,
    document_id: str | None,
    thread_id: str = "action",
    user_id: str = "action_user",
) -> dict:
    """Build RunnableConfig for direct tool invocation outside the agent graph."""
    cfg: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "user_token": user_token,
            "user_id": user_id,
        }
    }
    if document_id:
        cfg["configurable"]["document_id"] = document_id
    return cfg


async def _run_agent_once(
    *,
    agent: EdmsDocumentAgent,
    message: str,
    user_token: str,
    context_ui_id: str | None,
    thread_id: str,
    user_context: dict,
    file_path: str | None,
    file_name: str | None,
) -> str:
    """Run the LangGraph agent once and return the final assistant text.

    Used by /actions/summarize fallback paths only. These paths are not
    expected to require HITL — they ask the agent to perform a single
    file-extraction or analysis turn and consume its final AIMessage. If a
    tool ever does suspend with ``interrupt()`` here, ``ainvoke`` raises
    ``GraphInterrupt``; we surface this as an empty string so the caller
    can fall back gracefully.
    """
    from langchain_core.messages import AIMessage
    from langgraph.errors import GraphInterrupt

    inputs, _ctx = agent.build_initial_inputs(
        message=message,
        user_token=user_token,
        context_ui_id=context_ui_id,
        thread_id=thread_id,
        user_context=user_context,
        file_path=file_path,
        file_name=file_name,
    )
    configurable: dict[str, str] = {
        "thread_id": thread_id,
        "user_token": user_token,
    }
    if context_ui_id:
        configurable["document_id"] = context_ui_id
    config = {"configurable": configurable}
    try:
        final_state = await agent.graph.ainvoke(inputs, config=config)
    except GraphInterrupt:
        logger.warning(
            "actions._run_agent_once: tool suspended on a non-interactive "
            "path (thread=%s); returning empty content",
            thread_id,
        )
        return ""
    messages = (final_state or {}).get("messages", []) or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = str(msg.content or "").strip()
            if content:
                return content
    return ""


async def _resolve_file_bytes(
    *,
    current_path: str,
    is_uuid: bool,
    user_input: UserInput,
    deps: DepsDep,
) -> tuple[bytes | None, str]:
    """Достаёт байты документа: локальный файл или EDMS-вложение.

    Возвращает (file_bytes, file_name). bytes=None если резолв не удался.
    """
    file_name = "document"
    file_bytes: bytes | None = None

    if current_path and not is_uuid and Path(current_path).exists():
        async with aiofiles.open(current_path, "rb") as f:
            file_bytes = await f.read()
        file_name = Path(current_path).name
        return file_bytes, file_name

    if is_uuid and user_input.context_ui_id:
        try:
            try:
                uid = str(extract_user_id_from_token(user_input.user_token))
            except Exception:
                uid = "action_user"

            tool_config = _make_tool_config(
                user_token=user_input.user_token,
                document_id=user_input.context_ui_id,
                thread_id="action_resolve",
                user_id=uid,
            )
            doc_get_file_content = create_attachment_fetch_tool(deps)
            raw_result = await doc_get_file_content.ainvoke(
                {"attachment_id": current_path},
                config=tool_config,
            )
            if isinstance(raw_result, bytes):
                file_bytes = raw_result
            elif isinstance(raw_result, str):
                file_bytes = raw_result.encode("utf-8")
            elif isinstance(raw_result, dict):
                content_text = raw_result.get("content", "")
                if content_text:
                    file_bytes = content_text.encode("utf-8")
            else:
                file_bytes = str(raw_result).encode("utf-8")

            if file_bytes:
                file_name = "extracted_text.txt"
        except Exception as exc:
            logger.warning("Failed to read EDMS attachment: %s", exc)

    return file_bytes, file_name


@router.post(
    "/actions/summarize",
    response_model=AssistantResponse,
    summary="Trigger file summarization via v2 pipeline",
)
async def api_direct_summarize(
    request: Request,
    user_input: UserInput,
    background_tasks: BackgroundTasks,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
    deps: Annotated[AppDeps, Depends(get_deps)],
) -> AssistantResponse:
    current_path = (user_input.file_path or "").strip()
    is_uuid = is_system_attachment(current_path)
    summary_type = user_input.preferred_summary_format or "extractive"

    user_id = extract_user_id_from_token(user_input.user_token)
    new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:8]}"
    file_identifier: str | None = None

    # Build tool config for direct tool invocations
    tool_config = _make_tool_config(
        user_token=user_input.user_token,
        document_id=user_input.context_ui_id,
        thread_id=new_thread_id,
        user_id=str(user_id),
    )

    logger.info(
        "START SUMMARIZE",
        extra={
            "path_prefix": current_path[:40],
            "is_uuid": is_uuid,
            "context": user_input.context_ui_id,
            "summary_type": summary_type,
        },
    )

    # ── 1. Resolve stable file_identifier ─────────────────────────────────────
    if is_uuid:
        file_identifier = current_path
    elif current_path and Path(current_path).exists():
        try:
            file_identifier = get_file_hash(current_path)
        except Exception as exc:
            logger.warning("Could not hash local file: %s", exc)
    elif user_input.context_ui_id:
        try:
            doc_client = deps.document_client
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

    # ── 2. Try pipeline ─────────────────────────────────────────────────────
    service = getattr(request.app.state, "summarization_service", None)

    if service is not None:
        try:
            mode = _MODE_MAP.get(
                (user_input.preferred_summary_format or "abstractive").lower(),
                SummaryMode.ABSTRACTIVE,
            )

            file_bytes: bytes | None = None
            file_name = "document"

            if current_path and not is_uuid and Path(current_path).exists():
                async with aiofiles.open(current_path, "rb") as f:
                    file_bytes = await f.read()
                file_name = Path(current_path).name

            elif is_uuid and user_input.context_ui_id:
                try:
                    doc_get_file_content = create_attachment_fetch_tool(deps)
                    raw_result = await doc_get_file_content.ainvoke(
                        {"attachment_id": current_path},
                        config=tool_config,
                    )

                    if isinstance(raw_result, bytes):
                        file_bytes = raw_result
                    elif isinstance(raw_result, str):
                        file_bytes = raw_result.encode("utf-8")
                    elif isinstance(raw_result, dict):
                        content_text = raw_result.get("content", "")
                        if content_text:
                            file_bytes = content_text.encode("utf-8")
                        else:
                            logger.warning(
                                "doc_get_file_content returned dict without 'content'"
                            )
                    else:
                        file_bytes = str(raw_result).encode("utf-8")

                    if file_bytes:
                        file_name = "extracted_text.txt"

                    logger.info(
                        "Prepared %d bytes for pipeline (file_name=%s)",
                        len(file_bytes) if file_bytes else 0,
                        file_name,
                    )
                except Exception as exc:
                    logger.warning("Failed to read EDMS attachment for v2: %s", exc)

            if file_bytes and len(file_bytes) > 10:
                req = SummarizationRequest(
                    file_content=file_bytes,
                    file_name=file_name,
                    mode=mode,
                    language="ru",
                    request_id=str(uuid.uuid4()),
                    force_refresh=False,
                )

                resp = await service.summarize(req)
                # Return raw JSON for the frontend to render structured UI
                output_text = json.dumps(resp.output, ensure_ascii=False)

                if current_path and not is_uuid and Path(current_path).exists():
                    background_tasks.add_task(cleanup_file, current_path)

                return AssistantResponse(
                    status="success",
                    response=output_text,
                    thread_id=f"v2_{req.request_id[:8]}",
                    metadata={
                        "cache_file_identifier": resp.file_hash,
                        "cache_summary_type": mode.value,
                        "cache_context_ui_id": user_input.context_ui_id,
                        "from_cache": resp.cache_hit,
                        "pipeline": resp.chunking_strategy,
                        "cost_usd": resp.cost_usd,
                        "v2": True,
                    },
                )

            logger.warning(
                "file_bytes is empty for v2 pipeline, falling back to agent."
            )

        except Exception as exc:
            logger.warning(
                "v2 pipeline failed (%s) — falling back to agent", exc, exc_info=True
            )

    # ── 3. Agent fallback ──────────────────────────────────────────────────────
    logger.info("Using agent fallback for summarization")

    raw_text = ""
    try:
        if is_uuid and user_input.context_ui_id:
            doc_get_file_content = create_attachment_fetch_tool(deps)
            result = await doc_get_file_content.ainvoke(
                {"attachment_id": current_path},
                config=tool_config,
            )
            if isinstance(result, dict):
                raw_text = result.get("content", "") or ""
            elif isinstance(result, bytes):
                raw_text = result.decode("utf-8", errors="replace")
            else:
                raw_text = str(result)
        elif current_path and Path(current_path).exists():
            file_processor = deps.file_processor_service
            raw_text = await file_processor.extract_text_async(current_path)
    except Exception as exc:
        logger.warning(
            "Direct text extraction failed (%s), falling back to Agent...", exc
        )

    if not raw_text or len(raw_text.strip()) < 30:
        logger.info("Using Agent fallback for text extraction...")
        user_context = await resolve_user_context(user_input, user_id)
        instructions = f"Работай с вложением {current_path}. " if is_uuid else ""
        extract_msg = (
            f"{instructions}Прочитай файл и верни его полное текстовое содержимое. "
            "Не суммаризируй — верни исходный текст."
        )
        extract_result = await _run_agent_once(
            agent=agent,
            message=extract_msg,
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
            thread_id=new_thread_id,
            user_context=user_context,
            file_path=current_path,
            file_name=None,
        )
        raw_text = unwrap_text_from_agent_result(
            extract_result or ""
        )

    if not raw_text or len(raw_text.strip()) < 30:
        logger.warning("Text extraction returned < 30 chars — cannot summarize")
        return AssistantResponse(
            status="error",
            response="Не удалось извлечь текст из файла. Пожалуйста, убедитесь, что файл содержит текстовое содержимое.",
            thread_id=new_thread_id,
            metadata={
                "error": "text_extraction_failed",
                "cache_file_identifier": file_identifier,
                "from_cache": False,
                "pipeline": "agent_fallback",
            },
        )

    _type_labels = {
        "extractive": "ключевые факты, даты, суммы",
        "abstractive": "краткое изложение своими словами",
        "thesis": "структурированный тезисный план",
    }
    type_label = _type_labels.get(summary_type, summary_type)
    instructions = f"Работай с вложением {current_path}. " if is_uuid else ""
    user_context = await resolve_user_context(user_input, user_id)

    user_context = dict(user_context)
    user_context["preferred_summary_format"] = summary_type
    response_text = await _run_agent_once(
        agent=agent,
        message=f"{instructions}Проанализируй этот файл и выдели {type_label}.",
        user_token=user_input.user_token,
        context_ui_id=user_input.context_ui_id,
        thread_id=new_thread_id,
        user_context=user_context,
        file_path=current_path,
        file_name=None,
    )
    fallback_result: dict = {"content": response_text}
    _legacy_ignored = (
        fallback_result.get("content") or fallback_result.get("response") or ""
    )

    if current_path and not is_uuid and Path(current_path).exists():
        background_tasks.add_task(cleanup_file, current_path)

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


# ─────────────────────────────────────────────────────────────────────────────
# SSE streaming endpoint
# ─────────────────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Формирует одну SSE data-строку."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post(
    "/actions/summarize/stream",
    summary="Streamed file summarization (SSE)",
    response_class=StreamingResponse,
)
async def api_direct_summarize_stream(
    request: Request,
    user_input: UserInput,
    deps: Annotated[AppDeps, Depends(get_deps)],
) -> StreamingResponse:
    """
    SSE-стриминг суммаризации с тем же JSON-контрактом, что /actions/summarize.

    Эмитит:
      data: {"event":"delta","text":"..."}      — токены LLM по мере генерации
      data: {"event":"result","response":"...", "metadata":{...},
             "thread_id":"...","status":"success"}
                                                  — финальный ответ (как в non-stream)
      data: {"event":"error","message":"..."}   — ошибка (4xx-friendly или generic 5xx)
      data: [DONE]                              — терминатор стрима
    """
    service = getattr(request.app.state, "summarization_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SummarizationService недоступен; используйте /actions/summarize.",
        )

    current_path = (user_input.file_path or "").strip()
    is_uuid_input = is_system_attachment(current_path)
    summary_type = (user_input.preferred_summary_format or "abstractive").lower()
    mode = _MODE_MAP.get(summary_type, SummaryMode.ABSTRACTIVE)

    user_id = extract_user_id_from_token(user_input.user_token)
    new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:8]}"

    file_identifier: str | None = None
    if is_uuid_input:
        file_identifier = current_path
    elif current_path and Path(current_path).exists():
        try:
            file_identifier = get_file_hash(current_path)
        except Exception as exc:
            logger.warning("Could not hash local file: %s", exc)

    async def event_gen() -> AsyncIterator[str]:
        try:
            from edms_ai_assistant.api.sse import SSE_KEEPALIVE
            yield SSE_KEEPALIVE.decode()

            file_bytes, file_name = await _resolve_file_bytes(
                current_path=current_path,
                is_uuid=is_uuid_input,
                user_input=user_input,
                deps=deps,
            )
            if not file_bytes or len(file_bytes) <= 10:
                yield _sse({
                    "event": "error",
                    "message": "Не удалось прочитать содержимое документа.",
                })
                yield "data: [DONE]\n\n"
                return

            req = SummarizationRequest(
                file_content=file_bytes,
                file_name=file_name,
                mode=mode,
                language="ru",
                request_id=str(uuid.uuid4()),
                force_refresh=False,
            )

            final: SummarizationResponse | None = None
            async for event in service.summarize_stream(req):
                yield SSE_KEEPALIVE.decode()
                if isinstance(event, StreamEvent):
                    if event.kind == "delta" and event.text:
                        yield _sse({"event": "delta", "text": event.text})
                    elif event.kind == "error":
                        yield _sse({
                            "event": "error",
                            "message": "LLM streaming error",
                        })
                elif isinstance(event, SummarizationResponse):
                    final = event

            if final is None:
                yield _sse({
                    "event": "error",
                    "message": "Стриминг не вернул финальный результат.",
                })
                yield "data: [DONE]\n\n"
                return

            # Return raw JSON for the frontend to render structured UI
            response_text = json.dumps(final.output, ensure_ascii=False)
            yield _sse({
                "event": "result",
                "status": "success",
                "response": response_text or "{}",
                "thread_id": new_thread_id,
                "metadata": {
                    "cache_file_identifier": file_identifier or final.file_hash,
                    "cache_summary_type": mode.value,
                    "cache_context_ui_id": user_input.context_ui_id,
                    "from_cache": final.cache_hit,
                    "pipeline": final.chunking_strategy,
                    "cost_usd": final.cost_usd,
                    "input_tokens": final.input_tokens,
                    "output_tokens": final.output_tokens,
                    "latency_ms": final.latency_ms,
                    "v2": True,
                },
            })
            yield "data: [DONE]\n\n"

        except SummarizerError as exc:
            logger.warning("Stream summarization rejected: %s", exc)
            yield _sse({"event": "error", "message": str(exc)})
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Stream summarization failed: %s", exc, exc_info=True)
            yield _sse({"event": "error", "message": "Внутренняя ошибка сервера."})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
