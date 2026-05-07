# edms_ai_assistant/api/routes/actions.py
"""
Actions API routes.

Endpoints:
    POST /actions/summarize — trigger file summarization via v2 pipeline with agent fallback
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from edms_ai_assistant.agent.agent import EdmsDocumentAgent
from edms_ai_assistant.api.deps import get_agent
from edms_ai_assistant.api.helpers import (
    cleanup_file,
    format_output_as_text,
    is_system_attachment,
    resolve_user_context,
    unwrap_text_from_agent_result,
)
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.model import AssistantResponse, UserInput
from edms_ai_assistant.security import extract_user_id_from_token
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.summarizer.service import SummarizationRequest
from edms_ai_assistant.summarizer.structured.models import SummaryMode
from edms_ai_assistant.tools.attachment import doc_get_file_content
from edms_ai_assistant.utils.hash_utils import get_file_hash

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Actions"])

_MODE_MAP: dict[str, SummaryMode] = {
    "extractive": SummaryMode.EXTRACTIVE,
    "abstractive": SummaryMode.ABSTRACTIVE,
    "thesis": SummaryMode.THESIS,
}


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
) -> AssistantResponse:
    current_path = (user_input.file_path or "").strip()
    is_uuid = is_system_attachment(current_path)
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
                            str((first.get("id", "") if isinstance(first, dict) else getattr(first, "id", "")) or "")
                            or None
                        )

                    if file_identifier:
                        current_path = file_identifier
                        is_uuid = True
        except Exception as exc:
            logger.error("Error resolving EDMS attachments: %s", exc)

    # ── 2. Try v2 pipeline ─────────────────────────────────────────────────────
    service = getattr(request.app.state, "summarization_service", None)

    if service is not None:
        try:
            mode = _MODE_MAP.get(
                (user_input.human_choice or "abstractive").lower(),
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
                    tool_input = {
                        "token": user_input.user_token,
                        "document_id": user_input.context_ui_id,
                        "attachment_id": current_path,
                    }
                    raw_result = await doc_get_file_content.ainvoke(tool_input)

                    if isinstance(raw_result, bytes):
                        file_bytes = raw_result
                    elif isinstance(raw_result, str):
                        file_bytes = raw_result.encode("utf-8")
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
                output_text = format_output_as_text(resp)

                if current_path and not is_uuid and Path(current_path).exists():
                    background_tasks.add_task(cleanup_file, current_path)

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

            logger.warning("file_bytes is empty for v2 pipeline, falling back to agent.")

        except Exception as exc:
            logger.warning(
                "v2 pipeline failed (%s) — falling back to agent", exc, exc_info=True
            )

    # ── 3. Agent fallback ──────────────────────────────────────────────────────
    logger.info("Using agent fallback for summarization")

    raw_text = ""
    try:
        if is_uuid and user_input.context_ui_id:
            tool_input = {
                "token": user_input.user_token,
                "document_id": user_input.context_ui_id,
                "attachment_id": current_path,
            }
            raw_text = str(await doc_get_file_content.ainvoke(tool_input))
        elif current_path and Path(current_path).exists():
            raw_text = await FileProcessorService.extract_text_async(current_path)
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
        extract_result = await agent.chat(
            message=extract_msg,
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
            thread_id=new_thread_id,
            user_context=user_context,
            file_path=current_path,
            human_choice=None,
        )
        raw_text = unwrap_text_from_agent_result(
            extract_result.get("content") or extract_result.get("response") or ""
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
