# edms_ai_assistant/api/routes/chat.py
"""
Chat API routes.

Endpoints:
    POST /chat                      — main agent conversation endpoint
    GET  /chat/history/{thread_id}  — retrieve conversation history
    POST /chat/new                  — create a new conversation thread
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from edms_ai_assistant.agent.agent import EdmsDocumentAgent
from edms_ai_assistant.api.deps import get_agent
from edms_ai_assistant.api.helpers import (
    cleanup_file,
    is_system_attachment,
    resolve_user_context,
)
from edms_ai_assistant.model import AssistantResponse, NewChatRequest, UserInput
from edms_ai_assistant.security import extract_user_id_from_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

_FILE_OPERATION_KEYWORDS: tuple[str, ...] = (
    "сравни", "сравнение", "сравн", "compare", "отличи", "анализ",
    "проанализируй", "суммаризир", "прочит", "содержим", "прочти",
    "что в файл", "читай", "изучи",
)


@router.post(
    "/chat",
    response_model=AssistantResponse,
    summary="Send a message to the EDMS AI assistant",
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

    user_context = await resolve_user_context(user_input, user_id)

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

    _is_file_operation = any(
        kw in (user_input.message or "").lower() for kw in _FILE_OPERATION_KEYWORDS
    )
    _is_continuation = bool(user_input.human_choice)

    if user_input.file_path and not is_system_attachment(user_input.file_path):
        _is_disambiguation = result.get("action_type") in (
            "requires_disambiguation",
            "summarize_selection",
        )
        _should_cleanup = (
            result.get("status", "success") not in ("requires_action",)
            and not _is_file_operation
            and not _is_continuation
            and not _is_disambiguation
            and result.get("requires_reload", False)
        )
        if _should_cleanup:
            background_tasks.add_task(cleanup_file, user_input.file_path)

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


@router.get(
    "/chat/history/{thread_id}",
    summary="Get conversation history for a thread",
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


@router.post("/chat/new", summary="Create a new conversation thread")
async def create_new_thread(request: NewChatRequest) -> dict:
    try:
        user_id = extract_user_id_from_token(request.user_token)
        new_thread_id = f"chat_{user_id}_{uuid.uuid4().hex[:8]}"
        return {"status": "success", "thread_id": new_thread_id}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
