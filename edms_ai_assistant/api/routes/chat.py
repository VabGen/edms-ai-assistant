# edms_ai_assistant/api/routes/chat.py
"""
Chat API — native LangGraph HITL contract over SSE.

Endpoints:
    POST /chat/stream            — start a new turn, stream tokens + interrupts
    POST /chat/resume            — resume a paused graph with a typed value
    GET  /chat/state/{thread_id} — current snapshot + pending interrupts (reconnect)
    GET  /chat/history/{thread_id} — flat message list (UI history rendering)
    POST /chat/new               — allocate a fresh thread_id

Design contract:
    - The frontend never parses LLM text. Interruptions arrive as structured
      ``event: interrupt`` payloads matching
      ``edms_ai_assistant.agent.interrupt_contract.InterruptPayload``.
    - Structured tool outputs (compliance, navigate) arrive as
      ``event: ui_component`` payloads — the frontend renders interactive
      widgets directly instead of relying on LLM prose.
    - Resume is a pure POST that injects ``Command(resume=value)`` into the
      same ``thread_id``. No history mutation. No LLM re-invocation.
    - Both streaming endpoints respond with ``text/event-stream``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated, Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, Interrupt
from pydantic import BaseModel, Field, ValidationError

from edms_ai_assistant.agent.agent import EdmsDocumentAgent
from edms_ai_assistant.agent.interrupt_contract import (
    InterruptPayloadAdapter,
    ResumeValueAdapter,
)
from edms_ai_assistant.api.deps import get_agent
from edms_ai_assistant.api.helpers import resolve_user_context
from edms_ai_assistant.api.sse import SSE_KEEPALIVE, format_sse

from edms_ai_assistant.api.sse_events import (
    build_compliance_sse_event,
    build_navigate_sse_event,
    extract_compliance_from_tool_message,
    extract_navigate_url_from_tool_message,
    _parse_tool_content,
)

from edms_ai_assistant.model import NewChatRequest, UserInput
from edms_ai_assistant.security import extract_user_id_from_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


# ── Request / response schemas ────────────────────────────────────────────


class _UserContext(BaseModel):
    firstName: str | None = None
    lastName: str | None = None
    middleName: str | None = None
    role: str | None = None
    post: str | None = None


class ChatStreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    user_token: str = Field(..., min_length=10)
    thread_id: str | None = Field(None, max_length=255)
    context_ui_id: str | None = None
    context: _UserContext | None = None
    file_path: str | None = Field(None, max_length=500)
    file_name: str | None = None
    preferred_summary_format: str | None = Field(None, max_length=32)


class ChatResumeRequest(BaseModel):
    """Resume a paused graph with a structured value.

    ``resume_value`` must validate against ``ResumeValue`` (discriminated
    union by ``kind``). The ``interrupt_id`` is optional but recommended
    for diagnostics — the engine itself routes by checkpoint, not by id.
    """

    thread_id: str = Field(..., min_length=1, max_length=255)
    user_token: str = Field(..., min_length=10)
    resume_value: dict[str, Any]
    interrupt_id: str | None = None
    context_ui_id: str | None = Field(
        None,
        description="UUID активного документа в UI EDMS (для автоинъекции document_id)",
    )


class ChatStateResponse(BaseModel):
    thread_id: str
    messages: list[dict[str, Any]]
    pending_interrupts: list[dict[str, Any]] = []
    active_ui_directives: list[dict[str, Any]] = []


# ── Internal helpers ──────────────────────────────────────────────────────

async def _ensure_clean_state(
    agent: EdmsDocumentAgent,
    config: dict[str, Any],
) -> None:
    """Repair the graph state if it contains dangling tool_calls.

    This happens if a previous stream was aborted (e.g. user clicked Stop)
    while the LLM had requested a tool call or while the tool node was
    executing.  Without this repair the thread becomes permanently unusable.

    IMPORTANT: If there are pending interrupts, the graph is NOT corrupted —
    it is legitimately waiting for a HITL resume.  Do NOT repair in that case,
    otherwise we break the interrupt/resume cycle and cause infinite loops.
    """
    snapshot = await agent.graph.aget_state(config)
    if not snapshot or not snapshot.values:
        return

    for task in getattr(snapshot, "tasks", []) or []:
        for itr in getattr(task, "interrupts", []) or []:
            if isinstance(itr, Interrupt):
                logger.debug(
                    "Pending interrupt found (id=%s) — skipping state repair",
                    getattr(itr, "id", "?"),
                )
                return

    messages = snapshot.values.get("messages", [])
    if not messages:
        return

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return

    tool_messages = []
    for tc in last_msg.tool_calls:
        tool_messages.append(
            ToolMessage(
                content="Выполнение инструмента было прервано пользователем или системой.",
                tool_call_id=tc["id"],
                name=tc.get("name"),
                status="error",
            )
        )

    agent.graph.update_state(
        config,
        {"messages": tool_messages},
        as_node="tools",
    )
    logger.warning(
        "Repaired dangling tool calls: %s",
        [tc["id"] for tc in last_msg.tool_calls],
    )

def _make_config(
        thread_id: str,
        user_token: str,
        user_id: str,
        document_id: str | None = None,
) -> dict[str, Any]:
    """Build the RunnableConfig consumed by graph nodes and tools.

    ``configurable`` is the canonical place to pass per-request data into
    tools (LangGraph injects it as ``config: RunnableConfig``).
    """
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


def _serialise_message(msg: BaseMessage) -> dict[str, Any] | None:
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": str(msg.content)}
    if isinstance(msg, AIMessage):
        content = str(msg.content or "").strip()
        if not content:
            return None
        return {"role": "assistant", "content": content}
    return None


def _extract_pending_interrupts(snapshot: Any) -> list[dict[str, Any]]:
    """Pluck all pending Interrupts from a StateSnapshot.

    LangGraph stores pending interrupts on ``snapshot.tasks[*].interrupts``.
    We surface all of them — supports fan-out (Send) scenarios where
    multiple branches may be paused simultaneously.
    """
    results: list[dict[str, Any]] = []
    for task in getattr(snapshot, "tasks", []) or []:
        for itr in getattr(task, "interrupts", []) or []:
            if not isinstance(itr, Interrupt):
                continue
            results.append({
                "interrupt_id": getattr(itr, "id", None)
                                or getattr(itr, "ns", [""])[-1],
                "payload": itr.value,
            })
    return results


async def _validate_resume_kind(
        graph: Any,
        config: dict[str, Any],
        interrupt_id: str | None,
        resume_kind: str,
) -> None:
    """Return silently on match, raise 409 on mismatch."""
    if not interrupt_id:
        return  # Can't validate without an id
    snapshot = await graph.aget_state(config)
    for task in snapshot.tasks or []:
        for itr in task.interrupts or []:
            value = getattr(itr, "value", itr)
            if isinstance(value, dict):
                pending_kind = value.get("kind")
                if pending_kind and pending_kind != resume_kind:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Kind mismatch: expected {pending_kind!r}, got {resume_kind!r}",
                    )
                return  # found, matches


# ── Streaming core ────────────────────────────────────────────────────────

async def _stream_graph_events(
    agent: EdmsDocumentAgent,
    payload: Any,
    config: dict[str, Any],
    thread_id: str,
) -> AsyncIterator[str]:
    """Drive graph.astream and translate events into SSE.

    Guarantees:
      - Always emits a terminal event (done/error) so the frontend
        never gets stuck in loading state.
      - Handles GraphInterrupt, CancelledError, and unexpected exceptions.
      - Sends ui_component events for compliance/navigate ToolMessages.
    """
    compliance_sent = False
    navigate_sent = False
    sent_done = False

    try:
        async for mode, chunk in agent.graph.astream(
            payload,
            config=config,
            stream_mode=["updates", "custom"],
        ):
            # Send keepalive to prevent timeouts during long tool execution (e.g. OCR)
            yield SSE_KEEPALIVE.decode()

            # ── Custom channel: UIDirective ──────────────────────────────
            if mode == "custom":
                if isinstance(chunk, dict) and "ui" in chunk:
                    directive = chunk["ui"]
                    yield format_sse(
                        "ui_update",
                        {
                            "directive_id": directive.get("directive_id"),
                            "thread_id": thread_id,
                            "directive": directive,
                        },
                    )
                continue

            # ── Updates channel ──────────────────────────────────────────
            if mode != "updates" or not isinstance(chunk, dict):
                continue

            # ── Interrupts ──────────────────────────────────────────────
            interrupts = chunk.get("__interrupt__")
            if interrupts:
                for itr in interrupts:
                    value = getattr(itr, "value", itr)
                    interrupt_id = getattr(itr, "id", None)
                    yield format_sse(
                        "interrupt",
                        {
                            "interrupt_id": interrupt_id,
                            "thread_id": thread_id,
                            "payload": value,
                        },
                    )
                yield format_sse("done", {"thread_id": thread_id, "paused": True})
                return

            # ── Scan ALL nodes for ToolMessages with structured data ────
            for node_name, node_update in chunk.items():
                if node_name == "__interrupt__":
                    continue
                if not isinstance(node_update, dict):
                    continue

                messages = node_update.get("messages", []) or []
                for msg in messages:
                    if not isinstance(msg, ToolMessage):
                        continue

                    logger.debug(
                        "ToolMessage from node=%s name=%s content_type=%s",
                        node_name,
                        getattr(msg, "name", "?"),
                        type(msg.content).__name__,
                    )

                    # Compliance data
                    if not compliance_sent:
                        compliance_data = extract_compliance_from_tool_message(msg)
                        if compliance_data:
                            yield build_compliance_sse_event(compliance_data)
                            compliance_sent = True
                            logger.info(
                                "Compliance UI event sent: overall=%s fields=%d",
                                compliance_data.get("overall"),
                                len(compliance_data.get("fields", [])),
                            )

                    # Navigate URL
                    if not navigate_sent:
                        nav_url = extract_navigate_url_from_tool_message(msg)
                        if nav_url:
                            yield build_navigate_sse_event(nav_url)
                            navigate_sent = True
                            logger.info("Navigate UI event sent: %s", nav_url)
                        else:
                            data = _parse_tool_content(msg.content)
                            if isinstance(data, dict) and data.get("status") == "success":
                                logger.debug(
                                    "ToolMessage success but no navigate derived: "
                                    "keys=%s document_id=%s has_overall=%s has_fields=%s",
                                    list(data.keys()),
                                    data.get("document_id"),
                                    "overall" in data,
                                    "fields" in data,
                                )

            # ── Agent messages (text responses) ─────────────────────────
            agent_update = chunk.get("agent")
            if isinstance(agent_update, dict):
                for msg in agent_update.get("messages", []) or []:
                    rendered = _serialise_message(msg)
                    if rendered is not None and rendered["role"] == "assistant":
                        logger.debug(
                            "Agent message sent: content_len=%d",
                            len(rendered.get("content", "")),
                        )
                        yield format_sse("message", rendered)

    except GraphInterrupt as exc:
        for itr in getattr(exc, "args", []):
            if isinstance(itr, Interrupt):
                yield format_sse(
                    "interrupt",
                    {
                        "interrupt_id": getattr(itr, "id", None),
                        "thread_id": thread_id,
                        "payload": itr.value,
                    },
                )
        yield format_sse("done", {"thread_id": thread_id, "paused": True})
        return
    except asyncio.CancelledError:
        # Stream was aborted by user clicking Stop — send done so frontend resets
        logger.info("Stream cancelled for thread=%s", thread_id)
        yield format_sse("done", {"thread_id": thread_id, "paused": False})
        return
    except Exception as exc:
        logger.exception("chat stream failed for thread=%s", thread_id)
        yield format_sse(
            "error",
            {"code": "INTERNAL", "message": str(exc), "thread_id": thread_id},
        )
        return

    yield format_sse("done", {"thread_id": thread_id, "paused": False})


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post(
    "/chat/stream",
    summary="Start a new chat turn (SSE)",
    response_class=StreamingResponse,
)
async def chat_stream(
        body: ChatStreamRequest,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> StreamingResponse:
    try:
        user_id = extract_user_id_from_token(body.user_token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    thread_id = (
            body.thread_id
            or f"user_{user_id}_doc_{body.context_ui_id or 'general'}"
    )

    if body.context is not None:
        user_context = body.context.model_dump(exclude_none=True)
    else:
        bridged = UserInput(message=body.message, user_token=body.user_token)
        user_context = await resolve_user_context(bridged, user_id)

    if (
            body.preferred_summary_format
            and body.preferred_summary_format != "ask"
    ):
        user_context["preferred_summary_format"] = body.preferred_summary_format

    inputs, _ctx = agent.build_initial_inputs(
        message=body.message,
        user_token=body.user_token,
        context_ui_id=body.context_ui_id,
        thread_id=thread_id,
        user_context=user_context,
        file_path=body.file_path,
        file_name=body.file_name,
    )

    config = _make_config(thread_id, body.user_token, str(user_id), body.context_ui_id)
    await _ensure_clean_state(agent, config)

    return StreamingResponse(
        _stream_graph_events(agent, inputs, config, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/chat/resume",
    summary="Resume a paused graph (SSE)",
    response_class=StreamingResponse,
)
async def chat_resume(
        body: ChatResumeRequest,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> StreamingResponse:
    try:
        validated = ResumeValueAdapter.validate_python(body.resume_value)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        user_id = extract_user_id_from_token(body.user_token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    document_id = body.context_ui_id
    if not document_id:
        try:
            snapshot = await agent.graph.aget_state(
                {"configurable": {"thread_id": body.thread_id}}
            )
            if snapshot and snapshot.values:
                document_id = snapshot.values.get("document_id")
        except Exception as exc:
            logger.warning("Could not fetch state to restore document_id: %s", exc)

    config = _make_config(body.thread_id, body.user_token, str(user_id), document_id)

    await _validate_resume_kind(
        agent.graph, config, body.interrupt_id, validated.kind
    )

    resume_value = validated.model_dump(mode="json")

    await _ensure_clean_state(agent, config)

    return StreamingResponse(
        _stream_graph_events(
            agent,
            Command(resume=resume_value),
            config,
            body.thread_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/chat/state/{thread_id}",
    response_model=ChatStateResponse,
    summary="Get current thread snapshot (for reconnect)",
)
async def chat_state(
        thread_id: str,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> ChatStateResponse:
    """Return the latest state for ``thread_id``.

    Used by the frontend on tab re-open to:
      1. Render the message history.
      2. Detect and re-render a pending HITL card without sending anything.
    """
    try:
        snapshot = await agent.graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
    except Exception as exc:
        logger.warning("state fetch failed: thread=%s err=%s", thread_id, exc)
        return ChatStateResponse(thread_id=thread_id, messages=[])

    raw_messages: list[BaseMessage] = (snapshot.values or {}).get("messages", []) or []
    rendered: list[dict[str, Any]] = []
    for m in raw_messages:
        item = _serialise_message(m)
        if item is not None:
            rendered.append(item)

    pending_list = _extract_pending_interrupts(snapshot)
    valid_pending: list[dict[str, Any]] = []
    for p in pending_list:
        try:
            InterruptPayloadAdapter.validate_python(p["payload"])
            valid_pending.append(p)
        except ValidationError:
            logger.warning(
                "stale interrupt payload on thread=%s — dropping", thread_id
            )

    raw_directives = (snapshot.values or {}).get("last_ui_directives") or {}
    active_ui = [
        {"directive_id": did, "component": comp}
        for did, comp in raw_directives.items()
    ]

    return ChatStateResponse(
        thread_id=thread_id,
        messages=rendered,
        pending_interrupts=valid_pending,
        active_ui_directives=active_ui,
    )


@router.get(
    "/chat/history/{thread_id}",
    summary="Get conversation history for a thread (flat list)",
)
async def get_history(
        thread_id: str,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    try:
        snapshot = await agent.graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        raw: list[BaseMessage] = (snapshot.values or {}).get("messages", []) or []
        filtered: list[dict[str, str]] = []
        for m in raw:
            if isinstance(m, HumanMessage):
                filtered.append({"type": "human", "content": str(m.content)})
            elif isinstance(m, AIMessage):
                content = str(m.content or "").strip()
                if content:
                    filtered.append({"type": "ai", "content": content})
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
