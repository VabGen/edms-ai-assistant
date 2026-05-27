# edms_ai_assistant/agent/messages_utils.py
"""
Message-trimming and validation utilities for the LangGraph agent.

Centralises the pairwise-trim algorithm and dangling-tool_calls check
so that ``call_model`` stays a pure history->LLM->response function.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger(__name__)


class StateCorruptedError(RuntimeError):
    """Raised when message history contains structurally invalid sequences."""


def trim_pairwise(messages: list[BaseMessage], max_count: int) -> list[BaseMessage]:
    """Trim messages from the front, preserving ``AIMessage->ToolMessage`` pairs.

    An ``AIMessage`` that carries ``tool_calls`` and **all** of its
    corresponding ``ToolMessage`` responses form an atomic group — they
    are either all kept or all removed.

    This replaces the previous flat ``non_sys[-N:]`` trim which could
    slice a ``ToolMessage`` off its parent ``AIMessage``, producing the
    "dangling tool_calls" state that the sanitizer loop had to patch.

    Args:
        messages: Non-system message list (already filtered).
        max_count: Maximum number of messages to keep.

    Returns:
        Trimmed list with no broken AIMessage/ToolMessage pairs.
    """
    if len(messages) <= max_count:
        return list(messages)

    trimmed = list(messages[-max_count:])

    while trimmed:
        first_tool_msg = trimmed[0]
        if not isinstance(first_tool_msg, ToolMessage):
            break
        has_parent = any(
            isinstance(m, AIMessage)
            and any(
                tc.get("id") == first_tool_msg.tool_call_id
                for tc in getattr(m, "tool_calls", []) or []
            )
            for m in trimmed
        )
        if has_parent:
            break
        trimmed.pop(0)

    first_msg = trimmed[0] if trimmed else None
    if isinstance(first_msg, AIMessage) and first_msg.tool_calls:
        tool_call_ids = {tc.get("id") for tc in first_msg.tool_calls}
        present_ids = {
            m.tool_call_id
            for m in trimmed[1:]
            if isinstance(m, ToolMessage) and m.tool_call_id in tool_call_ids
        }
        if present_ids != tool_call_ids:
            logger.warning(
                "trim_pairwise: dropping orphaned AIMessage with tool_calls "
                "(%d of %d ToolMessages present)",
                len(present_ids),
                len(tool_call_ids),
            )
            trimmed.pop(0)

    return trimmed


def validate_no_dangling_tool_calls(
    messages: list[BaseMessage], *, fail_loud: bool = False
) -> bool:
    """Check that every ``AIMessage.tool_calls`` has matching ``ToolMessage`` s.

    Args:
        messages: Full message list (system + non-system).
        fail_loud: If ``True``, raise :class:`StateCorruptedError` on violation.
            In production, set to ``False`` — the function logs an error and
            returns ``False`` so the caller can decide.

    Returns:
        ``True`` if the list is structurally valid.
    """
    for i, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue

        expected_ids = {tc.get("id") for tc in tool_calls}
        found_ids: set[str] = set()

        for j in range(i + 1, len(messages)):
            candidate = messages[j]
            if (
                isinstance(candidate, ToolMessage)
                and candidate.tool_call_id in expected_ids
            ):
                found_ids.add(candidate.tool_call_id)
            elif isinstance(candidate, AIMessage):
                break  # next turn — stop looking

        missing = expected_ids - found_ids
        if missing:
            logger.error(
                "Dangling tool_calls: AIMessage at position %d is missing "
                "ToolMessage(s) for ids %s",
                i,
                missing,
            )
            if fail_loud:
                raise StateCorruptedError(
                    f"Dangling tool_calls at position {i}: missing {missing}"
                )
            return False

    return True
