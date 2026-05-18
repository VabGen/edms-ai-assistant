# edms_ai_assistant/agent/hitl_primitives.py
"""
Universal Human-in-the-Loop primitive for tools.

Tools call ``ask_human(payload)`` to suspend graph execution and request input
from the user. The suspension is performed by LangGraph's native ``interrupt()``
mechanism — the checkpointer atomically persists state, the API streams a
structured event to the frontend, and the tool resumes on the very same line
once ``Command(resume=...)`` is dispatched.

Usage inside a ``@tool``-decorated function:

    from edms_ai_assistant.agent.hitl_primitives import ask_human
    from edms_ai_assistant.agent.interrupt_contract import (
        CardSelectInterrupt, CardSelectResume, InterruptCard,
    )

    @tool
    async def employee_search_tool(last_name: str) -> dict:
        matches = await client.search(last_name)
        if len(matches) > 1:
            resume = ask_human(CardSelectInterrupt(
                prompt=f"Уточните «{last_name}»",
                cards=[InterruptCard(id=m.id, label=m.full_name,
                                     description=m.department) for m in matches],
            ))
            assert isinstance(resume, CardSelectResume)
            matches = [m for m in matches if m.id in resume.selected_ids]
        ...

The function is intentionally synchronous-looking — ``interrupt()`` raises
``GraphInterrupt`` which LangGraph catches and re-injects the resume value on
re-execution. Idempotency is guaranteed by the engine.
"""

from __future__ import annotations

from langgraph.types import interrupt as _lg_interrupt

from edms_ai_assistant.agent.interrupt_contract import (
    AbortResume,
    InterruptPayload,
    InterruptPayloadAdapter,
    ResumeValue,
    ResumeValueAdapter,
)


class ToolAborted(Exception):
    """Raised when the user (or system) cancels a HITL request.

    Tools may either let this propagate (the framework converts it into a
    graceful ``{"status": "cancelled"}`` ToolMessage) or catch it to perform
    custom cleanup.
    """

    def __init__(self, reason: str | None = None) -> None:
        super().__init__(reason or "Aborted by user")
        self.reason = reason


def ask_human(payload: InterruptPayload) -> ResumeValue:
    """Suspend graph execution and request a structured answer from the user.

    Args:
        payload: One of the ``InterruptPayload`` variants (disambiguation,
            confirmation, text_input, select). Validated and serialised to
            JSON-safe dict before being handed to LangGraph.

    Returns:
        A typed ``ResumeValue`` (matching ``kind`` to the payload, or
        ``AbortResume``). Tools should narrow the type with ``isinstance`` or
        let ``ToolAborted`` propagate.

    Raises:
        ToolAborted: When the user explicitly cancels the request.
    """
    serialised: dict = InterruptPayloadAdapter.dump_python(
        InterruptPayloadAdapter.validate_python(payload),
        mode="json",
    )

    raw = _lg_interrupt(serialised)

    resume: ResumeValue = ResumeValueAdapter.validate_python(raw)

    if isinstance(resume, AbortResume):
        raise ToolAborted(resume.reason)

    return resume


__all__ = ["ToolAborted", "ask_human"]
