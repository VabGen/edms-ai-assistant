# edms_ai_assistant/agent/state_manager.py
"""
AgentStateManager — all LangGraph state operations in one place.

Separating state management from orchestration logic means:
- The checkpoint backend (MemorySaver vs AsyncPostgresSaver) is a
  construction-time concern, not baked into the agent.
- Thread repair logic is independently testable.
- Orchestration code never touches the LangGraph API directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class AgentStateManager:
    """Manages LangGraph graph state: invocation, inspection, and repair.

    Args:
        graph: Compiled LangGraph state graph.
        checkpointer: Any ``BaseCheckpointSaver`` implementation.
            Pass ``MemorySaver()`` for development / tests.
            Pass ``AsyncPostgresSaver(...)`` for production.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        if graph is None:
            raise ValueError("graph cannot be None")
        if checkpointer is None:
            raise ValueError("checkpointer cannot be None")
        self._graph = graph
        self._checkpointer = checkpointer
        logger.debug(
            "AgentStateManager ready",
            extra={
                "graph_type": type(graph).__name__,
                "checkpointer_type": type(checkpointer).__name__,
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> CompiledStateGraph:
        """Expose the compiled graph for GraphBuilder.set_model() access."""
        return self._graph

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def get_state(self, thread_id: str) -> Any:
        """Return the current graph state snapshot for *thread_id*.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            ``StateSnapshot`` with ``.values`` (messages) and ``.next`` (pending nodes).
        """
        return await self._graph.aget_state(self._config(thread_id))

    async def update_state(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        as_node: str = "agent",
    ) -> None:
        """Merge *messages* into the graph state for *thread_id*.

        Args:
            thread_id: Conversation thread identifier.
            messages: Messages to merge (LangGraph reducer handles deduplication).
            as_node: Node to attribute the update to.
        """
        await self._graph.aupdate_state(
            self._config(thread_id),
            {"messages": messages},
            as_node=as_node,
        )

    async def invoke(
        self,
        inputs: dict[str, Any] | None,
        thread_id: str,
        timeout: float = 120.0,
    ) -> None:
        """Invoke the graph for *thread_id*.

        Passing ``inputs=None`` resumes execution from the last interrupt point
        (i.e. after ``interrupt_before=["tools"]``).

        Args:
            inputs: Initial inputs for a fresh invocation, or ``None`` to resume.
            thread_id: Conversation thread identifier.
            timeout: Maximum wall-clock seconds before ``asyncio.TimeoutError``.

        Raises:
            asyncio.TimeoutError: Execution exceeded *timeout*.
        """
        await asyncio.wait_for(
            self._graph.ainvoke(inputs, config=self._config(thread_id)),
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Thread health
    # ------------------------------------------------------------------

    async def is_thread_broken(self, thread_id: str) -> bool:
        """Check whether a thread has a dangling AIMessage with unresolved tool_calls.

        A "broken" thread arises when the agent produced tool_calls but the
        graph was interrupted (e.g. timeout, process restart) before the
        ``ToolMessage`` responses were injected.  The LLM API will reject any
        further invocations with::

            "An assistant message with tool_calls must be followed by tool messages"

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            ``True`` if the thread needs repair before the next invocation.
        """
        try:
            state = await self.get_state(thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])
            if not messages:
                return False
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                # Broken if the preceding message is not a ToolMessage.
                if len(messages) < 2 or not isinstance(messages[-2], ToolMessage):
                    return True
            return False
        except Exception:
            # If we cannot read state, assume it is fine and let the next
            # invocation surface the real error.
            return False

    async def repair_thread(self, thread_id: str) -> bool:
        """Repair a broken thread by injecting synthetic ToolMessage error responses.

        For each unresolved tool_call in the last AIMessage, a synthetic
        ``ToolMessage`` carrying a graceful error payload is injected.  This
        satisfies the LLM API constraint so the graph can resume normally on
        the next user message.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            ``True`` if repair succeeded, ``False`` on failure.
        """
        try:
            state = await self.get_state(thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])
            if not messages:
                return False

            last = messages[-1]
            if not isinstance(last, AIMessage):
                return False

            tool_calls: list[dict[str, Any]] = getattr(last, "tool_calls", []) or []
            if not tool_calls:
                return False

            synthetic: list[ToolMessage] = [
                ToolMessage(
                    content=json.dumps(
                        {
                            "status": "error",
                            "message": (
                                "Выполнение инструмента прервано: предыдущий запрос "
                                "завершился некорректно. Пожалуйста, повторите запрос."
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in tool_calls
            ]

            await self.update_state(thread_id, synthetic, as_node="tools")  # type: ignore[arg-type]
            logger.warning(
                "Thread repaired: injected %d synthetic ToolMessage(s)",
                len(synthetic),
                extra={"thread_id": thread_id},
            )
            return True

        except Exception as exc:
            logger.error(
                "Thread repair failed: %s",
                exc,
                extra={"thread_id": thread_id},
                exc_info=True,
            )
            return False
