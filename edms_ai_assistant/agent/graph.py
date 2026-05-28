# edms_ai_assistant/agent/graph.py
"""
GraphBuilder — компилирует LangGraph ReAct workflow.

Hot-path node ``call_model`` is now a **pure** function:
  history (trimmed) -> LLM -> response

No heuristic string-parsing, no dynamic SystemMessage injection,
no sanitizer loop, no synthetic AIMessage validator.  All
message-history invariants are enforced by ``trim_pairwise`` and
``validate_no_dangling_tool_calls`` in ``messages_utils``.

Tools are expected to return self-describing, user-friendly content
strings in their ToolMessages. The LLM interprets these natively.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from opentelemetry import trace

from edms_ai_assistant.agent.messages_utils import (
    trim_pairwise,
    validate_no_dangling_tool_calls,
)
from edms_ai_assistant.config import settings
from edms_ai_assistant.model import AgentState

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class GraphBuilder:
    """Компилирует LangGraph workflow с thread-safe model binding."""

    def __init__(
        self,
        tools: list[Any],
        checkpointer: BaseCheckpointSaver[Any],
    ) -> None:
        self._tools = tools
        self._checkpointer = checkpointer
        self._model: Runnable[Any, BaseMessage] | None = None
        self._model_lock: asyncio.Lock | None = None

    @property
    def lock(self) -> asyncio.Lock:
        """Thread-safe lazy-init lock to ensure it's attached to the running event loop."""
        if self._model_lock is None:
            # Note: in a single-threaded asyncio app, this is safe.
            self._model_lock = asyncio.Lock()
        return self._model_lock

    async def set_model_async(self, model: Runnable[Any, BaseMessage]) -> None:
        async with self.lock:
            self._model = model

    def set_model(self, model: Runnable[Any, BaseMessage]) -> None:
        self._model = model

    def compile(self) -> CompiledStateGraph[AgentState, Any, Any]:
        """Компилирует state graph для native HITL pipeline.

        Тулы сами решают, когда приостановить граф, через
        ``ask_human()`` -> ``langgraph.types.interrupt()``. Никаких
        ``interrupt_before`` — пауза ставится в той точке кода тула,
        где она логически уместна.

        Returns:
            Скомпилированный граф.

        Raises:
            RuntimeError: При ошибке компиляции LangGraph.
        """
        workflow: StateGraph[AgentState, Any, Any] = StateGraph(AgentState)
        builder_ref = self

        # ── Node: call_model (PURE — no string parsing, no injection) ────

        async def call_model(state: AgentState) -> dict[str, Any]:
            """Вызывает LLM с парно-обрезанной историей."""
            with tracer.start_as_current_span("call_model") as span:
                all_sys: list[BaseMessage] = [
                    m for m in state["messages"] if isinstance(m, SystemMessage)
                ]
                non_sys: list[BaseMessage] = [
                    m for m in state["messages"] if not isinstance(m, SystemMessage)
                ]

                # Pair-aware trim
                non_sys = trim_pairwise(non_sys, settings.AGENT_MAX_CONTEXT_MESSAGES)

                sys_msgs: list[BaseMessage] = all_sys[-1:] if all_sys else []
                candidate: list[BaseMessage] = sys_msgs + non_sys

                # validate_no_dangling_tool_calls(
                #     candidate, fail_loud=getattr(settings, "DEBUG", False)
                # )
                validate_no_dangling_tool_calls(candidate, fail_loud=False)

                if not candidate:
                    logger.error(
                        "Attempted to invoke LLM with empty messages list. "
                        "Checkpoint might be missing or history over-trimmed."
                    )
                    span.set_attribute("error", True)
                    return {
                        "messages": [
                            AIMessage(
                                content=(
                                    "Контекст диалога был утерян (возможно, из-за перезапуска сервера). "
                                    "Пожалуйста, начните новый чат."
                                )
                            )
                        ]
                    }

                if builder_ref._model is None:
                    raise RuntimeError(
                        "Model not set. Call GraphBuilder.set_model() before invoking."
                    )

                span.set_attribute("messages.count", len(candidate))
                llm_timeout = getattr(settings, "AGENT_LLM_TIMEOUT", 30.0)

                try:
                    response = await asyncio.wait_for(
                        builder_ref._model.ainvoke(candidate),
                        timeout=llm_timeout,
                    )
                    return {"messages": [response]}
                except TimeoutError:
                    span.set_attribute("error", "timeout")
                    logger.error("LLM call timed out after %.1fs", llm_timeout)
                    return {
                        "messages": [
                            AIMessage(
                                content=(
                                    "Превышено время ожидания ответа. "
                                    "Попробуйте переформулировать запрос."
                                )
                            )
                        ]
                    }
                except Exception as exc:
                    span.set_attribute("error", str(exc))
                    logger.error("LLM invocation failed: %s", exc, exc_info=True)
                    return {
                        "messages": [
                            AIMessage(
                                content=(
                                    "Произошла ошибка при обращении к языковой модели. "
                                    "Пожалуйста, попробуйте еще раз или переформулируйте запрос."
                                )
                            )
                        ]
                    }

        # ── Routing ──────────────────────────────────────────────────────

        def should_continue(state: AgentState) -> str:
            last: BaseMessage = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        # ── Wire graph ───────────────────────────────────────────────────
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools=self._tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        try:
            return workflow.compile(checkpointer=self._checkpointer)
        except Exception as exc:
            logger.error("Graph compilation failed", exc_info=True)
            raise RuntimeError(f"Failed to compile LangGraph: {exc}") from exc
