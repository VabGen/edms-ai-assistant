# edms_ai_assistant/agent/graph.py
"""
GraphBuilder — компилирует LangGraph ReAct workflow.

Hot-path node ``call_model`` is now a **pure** function:
  history (trimmed) → LLM → response

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
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from edms_ai_assistant.agent.messages_utils import (
    trim_pairwise,
    validate_no_dangling_tool_calls,
)
from edms_ai_assistant.config import settings
from edms_ai_assistant.model import AgentState

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Компилирует LangGraph workflow с thread-safe model binding."""

    def __init__(
        self,
        tools: list[Any],
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self._tools = tools
        self._checkpointer = checkpointer
        self._model: BaseLanguageModel | None = None
        self._model_lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-init lock — avoids RuntimeError outside running event loop."""
        if self._model_lock is None:
            self._model_lock = asyncio.Lock()
        return self._model_lock

    async def set_model_async(self, model: BaseLanguageModel) -> None:
        async with self._get_lock():
            self._model = model

    def set_model(self, model: BaseLanguageModel) -> None:
        self._model = model

    def compile(self) -> CompiledStateGraph:
        """Компилирует state graph для native HITL pipeline.

        Тулы сами решают, когда приостановить граф, через
        ``ask_human()`` → ``langgraph.types.interrupt()``. Никаких
        ``interrupt_before`` — пауза ставится в той точке кода тула,
        где она логически уместна.

        Returns:
            Скомпилированный граф.

        Raises:
            RuntimeError: При ошибке компиляции LangGraph.
        """
        workflow: StateGraph = StateGraph(AgentState)
        builder_ref = self

        # ── Node: call_model (PURE — no string parsing, no injection) ────

        async def call_model(state: AgentState) -> dict[str, Any]:
            """Вызывает LLM с парно-обрезанной историей.

            Единственная ответственность — подготовить messages и вызвать
            модель. Никакого парсинга ToolMessage, никаких динамических
            SystemMessage-инъекций.
            """
            all_sys: list[BaseMessage] = [
                m for m in state["messages"] if isinstance(m, SystemMessage)
            ]
            non_sys: list[BaseMessage] = [
                m for m in state["messages"] if not isinstance(m, SystemMessage)
            ]

            # Pair-aware trim: AIMessage(tool_calls) + ToolMessage* — атомарны
            non_sys = trim_pairwise(
                non_sys, settings.AGENT_MAX_CONTEXT_MESSAGES
            )

            # Последний system prompt (без динамических инъекций)
            sys_msgs: list[BaseMessage] = all_sys[-1:] if all_sys else []

            candidate: list[BaseMessage] = sys_msgs + non_sys

            # Validate structural invariant — fail loud in dev, log in prod
            validate_no_dangling_tool_calls(
                candidate, fail_loud=getattr(settings, "DEBUG", False)
            )

            if builder_ref._model is None:
                raise RuntimeError(
                    "Model not set. Call GraphBuilder.set_model() before invoking."
                )

            llm_timeout = getattr(settings, "AGENT_LLM_TIMEOUT", 30.0)
            try:
                response = await asyncio.wait_for(
                    builder_ref._model.ainvoke(candidate),
                    timeout=llm_timeout,
                )
                return {"messages": [response]}
            except asyncio.TimeoutError:
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

        # ── Routing ──────────────────────────────────────────────────────

        def should_continue(state: AgentState) -> str:
            last: BaseMessage = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        # ── Wire graph ───────────────────────────────────────────────────
        #
        # Канонический ReAct цикл:
        #   agent -> (tools or END) -> tools -> agent
        #
        # Узел validator удалён. Тулы обязаны возвращать понятный текст
        # (или структурированный JSON с понятными полями), который LLM
        # сможет интерпретировать самостоятельно без инъекций подсказок.

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