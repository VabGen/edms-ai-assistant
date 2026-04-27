"""
GraphBuilder — compiles the LangGraph ReAct workflow.

Extracted from EdmsDocumentAgent so that:
- The compiled graph can be rebuilt without recreating the agent.
- The LLM model binding can be swapped per-intent without recompilation.
- Node logic (call_model, validator) is independently readable and testable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from edms_ai_assistant.model import AgentState

logger = logging.getLogger(__name__)

# Maximum number of non-system messages kept in the context window.
_MAX_HISTORY_MSGS: int = 40


class GraphBuilder:
    """Compiles the LangGraph ReAct workflow.

    The ``call_model`` node accesses the model through ``set_model()`` so that
    ``EdmsDocumentAgent`` can rebind tools per-intent without recompiling the
    entire graph.

    Args:
        tools: Full list of LangChain tool objects available in the graph.
        checkpointer: Checkpoint backend (``MemorySaver`` or ``AsyncPostgresSaver``).
    """

    def __init__(
        self,
        tools: list[Any],
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self._tools = tools
        self._checkpointer = checkpointer
        # The current model binding; updated by set_model() per-intent.
        self._model: BaseLanguageModel | None = None

    def set_model(self, model: BaseLanguageModel) -> None:
        """Update the model used by the ``call_model`` node.

        Called by ``EdmsDocumentAgent`` at the start of each turn after
        the intent-specific tool subset has been bound.

        Args:
            model: A LangChain ``BaseLanguageModel`` already bound to tools.
        """
        self._model = model

    def compile(self) -> CompiledStateGraph:
        """Build and compile the state graph.

        Returns:
            Compiled graph with ``interrupt_before=["tools"]``.

        Raises:
            RuntimeError: If LangGraph compilation fails.
        """
        workflow: StateGraph = StateGraph(AgentState)

        # We capture ``self`` in the closures below so that ``set_model()``
        # updates are visible to every invocation of ``call_model``.
        builder_ref = self

        async def call_model(state: AgentState) -> dict[str, Any]:
            """Invoke the LLM with a trimmed, sanitized message history.

            Steps:
            1. Separate system messages from conversation history.
            2. Trim history to ``_MAX_HISTORY_MSGS``.
            3. Inject a compliance-fix hint when a compliance result is present.
            4. Strip dangling tool_calls that have no following ToolMessage.
            5. Invoke the model.
            """
            sys_msgs: list[BaseMessage] = [
                m for m in state["messages"] if isinstance(m, SystemMessage)
            ]
            non_sys: list[BaseMessage] = [
                m for m in state["messages"] if not isinstance(m, SystemMessage)
            ]

            if len(non_sys) > _MAX_HISTORY_MSGS:
                non_sys = non_sys[-_MAX_HISTORY_MSGS:]

            # --- Compliance-fix hint injection ---
            # When the most recent ToolMessage contains a compliance check result,
            # add a system hint so the LLM knows to call doc_update_field instead
            # of generating a plain-text response.
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    raw = str(msg.content)
                    if raw.startswith("{"):
                        try:
                            data: dict[str, Any] = json.loads(raw)
                            if data.get("status") == "success" and "fields" in data:
                                sys_msgs.append(SystemMessage(content=(
                                    "ВНИМАНИЕ: В истории есть результат compliance check. "
                                    "Если пользователь просит 'исправить', 'обновить' или 'применить': "
                                    "1. БЕРИ значения из 'correct_value'. "
                                    "2. ВЫЗЫВАЙ doc_update_field для КАЖДОГО поля с ошибкой. "
                                    "3. НЕ пиши ответ, пока не вызовешь все инструменты."
                                )))
                                logger.debug("Injected compliance-fix hint")
                        except json.JSONDecodeError:
                            pass
                    break

            # --- Sanitize dangling tool_calls ---
            # An AIMessage with tool_calls that is not immediately followed by a
            # ToolMessage would cause the LLM API to reject the request.  We strip
            # the tool_calls from such messages so the history is always valid.
            candidate: list[BaseMessage] = sys_msgs + non_sys
            final: list[BaseMessage] = []
            for i, msg in enumerate(candidate):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    next_msg = candidate[i + 1] if i + 1 < len(candidate) else None
                    if not isinstance(next_msg, ToolMessage):
                        final.append(AIMessage(content=msg.content or "", id=msg.id))
                        logger.warning(
                            "Sanitized dangling AIMessage tool_calls at history position %d", i
                        )
                        continue
                final.append(msg)

            assert builder_ref._model is not None, (
                "GraphBuilder.set_model() must be called before invoking the graph"
            )
            response = await builder_ref._model.ainvoke(final)
            return {"messages": [response]}

        async def validator(state: AgentState) -> dict[str, Any]:
            last: BaseMessage = state["messages"][-1]
            if not isinstance(last, ToolMessage):
                return {"messages": []}

            raw: str = str(last.content).strip()

            if not raw or raw in ("None", "{}", "null"):
                return {"messages": [AIMessage(content=(
                    "⚠️ Системное уведомление: инструмент вернул пустой результат. "
                    "Попробуй другой подход или сообщи пользователю."
                ))]}

            if raw.startswith("{"):
                try:
                    tool_data: dict[str, Any] = json.loads(raw)
                    status: str = tool_data.get("status", "")

                    # Compliance result
                    if status == "success" and "fields" in tool_data:
                        return {"messages": [AIMessage(
                            content="Анализ документа завершен. Обнаружены расхождения в полях."
                        )]}

                    # Interactive statuses
                    if status in ("requires_choice", "requires_disambiguation", "requires_action"):
                        logger.info("Validator: interactive status '%s' — stopping graph", status)
                        return {"messages": [AIMessage(content="")]}

                    # Already exists / duplicate — not an error, informational status.
                    # The LLM must inform the user explicitly instead of saying "added".
                    if status in ("already_exists", "already_added", "duplicate"):
                        detail = (
                            tool_data.get("message")
                            or tool_data.get("detail")
                            or "Запись уже существует."
                        )
                        return {"messages": [AIMessage(content=(
                            f"ИНФОРМАЦИЯ (показать пользователю): {detail}. "
                            "Сообщи об этом пользователю понятным языком."
                        ))]}

                except json.JSONDecodeError:
                    pass

            # Error result
            raw_lower = raw.lower()
            if '"status": "error"' in raw_lower or (
                    raw_lower.startswith("{") and '"error"' in raw_lower
            ):
                try:
                    err_msg: str = json.loads(raw).get("message", raw[:200])
                except (json.JSONDecodeError, KeyError):
                    err_msg = raw[:200]
                return {"messages": [AIMessage(content=(
                    f"⚠️ Системное уведомление: инструмент вернул ошибку: {err_msg}. "
                    "Проинформируй пользователя понятным языком."
                ))]}

            return {"messages": []}

        def should_continue(state: AgentState) -> str:
            """Route to 'tools' if the LLM produced tool_calls, otherwise END."""
            last: BaseMessage = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools=self._tools))
        workflow.add_node("validator", validator)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "validator")
        workflow.add_edge("validator", "agent")

        try:
            compiled = workflow.compile(
                checkpointer=self._checkpointer,
                interrupt_before=["tools"],
            )
            logger.debug("LangGraph compiled successfully")
            return compiled
        except Exception as exc:
            logger.error("Graph compilation failed", exc_info=True)
            raise RuntimeError(f"Failed to compile LangGraph: {exc}") from exc