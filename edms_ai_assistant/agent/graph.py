# edms_ai_assistant/agent/graph.py
"""
GraphBuilder — компилирует LangGraph ReAct workflow.

Решение: model передаётся в call_model через threading.local() или
через явный параметр в config["configurable"].

Выбрано: хранение последнего bound model в thread-safe структуре.
В production при asyncio concurrency (один event loop) достаточно
простого атрибута — asyncio не переключает корутины в середине
синхронного кода. Race condition возможен только при multi-thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

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
        self._model_lock = asyncio.Lock()

    async def set_model_async(self, model: BaseLanguageModel) -> None:
        """Thread-safe обновление модели.

        Args:
            model: LangChain модель с привязанными инструментами.
        """
        async with self._model_lock:
            self._model = model

    def set_model(self, model: BaseLanguageModel) -> None:
        """Синхронное обновление модели (для backward compatibility).

        Примечание: в asyncio context безопасно т.к. event loop
        не переключает корутины внутри синхронного кода.
        """
        self._model = model

    def compile(self) -> CompiledStateGraph:
        """Компилирует state graph.

        Returns:
            Скомпилированный граф с interrupt_before=["tools"].

        Raises:
            RuntimeError: При ошибке компиляции LangGraph.
        """
        workflow: StateGraph = StateGraph(AgentState)
        builder_ref = self  # Closure — ссылка на builder для доступа к _model

        async def call_model(state: AgentState) -> dict[str, Any]:
            """Вызывает LLM с обрезанной и санитизированной историей."""
            all_sys: list[BaseMessage] = [
                m for m in state["messages"] if isinstance(m, SystemMessage)
            ]
            non_sys: list[BaseMessage] = [
                m for m in state["messages"] if not isinstance(m, SystemMessage)
            ]

            # Обрезаем историю до лимита
            if len(non_sys) > settings.AGENT_MAX_CONTEXT_MESSAGES:
                non_sys = non_sys[-settings.AGENT_MAX_CONTEXT_MESSAGES:]

            # Оставляем только последний SystemMessage — каждый запрос добавляет
            # новый, старые содержат устаревший контекст (file_path, документ и т.д.)
            # и вводят LLM в заблуждение.
            sys_msgs: list[BaseMessage] = all_sys[-1:] if all_sys else []

            # Инъекция compliance-fix hint
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    raw = str(msg.content)
                    if raw.startswith("{"):
                        try:
                            data: dict[str, Any] = json.loads(raw)
                            if data.get("status") == "success" and "fields" in data:
                                sys_msgs.append(
                                    SystemMessage(
                                        content=(
                                            "ВНИМАНИЕ: В истории есть результат compliance check. "
                                            "При просьбе 'исправить': вызывай doc_update_field "
                                            "для каждого поля с ошибкой, используя 'correct_value'."
                                        )
                                    )
                                )
                        except json.JSONDecodeError:
                            pass
                    break

            # Санитизация висящих tool_calls
            candidate: list[BaseMessage] = sys_msgs + non_sys
            final: list[BaseMessage] = []
            for i, msg in enumerate(candidate):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    next_msg = candidate[i + 1] if i + 1 < len(candidate) else None
                    if not isinstance(next_msg, ToolMessage):
                        final.append(AIMessage(content=msg.content or "", id=msg.id))
                        logger.warning(
                            "Sanitized dangling tool_calls at position %d", i
                        )
                        continue
                final.append(msg)

            if builder_ref._model is None:
                raise RuntimeError(
                    "Model not set. Call GraphBuilder.set_model() before invoking."
                )
            return {"messages": [await builder_ref._model.ainvoke(final)]}

        async def validator(state: AgentState) -> dict[str, Any]:
            """Валидирует результат tool call и формирует guidance для LLM."""
            last: BaseMessage = state["messages"][-1]
            if not isinstance(last, ToolMessage):
                return {"messages": []}

            raw: str = str(last.content).strip()

            if not raw or raw in ("None", "{}", "null"):
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "⚠️ Инструмент вернул пустой результат. "
                                "Попробуй другой подход или сообщи пользователю."
                            )
                        )
                    ]
                }

            if raw.startswith("{"):
                try:
                    data: dict[str, Any] = json.loads(raw)
                    status: str = data.get("status", "")

                    if status == "success" and "fields" in data:
                        return {
                            "messages": [
                                AIMessage(
                                    content="Анализ документа завершён. Найдены расхождения."
                                )
                            ]
                        }

                    if status in (
                        "requires_choice",
                        "requires_disambiguation",
                        "requires_action",
                    ):
                        return {"messages": [AIMessage(content="")]}

                    if status in ("already_exists", "already_added", "duplicate"):
                        detail = (
                            data.get("message")
                            or data.get("detail")
                            or "Запись уже существует."
                        )
                        return {
                            "messages": [
                                AIMessage(
                                    content=f"ИНФОРМАЦИЯ: {detail}. Сообщи пользователю."
                                )
                            ]
                        }

                    if status == "error":
                        err_msg: str = data.get("message", raw[:200])
                        return {
                            "messages": [
                                AIMessage(
                                    content=(
                                        f"⚠️ Ошибка инструмента: {err_msg}. "
                                        "Проинформируй пользователя понятным языком."
                                    )
                                )
                            ]
                        }

                except json.JSONDecodeError:
                    pass

            return {"messages": []}

        def should_continue(state: AgentState) -> str:
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
            return workflow.compile(
                checkpointer=self._checkpointer,
                interrupt_before=["tools"],
            )
        except Exception as exc:
            logger.error("Graph compilation failed", exc_info=True)
            raise RuntimeError(f"Failed to compile LangGraph: {exc}") from exc
