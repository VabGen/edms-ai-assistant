# edms_ai_assistant/agent/orchestration/loop.py
"""
OrchestrationLoop v2 — Planning-first архитектура.

Главное изменение от v1:
  БЫЛО: NLP(keywords) → intent → router(subset) → LLM
  СТАЛО: IntentPlanner(LLM) → ExecutionPlan → executor/graph → response

Поток выполнения:
  1. IntentPlanner.plan() → ExecutionPlan
  2. can_answer_directly? → прямой LLM ответ (без tools, без graph)
  3. Иначе: bind_tools(ВСЕ tools) → LangGraph graph
     + если план содержит ParallelGroup → PlanExecutor.execute() параллельно
  4. ResponseBuilder → финальный ответ
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from edms_ai_assistant.agent.context import (
    AgentResponse,
    AgentStatus,
    ContextParams,
)
from edms_ai_assistant.agent.extraction.content_extractor import ContentExtractor
from edms_ai_assistant.agent.graph import GraphBuilder
from edms_ai_assistant.agent.orchestration.hitl import (
    HandlerResult,
    HumanChoiceHandler,
    find_pending_tool_call,
)
from edms_ai_assistant.agent.orchestration.response_builder import InteractiveStatusDetector, ResponseBuilder
from edms_ai_assistant.agent.orchestration.states import ExecutionState
from edms_ai_assistant.agent.planning import IntentPlanner, PlanExecutor
from edms_ai_assistant.agent.planning.models import ExecutionPlan, ParallelGroup
from edms_ai_assistant.agent.state_manager import AgentStateManager
from edms_ai_assistant.agent.tool_injector import (
    BROKEN_THREAD_SIGNALS,
    ToolCallInjector,
)
from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


def _check_requires_summary_choice(
    raw_calls: list[dict[str, Any]],
    context: ContextParams,
    last_tool_text: str | None,
    is_choice_active: bool = False,
) -> dict[str, Any] | None:
    """Pre-empt doc_summarize_text so the user always chooses the analysis format
    interactively, regardless of what summary_type the LLM guessed.

    Returns an AgentResponse dict (summarize_selection) to send to the frontend,
    or ``None`` when the tool should proceed normally.

    Returning non-None keeps the graph paused at ``interrupt_before=["tools"]``
    so the user\'s choice can be cleanly patched in via SummaryTypeHandler.

    Skip pre-emption only when:
    - ``is_choice_active`` is True — the user just clicked a button and the
      patched AIMessage is about to be resumed (the LLM did not guess the type).
    - The user has a fixed ``preferred_summary_format`` (not "ask") in settings.
    """
    for call in raw_calls:
        if call.get("name") != "doc_summarize_text":
            continue
        if is_choice_active:
            continue  # user just clicked — honour the patched choice
        args = call.get("args") or {}
        preferred: str | None = (context.user_context or {}).get("preferred_summary_format")
        if preferred and preferred != "ask":
            continue  # user has a fixed preference — injector will apply it
        # Always ask the user — ignore whatever summary_type the LLM may have set
        from edms_ai_assistant.tools.summarization import (  # noqa: PLC0415
            _build_requires_choice_response,
        )
        text = last_tool_text or args.get("text", "")
        data = _build_requires_choice_response(text)
        logger.info(
            "Pre-empting doc_summarize_text (llm_type=%r): returning summarize_selection",
            args.get("summary_type"),
        )
        return InteractiveStatusDetector._build_requires_choice(data)
    return None


class OrchestrationLoop:
    """
    Planning-first оркестрационный цикл.

    Ключевые отличия от v1:
    - IntentPlanner определяет нужны ли tools (LLM, не keywords)
    - PlanExecutor выполняет параллельные группы
    - При can_answer_directly — обходим graph полностью
    - LLM всегда видит ВСЕ tools когда они нужны
    """

    def __init__(
        self,
        state_manager: AgentStateManager,
        injector: ToolCallInjector,
        all_tools: list[Any],
        graph_builder: GraphBuilder,
        model: Any,
    ) -> None:
        self._state_manager = state_manager
        self._injector = injector
        self._all_tools = all_tools
        self._graph_builder = graph_builder
        self._model = model
        self._response_builder = ResponseBuilder()
        self._content_extractor = ContentExtractor()
        self._hitl_handler = HumanChoiceHandler()
        # Planning компоненты — новые
        self._planner = IntentPlanner(llm=model, tools=all_tools)
        self._executor = PlanExecutor(tools=all_tools)
        # Кэш binding для всех tools (один раз создаём)
        self._bound_model_all: Any | None = None

    def _refresh_model_if_needed(self) -> None:
        """Detect if get_chat_model() returned a new instance (after reset_chat_model())
        and refresh self._model, bound model cache, and planner accordingly."""
        from edms_ai_assistant.llm import get_chat_model  # noqa: PLC0415
        current = get_chat_model()
        if current is not self._model:
            self._model = current
            self._bound_model_all = None
            self._planner = IntentPlanner(llm=current, tools=self._all_tools)
            logger.info("OrchestrationLoop: LLM model refreshed after settings update")

    def _get_bound_model_all(self) -> Any:
        """Возвращает модель с ПОЛНЫМ набором tools (кэшировано)."""
        if self._bound_model_all is None:
            self._bound_model_all = self._model.bind_tools(self._all_tools)
            logger.info(
                "Full tool binding created: %d tools",
                len(self._all_tools),
            )
        return self._bound_model_all

    async def run(
        self,
        context: ContextParams,
        inputs: dict[str, Any] | None,
        is_choice_active: bool,
    ) -> dict[str, Any]:
        """
        Главный метод оркестрации.

        Алгоритм:
        1. Получаем историю диалога для контекста планировщика
        2. IntentPlanner.plan() → ExecutionPlan
        3. Если can_answer_directly → прямой ответ без tools
        4. Если есть параллельные шаги → PlanExecutor.execute()
        5. LangGraph graph для ReAct цикла с полным набором tools
        """
        self._refresh_model_if_needed()

        if inputs is None:
            # Resume from HITL interrupt — без planning
            self._graph_builder.set_model(self._get_bound_model_all())
            return await self._run_graph_loop(
                context=context,
                inputs=None,
                is_choice_active=is_choice_active,
            )

        # Получаем историю для планировщика
        history_length = await self._get_history_length(context.thread_id)

        # Извлекаем текст запроса из inputs
        user_message = self._extract_user_message(inputs)

        # === PLANNING ШАГ ===
        plan = await self._planner.plan(
            message=user_message,
            context=context,
            history_length=history_length,
        )

        # === ПРЯМОЙ ОТВЕТ (без tools) ===
        if plan.can_answer_directly:
            logger.info(
                "Direct answer path: %s",
                plan.reasoning,
            )
            return await self._direct_answer(inputs, context, plan)

        # === ПАРАЛЛЕЛЬНОЕ PRE-EXECUTION ===
        # Если план содержит параллельную группу в начале —
        # выполняем её до graph и инжектируем результаты
        parallel_results_context = ""
        if plan.parallel_capable and plan.steps:
            first_step = plan.steps[0]
            if isinstance(first_step, ParallelGroup):
                step_results = await self._executor.execute(
                    ExecutionPlan(
                        can_answer_directly=False,
                        parallel_capable=True,
                        steps=[first_step],
                        reasoning=plan.reasoning,
                    ),
                    context,
                )
                parallel_results_context = self._format_parallel_results(step_results)
                logger.info(
                    "Parallel pre-execution: %d tools completed",
                    len(step_results),
                )

        # === GRAPH EXECUTION с полным набором tools ===
        # Инжектируем результаты параллельного выполнения в inputs если есть
        if parallel_results_context:
            inputs = self._inject_parallel_context(inputs, parallel_results_context)

        # Биндим ВСЕ tools — LLM сама решает что использовать дальше
        self._graph_builder.set_model(self._get_bound_model_all())

        return await self._run_graph_loop(
            context=context,
            inputs=inputs,
            is_choice_active=is_choice_active,
        )

    async def _direct_answer(
        self,
        inputs: dict[str, Any],
        context: ContextParams,
        plan: ExecutionPlan,
    ) -> dict[str, Any]:
        """
        Прямой ответ через LLM без tools и без graph.

        Используется когда:
        - Общий вопрос (не связан с EDMS)
        - Юридический/экспертный анализ файла
        - Вопрос по уже полученным данным из истории
        """
        try:
            # Загружаем историю диалога для контекста
            history_messages = await self._get_history_messages(context.thread_id)

            # Системный промпт из inputs
            system_msgs = [
                m for m in inputs.get("messages", [])
                if isinstance(m, SystemMessage)
            ]
            human_msgs = [
                m for m in inputs.get("messages", [])
                if isinstance(m, HumanMessage)
            ]

            # Строим messages: system + история (без старых SystemMessage) + текущий запрос
            non_sys_history = [
                m for m in history_messages[-20:]
                if not isinstance(m, SystemMessage)
            ]
            all_messages = system_msgs + non_sys_history + human_msgs

            # Прямой вызов без tools
            response = await self._model.ainvoke(all_messages)
            content = str(response.content).strip()

            if content:
                # Сохраняем в state для истории
                await self._state_manager.update_state(
                    context.thread_id,
                    human_msgs + [response],
                    as_node="agent",
                )
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content=content,
                ).model_dump()

        except Exception as exc:
            logger.warning(
                "Direct answer failed, falling back to graph: %s",
                exc,
            )

        # Fallback — обычный graph
        self._graph_builder.set_model(self._get_bound_model_all())
        return await self._run_graph_loop(
            context=context,
            inputs=inputs,
            is_choice_active=False,
        )

    async def _run_graph_loop(
        self,
        context: ContextParams,
        inputs: dict[str, Any] | None,
        is_choice_active: bool,
    ) -> dict[str, Any]:
        """
        Основной ReAct цикл через LangGraph graph.
        ExecutionState используется для structured logging (observability).
        """
        current_inputs = inputs
        current_is_choice = is_choice_active
        _state = ExecutionState.INIT
        _log = lambda s: logger.debug(
            "[%s] state=%s iter=%s",
            context.thread_id,
            s.name,
            "—",
            extra={"execution_state": s.name, "thread_id": context.thread_id},
        )
        _log(_state)

        for iteration in range(settings.AGENT_MAX_ITERATIONS):
            _state = ExecutionState.INVOKING
            _log(_state)

            try:
                await self._state_manager.invoke(
                    inputs=current_inputs,
                    thread_id=context.thread_id,
                    timeout=settings.AGENT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                _state = ExecutionState.ERROR
                _log(_state)
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Превышено время ожидания выполнения.",
                ).model_dump()
            except Exception as exc:
                _state = ExecutionState.ERROR
                _log(_state)
                return await self._handle_error(exc, context, iteration)

            _state = ExecutionState.INSPECTING
            _log(_state)
            state = await self._state_manager.get_state(context.thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])

            if not messages:
                _state = ExecutionState.ERROR
                _log(_state)
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Пустое состояние агента.",
                ).model_dump()

            last_msg = messages[-1]
            last_is_tool = isinstance(last_msg, ToolMessage)
            last_has_calls = isinstance(last_msg, AIMessage) and bool(
                getattr(last_msg, "tool_calls", None)
            )
            is_finished = not state.next and not last_is_tool and not last_has_calls

            if is_finished or not last_has_calls:
                _state = ExecutionState.BUILDING_RESPONSE
                _log(_state)
                return self._response_builder.build(
                    messages, context, self._content_extractor
                )

            _state = ExecutionState.PATCHING
            _log(_state)
            raw_calls = list(last_msg.tool_calls)
            last_tool_text = self._content_extractor.extract_last_tool_text(messages)

            # Pre-empt: if doc_summarize_text needs user selection, return
            # selection buttons WITHOUT resuming — graph stays at interrupt
            _raw_for_check = [
                {"name": tc.get("name", ""), "args": dict(tc.get("args") or {}), "id": tc.get("id", "")}
                for tc in raw_calls
            ]
            choice_resp = _check_requires_summary_choice(_raw_for_check, context, last_tool_text, current_is_choice)
            if choice_resp:
                return choice_resp

            patched = self._injector.patch(
                raw_calls=[
                    {
                        "name": tc.get("name", ""),
                        "args": dict(tc.get("args", {})),
                        "id": tc.get("id", ""),
                    }
                    for tc in raw_calls
                ],
                context=context,
                messages=messages,
                last_tool_text=last_tool_text,
                is_choice_active=current_is_choice,
            )

            await self._state_manager.update_state(
                context.thread_id,
                [
                    AIMessage(
                        content=last_msg.content or "",
                        tool_calls=patched,
                        id=last_msg.id,
                    )
                ],
                as_node="agent",
            )

            current_is_choice = False
            current_inputs = None

        _state = ExecutionState.ERROR
        _log(_state)
        return AgentResponse(
            status=AgentStatus.ERROR,
            message="Превышен лимит итераций обработки.",
        ).model_dump()

    async def _get_history_length(self, thread_id: str) -> int:
        """Получить количество сообщений в истории диалога."""
        try:
            state = await self._state_manager.get_state(thread_id)
            messages = state.values.get("messages", [])
            return len([m for m in messages if isinstance(m, (HumanMessage, AIMessage))])
        except Exception:
            return 0

    async def _get_history_messages(self, thread_id: str) -> list[BaseMessage]:
        """Получить сообщения истории для прямого ответа."""
        try:
            state = await self._state_manager.get_state(thread_id)
            return state.values.get("messages", [])
        except Exception:
            return []

    @staticmethod
    def _extract_user_message(inputs: dict[str, Any]) -> str:
        """Извлечь текст пользователя из inputs."""
        for msg in reversed(inputs.get("messages", [])):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    @staticmethod
    def _format_parallel_results(results: list[Any]) -> str:
        """Форматировать результаты параллельного выполнения для инжекции."""
        parts = []
        for r in results:
            if r.error:
                parts.append(f"[{r.tool_name}] Error: {r.error}")
            else:
                result_str = json.dumps(r.result, ensure_ascii=False, default=str)[:3000]
                parts.append(f"[{r.tool_name}] Result: {result_str}")
        return "\n\n".join(parts)

    @staticmethod
    def _inject_parallel_context(
        inputs: dict[str, Any],
        parallel_context: str,
    ) -> dict[str, Any]:
        """Добавить результаты параллельного выполнения в system prompt."""
        new_messages = []
        for msg in inputs.get("messages", []):
            if isinstance(msg, SystemMessage):
                new_content = (
                    str(msg.content)
                    + f"\n\n<pre_fetched_data>\n{parallel_context}\n</pre_fetched_data>"
                )
                new_messages.append(SystemMessage(content=new_content))
            else:
                new_messages.append(msg)
        return {**inputs, "messages": new_messages}

    async def _handle_error(
        self,
        exc: Exception,
        context: ContextParams,
        iteration: int,
    ) -> dict[str, Any]:
        """Обработка ошибок — из v1 без изменений."""
        err_str = str(exc)
        logger.error(
            "Orchestration error at iteration %d: %s",
            iteration,
            err_str[:300],
            extra={"thread_id": context.thread_id},
            exc_info=True,
        )

        is_broken_thread = any(sig in err_str for sig in BROKEN_THREAD_SIGNALS)

        if iteration == 0 and is_broken_thread:
            repaired = await self._state_manager.repair_thread(context.thread_id)
            if repaired:
                return await self.run(
                    context=context,
                    inputs=None,
                    is_choice_active=False,
                )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Произошла техническая ошибка. Пожалуйста, начните новый диалог.",
            ).model_dump()

        if "TimeoutError" in type(exc).__name__:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания. Попробуйте ещё раз.",
            ).model_dump()

        return AgentResponse(
            status=AgentStatus.ERROR,
            message="Произошла внутренняя ошибка. Попробуйте переформулировать запрос.",
        ).model_dump()


async def handle_human_choice(
    context: ContextParams,
    human_choice: str,
    state_manager: AgentStateManager,
    loop: OrchestrationLoop,
) -> dict[str, Any]:
    """HITL обработка — из v1 без изменений."""
    choice = human_choice.strip()
    state = await state_manager.get_state(context.thread_id)
    messages: list[BaseMessage] = state.values.get("messages", [])
    pending_call = find_pending_tool_call(messages)
    hitl_handler = HumanChoiceHandler()
    result: HandlerResult = hitl_handler.classify_and_handle(
        choice=choice,
        pending_call=pending_call,
        messages=messages,
    )

    if result.patched_messages is not None:
        await state_manager.update_state(
            context.thread_id,
            result.patched_messages,
            as_node="agent",
        )
        return await loop.run(
            context=context,
            inputs=None,
            is_choice_active=True,
        )

    if result.new_inputs is not None:
        return await loop.run(
            context=context,
            inputs=result.new_inputs,
            is_choice_active=False,
        )

    return await loop.run(
        context=context,
        inputs={"messages": [HumanMessage(content=choice)]},
        is_choice_active=False,
    )