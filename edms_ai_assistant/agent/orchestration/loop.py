"""
OrchestrationLoop — state-machine реализация ReAct петли.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

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
from edms_ai_assistant.agent.orchestration.response_builder import ResponseBuilder
from edms_ai_assistant.agent.state_manager import AgentStateManager
from edms_ai_assistant.agent.tool_injector import (
    BROKEN_THREAD_SIGNALS,
    ToolCallInjector,
)
from edms_ai_assistant.services.nlp_service import UserIntent

logger = logging.getLogger(__name__)

_MAX_ITERATIONS: int = 10
_EXECUTION_TIMEOUT: float = 120.0

# Интенты при которых агент может ответить напрямую без вызова инструментов
_DIRECT_ANSWER_INTENTS: frozenset[UserIntent] = frozenset(
    {
        UserIntent.UNKNOWN,
    }
)

# Ключевые слова, указывающие, что нужны данные из EDMS (не давать прямой ответ)
_EDMS_REQUIRED_KEYWORDS: tuple[str, ...] = (
    "документ",
    "файл",
    "вложен",
    "сотрудник",
    "поручен",
    "ознакомлен",
    "карточк",
    "реестр",
    "поиск",
    "найди",
    "версии",
    "история",
    "контроль",
    "обращение",
    "договор",
    "совещание",
)


def _needs_edms_data(message: str) -> bool:
    """Проверяет, нужны ли данные из EDMS для ответа на сообщение."""
    lower = message.lower()
    return any(kw in lower for kw in _EDMS_REQUIRED_KEYWORDS)


class OrchestrationLoop:
    """
    Итеративная ReAct state-machine оркестрация.

    Основной поток:
    INIT → invoke graph → inspect state:
        ├── interactive status (requires_choice / disambiguation) → DONE
        ├── no tool_calls + graph finished → build response → DONE
        ├── tool_calls present → patch args → resume → next iteration
        └── error → repair/retry or ERROR → DONE
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
        self._tool_binding_cache: dict[str, Any] = {}

    async def run(
        self,
        context: ContextParams,
        inputs: dict[str, Any] | None,
        is_choice_active: bool,
    ) -> dict[str, Any]:
        """
        Запускает оркестрационный цикл.

        При UNKNOWN интенте без EDMS-ключевых слов — сначала пробует ответить
        напрямую через LLM без инструментов (быстрый путь).
        """
        if (
            inputs is not None
            and context.intent in _DIRECT_ANSWER_INTENTS
            and not context.file_path
            and not context.document_id
        ):
            human_msgs = [
                m for m in inputs.get("messages", []) if isinstance(m, HumanMessage)
            ]
            if human_msgs:
                last_text = str(human_msgs[-1].content)
                if not _needs_edms_data(last_text):
                    logger.info(
                        "Direct answer path: UNKNOWN intent, no EDMS keywords",
                        extra={"thread_id": context.thread_id},
                    )
                    return await self._direct_answer(inputs, context)

        current_inputs = inputs
        current_is_choice = is_choice_active

        for iteration in range(_MAX_ITERATIONS + 1):
            if iteration > _MAX_ITERATIONS:
                logger.error(
                    "Max iterations exceeded",
                    extra={"thread_id": context.thread_id, "max": _MAX_ITERATIONS},
                )
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Превышен лимит итераций обработки.",
                ).model_dump()

            if iteration == 0:
                self._bind_tools_for_intent(context)

            try:
                await self._state_manager.invoke(
                    inputs=current_inputs,
                    thread_id=context.thread_id,
                    timeout=_EXECUTION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Execution timeout at iteration %d",
                    iteration,
                    extra={"thread_id": context.thread_id},
                )
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Превышено время ожидания выполнения.",
                ).model_dump()
            except Exception as exc:
                return await self._handle_error(exc, context, iteration)

            state = await self._state_manager.get_state(context.thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])

            logger.debug(
                "Iteration %d: messages=%d last=%s next=%s",
                iteration,
                len(messages),
                type(messages[-1]).__name__ if messages else "—",
                list(state.next) if state.next else [],
            )

            if not messages:
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

            if is_finished:
                return self._response_builder.build(
                    messages, context, self._content_extractor
                )

            if not last_has_calls:
                return self._response_builder.build(
                    messages, context, self._content_extractor
                )

            raw_calls: list[dict[str, Any]] = list(last_msg.tool_calls)
            if len(raw_calls) > 1:
                logger.warning(
                    "Parallel tool_calls — keeping only first: kept=%s dropped=%s",
                    raw_calls[0].get("name"),
                    [tc.get("name") for tc in raw_calls[1:]],
                )
                raw_calls = raw_calls[:1]

            last_tool_text = self._content_extractor.extract_last_tool_text(messages)
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

            if current_is_choice and patched:
                last_patched_name = patched[-1].get("name", "")
                current_is_choice = last_patched_name in (
                    "doc_compare_attachment_with_local",
                    "doc_summarize_text",
                )
            else:
                current_is_choice = False

            current_inputs = None  # resume from interrupt

        return AgentResponse(
            status=AgentStatus.ERROR,
            message="Превышен лимит итераций обработки.",
        ).model_dump()

    async def _direct_answer(
        self,
        inputs: dict[str, Any],
        context: ContextParams,
    ) -> dict[str, Any]:
        """
        Быстрый путь: ответ напрямую через LLM без инструментов.

        Используется для общих вопросов (привет, что ты умеешь, объясни понятие и т.д.)
        когда инструменты EDMS не нужны.
        """
        try:
            no_tools_model = self._model
            response = await no_tools_model.ainvoke(inputs.get("messages", []))
            content = str(response.content).strip()
            if content:
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content=content,
                ).model_dump()
        except Exception as exc:
            logger.warning("Direct answer failed, falling back to normal path: %s", exc)

        self._bind_tools_for_intent(context)
        try:
            await self._state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=_EXECUTION_TIMEOUT,
            )
            state = await self._state_manager.get_state(context.thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])
            return self._response_builder.build(
                messages, context, self._content_extractor
            )
        except Exception as exc:
            logger.error("Direct answer fallback failed: %s", exc)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Произошла ошибка при обработке запроса.",
            ).model_dump()

    def _bind_tools_for_intent(self, context: ContextParams) -> None:
        """Выбирает минимальный набор инструментов для интента и биндит в модель."""
        from edms_ai_assistant.tools.router import (
            estimate_tools_tokens,
            get_tools_for_intent,
        )

        active_intent: UserIntent = context.intent or UserIntent.UNKNOWN
        is_appeal = context.user_context.get("doc_category", "") == "APPEAL"
        selected = get_tools_for_intent(
            active_intent, self._all_tools, include_appeal=is_appeal
        )

        cache_key = ",".join(sorted(getattr(t, "name", "") for t in selected))
        if cache_key not in self._tool_binding_cache:
            bound = self._model.bind_tools(selected)
            self._tool_binding_cache[cache_key] = bound
            logger.info(
                "Tool binding created: intent=%s tools=%d (~%d tokens)",
                active_intent.value,
                len(selected),
                estimate_tools_tokens(selected),
            )

        self._graph_builder.set_model(self._tool_binding_cache[cache_key])

    async def _handle_error(
        self,
        exc: Exception,
        context: ContextParams,
        iteration: int,
    ) -> dict[str, Any]:
        """Обрабатывает ошибки оркестрации с попыткой repair при broken thread."""
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
            logger.warning(
                "Broken thread detected — attempting repair",
                extra={"thread_id": context.thread_id},
            )
            repaired = await self._state_manager.repair_thread(context.thread_id)
            if repaired:
                logger.info("Thread repaired — retrying")
                return await self.run(
                    context=context,
                    inputs=None,
                    is_choice_active=False,
                )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=(
                    "Произошла техническая ошибка. " "Пожалуйста, начните новый диалог."
                ),
            ).model_dump()

        if "TimeoutError" in type(exc).__name__:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания. Попробуйте ещё раз.",
            ).model_dump()

        if "429" in err_str or "rate_limit" in err_str.lower():
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Сервис временно перегружен. Повторите через несколько секунд.",
            ).model_dump()

        return AgentResponse(
            status=AgentStatus.ERROR,
            message="Произошла внутренняя ошибка. Попробуйте переформулировать запрос.",
        ).model_dump()


# ---------------------------------------------------------------------------
# handle_human_choice — свободная функция (обратная совместимость)
# ---------------------------------------------------------------------------


async def handle_human_choice(
    context: ContextParams,
    human_choice: str,
    state_manager: AgentStateManager,
    loop: OrchestrationLoop,
) -> dict[str, Any]:
    """
    Обрабатывает HITL-выбор и возобновляет граф.

    Поддерживает форматы:
    - fix_field:<field>:<value>   → новый turn с инструкцией обновить поле
    - extractive|abstractive|thesis → инжект summary_type
    - UUID[,UUID...]              → инжект employee/attachment IDs
    - любой текст                → новое сообщение пользователя
    """
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
            inputs=None,  # resume from interrupt
            is_choice_active=True,
        )

    if result.new_inputs is not None:
        return await loop.run(
            context=context,
            inputs=result.new_inputs,
            is_choice_active=False,
        )

    logger.error("handle_human_choice: HandlerResult пустой, отправляем как сообщение")
    return await loop.run(
        context=context,
        inputs={"messages": [HumanMessage(content=choice)]},
        is_choice_active=False,
    )
