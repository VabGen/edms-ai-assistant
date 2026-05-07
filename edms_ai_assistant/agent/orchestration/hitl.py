"""
HumanChoiceHandler — стратегии обработки HITL-выборов.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from edms_ai_assistant.agent.context import is_valid_uuid
from edms_ai_assistant.agent.extraction.content_extractor import ContentExtractor
from edms_ai_assistant.agent.tool_injector import TOOL_DISAMBIG_ID_FIELD

logger = logging.getLogger(__name__)

# Инструменты, в которых summary_type управляется через HITL
_SUMMARIZE_TOOL = "doc_summarize_text"
_VALID_SUMMARY_TYPES = frozenset({"extractive", "abstractive", "thesis"})

# Инструменты, принимающие UUID через disambiguation
_DISAMBIG_TOOLS: frozenset[str] = frozenset(
    {
        "introduction_create_tool",
        "task_create_tool",
        "doc_control",
        "doc_compare_attachment_with_local",
    }
)


# ---------------------------------------------------------------------------
# HandlerResult
# ---------------------------------------------------------------------------


@dataclass
class HandlerResult:
    """Результат обработки HITL-выбора.

    Атрибуты:
        patched_messages: Если задан — обновить state и resume из interrupt.
        new_inputs: Если задан — начать новый turn графа с этими inputs.
        resume_from_interrupt: True = возобновить граф без новых inputs.
    """

    patched_messages: list[BaseMessage] | None = None
    new_inputs: dict[str, Any] | None = None
    resume_from_interrupt: bool = False


# ---------------------------------------------------------------------------
# Protocol для handlers
# ---------------------------------------------------------------------------


class ChoiceHandler(Protocol):
    """Протокол стратегии обработки конкретного типа HITL-выбора."""

    def can_handle(
        self,
        choice: str,
        pending_call: dict[str, Any] | None,
    ) -> bool: ...

    def handle(
        self,
        choice: str,
        pending_call: dict[str, Any] | None,
        messages: list[BaseMessage],
    ) -> HandlerResult: ...


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def _parse_uuids(choice: str) -> list[str]:
    """Парсит строку UUID(s) разделённых запятой или точкой с запятой."""
    return [
        s.strip()
        for s in choice.replace(";", ",").split(",")
        if s.strip() and is_valid_uuid(s.strip())
    ]


def _reconstruct_call(
    pending: dict[str, Any],
    new_args: dict[str, Any],
) -> list[dict[str, Any]]:
    """Собирает patched tool_call list из pending вызова и новых args."""
    return [
        {
            "name": pending["name"],
            "args": {**pending["args"], **new_args},
            "id": pending["id"],
        }
    ]


def _make_patched_ai_message(
    pending_ai: AIMessage,
    patched_calls: list[dict[str, Any]],
) -> AIMessage:
    """Создаёт новый AIMessage с обновлёнными tool_calls."""
    return AIMessage(
        content=pending_ai.content or "",
        tool_calls=patched_calls,
        id=pending_ai.id,
    )


# ---------------------------------------------------------------------------
# Конкретные обработчики
# ---------------------------------------------------------------------------


class FixFieldHandler:
    """Обрабатывает fix_field:<field>:<value> — прямое исправление поля документа."""

    def can_handle(self, choice: str, pending_call: dict[str, Any] | None) -> bool:
        return choice.lower().startswith("fix_field:")

    def handle(
        self,
        choice: str,
        pending_call: dict[str, Any] | None,
        messages: list[BaseMessage],
    ) -> HandlerResult:
        parts = choice.split(":", 2)
        if len(parts) != 3:
            logger.warning("Invalid fix_field format: %s", choice[:60])
            return HandlerResult(
                new_inputs={"messages": [HumanMessage(content=choice)]}
            )
        field_name, field_value = parts[1].strip(), parts[2].strip()
        fix_message = f"Обнови поле «{field_name}» документа значением «{field_value}»."
        logger.info("fix_field: %s = %r", field_name, field_value)
        return HandlerResult(
            new_inputs={"messages": [HumanMessage(content=fix_message)]}
        )


class SummaryTypeHandler:
    """Обрабатывает выбор формата суммаризации: extractive | abstractive | thesis."""

    def can_handle(self, choice: str, pending_call: dict[str, Any] | None) -> bool:
        return choice.lower() in _VALID_SUMMARY_TYPES

    def handle(
        self,
        choice: str,
        pending_call: dict[str, Any] | None,
        messages: list[BaseMessage],
    ) -> HandlerResult:
        summary_type = choice.lower()

        # Patch-and-resume path: works only while graph is paused at interrupt_before=["tools"].
        # After graph completes (requires_choice processed, validator + agent ran to END)
        # the subsequent messages make this ineffective — use explicit message instead.
        if pending_call is not None and pending_call["name"] == _SUMMARIZE_TOOL:
            pending_ai = _find_ai_message_for_call(messages, pending_call["id"])
            if pending_ai is not None:
                new_args: dict[str, Any] = {"summary_type": summary_type}
                last_tool_text = ContentExtractor.extract_last_tool_text(messages)
                if last_tool_text:
                    new_args["text"] = last_tool_text
                patched_calls = _reconstruct_call(pending_call, new_args)
                logger.info(
                    "SummaryTypeHandler: patch+resume summary_type=%s text_len=%d",
                    summary_type,
                    len(new_args.get("text", pending_call["args"].get("text", ""))),
                )
                return HandlerResult(
                    patched_messages=[_make_patched_ai_message(pending_ai, patched_calls)],
                    resume_from_interrupt=True,
                )

        # Fallback: graph already completed — send explicit instruction so the LLM
        # calls doc_summarize_text with the correct summary_type on the next turn.
        logger.info(
            "SummaryTypeHandler: explicit message fallback, summary_type=%s", summary_type
        )
        return HandlerResult(
            new_inputs={
                "messages": [
                    HumanMessage(
                        content=(
                            f"Пользователь выбрал формат анализа документа: «{summary_type}». "
                            f"Немедленно вызови инструмент doc_summarize_text "
                            f"с параметром summary_type='{summary_type}'. "
                            f"Используй текст документа из предыдущего шага. "
                            f"Не задавай вопросов — сразу выполни суммаризацию."
                        )
                    )
                ]
            }
        )


class UuidChoiceHandler:
    """Обрабатывает выбор по UUID — сотрудник, вложение или контролёр."""

    def can_handle(self, choice: str, pending_call: dict[str, Any] | None) -> bool:
        if pending_call is None:
            return False
        if pending_call["name"] not in _DISAMBIG_TOOLS:
            return False
        return bool(_parse_uuids(choice))

    def handle(
        self,
        choice: str,
        pending_call: dict[str, Any] | None,
        messages: list[BaseMessage],
    ) -> HandlerResult:
        if pending_call is None:
            return HandlerResult(new_inputs={"messages": [HumanMessage(content=choice)]})
        tool_name = pending_call["name"]
        id_field = TOOL_DISAMBIG_ID_FIELD.get(tool_name, "selected_employee_ids")
        valid_ids = _parse_uuids(choice)

        if not valid_ids:
            logger.warning(
                "UuidChoiceHandler: no valid UUIDs in choice for %s", tool_name
            )
            return HandlerResult(
                new_inputs={"messages": [HumanMessage(content=choice)]}
            )

        # doc_control принимает единственный UUID строкой, остальные — список
        injected_value: str | list[str] = (
            valid_ids[0] if tool_name == "doc_control" else valid_ids
        )
        patched_calls = _reconstruct_call(pending_call, {id_field: injected_value})
        logger.info(
            "Injected %s=%s for %s",
            id_field,
            valid_ids,
            tool_name,
        )

        pending_ai = _find_ai_message_for_call(messages, pending_call["id"])
        if pending_ai is None:
            return HandlerResult(
                new_inputs={"messages": [HumanMessage(content=choice)]}
            )

        return HandlerResult(
            patched_messages=[_make_patched_ai_message(pending_ai, patched_calls)],
            resume_from_interrupt=True,
        )


class PlainMessageHandler:
    """Fallback — отправляет выбор как новое сообщение пользователя."""

    def can_handle(self, choice: str, pending_call: dict[str, Any] | None) -> bool:
        return True

    def handle(
        self,
        choice: str,
        pending_call: dict[str, Any] | None,
        messages: list[BaseMessage],
    ) -> HandlerResult:
        logger.info(
            "PlainMessageHandler: routing '%s' as new HumanMessage",
            choice[:40],
        )
        return HandlerResult(new_inputs={"messages": [HumanMessage(content=choice)]})


# ---------------------------------------------------------------------------
# HumanChoiceHandler — роутер к конкретным стратегиям
# ---------------------------------------------------------------------------


class HumanChoiceHandler:
    """
    Роутер HITL-выборов.

    Порядок обработчиков:
    1. FixFieldHandler      — fix_field:name:value
    2. SummaryTypeHandler   — extractive | abstractive | thesis
    3. UuidChoiceHandler    — UUID(s) для disambiguation
    4. PlainMessageHandler  — всё остальное (fallback)
    """

    def __init__(self) -> None:
        self._handlers: list[ChoiceHandler] = [
            FixFieldHandler(),
            SummaryTypeHandler(),
            UuidChoiceHandler(),
            PlainMessageHandler(),
        ]

    def classify_and_handle(
        self,
        choice: str,
        pending_call: dict[str, Any] | None,
        messages: list[BaseMessage],
    ) -> HandlerResult:
        """Находит первый подходящий handler и выполняет обработку."""
        for handler in self._handlers:
            if handler.can_handle(choice, pending_call):
                return handler.handle(choice, pending_call, messages)
        # Никогда не достигается (PlainMessageHandler всегда can_handle=True)
        return HandlerResult(new_inputs={"messages": [HumanMessage(content=choice)]})


# ---------------------------------------------------------------------------
# Вспомогательные функции для поиска pending tool_call
# ---------------------------------------------------------------------------


def find_pending_tool_call(
    messages: list[BaseMessage],
) -> dict[str, Any] | None:
    """
    Находит последний AIMessage с tool_calls, после которого нет реального ToolMessage.

    Реальный ToolMessage — тот, у которого content не пустой и не является
    синтетической заглушкой.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue

        subsequent = messages[i + 1 :]
        has_real_result = any(
            isinstance(m, ToolMessage) and _is_real_tool_result(m) for m in subsequent
        )
        if not has_real_result:
            tc = tool_calls[0]
            return {
                "name": tc.get("name", ""),
                "args": dict(tc.get("args", {})),
                "id": tc.get("id", ""),
            }

    return None


def _find_ai_message_for_call(
    messages: list[BaseMessage],
    call_id: str,
) -> AIMessage | None:
    """Находит AIMessage содержащий tool_call с указанным id."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", []) or []:
            if tc.get("id") == call_id:
                return msg
    return None


def _is_real_tool_result(msg: ToolMessage) -> bool:
    """True если ToolMessage содержит реальный результат (не пустую заглушку).

    requires_choice не считается финальным результатом — инструмент ожидает
    выбора пользователя, поэтому AIMessage с pending tool_call остаётся «открытым».
    """
    content = str(msg.content).strip()
    if not (bool(content) and content not in ("", "{}", "null", "None") and len(content) >= 10):
        return False
    if content.startswith("{"):
        try:
            if json.loads(content).get("status") == "requires_choice":
                return False
        except (json.JSONDecodeError, TypeError):
            pass
    return True
