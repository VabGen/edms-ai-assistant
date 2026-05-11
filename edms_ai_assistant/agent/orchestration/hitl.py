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

_SUMMARIZE_TOOL = "doc_summarize_text"
_VALID_SUMMARY_TYPES = frozenset({"extractive", "abstractive", "thesis"})

_RETURN_DISAMBIG_TOOLS: frozenset[str] = frozenset(
    {
        "introduction_create_tool",
        "task_create_tool",
    }
)

_INTERRUPT_DISAMBIG_TOOLS: frozenset[str] = frozenset(
    {
        "doc_control",
        "doc_compare_attachment_with_local",
    }
)

_ACTION_TOOLS: frozenset[str] = frozenset(
    {
        "employee_search_tool",
    }
)

_DISAMBIG_TOOLS: frozenset[str] = (
    _RETURN_DISAMBIG_TOOLS | _INTERRUPT_DISAMBIG_TOOLS | _ACTION_TOOLS
)


# ---------------------------------------------------------------------------
# HandlerResult
# ---------------------------------------------------------------------------


@dataclass
class HandlerResult:
    patched_messages: list[BaseMessage] | None = None
    new_inputs: dict[str, Any] | None = None
    resume_from_interrupt: bool = False


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ChoiceHandler(Protocol):
    def can_handle(self, choice: str, pending_call: dict[str, Any] | None) -> bool: ...
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
    return [
        s.strip()
        for s in choice.replace(";", ",").split(",")
        if s.strip() and is_valid_uuid(s.strip())
    ]


def _reconstruct_call(
    pending: dict[str, Any], new_args: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "name": pending["name"],
            "args": {**pending["args"], **new_args},
            "id": pending["id"],
        }
    ]


def _make_patched_ai_message(
    pending_ai: AIMessage, patched_calls: list[dict[str, Any]]
) -> AIMessage:
    return AIMessage(
        content=pending_ai.content or "", tool_calls=patched_calls, id=pending_ai.id
    )


def _extract_disambiguation_meta(messages: list[BaseMessage]) -> dict[str, Any]:
    """Извлекает метаданные disambiguation из последнего ToolMessage.

    Возвращает:
    - current_group: текущая группа для выбора
    - remaining_groups: оставшиеся группы
    - already_selected_ids: уже выбранные UUID
    - already_resolved_uuids: UUID уже разрешённых сотрудников
    - original_executor_last_names / original_last_names: оригинальные имена
    - original_task_text: текст поручения (task_create_tool)
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(str(msg.content))
            if data.get("status") in _PENDING_STATUSES:
                return {
                    "current_group": data.get("current_group", ""),
                    "remaining_groups": data.get("remaining_groups", []),
                    "already_selected_ids": data.get("already_selected_ids", []),
                    "already_resolved_uuids": data.get("already_resolved_uuids", []),
                    "original_executor_last_names": data.get(
                        "original_executor_last_names", []
                    ),
                    "original_last_names": data.get("original_last_names", []),
                    "original_task_text": data.get("original_task_text", ""),
                }
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


class FixFieldHandler:
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
            return HandlerResult(
                new_inputs={"messages": [HumanMessage(content=choice)]}
            )
        field_name, field_value = parts[1].strip(), parts[2].strip()
        fix_message = f"Обнови поле «{field_name}» документа значением «{field_value}»."
        return HandlerResult(
            new_inputs={"messages": [HumanMessage(content=fix_message)]}
        )


class SummaryTypeHandler:
    def can_handle(self, choice: str, pending_call: dict[str, Any] | None) -> bool:
        return choice.lower() in _VALID_SUMMARY_TYPES

    def handle(
        self,
        choice: str,
        pending_call: dict[str, Any] | None,
        messages: list[BaseMessage],
    ) -> HandlerResult:
        summary_type = choice.lower()
        if pending_call is not None and pending_call["name"] == _SUMMARIZE_TOOL:
            pending_ai = _find_ai_message_for_call(messages, pending_call["id"])
            if pending_ai is not None:
                new_args: dict[str, Any] = {"summary_type": summary_type}
                last_tool_text = ContentExtractor.extract_last_tool_text(messages)
                if last_tool_text:
                    new_args["text"] = last_tool_text
                patched_calls = _reconstruct_call(pending_call, new_args)
                return HandlerResult(
                    patched_messages=[
                        _make_patched_ai_message(pending_ai, patched_calls)
                    ],
                    resume_from_interrupt=True,
                )
        return HandlerResult(
            new_inputs={
                "messages": [
                    HumanMessage(
                        content=f"Пользователь выбрал формат: «{summary_type}». "
                        f"Немедленно вызови doc_summarize_text с summary_type='{summary_type}'."
                    )
                ]
            }
        )


class UuidChoiceHandler:
    """Обрабатывает выбор по UUID с поддержкой последовательного disambiguation."""

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
            return HandlerResult(
                new_inputs={"messages": [HumanMessage(content=choice)]}
            )

        tool_name = pending_call["name"]
        valid_ids = _parse_uuids(choice)

        if not valid_ids:
            return HandlerResult(
                new_inputs={"messages": [HumanMessage(content=choice)]}
            )

        if tool_name in _ACTION_TOOLS:
            return self._handle_action_tool(tool_name, valid_ids)

        if tool_name in _INTERRUPT_DISAMBIG_TOOLS:
            return self._handle_interrupt_tool(
                tool_name, valid_ids, pending_call, messages
            )

        if tool_name in _RETURN_DISAMBIG_TOOLS:
            return self._handle_return_tool(
                tool_name, valid_ids, pending_call, messages
            )

        return HandlerResult(new_inputs={"messages": [HumanMessage(content=choice)]})

    # ── employee_search_tool ──────────────────────────────────────────

    def _handle_action_tool(
        self, tool_name: str, valid_ids: list[str]
    ) -> HandlerResult:
        selected_id = valid_ids[0]
        logger.info(
            "UuidChoiceHandler: employee_search → employee_id=%s", selected_id[:8]
        )
        return HandlerResult(
            new_inputs={
                "messages": [
                    HumanMessage(
                        content=f"Пользователь выбрал сотрудника с ID {selected_id}. "
                        f"Немедленно вызови employee_search_tool(employee_id='{selected_id}'). "
                        f"НЕ ищи по фамилии — используй ТОЛЬКО employee_id."
                    )
                ]
            }
        )

    # ── interrupt-based (doc_control) ─────────────────────────────────

    def _handle_interrupt_tool(
        self,
        tool_name: str,
        valid_ids: list[str],
        pending_call: dict[str, Any],
        messages: list[BaseMessage],
    ) -> HandlerResult:
        id_field = TOOL_DISAMBIG_ID_FIELD.get(tool_name, "selected_employee_ids")
        injected_value: str | list[str] = (
            valid_ids[0] if tool_name == "doc_control" else valid_ids
        )
        patched_calls = _reconstruct_call(pending_call, {id_field: injected_value})
        pending_ai = _find_ai_message_for_call(messages, pending_call["id"])
        if pending_ai is None:
            return HandlerResult(
                new_inputs={
                    "messages": [
                        HumanMessage(
                            content=f"Выбран ID {valid_ids[0]}. Вызови {tool_name} с {id_field}={valid_ids}."
                        )
                    ]
                }
            )
        return HandlerResult(
            patched_messages=[_make_patched_ai_message(pending_ai, patched_calls)],
            resume_from_interrupt=True,
        )

    # ── return-based с sequential disambiguation ─────────────────────

    def _handle_return_tool(
        self,
        tool_name: str,
        valid_ids: list[str],
        pending_call: dict[str, Any],
        messages: list[BaseMessage],
    ) -> HandlerResult:
        """
        Обрабатывает выбор для return-based инструментов.
        """
        meta = _extract_disambiguation_meta(messages)
        remaining_groups: list[str] = meta.get("remaining_groups", [])
        already_selected: list[str] = meta.get("already_selected_ids", [])
        already_resolved: list[str] = meta.get("already_resolved_uuids", [])
        current_group: str = meta.get("current_group", "")
        original_task_text: str = meta.get("original_task_text", "")

        all_selected = already_selected + already_resolved + valid_ids
        seen: set[str] = set()
        unique_selected: list[str] = []
        for eid in all_selected:
            if eid not in seen:
                seen.add(eid)
                unique_selected.append(eid)

        if remaining_groups:
            remaining_names_str = ", ".join(f"«{n}»" for n in remaining_groups)
            ids_str = str(unique_selected)
            remaining_str = str(remaining_groups)

            if tool_name == "task_create_tool":
                instruction = (
                    f"Пользователь выбрал сотрудника для «{current_group}». "
                    f"Осталось уточнить: {remaining_names_str}. "
                    f"НЕ создавай поручение — сначала нужно уточнить всех исполнителей.\n\n"
                    f"Вызови task_create_tool со следующими параметрами:\n"
                    f"• selected_employee_ids={ids_str}\n"
                    f"• executor_last_names={remaining_str}\n"
                    f"• task_text='{original_task_text}'\n"
                    f"Это позволит разрешить оставшихся исполнителей."
                )
            else:  # introduction_create_tool
                instruction = (
                    f"Пользователь выбрал сотрудника для «{current_group}». "
                    f"Осталось уточнить: {remaining_names_str}. "
                    f"НЕ создавай ознакомление — сначала уточните всех.\n\n"
                    f"Вызови introduction_create_tool со следующими параметрами:\n"
                    f"• selected_employee_ids={ids_str}\n"
                    f"• last_names={remaining_str}\n"
                    f"Это позволит разрешить оставшихся сотрудников."
                )

            logger.info(
                "UuidChoiceHandler: PARTIAL resolution for %s — "
                "current='%s', remaining=%s, accumulated_ids=%d",
                tool_name,
                current_group,
                remaining_groups,
                len(unique_selected),
            )

            return HandlerResult(
                new_inputs={"messages": [HumanMessage(content=instruction)]}
            )

        else:
            ids_str = str(unique_selected)

            if tool_name == "task_create_tool":
                instruction = (
                    f"Все исполнители выбраны. "
                    f"Немедленно вызови task_create_tool с параметрами:\n"
                    f"• selected_employee_ids={ids_str}\n"
                    f"• task_text='{original_task_text}'\n"
                    f"Не задавай вопросов — сразу создай поручение."
                )
            else:  # introduction_create_tool
                instruction = (
                    f"Все сотрудники выбраны. "
                    f"Немедленно вызови introduction_create_tool с параметром:\n"
                    f"• selected_employee_ids={ids_str}\n"
                    f"Не задавай вопросов — сразу создай ознакомление."
                )

            logger.info(
                "UuidChoiceHandler: COMPLETE resolution for %s — "
                "all_ids=%d, creating task/introduction",
                tool_name,
                len(unique_selected),
            )

            return HandlerResult(
                new_inputs={"messages": [HumanMessage(content=instruction)]}
            )


class PlainMessageHandler:
    def can_handle(self, choice: str, pending_call: dict[str, Any] | None) -> bool:
        return True

    def handle(
        self,
        choice: str,
        pending_call: dict[str, Any] | None,
        messages: list[BaseMessage],
    ) -> HandlerResult:
        logger.info(
            "PlainMessageHandler: routing '%s' as new HumanMessage", choice[:40]
        )
        return HandlerResult(new_inputs={"messages": [HumanMessage(content=choice)]})


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class HumanChoiceHandler:
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
        for handler in self._handlers:
            if handler.can_handle(choice, pending_call):
                return handler.handle(choice, pending_call, messages)
        return HandlerResult(new_inputs={"messages": [HumanMessage(content=choice)]})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PENDING_STATUSES: frozenset[str] = frozenset(
    {"requires_choice", "requires_disambiguation", "requires_action"}
)


def find_pending_tool_call(messages: list[BaseMessage]) -> dict[str, Any] | None:
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
            tc = _find_disambig_tool_call(tool_calls) or tool_calls[0]
            return {
                "name": tc.get("name", ""),
                "args": dict(tc.get("args", {})),
                "id": tc.get("id", ""),
            }
    return None


def _find_disambig_tool_call(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    for tc in tool_calls:
        if tc.get("name") in _DISAMBIG_TOOLS:
            return tc
    return None


def _find_ai_message_for_call(
    messages: list[BaseMessage], call_id: str
) -> AIMessage | None:
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", []) or []:
            if tc.get("id") == call_id:
                return msg
    return None


def _is_real_tool_result(msg: ToolMessage) -> bool:
    content = str(msg.content).strip()
    if not (
        bool(content)
        and content not in ("", "{}", "null", "None")
        and len(content) >= 10
    ):
        return False
    if content.startswith("{"):
        try:
            data = json.loads(content)
            if data.get("status") in _PENDING_STATUSES:
                return False
        except (json.JSONDecodeError, TypeError):
            pass
    return True
