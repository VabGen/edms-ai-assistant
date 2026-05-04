"""
ResponseBuilder — строит финальный AgentResponse из цепочки сообщений LangGraph.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from edms_ai_assistant.agent.context import AgentResponse, AgentStatus, ContextParams
from edms_ai_assistant.agent.orchestration.sanitizer import sanitize_technical_content

logger = logging.getLogger(__name__)

# Фразы-маркеры успешных мутирующих операций
_MUTATION_PHRASES: tuple[str, ...] = (
    "успешно добавлен",
    "успешно создан",
    "список ознакомления",
    "поручение создано",
    "поручение успешно",
    "обращение заполнено",
    "карточка заполнена",
    "добавлено в список",
    "ознакомление создано",
    "задача создана",
    "автозаполнен",
    "заголовок обновлен",
    "операция выполнена успешно",
)


def _is_mutation_response(content: str | None) -> bool:
    if not content:
        return False
    lower = content.lower()
    return any(phrase in lower for phrase in _MUTATION_PHRASES)


# ---------------------------------------------------------------------------
# InteractiveStatusDetector — детектирует статусы, требующие взаимодействия
# ---------------------------------------------------------------------------


class InteractiveStatusDetector:
    """
    Сканирует последний ToolMessage на наличие интерактивных статусов.

    Поддерживаемые статусы:
    - requires_choice       → action_type="summarize_selection" (выбор формата анализа)
    - requires_disambiguation → action_type="requires_disambiguation" (карточки сотрудников/вложений)
    - requires_action       → action_type="requires_disambiguation" (карточки из employee_search)
    """

    def detect(self, messages: list[BaseMessage]) -> dict[str, Any] | None:
        """Возвращает сериализованный AgentResponse или None."""
        last_tool = next(
            (m for m in reversed(messages) if isinstance(m, ToolMessage)),
            None,
        )
        if last_tool is None:
            return None

        raw = str(last_tool.content).strip()
        if not raw.startswith("{"):
            return None

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return None

        status: str = data.get("status", "")

        if status == "requires_choice":
            return self._build_requires_choice(data)
        if status in ("requires_disambiguation", "requires_action"):
            return self._build_disambiguation(data)

        return None

    @staticmethod
    def _build_requires_choice(data: dict[str, Any]) -> dict[str, Any]:
        """
        Формат выбора суммаризации — маппится во фронтенде на action_type=summarize_selection,
        что отображает кнопки «Пересказ», «Факты», «Тезисы».
        """
        options: list[dict[str, Any]] = data.get("options", [])
        hint: str = data.get("hint", "extractive")
        hint_reason: str = data.get("hint_reason", "")
        msg: str = data.get("message", "Выберите формат анализа документа:")

        option_lines = "\n".join(
            f"- **{opt['key']}** — {opt['label']}: {opt['description']}"
            for opt in options
            if isinstance(opt, dict)
        )
        hint_text = (
            f"\n\n💡 *Рекомендация:* **{hint}** — {hint_reason}" if hint_reason else ""
        )
        full_msg = (
            f"{msg}\n\n{option_lines}{hint_text}\n\n"
            "Ответьте: **extractive**, **abstractive** или **thesis**."
        )

        return AgentResponse(
            status=AgentStatus.REQUIRES_ACTION,
            action_type="summarize_selection",  # строка, не enum — фронтенд ожидает строку
            message=full_msg,
        ).model_dump()

    @staticmethod
    def _build_disambiguation(data: dict[str, Any]) -> dict[str, Any] | None:
        """
        Формирует карточки кандидатов для выбора сотрудника/вложения.
        Поддерживает как requires_disambiguation, так и requires_action (employee_search).
        """
        # Извлекаем кандидатов из всех известных полей
        available: list[dict[str, Any]] = (
            data.get("available_attachments")
            or data.get("available_employees")
            or data.get("choices")  # employee_search_tool → choices
            or data.get("candidates")
            or data.get("ambiguous_matches")
            or []
        )

        if not available:
            return None

        base_msg: str = data.get("message", "Уточните выбор:")
        # Убираем лишние UUID из сообщения
        import re

        base_msg = (
            re.sub(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                "",
                base_msg,
            )
            .strip()
            .rstrip("с «»")
            .strip()
            or "Уточните выбор:"
        )

        candidates = [
            _normalise_candidate(item) for item in available if isinstance(item, dict)
        ]
        candidates_json = json.dumps(candidates, ensure_ascii=False)

        return AgentResponse(
            status=AgentStatus.REQUIRES_ACTION,
            action_type="requires_disambiguation",
            message=f"{base_msg}\n\n<!--CANDIDATES:{candidates_json}-->",
        ).model_dump()


def _normalise_candidate(item: dict[str, Any]) -> dict[str, str]:
    """Нормализует словарь кандидата в {id, name, dept}."""
    display_name = (
        item.get("full_name")
        or item.get("fullName")
        or item.get("fio")
        or item.get("name")
        or ""
    ).strip()

    if not display_name:
        last = item.get("lastName", "")
        first = item.get("firstName", "")
        middle = item.get("middleName", "") or ""
        display_name = (
            " ".join(filter(None, [last, first, middle])).strip() or "Без имени"
        )

    dept = (
        item.get("department")
        or item.get("departmentName")
        or item.get("post")
        or item.get("position")
        or ""
    ).strip()

    item_id = str(item.get("id") or item.get("uuid") or item.get("employeeId") or "?")

    return {"id": item_id, "name": display_name, "dept": dept}


# ---------------------------------------------------------------------------
# ResponseBuilder
# ---------------------------------------------------------------------------


class ResponseBuilder:
    """Строит финальный AgentResponse из цепочки сообщений."""

    def __init__(self) -> None:
        self._interactive_detector = InteractiveStatusDetector()

    def build(
        self,
        messages: list[BaseMessage],
        context: ContextParams,
        content_extractor: Any,  # ContentExtractor — избегаем circular import
    ) -> dict[str, Any]:
        """
        Основной метод построения ответа.

        Порядок проверок:
        1. Интерактивный статус (requires_choice, disambiguation)
        2. Compliance данные из последнего ToolMessage
        3. Финальный текст (AIMessage или ToolMessage)
        4. navigate_url из ToolMessages
        """
        interactive = self._interactive_detector.detect(messages)
        if interactive:
            logger.info(
                "Interactive status detected: action_type=%s",
                interactive.get("action_type"),
                extra={"thread_id": context.thread_id},
            )
            return interactive

        compliance_data = self._extract_compliance(messages)

        final_content = content_extractor.extract_final_content(messages)
        navigate_url = content_extractor.extract_navigate_url(messages)

        metadata: dict[str, Any] = {}
        if compliance_data:
            metadata["compliance"] = compliance_data

        if final_content:
            final_content = content_extractor.clean_json_artifacts(final_content)
            final_content = sanitize_technical_content(final_content, context)
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                content=final_content,
                requires_reload=_is_mutation_response(final_content),
                navigate_url=navigate_url,
                metadata=metadata,
            ).model_dump()

        logger.warning(
            "No final content found",
            extra={"thread_id": context.thread_id},
        )
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            content="Операция завершена.",
            navigate_url=navigate_url,
            metadata=metadata,
        ).model_dump()

    @staticmethod
    def _extract_compliance(messages: list[BaseMessage]) -> dict[str, Any] | None:
        """Ищет compliance-результат в последних ToolMessages (staticmethod)."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, ToolMessage):
                try:
                    data: dict[str, Any] = json.loads(str(msg.content))
                    if (
                        data.get("status") == "success"
                        and isinstance(data.get("fields"), list)
                        and "overall" in data
                    ):
                        return data
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
        return None
