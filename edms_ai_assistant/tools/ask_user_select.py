# edms_ai_assistant/tools/ask_user_select.py
"""
Universal tool for rendering LLM-driven choices as clickable cards.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from edms_ai_assistant.agent.hitl_primitives import ToolAborted, ask_human

from langchain_core.runnables import RunnableConfig
from edms_ai_assistant.agent.interrupt_contract import (
    CardSelectInterrupt,
    CardSelectResume,
    InterruptCard,
    InterruptOption,
    SelectInterrupt,
    SelectResume,
)

logger = logging.getLogger(__name__)


class AskUserSelectInput(BaseModel):
    """Схема ввода для инструмента выбора."""

    prompt: str = Field(
        ...,
        description="Вопрос или пояснение для пользователя (заголовок карточек)",
        max_length=500,
    )
    options: list[str] = Field(
        ...,
        description=(
            "Список вариантов. Каждая опция может быть:\n"
            "  • простая строка: 'Заголовок'\n"
            "  • с описанием: 'Заголовок | Описание'\n"
            "  • со ссылкой: 'Заголовок | Описание | /document-form/UUID'\n"
            "  • JSON-объект-строка для богатой карточки документа:\n"
            '    \'{"label":"77Вх","description":"тема","attrs":{"Дата":"27.11.2025","Исполнитель":"ФИО"},"badges":["EXECUTION"],"url":"/document-form/UUID"}\'\n'
            "JSON-формат ОБЯЗАТЕЛЕН для списков найденных документов: даёт "
            "пользователю кнопку «Открыть в новой вкладке» и компактную "
            "таблицу атрибутов."
        ),
        min_length=2,
        max_length=20,
    )
    style: str = Field(
        default="cards",
        description=(
            "Стиль отображения: 'cards' — красивые карточки с кнопками, "
            "'list' — компактный список. По умолчанию 'cards'."
        ),
    )


@tool("ask_user_to_select", args_schema=AskUserSelectInput)
async def ask_user_to_select(
    prompt: str,
    options: list[str],
    style: str = "cards",
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> dict[str, Any]:
    """
    Показывает пользователю кликабельные карточки для выбора из списка вариантов.

    ИСПОЛЬЗУЙ ЭТОТ ИНСТРУМЕНТ ВСЕГДА, когда нужно предложить пользователю выбор
    из нескольких вариантов. НЕ выводи списки текстом!

    Примеры использования:
    - Выбор заголовка документа из нескольких вариантов
    - Выбор формата суммаризации
    - Выбор типа документа для создания
    - Любой выбор из 2+ вариантов

    Args:
        prompt: Вопрос или пояснение (отображается как заголовок).
        options: Список вариантов для выбора.
        style: Стиль отображения ('cards' или 'list').
        config: LangGraph RunnableConfig (инжектируется автоматически).

    Returns:
        Dict с выбранным вариантом:
        - selected: выбранный текст
        - selected_index: индекс выбранного варианта (начиная с 0)
    """
    logger.info(
        "ask_user_to_select: prompt='%s' options=%d style=%s",
        prompt[:50],
        len(options),
        style,
    )

    try:
        if style == "cards":
            cards = []
            for i, opt in enumerate(options):
                stripped = opt.strip()
                rich: dict[str, Any] | None = None
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        parsed = json.loads(stripped)
                        if isinstance(parsed, dict) and parsed.get("label"):
                            rich = parsed
                    except (json.JSONDecodeError, TypeError):
                        rich = None

                if rich is not None:
                    metadata: dict[str, Any] = {}
                    if url := rich.get("url"):
                        metadata["url"] = url
                    if cat := rich.get("category"):
                        metadata["category"] = cat

                    cards.append(
                        InterruptCard(
                            id=str(rich.get("id", i)),
                            label=str(rich["label"])[:100],
                            description=rich.get("description"),
                            badges=list(rich.get("badges") or []),
                            primary_attrs={
                                str(k): str(v)
                                for k, v in (rich.get("attrs") or {}).items()
                            },
                            metadata=metadata if metadata else None,
                        )
                    )
                else:
                    parts = [p.strip() for p in stripped.split(" | ", 2)]
                    label = parts[0]
                    description = parts[1] if len(parts) > 1 and parts[1] else None
                    url = parts[2] if len(parts) > 2 and parts[2] else None
                    cards.append(
                        InterruptCard(
                            id=str(i),
                            label=label[:100],
                            description=description,
                            metadata={"url": url} if url else None,
                        )
                    )

            resume = ask_human(
                CardSelectInterrupt(
                    prompt=prompt,
                    cards=cards,
                    multiple=False,
                    layout="list",
                )
            )

            if not isinstance(resume, CardSelectResume):
                return {"status": "cancelled", "message": "Выбор отменён"}

            selected_id = resume.selected_ids[0]
            selected_index = int(selected_id)
            selected_text = options[selected_index]

        else:
            interrupt_options = [
                InterruptOption(id=str(i), label=opt[:100])
                for i, opt in enumerate(options)
            ]

            resume = ask_human(
                SelectInterrupt(
                    prompt=prompt,
                    options=interrupt_options,
                )
            )

            if not isinstance(resume, SelectResume):
                return {"status": "cancelled", "message": "Выбор отменён"}

            selected_index = int(resume.selected_id)
            selected_text = options[selected_index]

        logger.info(
            "User selected: index=%d text='%s'",
            selected_index,
            selected_text[:50],
        )

        return {
            "status": "selected",
            "selected": selected_text,
            "selected_index": selected_index,
        }

    except ToolAborted:
        logger.info("User cancelled selection")
        return {"status": "cancelled", "message": "Пользователь отменил выбор"}
