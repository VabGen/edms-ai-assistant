# edms_ai_assistant/tools/ask_user_select.py
"""
Universal tool for rendering LLM-driven choices as clickable cards.
"""

from __future__ import annotations

import logging
from typing import Any, Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from edms_ai_assistant.agent.hitl_primitives import ask_human, ToolAborted
from edms_ai_assistant.agent.interrupt_contract import (
    SelectInterrupt,
    InterruptOption,
    SelectResume,
    CardSelectInterrupt,
    InterruptCard,
    CardSelectResume,
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
        description="Список вариантов для выбора",
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
            # Красивые карточки с заголовком и описанием
            cards = []
            for i, opt in enumerate(options):
                # Пытаемся разделить на заголовок и описание если есть разделитель
                parts = opt.split(" | ", 1)
                label = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else None

                cards.append(
                    InterruptCard(
                        id=str(i),
                        label=label[:100],
                        description=description,
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
            # Компактный список
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