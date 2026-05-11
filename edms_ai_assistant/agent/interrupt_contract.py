# edms_ai_assistant/agent/interrupt_contract.py
"""Универсальный контракт HITL прерываний для LangGraph."""

from __future__ import annotations

import enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class InterruptType(str, enum.Enum):
    ENTITY_DISAMBIGUATION = "entity_disambiguation"
    SUMMARY_TYPE_SELECTION = "summary_type_selection"
    FIELD_CORRECTION = "field_correction"
    PLAIN_CONFIRMATION = "plain_confirmation"


class DisambiguationOption(BaseModel):
    id: str = Field(description="UUID сущности")
    name: str = Field(description="Основной текст: ФИО, Название файла и т.д.")
    dept: str = Field(default="", description="Доп. контекст: Отдел, Тип файла и т.д.")
    search_term: str = Field(
        default="",
        description=(
            "Исходный поисковый запрос, по которому найдена эта сущность. "
            "Используется UI для группировки: «Уточните Петров (19 совпадений)», "
            "«Уточните Иванов (20 совпадений)»."
        ),
    )


class EntityDisambiguationPayload(BaseModel):
    type: Literal[InterruptType.ENTITY_DISAMBIGUATION] = (
        InterruptType.ENTITY_DISAMBIGUATION
    )
    entity_type: str = Field(
        description="employee | document | department | attachment"
    )
    options: list[DisambiguationOption] = Field(description="Список вариантов")
    multiple: bool = Field(default=False)


class SummaryTypePayload(BaseModel):
    type: Literal[InterruptType.SUMMARY_TYPE_SELECTION] = (
        InterruptType.SUMMARY_TYPE_SELECTION
    )
    options: list[str] = Field(default=["extractive", "abstractive", "thesis"])


InterruptPayload = Union[EntityDisambiguationPayload, SummaryTypePayload]


class EntityDisambiguationResume(BaseModel):
    type: Literal[InterruptType.ENTITY_DISAMBIGUATION] = (
        InterruptType.ENTITY_DISAMBIGUATION
    )
    selected_ids: list[str]


class SummaryTypeResume(BaseModel):
    type: Literal[InterruptType.SUMMARY_TYPE_SELECTION] = (
        InterruptType.SUMMARY_TYPE_SELECTION
    )
    summary_type: str


ResumePayload = Union[EntityDisambiguationResume, SummaryTypeResume]
