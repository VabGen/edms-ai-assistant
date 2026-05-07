# edms_ai_assistant/agent/planning/models.py
"""
ExecutionPlan — типизированная модель плана выполнения агента.

Заменяет NLP intent routing на LLM-based планирование.
LLM сама решает: нужны tools или можно ответить напрямую.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallStep(BaseModel):
    """Один вызов tool."""
    tool_name: str = Field(description="Имя инструмента из списка доступных")
    reason: str = Field(description="Почему нужен этот tool")
    args_hint: dict[str, Any] = Field(
        default_factory=dict,
        description="Подсказки по аргументам (token/document_id инжектируются автоматически)",
    )


class ParallelGroup(BaseModel):
    """Группа tools для параллельного выполнения."""
    steps: list[ToolCallStep] = Field(min_length=2)
    reason: str = Field(description="Почему эти tools можно выполнить параллельно")


class DirectAnswerStep(BaseModel):
    """Ответить напрямую без tools."""
    reason: str = Field(description="Почему tools не нужны")


class ExecutionPlan(BaseModel):
    """
    Структурированный план выполнения запроса.

    Производится Planning LLM-вызовом с response_format=json_object.
    Определяет: нужны ли tools, какие, в каком порядке/параллельно.
    """
    model_config = {"frozen": True}

    can_answer_directly: bool = Field(
        description=(
            "True если запрос можно обработать из знаний модели "
            "без обращения к EDMS API. "
            "Примеры для True: общие вопросы, юридический анализ загруженного файла, "
            "объяснение понятий, грамматика, математика. "
            "Примеры для False: данные документов EDMS, поиск сотрудников, "
            "создание поручений, ознакомление."
        )
    )
    parallel_capable: bool = Field(
        default=False,
        description="True если первые шаги можно выполнить параллельно",
    )
    steps: list[ToolCallStep | ParallelGroup | DirectAnswerStep] = Field(
        default_factory=list,
        description="Шаги выполнения. Пустой список если can_answer_directly=True.",
    )
    reasoning: str = Field(
        description="Краткое объяснение плана (1-2 предложения)",
        max_length=300,
    )
    estimated_complexity: Literal["simple", "medium", "complex"] = Field(
        default="simple",
        description="simple | medium | complex",
    )