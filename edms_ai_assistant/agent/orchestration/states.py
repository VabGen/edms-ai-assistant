# edms_ai_assistant/agent/orchestration/states.py
"""
Явная state machine для AgentExecutionState.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Final


class ExecutionState(Enum):
    """Состояния выполнения одного turn агента."""

    INIT = auto()  # Начало обработки запроса
    INVOKING = auto()  # Граф выполняется
    INSPECTING = auto()  # Анализ результата вызова
    PATCHING = auto()  # Подготовка аргументов tool_call
    AWAITING_HUMAN = auto()  # Ожидание выбора пользователя (HITL)
    BUILDING_RESPONSE = auto()  # Сборка финального ответа
    DONE = auto()  # Успешное завершение
    ERROR = auto()  # Ошибка, требующая ответа пользователю


# Допустимые переходы: source -> set of valid targets
VALID_TRANSITIONS: Final[dict[ExecutionState, frozenset[ExecutionState]]] = {
    ExecutionState.INIT: frozenset({ExecutionState.INVOKING, ExecutionState.ERROR}),
    ExecutionState.INVOKING: frozenset(
        {
            ExecutionState.INSPECTING,
            ExecutionState.ERROR,
        }
    ),
    ExecutionState.INSPECTING: frozenset(
        {
            ExecutionState.PATCHING,
            ExecutionState.AWAITING_HUMAN,
            ExecutionState.BUILDING_RESPONSE,
            ExecutionState.ERROR,
        }
    ),
    ExecutionState.PATCHING: frozenset({ExecutionState.INVOKING, ExecutionState.ERROR}),
    ExecutionState.AWAITING_HUMAN: frozenset({ExecutionState.DONE}),
    ExecutionState.BUILDING_RESPONSE: frozenset({ExecutionState.DONE}),
    ExecutionState.DONE: frozenset(),
    ExecutionState.ERROR: frozenset({ExecutionState.DONE}),
}


class InvalidTransitionError(Exception):
    """Попытка перехода в недопустимое состояние."""

    def __init__(
        self,
        from_state: ExecutionState,
        to_state: ExecutionState,
    ) -> None:
        super().__init__(
            f"Invalid transition: {from_state.name} → {to_state.name}. "
            f"Valid targets: {[s.name for s in VALID_TRANSITIONS.get(from_state, frozenset())]}"
        )
        self.from_state = from_state
        self.to_state = to_state
