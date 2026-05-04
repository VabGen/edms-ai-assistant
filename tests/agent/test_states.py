# tests/agent/test_states.py
"""Тесты state machine переходов."""
import pytest
from edms_ai_assistant.agent.orchestration.states import (
    ExecutionState,
    InvalidTransitionError,
    VALID_TRANSITIONS,
)
from edms_ai_assistant.agent.orchestration.loop import _StateMachine


class TestValidTransitions:
    def test_init_to_invoking(self):
        sm = _StateMachine("thread_1")
        sm.transition(ExecutionState.INVOKING)
        assert sm.current == ExecutionState.INVOKING
        assert sm.iteration == 1

    def test_invoking_to_inspecting(self):
        sm = _StateMachine("thread_1")
        sm.transition(ExecutionState.INVOKING)
        sm.transition(ExecutionState.INSPECTING)
        assert sm.current == ExecutionState.INSPECTING

    def test_full_happy_path(self):
        sm = _StateMachine("thread_1")
        sm.transition(ExecutionState.INVOKING)
        sm.transition(ExecutionState.INSPECTING)
        sm.transition(ExecutionState.BUILDING_RESPONSE)
        sm.transition(ExecutionState.DONE)
        assert sm.current == ExecutionState.DONE

    def test_iteration_increments_on_invoking(self):
        sm = _StateMachine("thread_1")
        sm.transition(ExecutionState.INVOKING)
        sm.transition(ExecutionState.INSPECTING)
        sm.transition(ExecutionState.PATCHING)
        sm.transition(ExecutionState.INVOKING)
        assert sm.iteration == 2


class TestInvalidTransitions:
    def test_init_cannot_go_to_done(self):
        sm = _StateMachine("thread_1")
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.transition(ExecutionState.DONE)
        assert exc_info.value.from_state == ExecutionState.INIT
        assert exc_info.value.to_state == ExecutionState.DONE

    def test_done_cannot_transition(self):
        sm = _StateMachine("thread_1")
        sm.transition(ExecutionState.INVOKING)
        sm.transition(ExecutionState.INSPECTING)
        sm.transition(ExecutionState.BUILDING_RESPONSE)
        sm.transition(ExecutionState.DONE)
        with pytest.raises(InvalidTransitionError):
            sm.transition(ExecutionState.INVOKING)

    def test_all_valid_transitions_covered(self):
        """Проверяем что все состояния присутствуют в VALID_TRANSITIONS."""
        for state in ExecutionState:
            assert state in VALID_TRANSITIONS, f"Missing: {state}"