"""
edms_ai_assistant.agent.orchestration — пакет оркестрации агента.

Публичный API:
    OrchestrationLoop   — основной цикл оркестрации (state machine)
    handle_human_choice — свободная функция обработки HITL-выборов
    HumanChoiceHandler  — роутер HITL-стратегий
    HandlerResult       — результат обработки HITL
    ResponseBuilder     — построитель финального ответа
    InteractiveStatusDetector — детектор интерактивных статусов
    sanitize_technical_content — очистка UUID/путей из ответа
"""

from edms_ai_assistant.agent.orchestration.hitl import (
    HandlerResult,
    HumanChoiceHandler,
    find_pending_tool_call,
)
from edms_ai_assistant.agent.orchestration.loop import (
    OrchestrationLoop,
    handle_human_choice,
)
from edms_ai_assistant.agent.orchestration.response_builder import (
    InteractiveStatusDetector,
    ResponseBuilder,
)
from edms_ai_assistant.agent.orchestration.sanitizer import sanitize_technical_content

__all__ = [
    "HandlerResult",
    "HumanChoiceHandler",
    "InteractiveStatusDetector",
    "OrchestrationLoop",
    "ResponseBuilder",
    "find_pending_tool_call",
    "handle_human_choice",
    "sanitize_technical_content",
]
