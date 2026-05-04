# edms_ai_assistant/agent/__init__.py
"""
Public API for the EDMS AI Agent package.

All external consumers import exclusively from this module.
Internal sub-modules are considered private implementation details.
"""

from edms_ai_assistant.agent.agent import EdmsDocumentAgent
from edms_ai_assistant.agent.context import (
    ActionType,
    AgentRequest,
    AgentResponse,
    AgentStatus,
    ContextParams,
)
from edms_ai_assistant.agent.orchestration import handle_human_choice
from edms_ai_assistant.agent.repositories import (
    DocumentRepository,
    IDocumentRepository,
)

__all__: list[str] = [
    "ActionType",
    "AgentRequest",
    "AgentResponse",
    "AgentStatus",
    "ContextParams",
    "DocumentRepository",
    "EdmsDocumentAgent",
    "IDocumentRepository",
    "handle_human_choice",
]
