"""
edms_ai_assistant.domain — domain/business models.

Replaces the ambiguously named `models/` package (which conflicted with
`model.py` — the HTTP/LangGraph contract layer at the same level).

Public API:
    AppealFields, DeclarantType, SubmissionFormAppeal — appeal document domain
    CreateTaskRequest, CreateTaskBatchRequest,
    CreateTaskRequestExecutor, TaskCreationResult, TaskType — task domain
"""

from edms_ai_assistant.domain.appeal_fields import (
    AppealFields,
    DeclarantType,
    SubmissionFormAppeal,
)
from edms_ai_assistant.domain.task_models import (
    CreateTaskBatchRequest,
    CreateTaskRequest,
    CreateTaskRequestExecutor,
    TaskCreationResult,
    TaskType,
)

__all__ = [
    "AppealFields",
    "CreateTaskBatchRequest",
    "CreateTaskRequest",
    "CreateTaskRequestExecutor",
    "CreateTaskRequestExecutor",
    "DeclarantType",
    "SubmissionFormAppeal",
    "TaskCreationResult",
    "TaskType",
]
