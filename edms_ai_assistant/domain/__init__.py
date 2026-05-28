"""
edms_ai_assistant.domain — domain/business models.

Replaces the ambiguously named `models/` package (which conflicted with
`model.py` — the HTTP/LangGraph contract layer at the same level).

Public API:
    AppealFields, DeclarantType, SubmissionFormAppeal — appeal document domain
    CreateTaskRequest, CreateTaskBatchRequest,
    CreateTaskRequestExecutor, TaskCreationResult, TaskType — task domain
    DocumentProfileDto, EmployeeDto, OrgDto — EDMS core domain
"""

from edms_ai_assistant.domain.base import EdmsBaseDto
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

from edms_ai_assistant.domain import employee, document, reference, report

_types_namespace = {}
for _mod in [employee, document, reference, report]:
    _types_namespace.update(
        {name: obj for name, obj in vars(_mod).items() if isinstance(obj, type)}
    )

for _name, _cls in _types_namespace.items():
    if isinstance(_cls, type) and issubclass(_cls, EdmsBaseDto):
        try:
            _cls.model_rebuild(_types_namespace=_types_namespace)
        except Exception:
            pass

del _types_namespace, _name, _cls, _mod

__all__ = [
    "AppealFields",
    "CreateTaskBatchRequest",
    "CreateTaskRequest",
    "CreateTaskRequestExecutor",
    "DeclarantType",
    "SubmissionFormAppeal",
    "TaskCreationResult",
    "TaskType",
]
