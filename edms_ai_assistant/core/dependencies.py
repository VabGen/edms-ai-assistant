# edms_ai_assistant/core/dependencies.py
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from edms_ai_assistant.clients.transport import HttpxTransport, IAsyncTransport
from edms_ai_assistant.clients.attachment_client import AttachmentClient
from edms_ai_assistant.clients.department_client import DepartmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.document_creator_client import DocumentCreatorClient
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.group_client import GroupClient
from edms_ai_assistant.clients.reference_client import ReferenceClient
from edms_ai_assistant.clients.task_client import TaskClient
from edms_ai_assistant.clients.access_grief_client import AccessGriefClient
from edms_ai_assistant.config import EdmsSettings, settings
from edms_ai_assistant.services.subject_service import SubjectService


# ── Настройки ────────────────────────────────────────────────────────────

@lru_cache
def get_settings() -> EdmsSettings:
    return settings


# ── Транспорт ────────────────────────────────────────────────────────────

@lru_cache
def get_transport() -> HttpxTransport:
    """Создает (и кэширует) единственный инстанс транспорта для приложения."""
    s = get_settings()
    return HttpxTransport(base_url=str(s.base_url), default_timeout=s.timeout)


# ── Клиенты ──────────────────────────────────────────────────────────────

def get_document_client(
        transport: Annotated[IAsyncTransport, Depends(get_transport)],
        edms_settings: Annotated[EdmsSettings, Depends(get_settings)],
) -> DocumentClient:
    return DocumentClient(transport, edms_settings)


def get_employee_client(
        transport: Annotated[IAsyncTransport, Depends(get_transport)],
        edms_settings: Annotated[EdmsSettings, Depends(get_settings)],
) -> EmployeeClient:
    return EmployeeClient(transport, edms_settings)


def get_reference_client(
        transport: Annotated[IAsyncTransport, Depends(get_transport)],
        edms_settings: Annotated[EdmsSettings, Depends(get_settings)],
) -> ReferenceClient:
    return ReferenceClient(transport, edms_settings)


def get_creator_client(
        transport: Annotated[IAsyncTransport, Depends(get_transport)],
        edms_settings: Annotated[EdmsSettings, Depends(get_settings)],
) -> DocumentCreatorClient:
    return DocumentCreatorClient(transport, edms_settings)


def get_task_client(
        transport: Annotated[IAsyncTransport, Depends(get_transport)],
        edms_settings: Annotated[EdmsSettings, Depends(get_settings)],
) -> TaskClient:
    return TaskClient(transport, edms_settings)


def get_department_client(
        transport: Annotated[IAsyncTransport, Depends(get_transport)],
        edms_settings: Annotated[EdmsSettings, Depends(get_settings)],
) -> DepartmentClient:
    return DepartmentClient(transport, edms_settings)


def get_group_client(
        transport: Annotated[IAsyncTransport, Depends(get_transport)],
        edms_settings: Annotated[EdmsSettings, Depends(get_settings)],
) -> GroupClient:
    return GroupClient(transport, edms_settings)


def get_attachment_client(
        transport: Annotated[IAsyncTransport, Depends(get_transport)],
        edms_settings: Annotated[EdmsSettings, Depends(get_settings)],
) -> AttachmentClient:
    return AttachmentClient(transport, edms_settings)


def get_access_grief_client(
        transport: Annotated[IAsyncTransport, Depends(get_transport)],
        edms_settings: Annotated[EdmsSettings, Depends(get_settings)],
) -> AccessGriefClient:
    return AccessGriefClient(transport, edms_settings)


# ── Сервисы ──────────────────────────────────────────────────────────────

def get_subject_service(
        ref_client: Annotated[ReferenceClient, Depends(get_reference_client)]
) -> SubjectService:
    return SubjectService(ref_client)


# ── Псевдонимы для удобной инъекции в роуты ─────────────────────────────

DocClient = Annotated[DocumentClient, Depends(get_document_client)]
EmpClient = Annotated[EmployeeClient, Depends(get_employee_client)]
RefClient = Annotated[ReferenceClient, Depends(get_reference_client)]
CreatorClient = Annotated[DocumentCreatorClient, Depends(get_creator_client)]
TaskClientDep = Annotated[TaskClient, Depends(get_task_client)]
DeptClient = Annotated[DepartmentClient, Depends(get_department_client)]
GroupClientDep = Annotated[GroupClient, Depends(get_group_client)]
AttachClient = Annotated[AttachmentClient, Depends(get_attachment_client)]
GriefClient = Annotated[AccessGriefClient, Depends(get_access_grief_client)]
SubjService = Annotated[SubjectService, Depends(get_subject_service)]
