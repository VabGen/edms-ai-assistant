from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from edms_ai_assistant.domain.base import EdmsBaseDto
from edms_ai_assistant.domain.enums import (
    AttachmentDocumentType,
    AttachmentType,
    CreateType,
    DocCategory,
    DocumentProcessType,
    DocumentStatus,
    JobStatus,
)


class DocumentTypeDto(EdmsBaseDto):
    id: int | None = None
    type_name: str | None = None
    doc_category_const: DocCategory | None = None
    active: bool | None = None
    object_type: str | None = Field(None, max_length=3, min_length=3, description="вид по таблицы 2 окб")
    create_date: datetime | None = None


class AttachmentDto(EdmsBaseDto):
    name: str | None = Field(None, description="Наименование вложенного файла")
    size: int | None = Field(None, description="Размер вложенного файла")


class DocumentDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    document_type: DocumentTypeDto | None = None
    short_summary: str | None = None
    reg_number: str | None = None
    reg_date: datetime | None = None
    out_number: str | None = None
    out_date: datetime | None = None
    status: DocumentStatus | None = None
    create_date: datetime | None = None
    author_id: UUID | None = None
    signed_by_id: UUID | None = None
    on_control: bool | None = None
    create_type: CreateType | None = None
    doc_category_const: DocCategory | None = None


class DocPermissionContainer(EdmsBaseDto):
    id: UUID | None = None
    can_read: bool | None = None
    can_edit: bool | None = None
    can_delete: bool | None = None
    can_sign: bool | None = None


class DocumentWithPermissions(EdmsBaseDto):
    document: DocumentDto
    permission: DocPermissionContainer


class DocumentPropertiesDto(EdmsBaseDto):
    id: UUID | None = None


class DocumentHistoryDto(EdmsBaseDto):
    id: UUID | None = None
    action_name: str | None = None
    employee_id: UUID | None = None
    create_date: datetime | None = None
    old_value: str | None = None
    new_value: str | None = None


class ControlDto(EdmsBaseDto):
    id: UUID | None = None
    control_type_id: UUID | None = None
    date_control_end: datetime | None = None
    control_days: int | None = None
    auto_control: bool | None = None


class BpmnProcessActivityDto(EdmsBaseDto):
    id: UUID | None = None
    process_id: UUID | None = None
    activity_type: DocumentProcessType | None = None


class BpmnProcessDirectoryDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    doc_category: DocCategory | None = None
    create_date: datetime | None = None
    active: bool | None = None


class TasksAndProjectsDto(EdmsBaseDto):
    tasks: list[dict[str, Any]] = []
    task_projects: list[dict[str, Any]] = []


class DocumentVersionDto(EdmsBaseDto):
    id: UUID | None = Field(None, description="Идентификатор версии")
    version: int | None = Field(None, description="Номер версии")
    document_id: UUID | None = Field(None, description="Идентификатор документа")
    document: DocumentDto | None = Field(None, description="Документ")
    deleted: bool | None = None


class DocumentRecipientDto(EdmsBaseDto):
    id: UUID | None = None
    recipient_id: UUID | None = None
    recipient_type: str | None = None


class ExecutionDocumentStatCount(EdmsBaseDto):
    total: int = 0
    executed: int = 0
    on_execution: int = 0
    overdue: int = 0


class DocumentUserColorDto(EdmsBaseDto):
    id: UUID | None = None
    document_id: UUID | None = None
    document_org_id: str | None = None
    color: str | None = None


class DocumentUserPropsDto(EdmsBaseDto):
    document_id: UUID | None = None
    employee_id: UUID | None = None
    create_task_count: int | None = None
    create_task_executed_count: int | None = None


class DocumentFormField(EdmsBaseDto):
    id: str | None = None
    payload: dict[str, dict[str, Any]] | None = None
