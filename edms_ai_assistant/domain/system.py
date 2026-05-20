from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from edms_ai_assistant.domain.base import EdmsBaseDto
from edms_ai_assistant.domain.enums import (
    ContentType,
    DocCategory,
    PanelType,
    Period,
    RecognitionFileTypeEnum,
    RecognitionLanguageEnum,
    ReportColumn,
    ReportType,
)
from edms_ai_assistant.domain.document import AttachmentDto


class TemplateDto(EdmsBaseDto):
    id: UUID | None = None
    template_name: str | None = None
    attachment: AttachmentDto | None = None
    doc_category_constant: DocCategory | None = None
    active: bool | None = None


class TemplateContractDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    attachments: list[AttachmentDto] | None = None
    active: bool | None = None


class RecognitionSetupDto(EdmsBaseDto):
    organization_id: str | None = None
    organization_name: str | None = None
    recognition_language: list[RecognitionLanguageEnum] | None = None
    doc_category_constant: list[DocCategory] | None = None
    recognition_file_type: list[RecognitionFileTypeEnum] | None = None
    active: bool | None = None


class UserDashboardDto(EdmsBaseDto):
    data: dict[str, Any] | None = None


class UserActionPanelBase(EdmsBaseDto):
    panel_type: PanelType | None = None
    period: Period | None = None


class UserActionPanelCountResponse(EdmsBaseDto):
    id: UUID | None = None
    data: UserActionPanelBase | None = None
    panel_type: PanelType | None = None
    sort: int | None = None
    count: int | None = None
    create_date: datetime | None = None


class ReportUserField(EdmsBaseDto):
    doc_profile_name: list[str] | None = None
    doc_type_name: list[str] | None = None
    destination_id: UUID | None = None
    signed_by_id: UUID | None = None


class ReportUserConfigDto(EdmsBaseDto):
    id: UUID | None = None
    type: ReportType | None = None
    name: str | None = None
    employee_id: UUID | None = None
    fields: dict[str, str] | None = None
    user_fields: ReportUserField | None = None
    columns: list[ReportColumn] | None = None


class TaskKanbanColumnDto(EdmsBaseDto):
    id: UUID | None = None
    employee_id: UUID | None = None
    name: str | None = None
    color: str | None = None
    order: int | None = None
