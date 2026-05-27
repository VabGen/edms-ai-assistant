from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

from pydantic import Field

from edms_ai_assistant.domain.base import EdmsBaseDto
from edms_ai_assistant.domain.enums import (
    AppealTypeReport,
    DeclarantType,
    DocCategory,
    DocumentStatus,
    ReportColumns,
    ReportField,
    ReportFormatType,
    ReportTaskControlField,
    ReportTaskStatus,
    ReportType,
    TaskStatus,
)

if TYPE_CHECKING:
    from edms_ai_assistant.domain.document import AttachmentDto


class ReportTaskDto(EdmsBaseDto):
    """DTO для задачи на формирование отчета V2."""

    id: UUID | None = None
    employee_id: UUID | None = None
    employee_org_id: str | None = None
    create_date: datetime | None = None
    status: ReportTaskStatus | None = None
    attachment: AttachmentDto | None = None
    data: dict[str, Any] | None = None
    end_date: datetime | None = None
    delayed_to: datetime | None = None
    attempt: int | None = None
    error_text: str | None = None
    type: ReportType | None = None


class ReportTaskFilter(EdmsBaseDto):
    """Фильтр для поиска отчетов пользователя."""

    status: ReportTaskStatus | None = None


class AbstractReportFilter(EdmsBaseDto):
    """Базовый фильтр для всех отчетов V2."""

    type: ReportFormatType


class CountOfExecutedAndUnexecutedControlTaskFilter(EdmsBaseDto):
    """Фильтр для отчета 'Данные о количестве исполненных и не исполненных контрольных поручений'."""

    in_archive: bool | None = None
    date_reg_start: datetime
    date_reg_end: datetime
    document_category: DocCategory | None = None
    document_profile_id: UUID | None = None
    task_executor_id: UUID | None = None
    task_executor_first_name: str | None = None
    task_executor_last_name: str | None = None
    task_executor_middle_name: str | None = None
    task_executor_department_id: UUID | None = None
    task_executor_department_name: str | None = None
    flag_chart: bool | None = None


class PerformingDisciplineReportFilter(EdmsBaseDto):
    """Фильтр для отчета 'Исполнительская дисциплина'."""

    in_archive: bool | None = None
    date_reg_start: datetime
    date_reg_end: datetime
    doc_category_constants: list[DocCategory]
    document_type_ids: list[int] | None = None
    flag_transcript: bool | None = None
    flag_chart: bool | None = None
    flag_io_or_secretary: bool | None = None
    flag_io_dismissal: bool | None = None
    parent_department_id: UUID | None = None
    executor_id: UUID | None = None


class ReceivedAppealsReportFilter(EdmsBaseDto):
    """Фильтр для отчета 'Поступившие обращения'."""

    in_archive: bool | None = None
    date_reg_start: datetime
    date_reg_end: datetime
    declarant_types: list[DeclarantType]
    applicant_text: str | None = None
    equals: bool | None = None
    country_appeal_ids: list[UUID] | None = None
    appeal_type_ids: list[UUID] | None = None
    appeal_type: AppealTypeReport | None = None
    delivery_method_names: list[str] | None = None
    subject_id: UUID | None = None
    solution_result_ids: list[UUID] | None = None
    statuses: list[DocumentStatus] | None = None
    flag_chart: bool | None = None


class VolumeOfDocumentFlowReportFilter(EdmsBaseDto):
    """Фильтр для отчета 'Объем документооборота'."""

    in_archive: bool | None = None
    date_reg_start: datetime
    date_reg_end: datetime
    flag_diagram_circular: bool
    flag_diagram_by_type: bool


class TaskOnControlReportFilter(AbstractReportFilter):
    """Фильтр для отчета 'Контроль поручений'."""

    in_archive: bool | None = None
    date_control_start: datetime
    date_control_end: datetime
    control_employee_id: UUID | None = None


class TaskOnStatusReportFilter(AbstractReportFilter):
    """Фильтр для отчета по поручениям."""

    in_archive: bool | None = None
    date_task_start: datetime
    date_task_end: datetime
    task_statuses: list[TaskStatus]
    task_author_id: UUID | None = None
    task_executor_id: UUID | None = None
    task_responsible_executor_id: UUID | None = None
    flag_endless: bool | None = None


class DocumentOnStatusReportFilter(AbstractReportFilter):
    """Фильтр для отчета по статусам документов."""

    in_archive: bool | None = None
    doc_category_constants: list[DocCategory]
    status: list[DocumentStatus]
    document_author_ids: list[UUID] | None = None
    document_type_ids: list[UUID] | None = None


class DocumentOnControlReportFilter(AbstractReportFilter):
    """Фильтр для отчета по всем контрольным документам."""

    in_archive: bool | None = None
    date_control_start: datetime
    date_control_end: datetime
    doc_category_constants: list[DocCategory]
    control_user_ids: list[UUID] | None = None


class DocumentOnRegistrationReportFilter(AbstractReportFilter):
    """Фильтр для отчета по всем зарегистрированным документам."""

    in_archive: bool | None = None
    date_reg_start: datetime
    date_reg_end: datetime
    doc_category_constants: list[DocCategory]


class ReportConstructRequest(EdmsBaseDto):
    """Запрос на формирование отчета через конструктор."""

    fields: dict[ReportField, str] | None = None
    columns: list[ReportColumns] | None = None


class IdsDto[T](EdmsBaseDto):
    """Общая модель для передачи списка идентификаторов."""

    ids: Annotated[list[T], Field(min_length=1)]


from edms_ai_assistant.domain.document import AttachmentDto

ReportTaskDto.model_rebuild()
