# edms_ai_assistant/domain/task_models.py
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from edms_ai_assistant.domain.base import EdmsBaseDto
from edms_ai_assistant.domain.enums import PeriodTaskInterval, TaskType

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID


class CreateTaskRequestExecutor(EdmsBaseDto):
    """Исполнитель поручения."""

    employee_id: Annotated[
        UUID | None, Field(description="UUID сотрудника-исполнителя")
    ] = None
    employeeId: Annotated[
        UUID | None, Field(description="Alias for employee_id", alias="employeeId")
    ] = None
    responsible: Annotated[
        bool | None, Field(description="Является ли сотрудник ответственным")
    ] = None
    stamp_text: str | None = None
    create_date: datetime | None = None
    executed_date: datetime | None = None


class CreateTaskRequest(EdmsBaseDto):
    """Request model for creating a single task."""

    task_text: Annotated[str, Field(description="Текст поручения", min_length=1)]
    planed_date_end: Annotated[
        datetime | None, Field(description="Плановая дата окончания")
    ] = None
    author_id: Annotated[UUID | None, Field(description="Автор поручения")] = None
    type: Annotated[TaskType, Field(description="Тип поручения")] = TaskType.GENERAL
    executors: Annotated[
        list[CreateTaskRequestExecutor] | None, Field(description="Исполнители")
    ] = None
    endless: Annotated[bool, Field(description="Бессрочное поручение")] = False
    period_task: Annotated[bool, Field(description="Периодическое поручение")] = False
    period: Annotated[
        PeriodTaskInterval | None, Field(description="Интервал создания поручения")
    ] = None
    create_task_for_each_executors: bool | None = False
    control_type_id: UUID | None = None
    control_employee_ids: list[UUID] | None = None
    control_plan_date_end: datetime | None = None


class UpdateTaskRequest(EdmsBaseDto):
    """Request model for updating a task."""

    id: Annotated[UUID, Field(description="ИД поручения")]
    task_text: Annotated[str, Field(description="Текст поручения", min_length=1)]
    planed_date_end: Annotated[
        datetime | None, Field(description="Планируемая дата исполнения")
    ] = None
    endless: Annotated[bool, Field(description="Бессрочное поручение")] = False
    period_task: Annotated[bool, Field(description="Поручение периодическое")] = False
    period: Annotated[
        PeriodTaskInterval | None, Field(description="Интервал создания")
    ] = None


class ExecuteTaskRequest(EdmsBaseDto):
    """Request model for executing a task."""

    stamp_text: Annotated[str | None, Field(description="Текст поручения")] = None


class TaskRevisionRequest(EdmsBaseDto):
    """Request model for task revision."""

    text: str | None = None
    ids: Annotated[list[UUID], Field(min_length=1)]


class ChangeResponsibleStatus(EdmsBaseDto):
    """Request model for changing responsible status."""

    responsible: bool


class CreateTaskBatchRequest(EdmsBaseDto):
    """Batch request for creating multiple tasks."""

    tasks: list[CreateTaskRequest] = Field(..., min_length=1)


class TaskCreationResult(EdmsBaseDto):
    """
    Result of task creation operation with disambiguation support.
    """

    success: bool
    status: str = "success"  # "success" | "requires_disambiguation" | "error"
    created_count: int = 0
    not_found_employees: list[str] = Field(default_factory=list)
    error_message: str | None = None
    ambiguous_matches: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of ambiguous employee matches requiring user selection",
    )
