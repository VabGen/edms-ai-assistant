# edms_ai_assistant/models/task_models.py
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class TaskType(str, Enum):
    GENERAL = "GENERAL"
    PROJECT = "PROJECT"
    CONTROL = "CONTROL"


class CreateTaskRequestExecutor(BaseModel):

    employeeId: UUID = Field(..., description="UUID сотрудника-исполнителя")
    responsible: bool = Field(
        default=False, description="Является ли сотрудник ответственным за поручение"
    )

    model_config = ConfigDict(
        json_encoders={UUID: str},
        use_enum_values=True,
    )


class CreateTaskRequest(BaseModel):
    """Request model for creating a single task."""

    taskText: str = Field(..., description="Текст поручения")
    planedDateEnd: datetime = Field(
        ..., description="Плановая дата окончания (ISO 8601)"
    )
    type: TaskType = Field(default=TaskType.GENERAL, description="Тип поручения")
    periodTask: bool = Field(default=False, description="Периодическое поручение")
    endless: bool = Field(default=False, description="Бессрочное поручение")
    executors: List[CreateTaskRequestExecutor] = Field(
        ..., min_length=1, description="Список исполнителей (минимум 1)"
    )

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda dt: dt.isoformat() if dt.tzinfo else dt.isoformat() + "Z",
        },
        use_enum_values=True,
    )


class CreateTaskBatchRequest(BaseModel):
    """Batch request for creating multiple tasks."""

    tasks: List[CreateTaskRequest] = Field(..., min_length=1)

    model_config = ConfigDict(
        json_encoders={UUID: str, datetime: lambda dt: dt.isoformat()},
        use_enum_values=True,
    )


class TaskCreationResult(BaseModel):
    """Result of task creation operation."""

    success: bool
    created_count: int = 0
    not_found_employees: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
