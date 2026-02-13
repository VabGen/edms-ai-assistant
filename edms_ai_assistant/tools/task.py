"""
EDMS AI Assistant - Task Creation Tool.

Инструмент для создания поручений к документам.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.models.task_models import TaskType
from edms_ai_assistant.services.task_service import TaskService

logger = logging.getLogger(__name__)


class TaskCreateInput(BaseModel):
    """Валидированная схема входных данных для создания поручения."""

    token: str = Field(..., description="JWT токен авторизации пользователя")
    document_id: str = Field(
        ...,
        description="UUID документа для создания поручения",
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )
    task_text: str = Field(
        ..., description="Текст поручения (обязательно)", min_length=3, max_length=2000
    )
    executor_last_names: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Фамилии исполнителей (минимум 1). Например: ['Иванов', 'Петров']",
    )
    responsible_last_name: Optional[str] = Field(
        None,
        max_length=100,
        description=(
            "Фамилия ответственного исполнителя из списка executor_last_names. "
            "Если не указано, первый исполнитель будет назначен ответственным."
        ),
    )
    planed_date_end: Optional[str] = Field(
        None,
        description=(
            "Плановая дата окончания в ISO 8601 (например: '2026-02-15T23:59:59Z'). "
            "Если не указано, будет установлен срок +7 дней от текущей даты."
        ),
    )
    task_type: Optional[TaskType] = Field(
        TaskType.GENERAL, description="Тип поручения (по умолчанию: GENERAL)"
    )

    @field_validator("task_text")
    @classmethod
    def validate_task_text(cls, v: str) -> str:
        """Валидация и нормализация текста поручения."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Текст поручения не может быть пустым")
        return cleaned

    @field_validator("executor_last_names")
    @classmethod
    def validate_executor_names(cls, v: List[str]) -> List[str]:
        """Валидация и очистка фамилий исполнителей."""
        cleaned = [name.strip() for name in v if name and name.strip()]
        if not cleaned:
            raise ValueError("Необходимо указать хотя бы одного исполнителя")
        return cleaned

    @field_validator("responsible_last_name")
    @classmethod
    def validate_responsible_name(cls, v: Optional[str]) -> Optional[str]:
        """Валидация фамилии ответственного."""
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None

    @field_validator("planed_date_end")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Валидация формата даты."""
        if v is None:
            return None
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError as e:
            raise ValueError(
                f"Неверный формат даты. Ожидается ISO 8601 "
                f"(например: '2026-02-15T23:59:59Z'): {e}"
            )


@tool("task_create_tool", args_schema=TaskCreateInput)
async def task_create_tool(
    token: str,
    document_id: str,
    task_text: str,
    executor_last_names: List[str],
    responsible_last_name: Optional[str] = None,
    planed_date_end: Optional[str] = None,
    task_type: Optional[TaskType] = TaskType.GENERAL,
) -> Dict[str, Any]:
    """
    Создает поручение к документу с назначением исполнителей и сроков.

    Бизнес-правила:
    - Минимум 1 исполнитель обязателен
    - Если ответственный не указан → первый исполнитель становится ответственным
    - Если срок не указан → +7 дней от текущей даты
    - Текст поручения капитализируется автоматически

    Args:
        token: JWT токен авторизации
        document_id: UUID документа
        task_text: Текст поручения (3-2000 символов)
        executor_last_names: Список фамилий исполнителей (1-20)
        responsible_last_name: Фамилия ответственного (опционально)
        planed_date_end: Дата завершения в ISO 8601 (опционально)
        task_type: Тип поручения (по умолчанию GENERAL)

    Returns:
        Dict с ключами:
        - status: "success" | "error"
        - message: Информационное сообщение
        - created_count: int (количество созданных поручений)
        - not_found_employees: List[str] (не найденные сотрудники)

    Examples:
         # Простое создание поручения
         result = await task_create_tool(
             token="jwt_token",
             document_id="doc_uuid",
             task_text="Подготовить отчет",
             executor_last_names=["Иванов"]
         )
         # {"status": "success", "message": "Поручение успешно создано...", ...}

         # С указанием ответственного и срока
         result = await task_create_tool(
             token="jwt_token",
             document_id="doc_uuid",
             task_text="Согласовать бюджет",
             executor_last_names=["Иванов", "Петров"],
             responsible_last_name="Петров",
             planed_date_end="2026-03-01T23:59:59Z"
         )
    """
    logger.info(
        "Creating task",
        extra={
            "document_id": document_id,
            "executors_count": len(executor_last_names),
            "has_responsible": bool(responsible_last_name),
            "has_deadline": bool(planed_date_end),
        },
    )

    try:
        deadline = _parse_deadline(planed_date_end) if planed_date_end else None

        async with TaskService() as service:
            result = await service.create_task(
                token=token,
                document_id=document_id,
                task_text=task_text,
                executor_last_names=executor_last_names,
                planed_date_end=deadline,
                responsible_last_name=responsible_last_name,
                task_type=task_type or TaskType.GENERAL,
            )

            if result.success:
                return {
                    "status": "success",
                    "message": (
                        f"✅ Поручение успешно создано. "
                        f"Исполнителей: {result.created_count}"
                    ),
                    "created_count": result.created_count,
                }

            return {
                "status": "error",
                "message": result.error_message or "❌ Не удалось создать поручение",
                "not_found_employees": result.not_found_employees or [],
            }

    except ValueError as e:
        logger.warning(f"Validation error: {e}", extra={"document_id": document_id})
        return {
            "status": "error",
            "message": f"❌ Ошибка валидации: {str(e)}",
        }
    except Exception as e:
        logger.error(
            f"Task creation failed: {e}",
            exc_info=True,
            extra={"document_id": document_id},
        )
        return {
            "status": "error",
            "message": f"❌ Произошла ошибка при создании поручения: {str(e)}",
        }


def _parse_deadline(date_str: str) -> datetime:
    """
    Парсит ISO 8601 дату с обработкой timezone.

    Args:
        date_str: Строка даты в формате ISO 8601

    Returns:
        datetime объект с UTC timezone

    Raises:
        ValueError: Если формат даты некорректен
    """
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Неверный формат даты planed_date_end. "
            f"Ожидается ISO 8601 (например: '2026-02-15T23:59:59Z'). "
            f"Ошибка: {e}"
        )
