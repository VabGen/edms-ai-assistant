# edms_ai_assistant/tools/task.py
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.services.task_service import TaskService
from edms_ai_assistant.models.task_models import TaskType

logger = logging.getLogger(__name__)


class TaskCreateInput(BaseModel):
    token: str = Field(..., description="Токен авторизации пользователя (JWT)")
    document_id: str = Field(..., description="UUID документа для создания поручения")

    task_text: str = Field(..., description="Текст поручения (обязательно)")
    executor_last_names: List[str] = Field(
        ...,
        min_length=1,
        description="Фамилии исполнителей (минимум 1). Например: ['Иванов', 'Петров']",
    )

    responsible_last_name: Optional[str] = Field(
        None,
        description=(
            "Фамилия ответственного исполнителя из списка executor_last_names. "
            "Если не указано, первый исполнитель будет назначен ответственным."
        ),
    )

    planed_date_end: Optional[str] = Field(
        None,
        description=(
            "Плановая дата окончания поручения в формате ISO 8601 (например: '2026-02-15T23:59:59Z'). "
            "Если не указано, будет установлен срок +7 дней от текущей даты."
        ),
    )

    task_type: Optional[TaskType] = Field(
        TaskType.GENERAL, description="Тип поручения (по умолчанию: GENERAL)"
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
        Создает новую задачу в системе на основании данных документа.
    """
    logger.info(
        f"[TASK-TOOL] Creating task for document {document_id}. "
        f"Executors: {executor_last_names}, Responsible: {responsible_last_name}"
    )

    # Validation
    if not executor_last_names:
        return {
            "status": "error",
            "message": "Необходимо указать хотя бы одного исполнителя (executor_last_names).",
        }

    if not task_text or not task_text.strip():
        return {
            "status": "error",
            "message": "Текст поручения (task_text) не может быть пустым.",
        }

    deadline = None
    if planed_date_end:
        try:
            deadline = datetime.fromisoformat(planed_date_end.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Неверный формат даты planed_date_end. Ожидается ISO 8601 (например: '2026-02-15T23:59:59Z'). Ошибка: {e}",
            }

    try:
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
                    "message": f"Поручение успешно создано. Исполнителей: {result.created_count}",
                }
            else:
                return {
                    "status": "error",
                    "message": result.error_message,
                    "not_found_employees": result.not_found_employees,
                }

    except Exception as e:
        logger.error(f"[TASK-TOOL] Error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Произошла ошибка при создании поручения: {str(e)}",
        }
