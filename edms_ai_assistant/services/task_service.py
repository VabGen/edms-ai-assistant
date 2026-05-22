# edms_ai_assistant/services/task_service.py
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from edms_ai_assistant.domain.task_models import (
    CreateTaskRequest,
    CreateTaskRequestExecutor,
    TaskCreationResult,
    TaskType,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edms_ai_assistant.services.resolution_service import ResolutionService
    from edms_ai_assistant.clients.task_client import TaskClient
    from uuid import UUID

logger = logging.getLogger(__name__)


class TaskService:
    """Service for managing document tasks with Disambiguation."""

    def __init__(self, resolution_service: ResolutionService, task_client: TaskClient):
        self._resolution = resolution_service
        self._task_client = task_client

    async def create_task(
            self,
            token: str,
            document_id: str,
            task_text: str,
            executor_last_names: list[str],
            planed_date_end: datetime | None = None,
            responsible_last_name: str | None = None,
            task_type: TaskType = TaskType.GENERAL,
    ) -> TaskCreationResult:
        if not executor_last_names:
            return TaskCreationResult(success=False, status="error",
                                      error_message="Необходимо указать хотя бы одного исполнителя.")
        if not task_text or not task_text.strip():
            return TaskCreationResult(success=False, status="error",
                                      error_message="Текст поручения не может быть пустым.")

        # Резолвим всех исполнителей
        emp_ids, not_found, ambiguous = await self._resolution.resolve_employees(token, executor_last_names)

        if ambiguous:
            return TaskCreationResult(
                success=False, status="requires_disambiguation",
                ambiguous_matches=[am.__dict__ for am in ambiguous],
                not_found_employees=not_found,
            )

        if not emp_ids:
            return TaskCreationResult(
                success=False, status="error",
                error_message=f"Не найдены сотрудники: {', '.join(not_found)}",
                not_found_employees=not_found,
            )

        # Резолвим ответственного
        responsible_id: UUID | None = None
        if responsible_last_name:
            resp_ids, _, resp_ambiguous = await self._resolution.resolve_employees(token, [responsible_last_name])
            if resp_ids:
                responsible_id = resp_ids.pop()
            elif resp_ambiguous:
                return TaskCreationResult(
                    success=False, status="requires_disambiguation",
                    ambiguous_matches=[am.__dict__ for am in resp_ambiguous],
                )

        if not responsible_id:
            responsible_id = next(iter(emp_ids))

        # Формируем DTO
        executors = [
            CreateTaskRequestExecutor(employeeId=emp_id, responsible=(emp_id == responsible_id))
            for emp_id in emp_ids
        ]

        return await self._submit_task(
            token=token, document_id=document_id, task_text=task_text,
            executors=executors, planed_date_end=planed_date_end,
            task_type=task_type, not_found=not_found,
        )

    async def create_task_by_employee_ids(
            self, token: str, document_id: str, task_text: str,
            employee_ids: list[UUID], planed_date_end: datetime | None = None,
            responsible_employee_id: UUID | None = None, task_type: TaskType = TaskType.GENERAL,
    ) -> TaskCreationResult:
        if not employee_ids:
            return TaskCreationResult(success=False, status="error",
                                      error_message="Необходимо указать хотя бы одного исполнителя.")

        responsible_id = responsible_employee_id or employee_ids[0]
        executors = [
            CreateTaskRequestExecutor(employeeId=emp_id, responsible=(emp_id == responsible_id))
            for emp_id in employee_ids
        ]
        return await self._submit_task(token, document_id, task_text, executors, planed_date_end, task_type)

    async def resolve_bulk_executors(self, token: str, **kwargs) -> dict:
        """Прокси метод для ResolutionService, возвращает доменную модель."""
        result = await self._resolution.resolve_bulk(token, **kwargs)
        return {
            "employee_ids": result.employee_ids,
            "not_found": result.not_found,
            "resolved_summary": result.resolved_summary,
        }

    async def _submit_task(
            self, token: str, document_id: str, task_text: str,
            executors: list[CreateTaskRequestExecutor], planed_date_end: datetime | None,
            task_type: TaskType, not_found: list[str] | None = None,
    ) -> TaskCreationResult:
        if not planed_date_end:
            planed_date_end = datetime.now(UTC) + timedelta(days=5)
        elif planed_date_end.tzinfo is None:
            planed_date_end = planed_date_end.replace(tzinfo=UTC)

        formatted_text = task_text[0].upper() + task_text[1:] if len(task_text) > 1 else task_text.upper()

        task_request = CreateTaskRequest(
            taskText=formatted_text, planedDateEnd=planed_date_end,
            type=task_type, periodTask=False, endless=False, executors=executors,
        )

        try:
            success = await self._task_client.create_tasks_batch(token, document_id, [task_request])
            if success:
                return TaskCreationResult(success=True, status="success", created_count=1,
                                          not_found_employees=not_found or [])
            return TaskCreationResult(success=False, status="error", error_message="Не удалось создать поручение.")
        except Exception as exc:
            logger.error("Task creation failed: %s", exc, exc_info=True)
            return TaskCreationResult(success=False, status="error", error_message=f"Ошибка API: {exc}")
