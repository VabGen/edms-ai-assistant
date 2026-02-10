# edms_ai_assistant/services/task_service.py
import logging
from typing import List, Optional, Set, Tuple
from uuid import UUID
from datetime import datetime, timezone

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.task_client import TaskClient
from edms_ai_assistant.models.task_models import (
    CreateTaskRequest,
    CreateTaskRequestExecutor,
    TaskType,
    TaskCreationResult,
)

logger = logging.getLogger(__name__)


class TaskService:
    """
    Service for managing document tasks (поручения).

    Orchestrates:
    - Employee search by last name
    - Responsible assignment logic
    - Duplicate prevention
    - Task creation via API
    """

    def __init__(self):
        self.employee_client = EmployeeClient()
        self.task_client = TaskClient()

    async def __aenter__(self):
        await self.employee_client.__aenter__()
        await self.task_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.employee_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.task_client.__aexit__(exc_type, exc_val, exc_tb)

    async def collect_executors(
        self,
        token: str,
        last_names: List[str],
        responsible_last_name: Optional[str] = None,
    ) -> Tuple[List[CreateTaskRequestExecutor], List[str]]:
        found_employees = []
        not_found = []

        for last_name in last_names:
            employee = await self.employee_client.find_by_last_name_fts(
                token, last_name
            )

            if not employee:
                employees = await self.employee_client.search_employees(
                    token, {"lastName": last_name}
                )

                if not employees:
                    not_found.append(last_name)
                    logger.warning(f"Employee not found: {last_name}")
                    continue

                employee = employees[0]
                if len(employees) > 1:
                    logger.info(
                        f"Multiple employees found for '{last_name}', using first: "
                        f"{employee.get('firstName', '')} {employee.get('lastName', '')}"
                    )

            found_employees.append(employee)

        if not_found:
            return [], not_found

        responsible_employee = None
        if responsible_last_name:
            responsible_employee = await self.employee_client.find_by_last_name_fts(
                token, responsible_last_name
            )

            if not responsible_employee:
                logger.warning(
                    f"Responsible employee '{responsible_last_name}' not found, "
                    f"will use first executor as responsible"
                )

        executors = []
        seen_ids: Set[UUID] = set()

        if responsible_employee:
            resp_id = (
                UUID(responsible_employee["id"])
                if isinstance(responsible_employee["id"], str)
                else responsible_employee["id"]
            )
            executors.append(
                CreateTaskRequestExecutor(employeeId=resp_id, responsible=True)
            )
            seen_ids.add(resp_id)

        for idx, emp in enumerate(found_employees):
            emp_id = UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]

            if emp_id in seen_ids:
                continue

            is_responsible = not responsible_employee and idx == 0

            executors.append(
                CreateTaskRequestExecutor(employeeId=emp_id, responsible=is_responsible)
            )
            seen_ids.add(emp_id)

        return executors, []

    async def create_task(
        self,
        token: str,
        document_id: str,
        task_text: str,
        executor_last_names: List[str],
        planed_date_end: Optional[datetime] = None,
        responsible_last_name: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
    ) -> TaskCreationResult:

        if not executor_last_names:
            return TaskCreationResult(
                success=False,
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )

        if not task_text or not task_text.strip():
            return TaskCreationResult(
                success=False, error_message="Текст поручения не может быть пустым."
            )

        executors, not_found = await self.collect_executors(
            token, executor_last_names, responsible_last_name
        )

        if not_found:
            return TaskCreationResult(
                success=False,
                error_message=f"Не найдены сотрудники: {', '.join(not_found)}",
                not_found_employees=not_found,
            )

        if not executors:
            return TaskCreationResult(
                success=False,
                error_message="Не удалось найти ни одного исполнителя.",
            )

        if not planed_date_end:
            from datetime import timedelta

            planed_date_end = datetime.now(timezone.utc) + timedelta(days=7)
        elif planed_date_end.tzinfo is None:
            planed_date_end = planed_date_end.replace(tzinfo=timezone.utc)

        task_text_formatted = (
            task_text[0].upper() + task_text[1:]
            if len(task_text) > 1
            else task_text.upper()
        )

        task_request = CreateTaskRequest(
            taskText=task_text_formatted,
            planedDateEnd=planed_date_end,
            type=task_type,
            periodTask=False,
            endless=False,
            executors=executors,
        )

        success = await self.task_client.create_tasks_batch(
            token, document_id, [task_request]
        )

        if success:
            return TaskCreationResult(
                success=True,
                created_count=1,
            )
        else:
            return TaskCreationResult(
                success=False,
                error_message="Не удалось создать поручение. Проверьте права доступа или корректность данных.",
            )
