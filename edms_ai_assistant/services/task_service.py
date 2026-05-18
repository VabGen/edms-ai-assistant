# edms_ai_assistant/services/task_service.py
"""
EDMS AI Assistant — Task Service with Disambiguation Support.

✅ v5: Добавлена поддержка личных групп (personal_group_names).
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from edms_ai_assistant.clients.department_client import DepartmentClient
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.group_client import GroupClient
from edms_ai_assistant.clients.task_client import TaskClient
from edms_ai_assistant.domain.task_models import (
    CreateTaskRequest,
    CreateTaskRequestExecutor,
    TaskCreationResult,
    TaskType,
)
from edms_ai_assistant.services.search_utils import (
    DEFAULT_PAGEABLE,
    build_employee_filter,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BulkResolutionResult:
    """Результат массового резолвинга исполнителей."""

    employee_ids: set[UUID] = field(default_factory=set)
    not_found: list[str] = field(default_factory=list)
    resolved_summary: list[str] = field(default_factory=list)


class TaskService:
    """Service for managing document tasks with Disambiguation."""

    def __init__(self) -> None:
        self.employee_client = EmployeeClient()
        self.task_client = TaskClient()
        self.department_client = DepartmentClient()
        self.group_client = GroupClient()

    async def __aenter__(self) -> "TaskService":
        await self.employee_client.__aenter__()
        await self.task_client.__aenter__()
        await self.department_client.__aenter__()
        await self.group_client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.employee_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.task_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.department_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.group_client.__aexit__(exc_type, exc_val, exc_tb)

    # ──────────────────────────────────────────────────────────────────────
    # Bulk Resolution
    # ──────────────────────────────────────────────────────────────────────

    async def resolve_bulk_executors(
        self,
        token: str,
        department_names: list[str] | None = None,
        group_names: list[str] | None = None,
        personal_group_names: list[str] | None = None,
        include_subordinates: bool = False,
    ) -> BulkResolutionResult:
        """Резолвит массовых исполнителей."""
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        summary: list[str] = []

        # ── Подразделения ───────────────────────────────────────────────
        if department_names:
            dept_ids, dept_not_found, dept_count = await self._resolve_departments(
                token, department_names
            )
            found_ids.update(dept_ids)
            not_found.extend(dept_not_found)
            if dept_count:
                summary.append(f"подразделения: {dept_count} сотрудников")

        # ── Группы ──────────────────────────────────────────────────────
        if group_names:
            group_ids, group_not_found, group_count = await self._resolve_groups(
                token, group_names, personal=False
            )
            found_ids.update(group_ids)
            not_found.extend(group_not_found)
            if group_count:
                summary.append(f"группы: {group_count} сотрудников")

        # ── Личные группы ───────────────────────────────────────────────
        if personal_group_names:
            pg_ids, pg_not_found, pg_count = await self._resolve_groups(
                token, personal_group_names, personal=True
            )
            found_ids.update(pg_ids)
            not_found.extend(pg_not_found)
            if pg_count:
                summary.append(f"личные группы: {pg_count} сотрудников")

        # ── Подчинённые ─────────────────────────────────────────────────
        if include_subordinates:
            sub_ids, sub_count = await self._resolve_subordinates(token)
            found_ids.update(sub_ids)
            if sub_count:
                summary.append(f"подчинённые: {sub_count} сотрудников")

        logger.info(
            "[TASK-SERVICE] Bulk resolution: %d employees (%s)",
            len(found_ids),
            "; ".join(summary) if summary else "none",
        )

        return BulkResolutionResult(
            employee_ids=found_ids,
            not_found=not_found,
            resolved_summary=summary,
        )

    async def _resolve_departments(
        self,
        token: str,
        department_names: list[str],
    ) -> tuple[set[UUID], list[str], int]:
        """Резолвит подразделения по названиям → employee UUIDs."""
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        total = 0

        for dept_name in department_names:
            ns = dept_name.strip()
            if not ns:
                continue

            try:
                dept = await self.department_client.find_by_name(token, ns)
                if not dept or not dept.get("id"):
                    not_found.append(f"Подразделение: {ns}")
                    continue

                dept_id = (
                    UUID(dept["id"]) if isinstance(dept["id"], str) else dept["id"]
                )
                employees = await self.department_client.get_employees_by_department_id(
                    token, dept_id
                )

                for emp in employees:
                    emp_id = (
                        UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
                    )
                    found_ids.add(emp_id)

                total += len(employees)
            except Exception:
                logger.warning(
                    "[TASK-SERVICE] Failed to resolve dept '%s'", ns, exc_info=True
                )
                not_found.append(f"Подразделение: {ns}")

        return found_ids, not_found, total

    async def _resolve_groups(
        self,
        token: str,
        group_names: list[str],
        *,
        personal: bool = False,
    ) -> tuple[set[UUID], list[str], int]:
        """Резолвит группы (обычные или личные) по названиям → employee UUIDs.

        Args:
            token: JWT токен
            group_names: Названия групп
            personal: True — искать в личных группах пользователя,
                      False — искать в общих группах

        Returns:
            (employee_ids, not_found_names, total_employee_count)
        """
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        group_ids: list[UUID] = []

        label = "Личная группа" if personal else "Группа"

        for group_name in group_names:
            ns = group_name.strip()
            if not ns:
                continue

            try:
                if personal:
                    group = await self.group_client.find_personal_by_name(token, ns)
                else:
                    group = await self.group_client.find_by_name(token, ns)

                if not group or not group.get("id"):
                    not_found.append(f"{label}: {ns}")
                    logger.warning("[TASK-SERVICE] %s not found: %s", label, ns)
                    continue

                group_id = (
                    UUID(group["id"]) if isinstance(group["id"], str) else group["id"]
                )
                group_ids.append(group_id)
            except Exception:
                logger.warning(
                    "[TASK-SERVICE] Failed to resolve %s '%s'",
                    label.lower(),
                    ns,
                    exc_info=True,
                )
                not_found.append(f"{label}: {ns}")

        total = 0
        if group_ids:
            try:
                if personal:
                    employees = (
                        await self.group_client.get_employees_by_personal_group_ids(
                            token, group_ids
                        )
                    )
                else:
                    employees = await self.group_client.get_employees_by_group_ids(
                        token, group_ids
                    )

                for emp in employees:
                    emp_id = (
                        UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
                    )
                    found_ids.add(emp_id)
                total = len(employees)
            except Exception:
                logger.warning(
                    "[TASK-SERVICE] Failed to get employees for %ss",
                    label.lower(),
                    exc_info=True,
                )

        return found_ids, not_found, total

    async def _resolve_subordinates(
        self,
        token: str,
    ) -> tuple[set[UUID], int]:
        """Резолвит подчинённых текущего пользователя."""
        try:
            current_user = await self.employee_client.get_current_user(token)
            if not current_user:
                return set(), 0

            user_id = current_user.get("id")
            dept_id = current_user.get("departmentId")

            if not dept_id:
                return set(), 0

            search_filter = {
                "departmentId": [str(dept_id)],
                "includes": ["POST", "DEPARTMENT"],
            }

            employees = await self.employee_client.search_employees_post(
                token=token,
                employee_filter=search_filter,
                pageable={"page": 0, "size": 100, "sort": "lastName,ASC"},
            )

            found_ids: set[UUID] = set()
            for emp in employees:
                emp_id_str = str(emp.get("id", ""))
                if emp_id_str and emp_id_str != str(user_id):
                    found_ids.add(UUID(emp_id_str))

            return found_ids, len(found_ids)
        except Exception:
            logger.warning(
                "[TASK-SERVICE] Failed to resolve subordinates", exc_info=True
            )
            return set(), 0

    # ──────────────────────────────────────────────────────────────────────
    # Individual Resolution
    # ──────────────────────────────────────────────────────────────────────

    async def collect_executors(
        self,
        token: str,
        last_names: list[str],
        responsible_last_name: str | None = None,
    ) -> tuple[
        list[CreateTaskRequestExecutor] | None,
        list[str],
        list[dict[str, Any]],
    ]:
        """Resolves executor employees by last name or full name."""
        found_employees: list[dict[str, Any]] = []
        not_found: list[str] = []
        ambiguous_results: list[dict[str, Any]] = []

        for name_query in last_names:
            search_filter = build_employee_filter(name_query=name_query)

            employees = await self.employee_client.search_employees_post(
                token=token,
                employee_filter=search_filter,
                pageable=DEFAULT_PAGEABLE,
            )

            if not employees:
                not_found.append(name_query)
                continue

            if len(employees) > 1:
                ambiguous_results.append(
                    {
                        "search_query": name_query,
                        "matches": [self._format_employee_match(e) for e in employees],
                    }
                )
                continue

            found_employees.append(employees[0])

        if ambiguous_results:
            return None, not_found, ambiguous_results

        if not found_employees:
            return None, not_found, []

        responsible_employee: dict[str, Any] | None = None
        if responsible_last_name:
            resp_filter = build_employee_filter(name_query=responsible_last_name)
            resp_results = await self.employee_client.search_employees_post(
                token=token,
                employee_filter=resp_filter,
                pageable=DEFAULT_PAGEABLE,
            )
            if resp_results and len(resp_results) == 1:
                responsible_employee = resp_results[0]
            else:
                responsible_employee = await self.employee_client.find_by_last_name_fts(
                    token, responsible_last_name
                )

        executors: list[CreateTaskRequestExecutor] = []
        seen_ids: set[UUID] = set()

        if responsible_employee:
            resp_id = _to_uuid(responsible_employee["id"])
            executors.append(
                CreateTaskRequestExecutor(employeeId=resp_id, responsible=True)
            )
            seen_ids.add(resp_id)

        for idx, emp in enumerate(found_employees):
            emp_id = _to_uuid(emp["id"])
            if emp_id in seen_ids:
                continue
            is_responsible = not responsible_employee and idx == 0
            executors.append(
                CreateTaskRequestExecutor(employeeId=emp_id, responsible=is_responsible)
            )
            seen_ids.add(emp_id)

        return executors, not_found, []

    # ──────────────────────────────────────────────────────────────────────
    # Task Creation
    # ──────────────────────────────────────────────────────────────────────

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
        logger.info("[TASK-SERVICE] Creating task. Executors: %s", executor_last_names)

        if not executor_last_names:
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )

        if not task_text or not task_text.strip():
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Текст поручения не может быть пустым.",
            )

        executors, not_found, ambiguous_results = await self.collect_executors(
            token, executor_last_names, responsible_last_name
        )

        if ambiguous_results:
            return TaskCreationResult(
                success=False,
                status="requires_disambiguation",
                ambiguous_matches=ambiguous_results,
                not_found_employees=not_found,
            )

        if not executors:
            return TaskCreationResult(
                success=False,
                status="error",
                error_message=f"Не найдены сотрудники: {', '.join(not_found)}",
                not_found_employees=not_found,
            )

        return await self._submit_task(
            token=token,
            document_id=document_id,
            task_text=task_text,
            executors=executors,
            planed_date_end=planed_date_end,
            task_type=task_type,
            not_found=not_found,
        )

    async def create_task_by_employee_ids(
        self,
        token: str,
        document_id: str,
        task_text: str,
        employee_ids: list[UUID],
        planed_date_end: datetime | None = None,
        responsible_employee_id: UUID | None = None,
        task_type: TaskType = TaskType.GENERAL,
    ) -> TaskCreationResult:
        if not employee_ids:
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )

        responsible_id = responsible_employee_id or employee_ids[0]
        executors: list[CreateTaskRequestExecutor] = []
        seen_ids: set[UUID] = set()
        for emp_id in employee_ids:
            if emp_id in seen_ids:
                continue
            executors.append(
                CreateTaskRequestExecutor(
                    employeeId=emp_id,
                    responsible=(emp_id == responsible_id),
                )
            )
            seen_ids.add(emp_id)

        return await self._submit_task(
            token=token,
            document_id=document_id,
            task_text=task_text,
            executors=executors,
            planed_date_end=planed_date_end,
            task_type=task_type,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────

    async def _submit_task(
        self,
        token: str,
        document_id: str,
        task_text: str,
        executors: list[CreateTaskRequestExecutor],
        planed_date_end: datetime | None,
        task_type: TaskType,
        not_found: list[str] | None = None,
    ) -> TaskCreationResult:
        if not planed_date_end:
            planed_date_end = datetime.now(UTC) + timedelta(days=5)
        elif planed_date_end.tzinfo is None:
            planed_date_end = planed_date_end.replace(tzinfo=UTC)

        formatted_text = (
            task_text[0].upper() + task_text[1:]
            if len(task_text) > 1
            else task_text.upper()
        )

        task_request = CreateTaskRequest(
            taskText=formatted_text,
            planedDateEnd=planed_date_end,
            type=task_type,
            periodTask=False,
            endless=False,
            executors=executors,
        )

        try:
            success = await self.task_client.create_tasks_batch(
                token, document_id, [task_request]
            )
            if success:
                logger.info(
                    "[TASK-SERVICE] Task created. Executors: %d", len(executors)
                )
                return TaskCreationResult(
                    success=True,
                    status="success",
                    created_count=1,
                    not_found_employees=not_found or [],
                )
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Не удалось создать поручение.",
            )
        except Exception as exc:
            logger.error("[TASK-SERVICE] Task creation failed: %s", exc, exc_info=True)
            return TaskCreationResult(
                success=False,
                status="error",
                error_message=f"Ошибка: {exc}",
            )

    @staticmethod
    def _format_employee_match(employee: dict[str, Any]) -> dict[str, Any]:
        last_name = employee.get("lastName", "")
        first_name = employee.get("firstName", "")
        middle_name = employee.get("middleName") or ""
        full_name = f"{last_name} {first_name} {middle_name}".strip()
        post_data = employee.get("post") or {}
        post_name = (
            post_data.get("postName", "Не указана")
            if isinstance(post_data, dict)
            else "Не указана"
        )
        dept_data = employee.get("department") or {}
        dept_name = (
            dept_data.get("name", "Не указан")
            if isinstance(dept_data, dict)
            else "Не указан"
        )
        return {
            "id": str(employee["id"]),
            "full_name": full_name,
            "post": post_name,
            "department": dept_name,
        }


def _to_uuid(value: Any) -> UUID:
    return UUID(value) if isinstance(value, str) else value
