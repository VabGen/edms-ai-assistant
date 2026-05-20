# edms_ai_assistant/services/resolution_service.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from edms_ai_assistant.clients.department_client import DepartmentClient
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.group_client import GroupClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import EmployeeDto
from edms_ai_assistant.services.search_utils import (
    DEFAULT_PAGEABLE,
    build_employee_filter,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolutionResult:
    """Результат массового резолвинга исполнителей."""
    employee_ids: set[UUID] = field(default_factory=set)
    not_found: list[str] = field(default_factory=list)
    resolved_summary: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AmbiguousMatch:
    """Структура для неоднозначного совпадения при поиске сотрудника."""
    search_query: str
    matches: list[dict]


class ResolutionService:
    """Единая точка входа для резолвинга сотрудников, отделов и групп."""

    def __init__(
            self,
            employee_client: EmployeeClient,
            department_client: DepartmentClient,
            group_client: GroupClient,
    ):
        self._employee_client = employee_client
        self._department_client = department_client
        self._group_client = group_client

    async def resolve_employees(
            self,
            token: str,
            last_names: list[str],
    ) -> tuple[set[UUID], list[str], list[AmbiguousMatch]]:
        """Резолвит сотрудников по фамилиям или ФИО."""
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        ambiguous: list[AmbiguousMatch] = []

        for name_query in last_names:
            search_filter = build_employee_filter(name_query=name_query)
            try:
                employees = await self._employee_client.search_employees_post(
                    token=token, employee_filter=search_filter, pageable=DEFAULT_PAGEABLE
                )
            except Exception:
                logger.warning("Employee search failed for '%s'", name_query, exc_info=True)
                not_found.append(name_query)
                continue

            if not employees:
                not_found.append(name_query)
            elif len(employees) == 1:
                if employees[0].id:
                    found_ids.add(employees[0].id)
            else:
                ambiguous.append(
                    AmbiguousMatch(
                        search_query=name_query,
                        matches=[self._format_employee_match(emp) for emp in employees],
                    )
                )
        return found_ids, not_found, ambiguous

    async def resolve_departments(
            self, token: str, department_names: list[str]
    ) -> tuple[set[UUID], list[str], int]:
        """Резолвит сотрудников по названиям отделов."""
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        total = 0

        for dept_name in department_names:
            ns = dept_name.strip()
            if not ns:
                continue
            try:
                dept = await self._department_client.find_by_name(token, ns)
                if not dept or not dept.id:
                    not_found.append(f"Департамент: {ns}")
                    continue

                employees = await self._department_client.get_employees_by_department_id(token, dept.id)
                for emp in employees:
                    if emp.id:
                        found_ids.add(emp.id)
                total += len(employees)
            except EdmsNotFoundError:
                not_found.append(f"Департамент: {ns}")
            except Exception:
                logger.warning("Failed to resolve dept '%s'", ns, exc_info=True)
                not_found.append(f"Департамент: {ns}")

        return found_ids, not_found, total

    async def resolve_groups(
            self, token: str, group_names: list[str], *, personal: bool = False
    ) -> tuple[set[UUID], list[str], int]:
        """Резолвит группы (обычные или личные) по названиям."""
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
                    group = await self._group_client.find_personal_by_name(token, ns)
                else:
                    group = await self._group_client.find_by_name(token, ns)

                if not group:
                    not_found.append(f"{label}: {ns}")
                    continue

                group_id = group.get("id")
                if group_id:
                    group_ids.append(UUID(str(group_id)))
            except EdmsNotFoundError:
                not_found.append(f"{label}: {ns}")
            except Exception:
                logger.warning("Failed to resolve %s '%s'", label.lower(), ns, exc_info=True)
                not_found.append(f"{label}: {ns}")

        total = 0
        if group_ids:
            try:
                if personal:
                    employees = await self._group_client.get_employees_by_personal_group_ids(token, group_ids)
                else:
                    employees = await self._group_client.get_employees_by_group_ids(token, group_ids)
                for emp in employees:
                    if emp.id:
                        found_ids.add(emp.id)
                total = len(employees)
            except Exception:
                logger.warning("Failed to get employees for %ss", label.lower(), exc_info=True)

        return found_ids, not_found, total

    async def resolve_subordinates(self, token: str) -> tuple[set[UUID], int]:
        """Резолвит подчинённых текущего пользователя."""
        try:
            current_user = await self._employee_client.get_current_user(token)
            if not current_user or not current_user.employee:
                return set(), 0

            user_id = current_user.employee.id

            dept_id = current_user.employee.department.id if current_user.employee.department else None
            if not dept_id:
                return set(), 0

            search_filter = {"departmentId": [str(dept_id)], "includes": ["POST", "DEPARTMENT"]}
            employees = await self._employee_client.search_employees_post(
                token=token, employee_filter=search_filter, pageable={"page": 0, "size": 100, "sort": "lastName,ASC"}
            )

            found_ids = {UUID(str(emp.id)) for emp in employees if emp.id and str(emp.id) != str(user_id)}
            return found_ids, len(found_ids)
        except Exception:
            logger.warning("Failed to resolve subordinates", exc_info=True)
            return set(), 0

    async def resolve_bulk(
            self,
            token: str,
            department_names: list[str] | None = None,
            group_names: list[str] | None = None,
            personal_group_names: list[str] | None = None,
            include_subordinates: bool = False,
    ) -> ResolutionResult:
        """Единый метод массового резолвинга по всем критериям."""
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        summary: list[str] = []

        if department_names:
            ids, nf, cnt = await self.resolve_departments(token, department_names)
            found_ids.update(ids)
            not_found.extend(nf)
            if cnt: summary.append(f"подразделения: {cnt} сотр.")

        if group_names:
            ids, nf, cnt = await self.resolve_groups(token, group_names, personal=False)
            found_ids.update(ids)
            not_found.extend(nf)
            if cnt: summary.append(f"группы: {cnt} сотр.")

        if personal_group_names:
            ids, nf, cnt = await self.resolve_groups(token, personal_group_names, personal=True)
            found_ids.update(ids)
            not_found.extend(nf)
            if cnt: summary.append(f"личные группы: {cnt} сотр.")

        if include_subordinates:
            ids, cnt = await self.resolve_subordinates(token)
            found_ids.update(ids)
            if cnt: summary.append(f"подчинённые: {cnt} сотр.")

        return ResolutionResult(
            employee_ids=found_ids, not_found=not_found, resolved_summary=summary
        )

    @staticmethod
    def _format_employee_match(employee: EmployeeDto) -> dict:
        """Форматирует данные сотрудника для disambiguation response."""
        post_name = employee.post.post_name if employee.post and employee.post.post_name else "Не указана"
        dept_name = employee.department.name if employee.department and employee.department.name else "Не указан"

        return {
            "id": str(employee.id),
            "full_name": f"{employee.last_name or ''} {employee.first_name or ''} {employee.middle_name or ''}".strip(),
            "post": post_name,
            "department": dept_name,
        }
