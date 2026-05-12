"""
EDMS AI Assistant - Introduction Service.
"""

import logging
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.clients.department_client import DepartmentClient
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.group_client import GroupClient
from edms_ai_assistant.services.search_utils import (
    DEFAULT_PAGEABLE,
    build_employee_filter,
)

logger = logging.getLogger(__name__)


class PostIntroductionRequest(BaseModel):
    """Request DTO для создания ознакомления через API."""

    executorListIds: list[UUID] = Field(
        ..., description="UUID сотрудников для добавления в список ознакомления"
    )
    comment: str = Field(
        default="",
        description="Комментарий к ознакомлению (пустая строка если не указан)",
    )

    model_config = ConfigDict(
        json_encoders={UUID: str},
        use_enum_values=True,
    )


@dataclass(frozen=True)
class IntroductionResult:
    """Immutable результат создания ознакомления."""

    success: bool
    added_count: int = 0
    error_message: str | None = None


@dataclass(frozen=True)
class EmployeeResolutionResult:
    """Результат резолвинга сотрудников с обработкой неоднозначностей."""

    employee_ids: set[UUID] = field(default_factory=set)
    not_found: list[str] = field(default_factory=list)
    ambiguous: list[dict] = field(default_factory=list)


@dataclass
class AmbiguousMatch:
    """Структура для неоднозначного совпадения при поиске сотрудника."""

    search_query: str
    matches: list[dict]


class IntroductionService:
    """Сервисный слой для управления списками ознакомления."""

    def __init__(self):
        self.employee_client = EmployeeClient()
        self.department_client = DepartmentClient()
        self.group_client = GroupClient()

    async def __aenter__(self):
        await self.employee_client.__aenter__()
        await self.department_client.__aenter__()
        await self.group_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.employee_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.department_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.group_client.__aexit__(exc_type, exc_val, exc_tb)

    async def resolve_employees(
        self,
        token: str,
        last_names: list[str],
        department_names: list[str],
        group_names: list[str],
        personal_group_names: list[str] | None = None,
        include_subordinates: bool = False,
    ) -> EmployeeResolutionResult:
        """Резолвит сотрудников по множественным критериям поиска."""
        found_ids: set[UUID] = set()
        not_found: list[str] = []
        ambiguous_results: list[dict] = []

        if last_names:
            for last_name in last_names:
                result = await self._resolve_by_last_name(token, last_name)
                if result["status"] == "not_found":
                    not_found.append(f"Сотрудник: {last_name}")
                elif result["status"] == "found":
                    found_ids.add(result["employee_id"])
                elif result["status"] == "ambiguous":
                    ambiguous_results.append(result["ambiguous_data"])

        if department_names:
            dept_ids, dept_not_found = await self._resolve_departments(
                token, department_names
            )
            found_ids.update(dept_ids)
            not_found.extend(dept_not_found)

        if group_names:
            group_ids, group_not_found = await self._resolve_groups(
                token, group_names, personal=False
            )
            found_ids.update(group_ids)
            not_found.extend(group_not_found)

        if personal_group_names:
            pg_ids, pg_not_found = await self._resolve_groups(
                token, personal_group_names, personal=True
            )
            found_ids.update(pg_ids)
            not_found.extend(pg_not_found)

        if include_subordinates:
            sub_ids, sub_count = await self._resolve_subordinates(token)
            found_ids.update(sub_ids)

        return EmployeeResolutionResult(
            employee_ids=found_ids,
            not_found=not_found,
            ambiguous=ambiguous_results,
        )

    async def _resolve_by_last_name(self, token: str, name_query: str) -> dict:
        """Резолвит сотрудника по фамилии или полному ФИО."""
        search_filter = build_employee_filter(name_query=name_query)

        employees = await self.employee_client.search_employees_post(
            token=token,
            employee_filter=search_filter,
            pageable=DEFAULT_PAGEABLE,
        )

        if not employees:
            logger.warning("Employee not found: %s", name_query)
            return {"status": "not_found"}

        if len(employees) == 1:
            emp = employees[0]
            emp_id = UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
            logger.debug("Found single employee: %s -> %s", name_query, emp_id)
            return {"status": "found", "employee_id": emp_id}

        logger.info(
            "Found %d employees for '%s' - disambiguation required",
            len(employees),
            name_query,
        )

        return {
            "status": "ambiguous",
            "ambiguous_data": {
                "search_query": name_query,
                "matches": [self._format_employee_match(emp) for emp in employees],
            },
        }

    async def _resolve_departments(
        self, token: str, department_names: list[str]
    ) -> tuple[set[UUID], list[str]]:
        """Резолвит сотрудников по названиям отделов."""
        found_ids: set[UUID] = set()
        not_found: list[str] = []

        for dept_name in department_names:
            ns = dept_name.strip()
            if not ns:
                continue
            try:
                dept = await self.department_client.find_by_name(token, ns)
                if not dept or not dept.get("id"):
                    not_found.append(f"Департамент: {ns}")
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

                logger.info(
                    "Resolved department '%s': %d employees", ns, len(employees)
                )
            except Exception:
                logger.warning("Failed to resolve department '%s'", ns, exc_info=True)
                not_found.append(f"Департамент: {ns}")

        return found_ids, not_found

    async def _resolve_groups(
        self,
        token: str,
        group_names: list[str],
        *,
        personal: bool = False,
    ) -> tuple[set[UUID], list[str]]:
        """Резолвит группы (обычные или личные) по названиям.

        Args:
            token: JWT токен
            group_names: Названия групп
            personal: True — личные группы, False — обычные группы
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
                    continue

                group_id = (
                    UUID(group["id"]) if isinstance(group["id"], str) else group["id"]
                )
                group_ids.append(group_id)
            except Exception:
                logger.warning(
                    "Failed to resolve %s '%s'", label.lower(), ns, exc_info=True
                )
                not_found.append(f"{label}: {ns}")

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

                logger.info(
                    "Resolved %d %ss: %d employees",
                    len(group_ids),
                    label.lower(),
                    len(employees),
                )
            except Exception:
                logger.warning(
                    "Failed to get employees for %ss",
                    label.lower(),
                    exc_info=True,
                )

        return found_ids, not_found

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
                logger.warning("Current user has no departmentId")
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
            logger.warning("Failed to resolve subordinates", exc_info=True)
            return set(), 0

    async def create_introduction(
        self,
        token: str,
        document_id: str,
        employee_ids: list[UUID],
        comment: str | None = None,
    ) -> IntroductionResult:
        """Создает список ознакомления через API EDMS."""
        if not employee_ids:
            logger.warning("Attempted to create introduction with empty employee list")
            return IntroductionResult(
                success=False,
                added_count=0,
                error_message="Не указаны сотрудники для добавления",
            )

        normalized_comment = self._normalize_comment(comment)

        request = PostIntroductionRequest(
            executorListIds=employee_ids, comment=normalized_comment
        )

        try:
            async with EdmsBaseClient() as client:
                endpoint = f"api/document/{document_id}/introduction"
                payload = request.model_dump(mode="json")

                logger.debug(
                    "Creating introduction",
                    extra={
                        "document_id": document_id,
                        "employee_count": len(employee_ids),
                        "has_comment": bool(normalized_comment),
                    },
                )

                await client._make_request("POST", endpoint, token=token, json=payload)

            logger.info(
                "Introduction created successfully",
                extra={"document_id": document_id, "added_count": len(employee_ids)},
            )

            return IntroductionResult(success=True, added_count=len(employee_ids))

        except Exception as e:
            logger.error(
                f"Failed to create introduction: {e}",
                exc_info=True,
                extra={"document_id": document_id},
            )
            return IntroductionResult(
                success=False,
                added_count=0,
                error_message=f"Ошибка API: {e!s}",
            )

    # ──────────────────────────────────────────────────────────────────────
    # Private: utils
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_comment(comment: str | None) -> str:
        """Нормализует комментарий перед отправкой в API."""
        if not comment:
            return ""

        comment = comment.strip()

        template_phrases = [
            "не указан комментарий к ознакомлению",
            "не указан комментарий",
            "комментарий к ознакомлению",
        ]

        if comment.lower() in template_phrases:
            return ""

        if len(comment) > 1:
            return comment[0].upper() + comment[1:]

        return comment.upper()

    @staticmethod
    def _format_employee_match(employee: dict) -> dict:
        """Форматирует данные сотрудника для disambiguation response."""
        first_name = employee.get("firstName", "")
        last_name = employee.get("lastName", "")
        middle_name = employee.get("middleName", "") or ""
        full_name = f"{last_name} {first_name} {middle_name}".strip()

        post_data = employee.get("post", {})
        post_name = (
            post_data.get("postName", "Не указана")
            if isinstance(post_data, dict)
            else "Не указана"
        )

        dept_data = employee.get("department", {})
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
