# edms_ai_assistant/services/introduction_service.py (FIXED VERSION)
import logging
from typing import List, Dict, Any, Optional, Set
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.department_client import DepartmentClient
from edms_ai_assistant.clients.group_client import GroupClient
from edms_ai_assistant.clients.base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class PostIntroductionRequest(BaseModel):
    """Request model for creating introduction."""

    executorListIds: List[UUID] = Field(
        ..., description="List of employee UUIDs to add to introduction"
    )
    # CRITICAL FIX: Java requires non-null comment, use empty string as default
    comment: str = Field(
        default="",  # ← CHANGED from Optional[str] = None
        description="Comment for introduction (empty string if not provided)",
    )

    model_config = ConfigDict(
        json_encoders={UUID: str},
        use_enum_values=True,
    )


class IntroductionResult(BaseModel):

    success: bool
    added_count: int = 0
    not_found: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class IntroductionService:

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

    async def collect_employees(
        self,
        token: str,
        last_names: Optional[List[str]] = None,
        department_names: Optional[List[str]] = None,
        group_names: Optional[List[str]] = None,
    ) -> tuple[Set[UUID], List[str], List[Dict[str, Any]]]:
        found_ids: Set[UUID] = set()
        not_found: List[str] = []
        ambiguous_results: List[Dict[str, Any]] = []

        if last_names:
            for last_name in last_names:
                employees = await self.employee_client.search_employees(
                    token, {"lastName": last_name, "includes": ["POST", "DEPARTMENT"]}
                )

                if not employees:
                    not_found.append(f"Сотрудник: {last_name}")
                    logger.warning(f"Employee not found: {last_name}")

                elif len(employees) == 1:
                    emp = employees[0]
                    emp_id = (
                        UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
                    )
                    found_ids.add(emp_id)
                    logger.debug(f"Found single employee: {last_name} -> {emp_id}")

                else:
                    logger.info(
                        f"Found {len(employees)} employees with last name '{last_name}' - requires disambiguation"
                    )
                    ambiguous_results.append(
                        {
                            "search_query": last_name,
                            "matches": [
                                {
                                    "id": str(emp["id"]),
                                    "full_name": f"{emp.get('lastName', '')} {emp.get('firstName', '')} {emp.get('middleName', '') or ''}".strip(),
                                    "post": (
                                        emp.get("post", {}).get(
                                            "postName", "Не указана"
                                        )
                                        if isinstance(emp.get("post"), dict)
                                        else "Не указана"
                                    ),
                                    "department": (
                                        emp.get("department", {}).get(
                                            "name", "Не указан"
                                        )
                                        if isinstance(emp.get("department"), dict)
                                        else "Не указан"
                                    ),
                                }
                                for emp in employees
                            ],
                        }
                    )

        if department_names:
            for dept_name in department_names:
                dept = await self.department_client.find_by_name(token, dept_name)

                if not dept:
                    not_found.append(f"Департамент: {dept_name}")
                    logger.warning(f"Department not found: {dept_name}")
                else:
                    dept_id = (
                        UUID(dept["id"]) if isinstance(dept["id"], str) else dept["id"]
                    )
                    employees = (
                        await self.department_client.get_employees_by_department_id(
                            token, dept_id
                        )
                    )

                    for emp in employees:
                        emp_id = (
                            UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
                        )
                        found_ids.add(emp_id)

                    logger.info(
                        f"Added {len(employees)} employees from department '{dept_name}'"
                    )

        if group_names:
            group_ids = []
            for group_name in group_names:
                group = await self.group_client.find_by_name(token, group_name)

                if not group:
                    not_found.append(f"Группа: {group_name}")
                    logger.warning(f"Group not found: {group_name}")
                else:
                    group_id = (
                        UUID(group["id"])
                        if isinstance(group["id"], str)
                        else group["id"]
                    )
                    group_ids.append(group_id)

            if group_ids:
                employees = await self.group_client.get_employees_by_group_ids(
                    token, group_ids
                )

                for emp in employees:
                    emp_id = (
                        UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]
                    )
                    found_ids.add(emp_id)

                logger.info(
                    f"Added {len(employees)} employees from {len(group_ids)} groups"
                )

        return found_ids, not_found, ambiguous_results

    async def create_introduction(
        self,
        token: str,
        document_id: str,
        employee_ids: List[UUID],
        comment: Optional[str] = None,
    ) -> bool:
        if not employee_ids:
            logger.warning("No employees to add to introduction")
            return False

        if not comment or comment.lower() in [
            "не указан комментарий к ознакомлению",
            "не указан комментарий",
            "комментарий к ознакомлению",
        ]:
            comment = ""

        if comment and comment.strip():
            comment = (
                comment[0].upper() + comment[1:]
                if len(comment) > 1
                else comment.upper()
            )

        request = PostIntroductionRequest(executorListIds=employee_ids, comment=comment)

        try:
            async with EdmsHttpClient() as client:
                endpoint = f"api/document/{document_id}/introduction"

                payload = request.model_dump(mode="json")
                logger.debug(f"Introduction request payload: {payload}")

                await client._make_request("POST", endpoint, token=token, json=payload)

            logger.info(
                f"Successfully created introduction for {len(employee_ids)} employees"
            )
            return True

        except Exception as e:
            logger.error(f"Error creating introduction: {e}", exc_info=True)
            return False
