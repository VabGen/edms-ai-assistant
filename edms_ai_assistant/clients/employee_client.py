# edms_ai_assistant/clients/employee_client.py
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import (
    CurrentUserDto,
    EmployeeAccessGriefDto,
    EmployeeDto,
    RoleDto,
)

if TYPE_CHECKING:
    from edms_ai_assistant.config import EdmsSettings
    from edms_ai_assistant.clients.transport import IAsyncTransport

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 20
_DEFAULT_INCLUDES: list[str] = ["POST", "DEPARTMENT"]


class EmployeeClient(EdmsBaseClient):
    """Client for EDMS Employee API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def search_employees(
            self,
            token: str,
            employee_filter: dict[str, Any] | None = None,
            pageable: dict[str, Any] | None = None,
    ) -> list[EmployeeDto]:
        params = {
            "page": _DEFAULT_PAGE,
            "size": _DEFAULT_SIZE,
            "includes": _DEFAULT_INCLUDES
        }
        if pageable:
            params.update(pageable)
        if employee_filter:
            params.update(employee_filter)

        return await self._request_list(
            "GET", "api/employee", token, EmployeeDto, params=params
        )

    async def search_employees_post(
            self,
            token: str,
            employee_filter: dict[str, Any] | None = None,
            pageable: dict[str, Any] | None = None,
    ) -> list[EmployeeDto]:
        params = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if pageable:
            params.update(pageable)

        eff_filter = employee_filter or {}
        if "includes" not in eff_filter:
            eff_filter["includes"] = _DEFAULT_INCLUDES

        return await self._request_list(
            "POST",
            "api/employee/search",
            token,
            EmployeeDto,
            params=params,
            json_data=eff_filter,
        )

    async def get_employee(self, token: str, employee_id: str) -> EmployeeDto | None:
        try:
            return await self._request_dto(
                "GET", f"api/employee/{employee_id}", token, EmployeeDto
            )
        except EdmsNotFoundError:
            return None

    async def get_current_user(self, token: str) -> CurrentUserDto | None:
        try:
            return await self._request_dto("GET", "api/employee/me", token, CurrentUserDto)
        except EdmsNotFoundError:
            return None

    async def find_by_last_name_fts(
            self, token: str, last_name: str
    ) -> EmployeeDto | None:
        try:
            return await self._request_dto(
                "GET",
                "api/employee/fts-lastname",
                token,
                EmployeeDto,
                params={"fts": last_name.strip()},
            )
        except EdmsNotFoundError:
            return None

    async def get_employee_roles(
            self, token: str, employee_id: str
    ) -> list[RoleDto]:
        """Fetches roles for an employee."""
        try:
            return await self._request_list(
                "GET", f"api/employee/{employee_id}/role", token, RoleDto
            )
        except EdmsNotFoundError:
            return []

    async def get_employee_griefs(
            self, token: str, employee_id: str
    ) -> list[EmployeeAccessGriefDto]:
        """Fetches access griefs for an employee."""
        try:
            return await self._request_list(
                "GET", f"api/employee/{employee_id}/griefs", token, EmployeeAccessGriefDto
            )
        except EdmsNotFoundError:
            return []

    async def dismiss_employee(
            self, token: str, employee_id: str, delegate_to_id: str
    ) -> None:
        """Dismisses an employee."""
        await self.make_request(
            "POST", "api/employee/dismiss", token, json_data={"id": employee_id, "to": delegate_to_id}, is_json_response=False
        )

    async def recover_employee(self, token: str, employee_id: str) -> None:
        """Recovers an employee."""
        await self.make_request(
            "POST", "api/employee/recover", token, json_data={"id": employee_id}, is_json_response=False
        )
