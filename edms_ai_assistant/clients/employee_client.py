# edms_ai_assistant/clients/employee_client.py
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING, TypeVar
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import (
    CurrentUserDto,
    EmployeeAccessGriefDto,
    EmployeeDto,
    RoleDto,
    UserLoginHistoryEntryDto,
    SliceDto,
    EmployeeAddRequest,
    EmployeeUpdateRequest,
    EmployeeFilter
)

if TYPE_CHECKING:
    from edms_ai_assistant.config import EdmsSettings
    from edms_ai_assistant.clients.transport import IAsyncTransport

logger = logging.getLogger(__name__)

T = TypeVar("T")

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
            employee_filter: EmployeeFilter | dict[str, Any] | None = None,
            pageable: dict[str, Any] | None = None,
    ) -> list[EmployeeDto]:
        """Searches for employees using GET request."""
        logger.info("Searching employees via GET")
        params = {
            "page": _DEFAULT_PAGE,
            "size": _DEFAULT_SIZE,
            "includes": _DEFAULT_INCLUDES
        }
        if pageable:
            params.update(pageable)
        if employee_filter:
            if isinstance(employee_filter, EmployeeFilter):
                params.update(employee_filter.model_dump(exclude_none=True, by_alias=True))
            else:
                params.update(employee_filter)

        return await self._request_list(
            "GET", "api/employee", token, EmployeeDto, params=params
        )

    async def search_employees_post(
            self,
            token: str,
            employee_filter: EmployeeFilter | dict[str, Any] | None = None,
            pageable: dict[str, Any] | None = None,
    ) -> list[EmployeeDto]:
        """Searches for employees using POST request with filter body."""
        logger.info("Searching employees via POST")
        params = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if pageable:
            params.update(pageable)

        if isinstance(employee_filter, EmployeeFilter):
            eff_filter = employee_filter.model_dump(exclude_none=True, by_alias=True)
        else:
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

    async def get_employee(self, token: str, employee_id: str | UUID) -> EmployeeDto | None:
        """Fetches a single employee by ID."""
        logger.info(f"Fetching employee {employee_id}")
        try:
            return await self._request_dto(
                "GET", f"api/employee/{employee_id}", token, EmployeeDto
            )
        except EdmsNotFoundError:
            logger.error(f"Employee {employee_id} not found")
            return None

    async def create_employee(self, token: str, request: EmployeeAddRequest) -> EmployeeDto:
        """Creates a new employee."""
        logger.info("Creating new employee")
        return await self._request_dto(
            "POST", "api/employee", token, EmployeeDto, json_data=request.model_dump(by_alias=True)
        )

    async def update_employee(self, token: str, request: EmployeeUpdateRequest) -> EmployeeDto:
        """Updates an existing employee."""
        logger.info(f"Updating employee {request.employee.id}")
        return await self._request_dto(
            "PUT", "api/employee", token, EmployeeDto, json_data=request.model_dump(by_alias=True)
        )

    async def delete_employees(self, token: str, employee_ids: list[UUID]) -> None:
        """Deletes employees by IDs."""
        logger.info(f"Deleting employees: {employee_ids}")
        await self.make_request(
            "DELETE", "api/employee", token, json_data={"ids": employee_ids}, is_json_response=False
        )

    async def get_current_user(self, token: str) -> CurrentUserDto | None:
        """Fetches information about the current authenticated user."""
        logger.info("Fetching current user info")
        try:
            return await self._request_dto("GET", "api/employee/me", token, CurrentUserDto)
        except EdmsNotFoundError:
            logger.error("Current user information not found")
            return None

    async def find_by_last_name_fts(
            self, token: str, last_name: str
    ) -> EmployeeDto | None:
        """Full-text search for employee by last name."""
        logger.info(f"Searching employee by last name (FTS): {last_name}")
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
            self, token: str, employee_id: str | UUID
    ) -> list[RoleDto]:
        """Fetches roles for an employee."""
        logger.info(f"Fetching roles for employee {employee_id}")
        try:
            return await self._request_list(
                "GET", f"api/employee/{employee_id}/role", token, RoleDto
            )
        except EdmsNotFoundError:
            return []

    async def get_employee_griefs(
            self, token: str, employee_id: str | UUID
    ) -> list[EmployeeAccessGriefDto]:
        """Fetches access griefs for an employee."""
        logger.info(f"Fetching access griefs for employee {employee_id}")
        try:
            return await self._request_list(
                "GET", f"api/employee/{employee_id}/griefs", token, EmployeeAccessGriefDto
            )
        except EdmsNotFoundError:
            return []

    async def dismiss_employee(
            self, token: str, employee_id: str | UUID, delegate_to_id: str | UUID
    ) -> None:
        """Dismisses an employee and delegates their tasks."""
        logger.info(f"Dismissing employee {employee_id}, delegating to {delegate_to_id}")
        await self.make_request(
            "POST", "api/employee/dismiss", token, json_data={"id": str(employee_id), "to": str(delegate_to_id)}, is_json_response=False
        )

    async def recover_employee(self, token: str, employee_id: str | UUID) -> None:
        """Recovers a dismissed employee."""
        logger.info(f"Recovering employee {employee_id}")
        await self.make_request(
            "POST", "api/employee/recover", token, json_data={"id": str(employee_id)}, is_json_response=False
        )

    async def get_login_history(
        self, token: str, page: int = 0, size: int = 20
    ) -> SliceDto[UserLoginHistoryEntryDto]:
        """Fetches user login history."""
        logger.info("Fetching user login history")
        return await self._request_dto(
            "GET",
            "api/employee/login-history",
            token,
            SliceDto[UserLoginHistoryEntryDto],
            params={"page": page, "size": size}
        )

    async def get_user_actions(
        self, token: str, action_filter: dict[str, Any], page: int = 0, size: int = 20
    ) -> SliceDto[Any]:
        """Fetches user actions based on filter."""
        logger.info("Fetching user actions")
        return await self._request_dto(
            "POST",
            "api/employee/actions",
            token,
            SliceDto[Any],
            params={"page": page, "size": size},
            json_data=action_filter
        )
