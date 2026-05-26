# edms_ai_assistant/clients/department_client.py
from __future__ import annotations

import logging

from typing import Any, TYPE_CHECKING
from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import (
    DepartmentDto,
    EmployeeDto,
    DepartmentFilter,
    BasicSearchEmployeeRequest,
    DeputyLeaderDepartmentDto,
)
from edms_ai_assistant.domain.reference import BasicSearchRequest

if TYPE_CHECKING:
    from edms_ai_assistant.config import EdmsSettings
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from uuid import UUID

logger = logging.getLogger(__name__)


class DepartmentClient(EdmsBaseClient):
    """Client for EDMS Department API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def get_departments_extended(
        self, token: str, filter: DepartmentFilter | None = None, page: int = 0, size: int = 20
    ) -> list[DepartmentDto]:
        """GET api/department/extended"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        params.update({"page": page, "size": size})
        result = await self.make_request("GET", "api/department/extended", token=token, params=params)
        if isinstance(result, dict) and "content" in result:
            return [DepartmentDto.model_validate(item) for item in result["content"]]
        return [DepartmentDto.model_validate(item) for item in result]

    async def get_roots(
        self, token: str, filter: DepartmentFilter | None = None, page: int = 0, size: int = 20
    ) -> list[DepartmentDto]:
        """GET api/department/roots"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        params.update({"page": page, "size": size})
        result = await self.make_request("GET", "api/department/roots", token=token, params=params)
        if isinstance(result, dict) and "content" in result:
            return [DepartmentDto.model_validate(item) for item in result["content"]]
        return [DepartmentDto.model_validate(item) for item in result]

    async def get_children(
        self, token: str, department_id: UUID, filter: DepartmentFilter | None = None, page: int = 0, size: int = 20
    ) -> list[DepartmentDto]:
        """GET api/department/child-departments/{id}"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        params.update({"page": page, "size": size})
        result = await self.make_request("GET", f"api/department/child-departments/{department_id}", token=token, params=params)
        if isinstance(result, dict) and "content" in result:
            return [DepartmentDto.model_validate(item) for item in result["content"]]
        return [DepartmentDto.model_validate(item) for item in result]

    async def get_departments(
        self, token: str, search: BasicSearchRequest | None = None, page: int = 0, size: int = 20
    ) -> list[DepartmentDto]:
        """GET api/department"""
        params = search.model_dump(exclude_none=True) if search else {}
        params.update({"page": page, "size": size})
        result = await self.make_request("GET", "api/department", token=token, params=params)
        if isinstance(result, dict) and "content" in result:
            return [DepartmentDto.model_validate(item) for item in result["content"]]
        return [DepartmentDto.model_validate(item) for item in result]

    async def get_department(self, token: str, department_id: UUID, filter: DepartmentFilter | None = None) -> DepartmentDto:
        """GET api/department/{id}"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        result = await self.make_request("GET", f"api/department/{department_id}", token=token, params=params)
        return DepartmentDto.model_validate(result)

    async def get_employees(
        self, token: str, department_id: UUID, search: BasicSearchEmployeeRequest | None = None, page: int = 0, size: int = 20
    ) -> list[EmployeeDto]:
        """GET api/department/{id}/employees"""
        params = search.model_dump(exclude_none=True) if search else {}
        params.update({"page": page, "size": size})
        result = await self.make_request("GET", f"api/department/{department_id}/employees", token=token, params=params)
        if isinstance(result, dict) and "content" in result:
            return [EmployeeDto.model_validate(item) for item in result["content"]]
        return [EmployeeDto.model_validate(item) for item in result]

    async def get_employees_all(self, token: str, department_id: UUID, filter: Any | None = None) -> list[EmployeeDto]:
        """GET api/department/{id}/employees/all"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        result = await self.make_request("GET", f"api/department/{department_id}/employees/all", token=token, params=params)
        return [EmployeeDto.model_validate(item) for item in result]

    async def get_path(self, token: str, department_id: UUID) -> list[DepartmentDto]:
        """GET api/department/getPath/{id}"""
        result = await self.make_request("GET", f"api/department/getPath/{department_id}", token=token)
        return [DepartmentDto.model_validate(item) for item in result]

    async def get_without_children(self, token: str, department_id: UUID, page: int = 0, size: int = 20) -> list[DepartmentDto]:
        """GET api/department/getAllWithoutChild/{id}"""
        result = await self.make_request(
            "GET", f"api/department/getAllWithoutChild/{department_id}", token=token, params={"page": page, "size": size}
        )
        if isinstance(result, dict) and "content" in result:
            return [DepartmentDto.model_validate(item) for item in result["content"]]
        return [DepartmentDto.model_validate(item) for item in result]

    async def get_deputy_leaders(self, token: str, department_id: UUID) -> list[DeputyLeaderDepartmentDto]:
        """GET api/department/{id}/deputy-leader"""
        result = await self.make_request("GET", f"api/department/{department_id}/deputy-leader", token=token)
        return [DeputyLeaderDepartmentDto.model_validate(item) for item in result]

    async def create_department(self, token: str, department: DepartmentDto) -> DepartmentDto:
        """POST api/department"""
        result = await self.make_request("POST", "api/department", token=token, json_data=department.model_dump(exclude_none=True))
        return DepartmentDto.model_validate(result)

    async def update_department(self, token: str, department: DepartmentDto) -> DepartmentDto:
        """PUT api/department"""
        result = await self.make_request("PUT", "api/department", token=token, json_data=department.model_dump(exclude_none=True))
        return DepartmentDto.model_validate(result)

    async def delete_department(self, token: str, department_id: UUID, ignore_errors: bool = True):
        """DELETE api/department/{id}"""
        await self.make_request(
            "DELETE", f"api/department/{department_id}", token=token, params={"ignoreErrors": str(ignore_errors).lower()}
        )

    async def find_by_name(self, token: str, department_name: str) -> DepartmentDto | None:
        """GET api/department/fts-name"""
        try:
            result = await self.make_request("GET", "api/department/fts-name", token=token, params={"fts": department_name})
            return DepartmentDto.model_validate(result)
        except EdmsNotFoundError:
            return None

    # Legacy helper
    async def get_employees_by_department_id(self, token: str, department_id: UUID | str) -> list[EmployeeDto]:
        return await self.get_employees_all(token, UUID(str(department_id)))
