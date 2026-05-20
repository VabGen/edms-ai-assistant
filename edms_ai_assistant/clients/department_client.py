# edms_ai_assistant/clients/department_client.py
from __future__ import annotations

import logging
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.config import EdmsSettings
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import DepartmentDto, EmployeeDto

logger = logging.getLogger(__name__)


class DepartmentClient(EdmsBaseClient):
    """Client for EDMS Department API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def find_by_name(
            self, token: str, department_name: str
    ) -> DepartmentDto | None:
        """Ищет отдел по названию через FTS."""
        try:
            return await self._request_dto(
                "GET",
                "api/department/fts-name",
                token,
                DepartmentDto,
                params={"fts": department_name}
            )
        except EdmsNotFoundError:
            return None

    async def get_employees_by_department_id(
            self, token: str, department_id: UUID | str
    ) -> list[EmployeeDto]:
        """Получает список сотрудников отдела."""
        try:
            return await self._request_list(
                "GET",
                f"api/department/{department_id}/employees/all",
                token,
                EmployeeDto
            )
        except EdmsNotFoundError:
            return []
