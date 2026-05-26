# edms_ai_assistant/clients/acting_officer_client.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.domain.employee import (
    EmployeeIoDto,
    SecretaryRequest,
    EmployeeIoRequest,
)

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)

class EmployeeActingClient(EdmsBaseClient):
    """Клиент для работы с ИО (исполняющими обязанности) и секретарями."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def get_acting_for_target(self, token: str, target_id: UUID) -> list[EmployeeIoDto]:
        """
        Получить список сотрудников ИО для конкретного сотрудника.
        GET api/employee/io/{id}
        """
        result = await self.make_request("GET", f"api/employee/io/{target_id}", token=token)
        return [EmployeeIoDto.model_validate(item) for item in result]

    async def get_acting_targets(self, token: str, io_id: UUID) -> list[EmployeeIoDto]:
        """
        Получить список сотрудников, за которых текущий пользователь исполняет обязанности.
        GET api/employee/io/{id}/target
        """
        result = await self.make_request("GET", f"api/employee/io/{io_id}/target", token=token)
        return [EmployeeIoDto.model_validate(item) for item in result]

    async def add_acting_officer(self, token: str, emp: EmployeeIoDto) -> EmployeeIoDto:
        """
        Добавить сотрудника ИО.
        POST api/employee/io
        """
        result = await self.make_request(
            "POST", "api/employee/io", token=token, json=emp.model_dump(exclude_none=True)
        )
        return EmployeeIoDto.model_validate(result)

    async def add_secretaries(self, token: str, request: SecretaryRequest) -> list[EmployeeIoDto]:
        """
        Добавить список секретарей.
        POST api/employee/io/secretary
        """
        result = await self.make_request(
            "POST", "api/employee/io/secretary", token=token, json=request.model_dump(exclude_none=True)
        )
        return [EmployeeIoDto.model_validate(item) for item in result]

    async def delete_acting_officer(self, token: str, request: EmployeeIoRequest):
        """
        Удалить сотрудника ИО.
        DELETE api/employee/io
        """
        await self.make_request(
            "DELETE", "api/employee/io", token=token, json=request.model_dump(exclude_none=True)
        )

    async def delete_secretaries(self, token: str, request: SecretaryRequest):
        """
        Удалить список секретарей.
        DELETE api/employee/io/secretary
        """
        await self.make_request(
            "DELETE", "api/employee/io/secretary", token=token, json=request.model_dump(exclude_none=True)
        )
