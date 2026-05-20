# edms_ai_assistant/clients/group_client.py
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.config import EdmsSettings
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import EmployeeDto, GroupDto

logger = logging.getLogger(__name__)


class GroupClient(EdmsBaseClient):
    """Client for EDMS Group API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def find_by_name(self, token: str, group_name: str) -> GroupDto | None:
        """Поиск группы по названию через FTS."""
        try:
            return await self._request_dto(
                "GET", "api/group/fts-name", token, GroupDto, params={"fts": group_name}
            )
        except EdmsNotFoundError:
            return None

    async def get_employees_by_group_ids(
            self, token: str, group_ids: list[UUID] | list[str]
    ) -> list[EmployeeDto]:
        """Получает сотрудников из общих групп."""
        if not group_ids:
            return []

        try:
            # Специфичная логика маппинга для общих групп (вложенный employee)
            raw_data = await self._make_request(
                "GET",
                "api/group/employee/all",
                token,
                params={"ids": [str(gid) for gid in group_ids]}
            )
            if not isinstance(raw_data, list):
                return []

            employees = []
            for item in raw_data:
                if isinstance(item, dict) and "employee" in item:
                    employees.append(EmployeeDto.model_validate(item["employee"]))
            return employees
        except EdmsNotFoundError:
            return []

    async def find_personal_by_name(
            self, token: str, group_name: str
    ) -> dict[str, Any] | None:
        """Поиск личной группы по названию."""
        params = {"query": group_name, "page": 0, "size": 20}

        try:
            # Личные группы пока возвращают сырой dict, так как DTO может быть сложным
            result = await self._make_request("GET", "api/personal-group", token, params=params)

            items = []
            if isinstance(result, dict) and "content" in result:
                items = result["content"]
            elif isinstance(result, list):
                items = result

            if items:
                return _find_best_personal_group_match(items, group_name)
            return None
        except EdmsNotFoundError:
            return None

    async def get_employees_by_personal_group_ids(
            self, token: str, group_ids: list[UUID] | list[str]
    ) -> list[EmployeeDto]:
        """Получает сотрудников из личных групп."""
        if not group_ids:
            return []

        try:
            raw_data = await self._make_request(
                "GET",
                "api/personal-group/employee/all",
                token,
                params={"ids": [str(gid) for gid in group_ids]}
            )
            if not isinstance(raw_data, list):
                return []

            employees = []
            for item in raw_data:
                if isinstance(item, dict):
                    emp_data = item.get("employee")
                    if isinstance(emp_data, dict):
                        employees.append(EmployeeDto.model_validate(emp_data))
                    elif "id" in item and "lastName" in item:
                        employees.append(EmployeeDto.model_validate(item))
            return employees
        except EdmsNotFoundError:
            return []


def _find_best_personal_group_match(
        groups: list[dict[str, Any]],
        search_name: str,
) -> dict[str, Any] | None:
    search_lower = search_name.strip().lower()
    for g in groups:
        name = (g.get("name") or "").strip().lower()
        if name == search_lower:
            return g
    for g in groups:
        name = (g.get("name") or "").strip().lower()
        if name.startswith(search_lower):
            return g
    return groups[0] if groups else None
