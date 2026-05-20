# edms_ai_assistant/clients/group_client.py
"""
EDMS AI Assistant — Group HTTP Client.

──────────────────────────────────────────────────────────────────
  ОБЩИЕ ГРУППЫ
  GET  /api/group/fts-name              → find_by_name()
  GET  /api/group/employee/all          → get_employees_by_group_ids()

  ЛИЧНЫЕ ГРУППЫ
  GET  /api/personal-group              → find_personal_by_name()
  GET  /api/personal-group/employee/all → get_employees_by_personal_group_ids()
──────────────────────────────────────────────────────────────────
"""

import logging
from typing import Any
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import EmployeeDto

logger = logging.getLogger(__name__)


class GroupClient:
    """Concrete async HTTP client for EDMS Group API.

    Использует композицию: делегирует HTTP-логику базовому клиенту.
    Возвращает строгие Pydantic DTO для сотрудников.
    """

    def __init__(self, base_client: EdmsBaseClient):
        self._client = base_client

    # ── Общие группы ───────────────────────────────────────────────────

    async def find_by_name(self, token: str, group_name: str) -> dict[str, Any] | None:
        """Поиск группы по названию через FTS.

        Endpoint: GET /api/group/fts-name?fts={group_name}
        Response: GroupDto (один объект)
        """
        try:
            result = await self._client._make_request(
                "GET", "api/group/fts-name", token=token, params={"fts": group_name}
            )
            if isinstance(result, dict) and result:
                logger.info("Found group: %s", result.get("name", "Unknown"))
                return result
        except EdmsNotFoundError:
            logger.info("Group not found: '%s'", group_name)

        return None

    async def get_employees_by_group_ids(
            self, token: str, group_ids: list[UUID]
    ) -> list[EmployeeDto]:
        """Получает сотрудников из общих групп.

        Endpoint: GET /api/group/employee/all?ids={id1}&ids={id2}
        Response: List<GroupEmployeeDto> (внутри лежит {"employee": {...}})
        """
        if not group_ids:
            return []

        try:
            result = await self._client._make_request(
                "GET",
                "api/group/employee/all",
                token=token,
                params={"ids": [str(gid) for gid in group_ids]}
            )
            if isinstance(result, list):
                employees = []
                for item in result:
                    if isinstance(item, dict):
                        emp_data = item.get("employee")
                        if isinstance(emp_data, dict):
                            employees.append(EmployeeDto.model_validate(emp_data))
                return employees
        except EdmsNotFoundError:
            pass

        return []

    # ── Личные группы ──────────────────────────────────────────────────

    async def find_personal_by_name(
            self, token: str, group_name: str
    ) -> dict[str, Any] | None:
        """Поиск личной группы по названию.

        PersonalGroupController НЕ имеет /fts-name.
        Вместо этого используем GET /api/personal-group с BasicSearchRequest.

        Endpoint: GET /api/personal-group?query={name}&page=0&size=20
        Response: SliceDto { "content": [PersonalGroupDto, ...], ... }
        """
        params = {
            "query": group_name,
            "page": 0,
            "size": 20,
        }

        try:
            result = await self._client._make_request(
                "GET", "api/personal-group", token=token, params=params
            )

            items = []
            if isinstance(result, dict):
                content = result.get("content")
                if isinstance(content, list):
                    items = content
            elif isinstance(result, list):
                items = result

            if items:
                best = _find_best_personal_group_match(items, group_name)
                if best:
                    logger.info(
                        "Found personal group: %s (id=%s)",
                        best.get("name", "Unknown"),
                        str(best.get("id", ""))[:8],
                    )
                    return best

            logger.warning("Personal group not found: '%s'", group_name)
        except EdmsNotFoundError:
            logger.info("Personal group not found: '%s'", group_name)

        return None

    async def get_employees_by_personal_group_ids(
            self, token: str, group_ids: list[UUID]
    ) -> list[EmployeeDto]:
        """Получает сотрудников из личных групп.

        Endpoint: GET /api/personal-group/employee/all?ids={id1}&ids={id2}
        Response: List<PersonalGroupEmployeeDto>
        """
        if not group_ids:
            return []

        try:
            result = await self._client._make_request(
                "GET",
                "api/personal-group/employee/all",
                token=token,
                params={"ids": [str(gid) for gid in group_ids]}
            )

            if isinstance(result, list):
                employees = []
                for item in result:
                    if isinstance(item, dict):
                        emp_data = item.get("employee")
                        if isinstance(emp_data, dict):
                            employees.append(EmployeeDto.model_validate(emp_data))
                        elif "id" in item and "lastName" in item:
                            employees.append(EmployeeDto.model_validate(item))
                return employees
        except EdmsNotFoundError:
            pass

        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_best_personal_group_match(
        groups: list[dict[str, Any]],
        search_name: str,
) -> dict[str, Any] | None:
    """Находит лучшее совпадение личной группы по названию.

    Приоритет:
    1. Точное совпадение (case-insensitive)
    2. Начинается с поискового терма
    3. Содержит поисковый терм
    4. Первый из списка
    """
    search_lower = search_name.strip().lower()

    # Точное совпадение
    for g in groups:
        name = (g.get("name") or "").strip().lower()
        if name == search_lower:
            return g

    # Начинается с
    for g in groups:
        name = (g.get("name") or "").strip().lower()
        if name.startswith(search_lower):
            return g

    # Содержит
    for g in groups:
        name = (g.get("name") or "").strip().lower()
        if search_lower in name:
            return g

    # Первый из списка
    return groups[0] if groups else None
