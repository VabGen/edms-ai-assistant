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
from abc import abstractmethod
from typing import Any
from uuid import UUID

from .base_client import EdmsBaseClient

logger = logging.getLogger(__name__)


class BaseGroupClient(EdmsBaseClient):
    """Abstract interface for EDMS Group API clients."""

    @abstractmethod
    async def find_by_name(self, token: str, group_name: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def find_personal_by_name(
        self, token: str, group_name: str
    ) -> dict[str, Any] | None:
        """Находит личную группу по названию (через BasicSearchRequest)."""
        raise NotImplementedError

    @abstractmethod
    async def get_employees_by_group_ids(
        self, token: str, group_ids: list[UUID]
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_employees_by_personal_group_ids(
        self, token: str, group_ids: list[UUID]
    ) -> list[dict[str, Any]]:
        """Получает сотрудников из личных групп."""
        raise NotImplementedError


class GroupClient(BaseGroupClient, EdmsBaseClient):
    """Concrete async HTTP client for EDMS Group API."""

    # ── Общие группы ───────────────────────────────────────────────────

    async def find_by_name(self, token: str, group_name: str) -> dict[str, Any] | None:
        """Поиск группы по названию через FTS.

        Endpoint: GET /api/group/fts-name?fts={group_name}
        Response: GroupDto (один объект)
        """
        endpoint = "api/group/fts-name"
        params = {"fts": group_name}

        try:
            result = await self._make_request(
                "GET", endpoint, token=token, params=params
            )
            if result and isinstance(result, dict):
                logger.info(f"Found group: {result.get('name', 'Unknown')}")
                return result
            return None
        except Exception as e:
            logger.error(f"Error searching group '{group_name}': {e}")
            return None

    async def get_employees_by_group_ids(
        self, token: str, group_ids: list[UUID]
    ) -> list[dict[str, Any]]:
        """Получает сотрудников из общих групп.

        Endpoint: GET /api/group/employee/all?ids={id1}&ids={id2}
        Response: List<GroupEmployeeDto>
        GroupEmployeeDto = { "employee": {...}, "group": {...}, ... }
        """
        if not group_ids:
            return []

        endpoint = f"api/group/employee/all?ids={group_ids[0]}"
        for group_id in group_ids[1:]:
            endpoint += f"&ids={group_id}"

        try:
            result = await self._make_request("GET", endpoint, token=token)

            if isinstance(result, list):
                employees = []
                for item in result:
                    if isinstance(item, dict) and "employee" in item:
                        employees.append(item["employee"])

                logger.info(
                    f"Found {len(employees)} employees in {len(group_ids)} groups"
                )
                return employees
            return []
        except Exception as e:
            logger.error(f"Error fetching employees for groups {group_ids}: {e}")
            return []

    # ── Личные группы ──────────────────────────────────────────────────

    async def find_personal_by_name(
        self, token: str, group_name: str
    ) -> dict[str, Any] | None:
        """Поиск личной группы по названию.

        PersonalGroupController НЕ имеет /fts-name.
        Вместо этого используем GET /api/personal-group с BasicSearchRequest.

        BasicSearchRequest имеет поле 'query' для текстового поиска.
        Возвращает SliceDto<PersonalGroupDto> с пагинацией.

        Endpoint: GET /api/personal-group?query={name}&page=0&size=20
        Response: SliceDto { "content": [PersonalGroupDto, ...], ... }
        """
        endpoint = "api/personal-group"
        params = {
            "query": group_name,
            "page": 0,
            "size": 20,
        }

        try:
            result = await self._make_request(
                "GET", endpoint, token=token, params=params
            )

            if not result:
                return None

            # SliceDto: { "content": [...], "hasNext": bool }
            if isinstance(result, dict):
                content = result.get("content")
                if isinstance(content, list) and content:
                    best = _find_best_personal_group_match(content, group_name)
                    if best:
                        logger.info(
                            "Found personal group: %s (id=%s)",
                            best.get("name", "Unknown"),
                            str(best.get("id", ""))[:8],
                        )
                        return best

            if isinstance(result, list) and result:
                best = _find_best_personal_group_match(result, group_name)
                if best:
                    logger.info("Found personal group: %s", best.get("name", "Unknown"))
                    return best

            logger.warning("Personal group not found: %s", group_name)
            return None
        except Exception as e:
            logger.error("Error searching personal group '%s': %s", group_name, e)
            return None

    async def get_employees_by_personal_group_ids(
        self, token: str, group_ids: list[UUID]
    ) -> list[dict[str, Any]]:
        """Получает сотрудников из личных групп.

        Endpoint: GET /api/personal-group/employee/all?ids={id1}&ids={id2}
        Response: List<PersonalGroupEmployeeDto>
        PersonalGroupEmployeeDto = { "employee": {...}, ... }  (аналогично GroupEmployeeDto)
        """
        if not group_ids:
            return []

        endpoint = f"api/personal-group/employee/all?ids={group_ids[0]}"
        for group_id in group_ids[1:]:
            endpoint += f"&ids={group_id}"

        try:
            result = await self._make_request("GET", endpoint, token=token)

            if isinstance(result, list):
                employees = []
                for item in result:
                    if isinstance(item, dict):
                        emp = item.get("employee")
                        if emp and isinstance(emp, dict):
                            employees.append(emp)
                        elif "id" in item and "lastName" in item:
                            # Нет обёртки — сам объект сотрудника
                            employees.append(item)

                logger.info(
                    "Found %d employees in %d personal groups",
                    len(employees),
                    len(group_ids),
                )
                return employees
            return []
        except Exception as e:
            logger.error(
                "Error fetching employees for personal groups %s: %s",
                group_ids,
                e,
            )
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
