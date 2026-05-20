# edms_ai_assistant/clients/employee_client.py
"""
EDMS AI Assistant — Employee HTTP Client.

──────────────────────────────────────────────────────────────────────────────────────────────────
  ПОИСК И ЧТЕНИЕ
  GET  /                          → search_employees()        (SliceDto<EmployeeDto>)
  POST /search                    → search_employees_post()   (SliceDto<EmployeeDto>)
  GET  /{id}                      → get_employee()            (EmployeeDto)
  GET  /me                        → get_current_user()        (CurrentUserDto)
  GET  /fts-lastname              → find_by_last_name_fts()   (EmployeeDto, top-1)

  СВЯЗАННЫЕ ДАННЫЕ
  GET  /{id}/role                 → get_employee_roles()      (Set[RoleDto])
  GET  /{id}/griefs               → get_employee_griefs()     (List[EmployeeAccessGriefDto>)

  ЖИЗНЕННЫЙ ЦИКЛ
  POST /dismiss                   → dismiss_employee()        (204)
  POST /recover                   → recover_employee()        (204)
  POST /disable                   → disable_employee()        (204)
  POST /enable                    → enable_employee()         (204)
──────────────────────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Any

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.document import SpringSlice
from edms_ai_assistant.domain.employee import (
    CurrentUserDto,
    EmployeeAccessGriefDto,
    EmployeeDto,
    RoleDto,
)

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 20
_DEFAULT_INCLUDES: list[str] = ["POST", "DEPARTMENT"]


class EmployeeClient:
    """Concrete async HTTP client for EDMS Employee API.
    """

    def __init__(self, base_client: EdmsBaseClient):
        self._client = base_client

    # ── Поиск и чтение ────────────────────────────────────────────────────────

    async def search_employees(
            self,
            token: str,
            employee_filter: dict[str, Any] | None = None,
            pageable: dict[str, Any] | None = None,
    ) -> list[EmployeeDto]:
        effective_filter = _ensure_includes(employee_filter or {})
        params = _build_params(effective_filter, pageable)

        logger.debug(
            "Searching employees (GET)",
            extra={"filter_keys": list(effective_filter.keys())},
        )

        result = await self._client._make_request(
            "GET", "api/employee", token=token, params=params
        )
        return _extract_slice_content(result, endpoint="GET api/employee")

    async def search_employees_post(
            self,
            token: str,
            employee_filter: dict[str, Any] | None = None,
            pageable: dict[str, Any] | None = None,
    ) -> list[EmployeeDto]:
        effective_filter = _ensure_includes(employee_filter or {})
        pageable_params = _build_pageable_params(pageable)

        logger.debug(
            "Searching employees (POST)",
            extra={"filter_keys": list(effective_filter.keys())},
        )

        result = await self._client._make_request(
            "POST",
            "api/employee/search",
            token=token,
            params=pageable_params,
            json=effective_filter,
        )
        return _extract_slice_content(result, endpoint="POST api/employee/search")

    async def get_employee(self, token: str, employee_id: str) -> EmployeeDto | None:
        result = await self._client._make_request(
            "GET", f"api/employee/{employee_id}", token=token
        )
        if isinstance(result, dict) and result:
            return EmployeeDto.model_validate(result)
        return None

    async def get_current_user(self, token: str) -> CurrentUserDto | None:
        result = await self._client._make_request("GET", "api/employee/me", token=token)
        if isinstance(result, dict) and result:
            return CurrentUserDto.model_validate(result)
        return None

    async def find_by_last_name_fts(
            self, token: str, last_name: str
    ) -> EmployeeDto | None:
        try:
            result = await self._client._make_request(
                "GET",
                "api/employee/fts-lastname",
                token=token,
                params={"fts": last_name.strip()},
            )
            if isinstance(result, dict) and result:
                logger.debug(
                    "FTS employee found",
                    extra={"last_name": last_name, "found_id": str(result.get("id", ""))[:8]},
                )
                return EmployeeDto.model_validate(result)
            return None
        except EdmsNotFoundError:
            logger.info("FTS employee not found: %s", last_name)
            return None
        except Exception as exc:
            logger.warning("FTS search failed for '%s': %s", last_name, exc)
            return None

    # ── Связанные данные ──────────────────────────────────────────────────────

    async def get_employee_roles(
            self, token: str, employee_id: str
    ) -> list[RoleDto]:
        """Fetches roles for an employee. GET /api/employee/{id}/role."""
        try:
            result = await self._client._make_request(
                "GET", f"api/employee/{employee_id}/role", token=token
            )
            if isinstance(result, list):
                return [RoleDto.model_validate(r) for r in result]
            if isinstance(result, dict):
                return [RoleDto.model_validate(v) for v in result.values() if isinstance(v, dict)]
        except EdmsNotFoundError:
            pass
        return []

    async def get_employee_griefs(
            self, token: str, employee_id: str
    ) -> list[EmployeeAccessGriefDto]:
        """Fetches access griefs for an employee. GET /api/employee/{id}/griefs."""
        try:
            result = await self._client._make_request(
                "GET", f"api/employee/{employee_id}/griefs", token=token
            )
            if isinstance(result, list):
                return [EmployeeAccessGriefDto.model_validate(g) for g in result]
        except EdmsNotFoundError:
            pass
        return []

    # ── Жизненный цикл ────────────────────────────────────────────────────────

    async def dismiss_employee(
            self, token: str, employee_id: str, delegate_to_id: str
    ) -> bool:
        """Dismisses an employee. Raises on failure."""
        return await self._action_request(
            "POST", "api/employee/dismiss", token, {"id": employee_id, "to": delegate_to_id}
        )

    async def recover_employee(self, token: str, employee_id: str) -> bool:
        """Recovers an employee. Raises on failure."""
        return await self._action_request(
            "POST", "api/employee/recover", token, {"id": employee_id}
        )

    async def disable_employee(
            self, token: str, employee_id: str, delegate_to_id: str
    ) -> bool:
        """Disables an employee. Raises on failure."""
        return await self._action_request(
            "POST", "api/employee/disable", token, {"id": employee_id, "to": delegate_to_id}
        )

    async def enable_employee(self, token: str, employee_id: str) -> bool:
        """Enables an employee. Raises on failure."""
        return await self._action_request(
            "POST", "api/employee/enable", token, {"id": employee_id}
        )

    # ── Внутренние хелперы ────────────────────────────────────────────────────

    async def _action_request(
            self, method: str, endpoint: str, token: str, payload: dict
    ) -> bool:
        await self._client._make_request(
            method, endpoint, token=token, json=payload, is_json_response=False
        )
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════


def _ensure_includes(employee_filter: dict[str, Any]) -> dict[str, Any]:
    if not employee_filter.get("includes"):
        return {**employee_filter, "includes": _DEFAULT_INCLUDES}
    return employee_filter


def _build_pageable_params(pageable: dict[str, Any] | None) -> dict[str, Any]:
    effective: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
    if pageable:
        effective.update(pageable)
    return effective


def _build_params(
        employee_filter: dict[str, Any],
        pageable: dict[str, Any] | None,
) -> dict[str, Any]:
    return {**employee_filter, **_build_pageable_params(pageable)}


def _extract_slice_content(
        result: Any,
        endpoint: str = "api/employee",
) -> list[EmployeeDto]:
    """Извлекает список сотрудников из Spring Slice обертки."""
    if isinstance(result, dict):
        try:
            slice_dto = SpringSlice[EmployeeDto].model_validate(result)
            return slice_dto.content
        except Exception:
            if result.get("id") or result.get("lastName"):
                return [EmployeeDto.model_validate(result)]
            logger.warning(
                "Unexpected Slice response from %s",
                endpoint,
                extra={"response_keys": list(result.keys())},
            )
            return []

    if isinstance(result, list):
        return [EmployeeDto.model_validate(item) for item in result]

    return []
