"""
EDMS AI Assistant — Employee HTTP Client.

──────────────────────────────────────────────────────────────────
  ПОИСК И ЧТЕНИЕ
  GET  /                          → search_employees()        (SliceDto<EmployeeDto>)
  POST /search                    → search_employees_post()   (SliceDto<EmployeeDto>)
  GET  /{id}                      → get_employee()            (EmployeeDto)
  GET  /me                        → get_current_user()        (CurrentUser)
  GET  /fts-lastname              → find_by_last_name_fts()   (EmployeeDto, top-1)

  СВЯЗАННЫЕ ДАННЫЕ
  GET  /{id}/role                 → get_employee_roles()      (Set<RoleDto>)
  GET  /{id}/griefs               → get_employee_griefs()     (List<EmployeeAccessGriefDto>)

  ЖИЗНЕННЫЙ ЦИКЛ
  POST /dismiss                   → dismiss_employee()        (204)
  POST /recover                   → recover_employee()        (204)
  POST /disable                   → disable_employee()        (204)
  POST /enable                    → enable_employee()         (204)
──────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from .base_client import EdmsBaseClient, EdmsHttpClient

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 20
_DEFAULT_INCLUDES: list[str] = ["POST", "DEPARTMENT"]


# ══════════════════════════════════════════════════════════════════════════════
# Abstract Interface
# ══════════════════════════════════════════════════════════════════════════════


class BaseEmployeeClient(EdmsBaseClient):
    """Abstract interface for EDMS Employee API clients.

    Определяет контракт для всех read/write методов EmployeeController.java,
    релевантных для агента.
    """

    # ── Поиск и чтение ────────────────────────────────────────────────────────

    @abstractmethod
    async def search_employees(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def search_employees_post(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_employee(self, token: str, employee_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def get_current_user(self, token: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def find_by_last_name_fts(
        self, token: str, last_name: str
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    # ── Связанные данные ──────────────────────────────────────────────────────

    @abstractmethod
    async def get_employee_roles(
        self, token: str, employee_id: str
    ) -> list[dict[str, Any]]:
        """Fetches roles for an employee. GET /api/employee/{id}/role."""
        raise NotImplementedError

    @abstractmethod
    async def get_employee_griefs(
        self, token: str, employee_id: str
    ) -> list[dict[str, Any]]:
        """Fetches access griefs for an employee. GET /api/employee/{id}/griefs."""
        raise NotImplementedError

    # ── Жизненный цикл ────────────────────────────────────────────────────────

    @abstractmethod
    async def dismiss_employee(
        self, token: str, employee_id: str, delegate_to_id: str
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def recover_employee(self, token: str, employee_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def disable_employee(
        self, token: str, employee_id: str, delegate_to_id: str
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def enable_employee(self, token: str, employee_id: str) -> bool:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Concrete Implementation
# ══════════════════════════════════════════════════════════════════════════════


class EmployeeClient(BaseEmployeeClient, EdmsHttpClient):
    """Concrete async HTTP client for EDMS Employee API.

    Реализует взаимодействие с EmployeeController.java.

    Важно про пагинацию:
      GET  /api/employee          → Spring Slice<EmployeeDto> (НЕ Page — нет totalElements)
      POST /api/employee/search   → Spring Slice<EmployeeDto>

      Структура Slice: { "content": [...], "hasNext": bool, "hasPrevious": bool }
      Метод _extract_slice_content() корректно извлекает content.

    Важно про includes:
      Без includes=[POST, DEPARTMENT] в ответе не будет вложенных объектов.
    """

    # ── Поиск и чтение ────────────────────────────────────────────────────────

    async def search_employees(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        effective_filter = _ensure_includes(employee_filter or {})
        params = _build_params(effective_filter, pageable)

        logger.debug(
            "Searching employees (GET)",
            extra={"filter_keys": list(effective_filter.keys())},
        )

        result = await self._make_request(
            "GET", "api/employee", token=token, params=params
        )
        return _extract_slice_content(result, endpoint="GET api/employee")

    async def search_employees_post(
        self,
        token: str,
        employee_filter: dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        effective_filter = _ensure_includes(employee_filter or {})
        pageable_params = _build_pageable_params(pageable)

        logger.debug(
            "Searching employees (POST)",
            extra={"filter_keys": list(effective_filter.keys())},
        )

        result = await self._make_request(
            "POST",
            "api/employee/search",
            token=token,
            params=pageable_params,
            json=effective_filter,
        )
        return _extract_slice_content(result, endpoint="POST api/employee/search")

    async def get_employee(self, token: str, employee_id: str) -> dict[str, Any] | None:
        result = await self._make_request(
            "GET", f"api/employee/{employee_id}", token=token
        )
        return result if isinstance(result, dict) and result else None

    async def get_current_user(self, token: str) -> dict[str, Any] | None:
        result = await self._make_request("GET", "api/employee/me", token=token)
        return result if isinstance(result, dict) and result else None

    async def find_by_last_name_fts(
        self, token: str, last_name: str
    ) -> dict[str, Any] | None:
        try:
            result = await self._make_request(
                "GET",
                "api/employee/fts-lastname",
                token=token,
                params={"fts": last_name.strip()},
            )
            if isinstance(result, dict) and result:
                logger.debug(
                    "FTS employee found",
                    extra={
                        "last_name": last_name,
                        "found_id": str(result.get("id", ""))[:8],
                    },
                )
                return result
            return None
        except Exception:
            logger.warning("FTS search failed for '%s'", last_name, exc_info=True)
            return None

    # ── Связанные данные ──────────────────────────────────────────────────────

    async def get_employee_roles(
            self, token: str, employee_id: str
    ) -> list[dict[str, Any]]:
        """Fetches roles for an employee. GET /api/employee/{id}/role."""
        try:
            result = await self._make_request(
                "GET", f"api/employee/{employee_id}/role", token=token
            )
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                values = list(result.values())
                if values and isinstance(values[0], dict):
                    return values
            return []
        except Exception:
            logger.warning(
                "Failed to fetch roles for employee %s",
                employee_id[:8],
                exc_info=True,
            )
            return []

    async def get_employee_griefs(
            self, token: str, employee_id: str
    ) -> list[dict[str, Any]]:
        """Fetches access griefs for an employee. GET /api/employee/{id}/griefs."""
        try:
            result = await self._make_request(
                "GET", f"api/employee/{employee_id}/griefs", token=token
            )
            if isinstance(result, list):
                return result
            return []
        except Exception:
            logger.warning(
                "Failed to fetch griefs for employee %s",
                employee_id[:8],
                exc_info=True,
            )
            return []

    # ── Жизненный цикл ────────────────────────────────────────────────────────

    async def dismiss_employee(
        self,
        token: str,
        employee_id: str,
        delegate_to_id: str,
    ) -> bool:
        return await self._action_request(
            "POST",
            "api/employee/dismiss",
            token=token,
            json={"id": employee_id, "to": delegate_to_id},
            log_context={"employee_id": employee_id, "delegate_to": delegate_to_id},
        )

    async def recover_employee(self, token: str, employee_id: str) -> bool:
        return await self._action_request(
            "POST",
            "api/employee/recover",
            token=token,
            json={"id": employee_id},
            log_context={"employee_id": employee_id},
        )

    async def disable_employee(
        self,
        token: str,
        employee_id: str,
        delegate_to_id: str,
    ) -> bool:
        return await self._action_request(
            "POST",
            "api/employee/disable",
            token=token,
            json={"id": employee_id, "to": delegate_to_id},
            log_context={"employee_id": employee_id, "delegate_to": delegate_to_id},
        )

    async def enable_employee(self, token: str, employee_id: str) -> bool:
        return await self._action_request(
            "POST",
            "api/employee/enable",
            token=token,
            json={"id": employee_id},
            log_context={"employee_id": employee_id},
        )

    # ── Внутренние хелперы ────────────────────────────────────────────────────

    async def _action_request(
        self,
        method: str,
        endpoint: str,
        token: str,
        json: dict[str, Any],
        log_context: dict[str, Any],
    ) -> bool:
        try:
            await self._make_request(
                method,
                endpoint,
                token=token,
                json=json,
                is_json_response=False,
            )
            logger.info(
                "Employee action succeeded: %s %s",
                method,
                endpoint,
                extra=log_context,
            )
            return True
        except Exception:
            logger.error(
                "Employee action failed: %s %s",
                method,
                endpoint,
                exc_info=True,
                extra=log_context,
            )
            return False


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
) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            return content
        # Some endpoints return raw list without Slice wrapper
        if result.get("id") or result.get("lastName"):
            return [result]
        logger.warning(
            "Unexpected Slice response from %s: dict without 'content'",
            endpoint,
            extra={"response_keys": list(result.keys())},
        )
        return []

    if isinstance(result, list):
        return result

    return []