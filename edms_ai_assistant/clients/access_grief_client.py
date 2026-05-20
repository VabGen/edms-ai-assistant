# edms_ai_assistant/clients/access_grief_client.py
"""
EDMS AI Assistant — Access Grief HTTP Client.

Клиент для AccessGriefController.java.

  GET  /api/access-grief                    → search_griefs()    (SliceDto<AccessGriefDto>)
  GET  /api/access-grief/{id}               → get_grief()        (AccessGriefDto)
  GET  /api/access-grief/{id}/employees     → get_grief_employees() (SliceDto<EmployeeAccessGriefDto>)

Намеренно НЕ реализованы:
  - POST / PUT / DELETE — управление грифами не входит в сценарии агента
"""

from __future__ import annotations

import logging
from typing import Any

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.document import SpringSlice
from edms_ai_assistant.domain.employee import AccessGriefDto, EmployeeAccessGriefDto

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 50


class AccessGriefClient:
    """Async HTTP client for EDMS Access Grief API.
    """

    def __init__(self, base_client: EdmsBaseClient):
        self._client = base_client

    async def search_griefs(
            self,
            token: str,
            name: str | None = None,
            pageable: dict[str, Any] | None = None,
    ) -> list[AccessGriefDto]:
        """Searches access griefs. GET /api/access-grief.

        Args:
            token: JWT bearer token.
            name: Filter by grief name (partial match).
            pageable: {'page': int, 'size': int, 'sort': str}.

        Returns:
            List of AccessGriefDto.
        """
        params: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if name:
            params["name"] = name
        if pageable:
            params.update(pageable)

        logger.debug("Searching access griefs", extra={"name": name})

        result = await self._client._make_request(
            "GET", "api/access-grief", token=token, params=params
        )
        return self._extract_slice_content(result, endpoint="GET api/access-grief")

    async def get_grief(self, token: str, grief_id: str) -> AccessGriefDto | None:
        """Gets a single access grief by UUID. GET /api/access-grief/{id}.

        Args:
            token: JWT bearer token.
            grief_id: AccessGrief UUID string.

        Returns:
            AccessGriefDto or None if 404 Not Found.
        """
        try:
            result = await self._client._make_request(
                "GET", f"api/access-grief/{grief_id}", token=token
            )
            if isinstance(result, dict) and result:
                return AccessGriefDto.model_validate(result)
            return None
        except EdmsNotFoundError:
            logger.info("Access grief not found: %s", grief_id[:8])
            return None

    async def get_grief_employees(
            self,
            token: str,
            grief_id: str,
            pageable: dict[str, Any] | None = None,
    ) -> list[EmployeeAccessGriefDto]:
        """Gets employees with a specific access grief. GET /api/access-grief/{id}/employees.

        Args:
            token: JWT bearer token.
            grief_id: AccessGrief UUID string.
            pageable: Pagination params.

        Returns:
            List of EmployeeAccessGriefDto.
        """
        params: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if pageable:
            params.update(pageable)

        logger.debug("Fetching employees for grief", extra={"grief_id": grief_id[:8]})

        result = await self._client._make_request(
            "GET",
            f"api/access-grief/{grief_id}/employees",
            token=token,
            params=params,
        )
        return self._extract_employees_slice(result)

    # ── Внутренние хелперы ────────────────────────────────────────────

    @staticmethod
    def _extract_slice_content(result: Any, endpoint: str) -> list[AccessGriefDto]:
        if isinstance(result, dict):
            try:
                return SpringSlice[AccessGriefDto].model_validate(result).content
            except Exception:
                if result.get("id"):
                    return [AccessGriefDto.model_validate(result)]
                logger.warning("Unexpected response from %s", endpoint)
                return []
        if isinstance(result, list):
            return [AccessGriefDto.model_validate(item) for item in result]
        return []

    @staticmethod
    def _extract_employees_slice(result: Any) -> list[EmployeeAccessGriefDto]:
        if isinstance(result, dict):
            try:
                return SpringSlice[EmployeeAccessGriefDto].model_validate(result).content
            except Exception:
                logger.warning("Unexpected response from GET api/access-grief/{id}/employees")
                return []
        if isinstance(result, list):
            return [EmployeeAccessGriefDto.model_validate(item) for item in result]
        return []
