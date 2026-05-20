# edms_ai_assistant/clients/access_grief_client.py
from __future__ import annotations

import logging
from typing import Any

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.config import EdmsSettings
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import AccessGriefDto, EmployeeAccessGriefDto

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 50


class AccessGriefClient(EdmsBaseClient):
    """Client for EDMS Access Grief API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def search_griefs(
            self,
            token: str,
            name: str | None = None,
            pageable: dict[str, Any] | None = None,
    ) -> list[AccessGriefDto]:
        """Searches access griefs."""
        params: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if name:
            params["name"] = name
        if pageable:
            params.update(pageable)

        return await self._request_list("GET", "api/access-grief", token, AccessGriefDto, params=params)

    async def get_grief(self, token: str, grief_id: str) -> AccessGriefDto | None:
        """Gets a single access grief by UUID."""
        try:
            return await self._request_dto("GET", f"api/access-grief/{grief_id}", token, AccessGriefDto)
        except EdmsNotFoundError:
            return None

    async def get_grief_employees(
            self,
            token: str,
            grief_id: str,
            pageable: dict[str, Any] | None = None,
    ) -> list[EmployeeAccessGriefDto]:
        """Gets employees with a specific access grief."""
        params: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if pageable:
            params.update(pageable)

        return await self._request_list(
            "GET",
            f"api/access-grief/{grief_id}/employees",
            token,
            EmployeeAccessGriefDto,
            params=params,
        )
