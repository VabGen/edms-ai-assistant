# edms_ai_assistant/clients/access_grief_client.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import (
    AccessGriefDto,
    AccessGriefFilter,
    AccessGriefRequest,
    EmployeeAccessGriefDto,
    SliceDto,
)

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

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
        filter: AccessGriefFilter | dict[str, Any] | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[AccessGriefDto]:
        """Searches access griefs."""
        logger.info("Searching access griefs")
        params: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if pageable:
            params.update(pageable)

        if isinstance(filter, AccessGriefFilter):
            params.update(filter.model_dump(exclude_none=True, by_alias=True))
        elif filter:
            params.update(filter)

        return await self._request_list(
            "GET", "api/access-grief", token, AccessGriefDto, params=params
        )

    async def get_grief(
        self, token: str, grief_id: str | UUID
    ) -> AccessGriefDto | None:
        """Gets a single access grief by UUID."""
        logger.info(f"Fetching access grief {grief_id}")
        try:
            return await self._request_dto(
                "GET", f"api/access-grief/{grief_id}", token, AccessGriefDto
            )
        except EdmsNotFoundError:
            logger.error(f"Access grief {grief_id} not found")
            return None

    async def get_grief_employees(
        self,
        token: str,
        grief_id: str | UUID,
        pageable: dict[str, Any] | None = None,
    ) -> SliceDto[EmployeeAccessGriefDto]:
        """Gets employees with a specific access grief."""
        logger.info(f"Fetching employees for access grief {grief_id}")
        params: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if pageable:
            params.update(pageable)

        return await self._request_dto(
            "GET",
            f"api/access-grief/{grief_id}/employees",
            token,
            SliceDto[EmployeeAccessGriefDto],
            params=params,
        )

    async def create_grief(
        self, token: str, request: AccessGriefRequest
    ) -> AccessGriefDto:
        """Creates a new access grief."""
        logger.info("Creating new access grief")
        return await self._request_dto(
            "POST",
            "api/access-grief",
            token,
            AccessGriefDto,
            json_data=request.model_dump(by_alias=True),
        )

    async def update_grief(
        self, token: str, grief_id: str | UUID, request: AccessGriefRequest
    ) -> AccessGriefDto:
        """Updates an existing access grief."""
        logger.info(f"Updating access grief {grief_id}")
        return await self._request_dto(
            "PUT",
            f"api/access-grief/{grief_id}",
            token,
            AccessGriefDto,
            json_data=request.model_dump(by_alias=True),
        )

    async def delete_grief(self, token: str, grief_id: str | UUID) -> None:
        """Deletes an access grief (Note: Java controller throws UnsupportedOperationException currently)."""
        logger.info(f"Deleting access grief {grief_id}")
        await self.make_request(
            "DELETE", f"api/access-grief/{grief_id}", token, is_json_response=False
        )

    async def delete_griefs(self, token: str, grief_ids: list[UUID]) -> None:
        """Deletes a list of access griefs."""
        logger.info(f"Deleting access griefs: {grief_ids}")
        await self.make_request(
            "DELETE",
            "api/access-grief",
            token,
            json_data={"ids": grief_ids},
            is_json_response=False,
        )
