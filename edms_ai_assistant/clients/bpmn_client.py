# edms_ai_assistant/clients/bpmn_client.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.domain.document import (
    BpmnProcessDirectoryDto,
    BpmnProcessDirectoryRequest,
    BpmnSearchRequest,
)
from edms_ai_assistant.domain.employee import SliceDto

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)


class BpmnClient(EdmsBaseClient):
    """Client for EDMS BPMN Process Directory API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def get_definition(self, token: str) -> dict[str, Any]:
        """GET api/bpmn/definition"""
        logger.info("Fetching BPMN process definitions")
        return await self.make_request("GET", "api/bpmn/definition", token=token)

    async def upload(
        self, token: str, request: BpmnProcessDirectoryRequest
    ) -> BpmnProcessDirectoryDto:
        """POST api/bpmn"""
        logger.info(f"Uploading BPMN process directory: {request.name}")
        return await self._request_dto(
            "POST",
            "api/bpmn",
            token,
            BpmnProcessDirectoryDto,
            json_data=request.model_dump(exclude_none=True),
        )

    async def get_xml(self, token: str, bpmn_id: UUID) -> str:
        """GET api/bpmn/{id}/xml"""
        logger.info(f"Downloading XML for BPMN directory {bpmn_id}")
        result = await self.make_request("GET", f"api/bpmn/{bpmn_id}/xml", token=token)
        return result.get("xml", "")

    async def get_all(
        self,
        token: str,
        request: BpmnSearchRequest | None = None,
        page: int = 0,
        size: int = 20,
    ) -> SliceDto[BpmnProcessDirectoryDto]:
        """GET api/bpmn"""
        logger.info("Searching BPMN process directories")
        params = request.model_dump(exclude_none=True) if request else {}
        params.update({"page": page, "size": size})
        return await self._request_dto(
            "GET", "api/bpmn", token, SliceDto[BpmnProcessDirectoryDto], params=params
        )

    async def get_by_id(self, token: str, bpmn_id: UUID) -> BpmnProcessDirectoryDto:
        """GET api/bpmn/{id}"""
        logger.info(f"Fetching BPMN process directory by id: {bpmn_id}")
        return await self._request_dto(
            "GET", f"api/bpmn/{bpmn_id}", token, BpmnProcessDirectoryDto
        )

    async def edit(
        self, token: str, request: BpmnProcessDirectoryRequest
    ) -> BpmnProcessDirectoryDto:
        """PUT api/bpmn"""
        logger.info(f"Editing BPMN process directory: {request.id}")
        return await self._request_dto(
            "PUT",
            "api/bpmn",
            token,
            BpmnProcessDirectoryDto,
            json_data=request.model_dump(exclude_none=True),
        )

    async def copy(self, token: str, ids: list[UUID]) -> list[BpmnProcessDirectoryDto]:
        """POST api/bpmn/copy"""
        logger.info(f"Copying BPMN process directories: {ids}")
        return await self._request_list(
            "POST",
            "api/bpmn/copy",
            token,
            BpmnProcessDirectoryDto,
            json_data={"ids": [str(i) for i in ids]},
        )

    async def delete(self, token: str, bpmn_id: UUID) -> None:
        """DELETE api/bpmn/{id}"""
        logger.info(f"Deleting BPMN process directory: {bpmn_id}")
        await self.make_request(
            "DELETE", f"api/bpmn/{bpmn_id}", token=token, is_json_response=False
        )

    async def delete_list(self, token: str, ids: list[UUID]) -> None:
        """DELETE api/bpmn"""
        logger.info(f"Deleting list of BPMN process directories: {ids}")
        await self.make_request(
            "DELETE",
            "api/bpmn",
            token=token,
            json_data={"ids": [str(i) for i in ids]},
            is_json_response=False,
        )
