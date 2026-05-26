# edms_ai_assistant/clients/correspondent_client.py
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.domain.document import (
    CorrespondentDto,
    CorrespondentAddRequest,
    CorrespondentUpdateRequest,
    CorrespondentGroupDto,
    CorrespondentGroupAddRequest,
    CorrespondentGroupUpdateRequest,
    IntermediateCorrespondentDto,
    IntermediateCorrespondentRequest,
)
from edms_ai_assistant.domain.reference import BasicSearchRequest

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)

class CorrespondentClient(EdmsBaseClient):
    """Клиент для работы с адресатами и корреспондентами."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    # ══════════════════════════════════════════════════════════════════════════════
    # Correspondents
    # ══════════════════════════════════════════════════════════════════════════════

    async def get_correspondents(
        self, token: str, filter_params: dict[str, Any] | None = None, page: int = 0, size: int = 20
    ) -> SliceDto[CorrespondentDto]:
        """GET api/correspondent"""
        logger.info("Searching correspondents")
        params = filter_params or {}
        params.update({"page": page, "size": size})
        return await self._request_dto("GET", "api/correspondent", token, SliceDto[CorrespondentDto], params=params)

    async def get_correspondent(self, token: str, correspondent_id: UUID) -> CorrespondentDto:
        """GET api/correspondent/{id}"""
        logger.info(f"Fetching correspondent by id: {correspondent_id}")
        result = await self.make_request("GET", f"api/correspondent/{correspondent_id}", token=token)
        return CorrespondentDto.model_validate(result)

    async def create_correspondent(self, token: str, request: CorrespondentAddRequest) -> CorrespondentDto:
        """POST api/correspondent/v2"""
        logger.info("Creating new correspondent")
        result = await self.make_request(
            "POST", "api/correspondent/v2", token=token, json_data=request.model_dump(exclude_none=True)
        )
        return CorrespondentDto.model_validate(result)

    async def update_correspondent(self, token: str, request: CorrespondentUpdateRequest) -> CorrespondentDto:
        """PUT api/correspondent/v2"""
        logger.info(f"Updating correspondent {request.correspondent.id}")
        result = await self.make_request(
            "PUT", "api/correspondent/v2", token=token, json_data=request.model_dump(exclude_none=True)
        )
        return CorrespondentDto.model_validate(result)

    async def delete_correspondent(self, token: str, correspondent_id: UUID):
        """DELETE api/correspondent/{id}"""
        logger.info(f"Deleting correspondent {correspondent_id}")
        await self.make_request("DELETE", f"api/correspondent/{correspondent_id}", token=token)

    async def delete_correspondents_batch(self, token: str, ids: list[UUID]):
        """DELETE api/correspondent"""
        logger.info(f"Deleting correspondents: {ids}")
        await self.make_request("DELETE", "api/correspondent", token=token, json_data={"ids": [str(i) for i in ids]})

    async def search_correspondent_fts(self, token: str, fts: str) -> CorrespondentDto:
        """GET api/correspondent/fts-name"""
        logger.info(f"Searching correspondent (FTS): {fts}")
        result = await self.make_request("GET", "api/correspondent/fts-name", token=token, params={"fts": fts})
        return CorrespondentDto.model_validate(result)

    async def export_correspondents(self, token: str) -> bytes:
        """GET api/correspondent/export"""
        logger.info("Exporting correspondents to XLSX")
        return await self.make_request(
            "GET", "api/correspondent/export", token=token, is_json_response=False, long_timeout=True
        )

    async def get_import_template(self, token: str) -> bytes:
        """GET api/correspondent/template"""
        logger.info("Fetching correspondent import template")
        return await self.make_request(
            "GET", "api/correspondent/template", token=token, is_json_response=False
        )

    async def import_correspondents(self, token: str, file_content: bytes, filename: str):
        """POST api/correspondent/import"""
        logger.info("Importing correspondents from file")
        await self.make_request(
            "POST",
            "api/correspondent/import",
            token=token,
            files={"file": (filename, file_content)},
            is_json_response=False,
        )

    # ══════════════════════════════════════════════════════════════════════════════
    # Correspondent Groups
    # ══════════════════════════════════════════════════════════════════════════════

    async def get_correspondent_groups(
        self, token: str, search: BasicSearchRequest | None = None, page: int = 0, size: int = 20
    ) -> SliceDto[CorrespondentGroupDto]:
        """GET api/correspondent-group"""
        logger.info("Searching correspondent groups")
        params = search.model_dump(exclude_none=True) if search else {}
        params.update({"page": page, "size": size})
        return await self._request_dto("GET", "api/correspondent-group", token, SliceDto[CorrespondentGroupDto], params=params)

    async def get_correspondent_group(self, token: str, group_id: UUID) -> CorrespondentGroupDto:
        """GET api/correspondent-group/{id}"""
        logger.info(f"Fetching correspondent group by id: {group_id}")
        result = await self.make_request("GET", f"api/correspondent-group/{group_id}", token=token)
        return CorrespondentGroupDto.model_validate(result)

    async def create_correspondent_group(self, token: str, request: CorrespondentGroupAddRequest) -> CorrespondentGroupDto:
        """POST api/correspondent-group/v2"""
        logger.info("Creating new correspondent group")
        result = await self.make_request(
            "POST", "api/correspondent-group/v2", token=token, json_data=request.model_dump(exclude_none=True)
        )
        return CorrespondentGroupDto.model_validate(result)

    async def update_correspondent_group(self, token: str, request: CorrespondentGroupUpdateRequest) -> CorrespondentGroupDto:
        """PUT api/correspondent-group/v2"""
        logger.info(f"Updating correspondent group {request.correspondent_group.id}")
        result = await self.make_request(
            "PUT", "api/correspondent-group/v2", token=token, json_data=request.model_dump(exclude_none=True)
        )
        return CorrespondentGroupDto.model_validate(result)

    async def get_all_in_group(self, token: str, group_id: UUID) -> list[CorrespondentDto]:
        """GET api/correspondent-group/{groupId}/all"""
        logger.info(f"Fetching all correspondents in group {group_id}")
        result = await self.make_request("GET", f"api/correspondent-group/{group_id}/all", token=token)
        return [CorrespondentDto.model_validate(item) for item in result]

    async def get_group_links(
        self, token: str, group_id: UUID, page: int = 0, size: int = 20
    ) -> SliceDto[IntermediateCorrespondentDto]:
        """GET api/correspondent-group/group/{groupId}"""
        logger.info(f"Fetching group links for group {group_id}")
        return await self._request_dto(
            "GET", f"api/correspondent-group/group/{group_id}", token, SliceDto[IntermediateCorrespondentDto], params={"page": page, "size": size}
        )

    async def save_group_links(self, token: str, request: IntermediateCorrespondentRequest) -> list[IntermediateCorrespondentDto]:
        """POST api/correspondent-group/link"""
        logger.info(f"Saving group links for group {request.id}")
        result = await self.make_request(
            "POST", "api/correspondent-group/link", token=token, json_data=request.model_dump(exclude_none=True)
        )
        return [IntermediateCorrespondentDto.model_validate(item) for item in result]

    async def delete_group_links(self, token: str, request: IntermediateCorrespondentRequest):
        """DELETE api/correspondent-group/link"""
        logger.info(f"Deleting group links for group {request.id}")
        await self.make_request(
            "DELETE", "api/correspondent-group/link", token=token, json_data=request.model_dump(exclude_none=True)
        )

    async def delete_correspondent_groups_batch(self, token: str, ids: list[UUID]):
        """DELETE api/correspondent-group"""
        logger.info(f"Deleting correspondent groups: {ids}")
        await self.make_request("DELETE", "api/correspondent-group", token=token, json_data={"ids": [str(i) for i in ids]})
