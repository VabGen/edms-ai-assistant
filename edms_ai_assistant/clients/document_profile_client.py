# edms_ai_assistant/clients/document_profile_client.py
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.domain.document import (
    DocumentProfileDto,
    DocumentProfileFilter,
    ProfileAttachmentDto,
    ProfileAccessEmployeeDto,
    ProfileAccessGroupDto,
    DocumentProfileAccessEntryDto,
    ProfileContractAttachmentDto,
)
from edms_ai_assistant.domain.employee import SliceDto

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)

class DocumentProfileClient(EdmsBaseClient):
    """Client for EDMS Document Profile API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def get_profiles(
        self,
        token: str,
        filter: DocumentProfileFilter | None = None,
        list_attribute: bool = False,
        page: int = 0,
        size: int = 20
    ) -> list[DocumentProfileDto]:
        """GET api/doc-profile"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        params["listAttribute"] = str(list_attribute).lower()
        params.update({"page": page, "size": size})

        result = await self.make_request("GET", "api/doc-profile", token=token, params=params)
        if isinstance(result, dict) and "content" in result:
            return [DocumentProfileDto.model_validate(item) for item in result["content"]]
        return [DocumentProfileDto.model_validate(item) for item in result]

    async def get_profile(self, token: str, profile_id: UUID) -> DocumentProfileDto:
        """GET api/doc-profile/{id}"""
        return await self._request_dto("GET", f"api/doc-profile/{profile_id}", token, DocumentProfileDto)

    async def get_xml(self, token: str, profile_id: UUID) -> str:
        """GET api/doc-profile/{id}/xml"""
        result = await self.make_request("GET", f"api/doc-profile/{profile_id}/xml", token=token)
        return result.get("xml", "")

    async def get_attachments(self, token: str, profile_id: UUID) -> list[ProfileAttachmentDto]:
        """GET api/doc-profile/{id}/attachment"""
        return await self._request_list("GET", f"api/doc-profile/{profile_id}/attachment", token, ProfileAttachmentDto)

    async def delete_attachment(self, token: str, profile_id: UUID, attachment_id: UUID) -> None:
        """DELETE api/doc-profile/{id}/attachment/{attachmentId}"""
        await self.make_request("DELETE", f"api/doc-profile/{profile_id}/attachment/{attachment_id}", token=token, is_json_response=False)

    async def upload_attachment(self, token: str, profile_id: UUID, file_name: str, file_content: bytes) -> ProfileAttachmentDto:
        """POST api/doc-profile/{id}/attachment"""
        data = await self._upload_file(f"api/doc-profile/{profile_id}/attachment", token, file_name, file_content)
        return ProfileAttachmentDto.model_validate(data)

    async def download_attachment(self, token: str, profile_id: UUID, attachment_id: UUID) -> bytes:
        """GET api/doc-profile/{profileId}/attachment/{id}"""
        return await self.make_request("GET", f"api/doc-profile/{profile_id}/attachment/{attachment_id}", token=token, is_json_response=False)

    async def create_profile(self, token: str, profile: DocumentProfileDto) -> DocumentProfileDto:
        """POST api/doc-profile"""
        return await self._request_dto("POST", "api/doc-profile", token, DocumentProfileDto, json_data=profile.model_dump(exclude_none=True))

    async def update_profile(self, token: str, profile: DocumentProfileDto) -> DocumentProfileDto:
        """PUT api/doc-profile"""
        return await self._request_dto("PUT", "api/doc-profile", token, DocumentProfileDto, json_data=profile.model_dump(exclude_none=True))

    async def copy_profiles(self, token: str, profile_ids: list[UUID]) -> list[DocumentProfileDto]:
        """POST api/doc-profile/copy"""
        return await self._request_list("POST", "api/doc-profile/copy", token, DocumentProfileDto, json_data={"ids": [str(i) for i in profile_ids]})

    async def bpmn_deploy(self, token: str, profile_id: UUID) -> DocumentProfileDto:
        """PUT api/doc-profile/{id}/bpmn-deploy"""
        return await self._request_dto("PUT", f"api/doc-profile/{profile_id}/bpmn-deploy", token, DocumentProfileDto, json_data={"id": str(profile_id)})

    async def delete_profile(self, token: str, profile_id: UUID) -> None:
        """DELETE api/doc-profile/{id}"""
        await self.make_request("DELETE", f"api/doc-profile/{profile_id}", token=token, is_json_response=False)

    async def delete_profiles(self, token: str, profile_ids: list[UUID]) -> None:
        """DELETE api/doc-profile"""
        await self.make_request("DELETE", "api/doc-profile", token=token, json_data={"ids": [str(i) for i in profile_ids]}, is_json_response=False)

    async def get_access_employees(self, token: str, profile_id: UUID, page: int = 0, size: int = 20) -> SliceDto[ProfileAccessEmployeeDto]:
        """GET api/doc-profile/{id}/access-employee"""
        return await self._request_dto(
            "GET",
            f"api/doc-profile/{profile_id}/access-employee",
            token,
            SliceDto[ProfileAccessEmployeeDto],
            params={"page": page, "size": size}
        )

    async def get_access_groups(self, token: str, profile_id: UUID, page: int = 0, size: int = 20) -> SliceDto[ProfileAccessGroupDto]:
        """GET api/doc-profile/{id}/access-group"""
        return await self._request_dto(
            "GET",
            f"api/doc-profile/{profile_id}/access-group",
            token,
            SliceDto[ProfileAccessGroupDto],
            params={"page": page, "size": size}
        )

    async def get_access_entries(self, token: str, profile_id: UUID, page: int = 0, size: int = 20) -> SliceDto[DocumentProfileAccessEntryDto]:
        """GET api/doc-profile/{id}/access-entry"""
        return await self._request_dto(
            "GET",
            f"api/doc-profile/{profile_id}/access-entry",
            token,
            SliceDto[DocumentProfileAccessEntryDto],
            params={"page": page, "size": size}
        )

    async def add_access_entries(self, token: str, profile_id: UUID, entries: list[DocumentProfileAccessEntryDto]) -> None:
        """POST api/doc-profile/{id}/access-entry/batch"""
        await self.make_request(
            "POST",
            f"api/doc-profile/{profile_id}/access-entry/batch",
            token=token,
            json_data=[e.model_dump(exclude_none=True) for e in entries],
            is_json_response=False
        )

    async def delete_access_entries(self, token: str, profile_id: UUID, entry_ids: list[UUID]) -> None:
        """DELETE api/doc-profile/{id}/access-entry/batch"""
        await self.make_request(
            "DELETE",
            f"api/doc-profile/{profile_id}/access-entry/batch",
            token=token,
            json_data={"ids": [str(i) for i in entry_ids]},
            is_json_response=False
        )

    async def sync_access_entries(self, token: str, profile_id: UUID, entries: list[DocumentProfileAccessEntryDto]) -> None:
        """POST api/doc-profile/{id}/access-entry/sync"""
        await self.make_request(
            "POST",
            f"api/doc-profile/{profile_id}/access-entry/sync",
            token=token,
            json_data=[e.model_dump(exclude_none=True) for e in entries],
            is_json_response=False
        )

    async def remove_access_entries_sync(self, token: str, profile_id: UUID, entries: list[DocumentProfileAccessEntryDto]) -> None:
        """DELETE api/doc-profile/{id}/access-entry/sync"""
        await self.make_request(
            "DELETE",
            f"api/doc-profile/{profile_id}/access-entry/sync",
            token=token,
            json_data=[e.model_dump(exclude_none=True) for e in entries],
            is_json_response=False
        )

    async def add_groups(self, token: str, profile_id: UUID, group_ids: list[UUID]) -> None:
        """POST api/doc-profile/{id}/access-group/batch"""
        await self.make_request(
            "POST",
            f"api/doc-profile/{profile_id}/access-group/batch",
            token=token,
            json_data={"ids": [str(i) for i in group_ids]},
            is_json_response=False
        )

    async def add_employees(self, token: str, profile_id: UUID, employee_ids: list[UUID]) -> None:
        """POST api/doc-profile/{id}/access-employee/batch"""
        await self.make_request(
            "POST",
            f"api/doc-profile/{profile_id}/access-employee/batch",
            token=token,
            json_data={"ids": [str(i) for i in employee_ids]},
            is_json_response=False
        )

    async def delete_groups(self, token: str, profile_id: UUID, group_ids: list[UUID]) -> None:
        """DELETE api/doc-profile/{id}/access-group/batch"""
        await self.make_request(
            "DELETE",
            f"api/doc-profile/{profile_id}/access-group/batch",
            token=token,
            json_data={"ids": [str(i) for i in group_ids]},
            is_json_response=False
        )

    async def delete_employees(self, token: str, profile_id: UUID, employee_ids: list[UUID]) -> None:
        """DELETE api/doc-profile/{id}/access-employee/batch"""
        await self.make_request(
            "DELETE",
            f"api/doc-profile/{profile_id}/access-employee/batch",
            token=token,
            json_data={"ids": [str(i) for i in employee_ids]},
            is_json_response=False
        )

    async def get_contract_attachments(self, token: str, profile_id: UUID) -> list[ProfileContractAttachmentDto]:
        """GET api/doc-profile/{id}/contract-attachment"""
        return await self._request_list("GET", f"api/doc-profile/{profile_id}/contract-attachment", token, ProfileContractAttachmentDto)

    async def add_contract_attachments(self, token: str, profile_id: UUID, attachment_ids: list[UUID]) -> None:
        """POST api/doc-profile/{id}/contract-attachment"""
        await self.make_request(
            "POST",
            f"api/doc-profile/{profile_id}/contract-attachment",
            token=token,
            json_data={"ids": [str(i) for i in attachment_ids]},
            is_json_response=False
        )

    async def delete_contract_attachment(self, token: str, attachment_id: UUID) -> None:
        """DELETE api/doc-profile/contract-attachment/{id}"""
        await self.make_request("DELETE", f"api/doc-profile/contract-attachment/{attachment_id}", token=token, is_json_response=False)

    async def upload_contract_attachment(self, token: str, profile_id: UUID, file_name: str, file_content: bytes) -> ProfileContractAttachmentDto:
        """POST api/doc-profile/{id}/contract-attachment"""
        data = await self._upload_file(f"api/doc-profile/{profile_id}/contract-attachment", token, file_name, file_content)
        return ProfileContractAttachmentDto.model_validate(data)

    async def download_contract_attachment(self, token: str, attachment_id: UUID) -> bytes:
        """GET api/doc-profile/contract-attachment/{id}/download"""
        return await self.make_request("GET", f"api/doc-profile/contract-attachment/{attachment_id}/download", token=token, is_json_response=False)
