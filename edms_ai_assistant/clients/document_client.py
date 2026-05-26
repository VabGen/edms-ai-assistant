# edms_ai_assistant/clients/document_client.py
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
from uuid import UUID
from datetime import datetime

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.document import (
    BpmnProcessActivityDto,
    ControlDto,
    DocPermissionContainer,
    DocumentDto,
    DocumentHistoryDto,
    DocumentPropertiesDto,
    DocumentRecipientDto,
    DocumentVersionDto,
    DocumentWithPermissions,
    ExecutionDocumentStatCount,
    TasksAndProjectsDto,
    UserSmdoStat,
    DocumentUserColorDto,
    DocumentAccessEntryDto,
    CountResult,
    DocumentNextProcessRequest,
    DocumentCancelAction,
    RepeatIdenticalAppealDto,
    ContractVersionInfoDto,
    NomenclatureAffairDto,
    DocumentAismvRecreateRequest,
    DocumentBasedExistingBody,
    DocumentRecipientDeliveryHistoryDto,
    ContractControlPointDto,
    ContractControlPointFilter,
    ContractControlPointResponsibleDto,
    ContractControlPointLinkDto,
    ContractControlPointAttachmentDto,
    ControlPointMainFields,
    ControlPointRevisionRequest,
    ControlPointWithPermission,
)

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 10

FULL_DOC_INCLUDES: list[str] = [
    "DOCUMENT_TYPE",
    "DELIVERY_METHOD",
    "CORRESPONDENT",
    "RECIPIENT",
    "PRE_NOMENCLATURE_AFFAIRS",
    "CITIZEN_TYPE",
    "REGISTRATION_JOURNAL",
    "CURRENCY",
    "SOLUTION_RESULT",
    "PARENT_SUBJECT",
    "ADDITIONAL_DOCUMENT_AND_TYPE",
]

SEARCH_DOC_INCLUDES: list[str] = [
    "DOCUMENT_TYPE",
    "CORRESPONDENT",
    "REGISTRATION_JOURNAL",
]


class DocumentClient(EdmsBaseClient):
    """Client for EDMS Document API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def search_documents(
            self,
            token: str,
            doc_filter: dict[str, Any] | None = None,
            pageable: dict[str, Any] | None = None,
            includes: list[str] | None = None,
    ) -> list[DocumentDto]:
        """Searches documents. Returns list of DocumentDto."""
        logger.info("Searching documents with filter: %s", doc_filter)
        params: dict[str, Any] = {
            "page": _DEFAULT_PAGE,
            "size": _DEFAULT_SIZE,
            "includes": includes or SEARCH_DOC_INCLUDES
        }
        if pageable:
            params.update(pageable)
        if doc_filter:
            params.update(doc_filter)

        return await self._request_list(
            "GET", "api/document", token, DocumentDto, params=params
        )

    async def get_document_metadata(
            self,
            token: str,
            document_id: UUID | str,
            includes: list[str] | None = None,
    ) -> DocumentDto | None:
        """Fetches full document metadata by UUID."""
        logger.info("Fetching document metadata for: %s", document_id)
        params = {"includes": includes or FULL_DOC_INCLUDES}

        try:
            return await self._request_dto(
                "GET", f"api/document/{document_id}", token, DocumentDto, params=params
            )
        except EdmsNotFoundError:
            logger.info("Document not found: %s", document_id)
            return None

    async def get_document_with_permissions(
            self,
            token: str,
            document_id: UUID | str,
            includes: list[str] | None = None,
    ) -> DocumentWithPermissions | None:
        """Fetches document and its permissions in a single request."""
        params = {"includes": includes or FULL_DOC_INCLUDES}

        try:
            return await self._request_dto(
                "GET", f"api/document/{document_id}/all", token, DocumentWithPermissions, params=params
            )
        except EdmsNotFoundError:
            return None

    async def get_document_permissions(
            self, token: str, document_id: UUID | str
    ) -> DocPermissionContainer | None:
        """Fetches DocPermissionContainer for a document."""
        try:
            return await self._request_dto(
                "GET", f"api/document/{document_id}/permission", token, DocPermissionContainer
            )
        except EdmsNotFoundError:
            return None

    async def get_document_properties(
            self, token: str, document_id: UUID | str
    ) -> DocumentPropertiesDto | None:
        """Fetches extended document properties."""
        try:
            return await self._request_dto(
                "GET", f"api/document/{document_id}/properties", token, DocumentPropertiesDto
            )
        except EdmsNotFoundError:
            return None

    async def get_document_history(
            self, token: str, document_id: UUID | str
    ) -> list[DocumentHistoryDto]:
        """Fetches document processing protocol (history v1)."""
        try:
            return await self._request_list(
                "GET", f"api/document/{document_id}/history", token, DocumentHistoryDto
            )
        except EdmsNotFoundError:
            return []

    async def get_document_history_v2(
            self, token: str, document_id: UUID | str
    ) -> list[DocumentHistoryDto]:
        """Fetches document processing protocol (history v2, preferred)."""
        try:
            return await self._request_list(
                "GET", f"api/document/{document_id}/history/v2", token, DocumentHistoryDto
            )
        except EdmsNotFoundError:
            return []

    async def get_process_activity(
            self, token: str, document_id: UUID | str
    ) -> BpmnProcessActivityDto | None:
        """Fetches current BPMN process with active activities."""
        try:
            return await self._request_dto(
                "GET", f"api/document/{document_id}/bpmn", token, BpmnProcessActivityDto
            )
        except EdmsNotFoundError:
            return None

    async def get_tasks_and_projects(
            self, token: str, document_id: UUID | str
    ) -> TasksAndProjectsDto | None:
        """Fetches tasks and task-projects linked to a document."""
        try:
            return await self._request_dto(
                "GET", f"api/document/{document_id}/task-task-project", token, TasksAndProjectsDto
            )
        except EdmsNotFoundError:
            return None

    async def get_document_control(
            self, token: str, document_id: UUID | str
    ) -> ControlDto | None:
        """Fetches current control record (ControlDto) for a document."""
        try:
            return await self._request_dto(
                "GET", f"api/document/{document_id}/control", token, ControlDto
            )
        except EdmsNotFoundError:
            return None

    async def set_document_control(
            self,
            token: str,
            document_id: UUID | str,
            control_request: dict[str, Any],
    ) -> ControlDto:
        """Sets a document on control."""
        return await self._request_dto(
            "POST",
            f"api/document/{document_id}/control",
            token,
            ControlDto,
            json_data=control_request,
        )

    async def change_document_control(
            self,
            token: str,
            document_id: UUID | str,
            control_request: dict[str, Any],
    ) -> ControlDto:
        """Updates document control settings."""
        return await self._request_dto(
            "PUT",
            f"api/document/{document_id}/control",
            token,
            ControlDto,
            json_data=control_request,
        )

    async def remove_document_control(self, token: str, document_id: UUID | str) -> None:
        """Removes control mark. Raises on failure."""
        await self.make_request(
            "PUT",
            "api/document/control",
            token,
            json_data={"id": str(document_id)},
            is_json_response=False,
        )

    async def delete_document_control(self, token: str, document_id: UUID | str) -> None:
        """Deletes control record. Raises on failure."""
        await self.make_request(
            "DELETE",
            f"api/document/{document_id}/control",
            token,
            is_json_response=False,
        )

    # ══════════════════════════════════════════════════════════════════════════════
    # Contract Control Points
    # ══════════════════════════════════════════════════════════════════════════════

    async def get_control_points(
        self, token: str, document_id: UUID, filter: ContractControlPointFilter | None = None
    ) -> list[ContractControlPointDto]:
        """GET api/document/{documentId}/control-point"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        return await self._request_list(
            "GET", f"api/document/{document_id}/control-point", token, ContractControlPointDto, params=params
        )

    async def get_control_point(self, token: str, document_id: UUID, point_id: UUID) -> ContractControlPointDto:
        """GET api/document/{documentId}/control-point/{id}"""
        return await self._request_dto(
            "GET", f"api/document/{document_id}/control-point/{point_id}", token, ContractControlPointDto
        )

    async def get_control_point_with_permission(
        self, token: str, document_id: UUID, point_id: UUID
    ) -> ControlPointWithPermission:
        """GET api/document/{documentId}/control-point/{id}/all"""
        return await self._request_dto(
            "GET", f"api/document/{document_id}/control-point/{point_id}/all", token, ControlPointWithPermission
        )

    async def get_control_point_permissions(self, token: str, document_id: UUID, point_id: UUID) -> list[Any]:
        """GET api/document/{documentId}/control-point/{id}/permission"""
        return await self.make_request(
            "GET", f"api/document/{document_id}/control-point/{point_id}/permission", token
        )

    async def create_control_point(
        self, token: str, document_id: UUID, fields: ControlPointMainFields
    ) -> ContractControlPointDto:
        """POST api/document/{documentId}/control-point"""
        return await self._request_dto(
            "POST",
            f"api/document/{document_id}/control-point",
            token,
            ContractControlPointDto,
            json_data=fields.model_dump(exclude_none=True),
        )

    async def execute_control_point_operations(
        self, token: str, document_id: UUID, point_id: UUID, operations: list[dict[str, Any]]
    ) -> None:
        """POST api/document/{documentId}/control-point/{id}"""
        await self.make_request(
            "POST", f"api/document/{document_id}/control-point/{point_id}", token, json_data=operations, is_json_response=False
        )

    async def complete_control_point(self, token: str, document_id: UUID, point_id: UUID) -> ContractControlPointDto:
        """PUT api/document/{documentId}/control-point/{id}/complete"""
        return await self._request_dto(
            "PUT", f"api/document/{document_id}/control-point/{point_id}/complete", token, ContractControlPointDto
        )

    async def revision_control_point(
        self, token: str, document_id: UUID, point_id: UUID, request: ControlPointRevisionRequest
    ) -> ContractControlPointDto:
        """PUT api/document/{documentId}/control-point/{id}/revision"""
        return await self._request_dto(
            "PUT",
            f"api/document/{document_id}/control-point/{point_id}/revision",
            token,
            ContractControlPointDto,
            json_data=request.model_dump(exclude_none=True),
        )

    async def delete_control_point(self, token: str, document_id: UUID, point_id: UUID) -> None:
        """DELETE api/document/{documentId}/control-point/{id}"""
        await self.make_request(
            "DELETE", f"api/document/{document_id}/control-point/{point_id}", token, is_json_response=False
        )

    async def move_control_point(
        self, token: str, document_id: UUID, move_id: UUID, target_id: UUID
    ) -> list[ContractControlPointDto]:
        """PUT api/document/{documentId}/control-point/move/{moveId}/{targetId}"""
        return await self._request_list(
            "PUT",
            f"api/document/{document_id}/control-point/move/{move_id}/{target_id}",
            token,
            ContractControlPointDto,
        )

    async def get_control_point_responsibles(
        self, token: str, document_id: UUID, point_id: UUID
    ) -> list[ContractControlPointResponsibleDto]:
        """GET api/document/{documentId}/control-point/{id}/responsible"""
        return await self._request_list(
            "GET",
            f"api/document/{document_id}/control-point/{point_id}/responsible",
            token,
            ContractControlPointResponsibleDto,
        )

    async def get_control_point_links(self, token: str, document_id: UUID, point_id: UUID) -> list[ContractControlPointLinkDto]:
        """GET api/document/{documentId}/control-point/{id}/link"""
        return await self._request_list(
            "GET", f"api/document/{document_id}/control-point/{point_id}/link", token, ContractControlPointLinkDto
        )

    async def get_control_point_attachments(
        self, token: str, document_id: UUID, point_id: UUID
    ) -> list[ContractControlPointAttachmentDto]:
        """GET api/document/{documentId}/control-point/{id}/attachment"""
        return await self._request_list(
            "GET", f"api/document/{document_id}/control-point/{point_id}/attachment", token, ContractControlPointAttachmentDto
        )

    async def download_control_point_attachment(self, token: str, document_id: UUID, attach_id: UUID) -> bytes:
        """GET api/document/{documentId}/control-point/attachment/{attachId}/download"""
        return await self.make_request(
            "GET",
            f"api/document/{document_id}/control-point/attachment/{attach_id}/download",
            token=token,
            is_json_response=False,
            long_timeout=True,
        )

    async def delete_control_point_attachment(self, token: str, document_id: UUID, point_id: UUID, attach_id: UUID) -> None:
        """DELETE api/document/{documentId}/control-point/{pointId}/attachment/{attachId}"""
        await self.make_request(
            "DELETE",
            f"api/document/{document_id}/control-point/{point_id}/attachment/{attach_id}",
            token=token,
            is_json_response=False,
        )

    async def get_document_colors(self, token: str) -> list[DocumentUserColorDto]:
        """Fetches all document colors."""
        return await self._request_list("GET", "api/document/color", token, DocumentUserColorDto)

    async def create_document_color(
            self,
            token: str,
            document_id: UUID | str,
            color: str,
    ) -> DocumentUserColorDto:
        """Sets a color for a document."""
        return await self._request_dto(
            "POST",
            f"api/document/{document_id}/color",
            token,
            DocumentUserColorDto,
            json_data={"color": color, "documentId": str(document_id)},
        )

    async def update_document_color(
            self,
            token: str,
            document_id: UUID | str,
            color_id: UUID | str,
            color: str,
    ) -> DocumentUserColorDto:
        """Updates a document color."""
        return await self._request_dto(
            "PUT",
            f"api/document/{document_id}/color",
            token,
            DocumentUserColorDto,
            json_data={"id": str(color_id), "color": color, "documentId": str(document_id)},
        )

    async def create_document_colors_batch(
            self,
            token: str,
            colors: list[DocumentUserColorDto],
    ) -> list[DocumentUserColorDto]:
        """Creates or updates document colors in batch."""
        return await self._request_list(
            "POST",
            "api/document/color/batch",
            token,
            DocumentUserColorDto,
            json_data=[c.model_dump(by_alias=True) for c in colors],
        )

    async def delete_document_colors(self, token: str, color_ids: list[UUID | str]) -> None:
        """Deletes specified colors."""
        await self.make_request(
            "DELETE",
            "api/document/color",
            token,
            json_data={"ids": [str(cid) for cid in color_ids]},
            is_json_response=False,
        )

    async def get_access_entries(self, token: str, document_id: UUID | str) -> list[DocumentAccessEntryDto]:
        """Fetches access list entries for a document."""
        return await self._request_list(
            "GET", f"api/document/{document_id}/access-entry", token, DocumentAccessEntryDto
        )

    async def add_access_entries(
            self,
            token: str,
            document_id: UUID | str,
            entries: list[DocumentAccessEntryDto],
    ) -> None:
        """Adds access list entries."""
        await self.make_request(
            "POST",
            f"api/document/{document_id}/access-entry/batch",
            token,
            json_data=[e.model_dump(by_alias=True) for e in entries],
            is_json_response=False,
        )

    async def delete_access_entries(
            self,
            token: str,
            document_id: UUID | str,
            entry_ids: list[UUID | str],
    ) -> None:
        """Deletes access list entries."""
        await self.make_request(
            "DELETE",
            f"api/document/{document_id}/access-entry/batch",
            token,
            json_data={"ids": [str(eid) for eid in entry_ids]},
            is_json_response=False,
        )

    async def delete_document(self, token: str, document_id: UUID | str) -> None:
        """Deletes a single document."""
        await self.make_request(
            "DELETE",
            f"api/document/{document_id}",
            token,
            is_json_response=False,
        )

    async def delete_documents_batch(self, token: str, document_ids: list[UUID | str]) -> CountResult:
        """Deletes documents in batch."""
        return await self._request_dto(
            "DELETE",
            "api/document",
            token,
            CountResult,
            json_data={"ids": [str(did) for did in document_ids]},
        )

    async def create_document(self, token: str, profile_id: UUID | str) -> DocumentWithPermissions:
        """Creates a new document from profile."""
        return await self._request_dto(
            "POST",
            "api/document",
            token,
            DocumentWithPermissions,
            json_data={"id": str(profile_id)},
        )

    async def create_new_version(
            self,
            token: str,
            doc_id: UUID | str,
            body: DocumentBasedExistingBody,
    ) -> DocumentDto:
        """Creates a new version of an existing document."""
        return await self._request_dto(
            "POST",
            f"api/document/{doc_id}/version",
            token,
            DocumentDto,
            json_data=body.model_dump(by_alias=True),
        )

    async def get_document_nomenclatures(self, token: str, document_id: UUID | str) -> list[NomenclatureAffairDto]:
        """Fetches affairs where document is added (affairs)."""
        return await self._request_list(
            "GET", f"api/document/{document_id}/nomenclature-affair", token, NomenclatureAffairDto
        )

    async def get_nomenclature(self, token: str, document_id: UUID | str) -> list[Any]:
        """Fetches nomenclature list for a document."""
        return await self.make_request("GET", f"api/document/{document_id}/nomenclature", token)

    async def write_off_document(
            self,
            token: str,
            document_id: UUID | str,
            affair_ids: list[UUID | str],
    ) -> list[Any]:
        """Writes off document to nomenclature affairs."""
        return await self.make_request(
            "PUT",
            f"api/document/{document_id}/write-off",
            token,
            json_data={"ids": [str(aid) for aid in affair_ids]},
        )

    async def extract_nomenclature(
            self,
            token: str,
            document_id: UUID | str,
            affair_ids: list[UUID | str],
    ) -> None:
        """Extracts document from nomenclature affairs."""
        await self.make_request(
            "PUT",
            f"api/document/{document_id}/extract-nomenclature",
            token,
            json_data={"ids": [str(aid) for aid in affair_ids]},
            is_json_response=False,
        )

    async def set_archive_state(
            self,
            token: str,
            document_id: UUID | str,
            in_archive: bool,
    ) -> None:
        """Sets document archive state."""
        await self.make_request(
            "PUT",
            f"api/document/{document_id}/archive",
            token,
            json_data={"inArchive": in_archive},
            is_json_response=False,
        )

    async def archive_documents_batch(self, token: str, document_ids: list[UUID | str]) -> CountResult:
        """Moves documents to archive in batch."""
        return await self._request_dto(
            "PUT",
            "api/document/archive",
            token,
            CountResult,
            json_data={"ids": [str(did) for did in document_ids]},
        )

    async def extract_archive_batch(self, token: str, document_ids: list[UUID | str]) -> CountResult:
        """Extracts documents from archive in batch."""
        return await self._request_dto(
            "DELETE",
            "api/document/archive",
            token,
            CountResult,
            json_data={"ids": [str(did) for did in document_ids]},
        )

    async def get_contract_version_info(self, token: str, document_id: UUID | str) -> list[ContractVersionInfoDto]:
        """Fetches contract version info."""
        return await self._request_list(
            "GET", f"api/document/{document_id}/contract-version-info", token, ContractVersionInfoDto
        )

    async def get_view_documents(
            self,
            token: str,
            view_id: UUID | str,
            pageable: dict[str, Any] | None = None,
            doc_filter: dict[str, Any] | None = None,
    ) -> list[DocumentDto]:
        """Fetches documents for a custom view."""
        params = pageable or {}
        if doc_filter:
            params.update(doc_filter)
        return await self._request_list(
            "GET", f"api/document/view/{view_id}/entry", token, DocumentDto, params=params
        )

    async def get_document_years(self, token: str, doc_filter: dict[str, Any] | None = None) -> list[int]:
        """Fetches years for which documents exist."""
        return await self.make_request("GET", "api/document/year", token, params=doc_filter)

    async def find_recipients(self, token: str, doc_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Searches for document recipients."""
        return await self.make_request("GET", "api/document/recipient", token, params=doc_filter)

    async def find_statuses(self, token: str, doc_filter: dict[str, Any] | None = None) -> list[str]:
        """Searches for document statuses."""
        return await self.make_request("GET", "api/document/status", token, params=doc_filter)

    async def get_nomenclature_affair_v2(self, token: str, document_id: UUID | str) -> list[NomenclatureAffairDto]:
        """Fetches nomenclature affairs for a document (v2)."""
        return await self._request_list(
            "GET", f"api/document/{document_id}/nomenclature/v2", token, NomenclatureAffairDto
        )

    async def get_contract_responsibles(self, token: str, document_id: UUID | str) -> list[Any]:
        """Fetches responsible persons for a contract."""
        return await self.make_request("GET", f"api/document/{document_id}/responsible", token)

    async def get_online_users(self, token: str, document_id: UUID | str) -> list[UUID]:
        """Fetches IDs of users currently viewing the document."""
        data = await self.make_request("GET", f"api/document/{document_id}/online-user", token)
        return [UUID(uid) for uid in data]

    async def get_nomenclature_affair_with_links(self, token: str, document_id: UUID | str) -> dict[str, Any]:
        """Fetches nomenclature affairs and document links."""
        return await self.make_request("GET", f"api/document/{document_id}/nomenclature-affair-document-link", token)

    async def get_aismv_ack_delivery(self, token: str, document_id: UUID | str) -> list[Any]:
        """Fetches AISMV acknowledgement delivery history."""
        return await self.make_request("GET", f"api/document/{document_id}/aismv-ack-delivery", token)

    async def retry_aismv_ack_delivery(self, token: str, document_id: UUID | str, delivery_id: UUID | str) -> None:
        """Retries AISMV acknowledgement delivery."""
        await self.make_request(
            "POST", f"api/document/{document_id}/aismv-ack-delivery/{delivery_id}/retry", token, is_json_response=False
        )

    async def get_recipient_aismv_delivery_history(
            self, token: str, document_id: UUID | str, recipient_id: UUID | str
    ) -> list[DocumentRecipientDeliveryHistoryDto]:
        """Fetches AISMV delivery history for a specific recipient."""
        return await self._request_list(
            "GET",
            f"api/document/{document_id}/recipient/{recipient_id}/aismv-delivery-history",
            token,
            DocumentRecipientDeliveryHistoryDto
        )

    async def cancel_recipient_aismv_delivery(
            self, token: str, document_id: UUID | str, recipient_id: UUID | str, delivery_id: UUID | str
    ) -> None:
        """Cancels AISMV delivery for a recipient."""
        await self.make_request(
            "POST",
            f"api/document/{document_id}/recipient/{recipient_id}/aismv-delivery-history/{delivery_id}/cancel",
            token,
            is_json_response=False
        )

    async def confirm_recipient_aismv_delivery(
            self, token: str, document_id: UUID | str, recipient_id: UUID | str, delivery_id: UUID | str
    ) -> None:
        """Confirms AISMV delivery for a recipient."""
        await self.make_request(
            "POST",
            f"api/document/{document_id}/recipient/{recipient_id}/aismv-delivery-history/{delivery_id}/confirm",
            token,
            is_json_response=False
        )

    async def retry_recipient_aismv_delivery(
            self, token: str, document_id: UUID | str, recipient_id: UUID | str, delivery_id: UUID | str
    ) -> None:
        """Retries AISMV delivery for a recipient."""
        await self.make_request(
            "POST",
            f"api/document/{document_id}/recipient/{recipient_id}/aismv-delivery-history/{delivery_id}/retry",
            token,
            is_json_response=False
        )

    async def get_not_sent_recipients(self, token: str, document_id: UUID | str) -> list[DocumentRecipientDto]:
        """Fetches recipients to whom the document has not been sent yet."""
        return await self._request_list(
            "GET", f"api/document/{document_id}/not-sent-recipient", token, DocumentRecipientDto
        )

    async def get_recipient_history(
            self, token: str, document_id: UUID | str, recipient_id: UUID | str
    ) -> list[DocumentRecipientDeliveryHistoryDto]:
        """Fetches delivery history for a recipient."""
        return await self._request_list(
            "GET",
            f"api/document/{document_id}/recipient/{recipient_id}/history",
            token,
            DocumentRecipientDeliveryHistoryDto
        )

    async def recipient_add_documents(
            self, token: str, document_id: UUID | str, recipient_id: UUID | str, attachment_ids: list[UUID | str]
    ) -> None:
        """Sends additional documents to a recipient."""
        await self.make_request(
            "POST",
            f"api/document/{document_id}/recipient/{recipient_id}/add-documents",
            token,
            json_data={"ids": [str(aid) for aid in attachment_ids]},
            is_json_response=False
        )

    async def get_repeat_identical_appeals(self, token: str, document_id: UUID | str) -> list[RepeatIdenticalAppealDto]:
        """Fetches repeat and identical appeals linked to a document."""
        return await self._request_list(
            "GET", f"api/document/{document_id}/repeat-identical", token, RepeatIdenticalAppealDto
        )

    async def get_pre_nomenclature(self, token: str, document_id: UUID | str) -> list[DocumentPreNomenclatureDto]:
        """Fetches preliminary nomenclature for a document."""
        return await self._request_list(
            "GET", f"api/document/{document_id}/pre-nomenclature", token, DocumentPreNomenclatureDto
        )

    async def change_document_author(self, token: str, document_id: UUID | str, author_id: UUID | str) -> DocumentDto:
        """Changes document author."""
        return await self._request_dto(
            "PUT",
            f"api/document/{document_id}/change-document-author",
            token,
            DocumentDto,
            json_data={"id": str(author_id)},
        )

    async def notify_to_preparation(self, token: str, document_id: UUID | str, employee_id: UUID | str) -> None:
        """Notifies and moves document to preparation stage (Meeting)."""
        await self.make_request(
            "POST",
            f"api/document/{document_id}/notify-to-preparation",
            token,
            json_data={"id": str(employee_id)},
            is_json_response=False
        )

    async def get_employees_for_notify(self, token: str, document_id: UUID | str) -> list[Any]:
        """Fetches employees eligible for notification."""
        return await self.make_request("GET", f"api/document/{document_id}/employees-for-notify", token)

    async def notify_meeting(self, token: str, document_id: UUID | str, body: dict[str, Any]) -> None:
        """Sends meeting notification."""
        await self.make_request(
            "POST", f"api/document/{document_id}/notify-meeting", token, json_data=body, is_json_response=False
        )

    async def create_based_existing(
            self, token: str, document_id: UUID | str, body: DocumentBasedExistingBody
    ) -> DocumentDto:
        """Creates a new document based on an existing one."""
        return await self._request_dto(
            "POST",
            f"api/document/{document_id}/creating-based-existing",
            token,
            DocumentDto,
            json_data=body.model_dump(by_alias=True),
        )

    async def get_status_group_count(self, token: str, doc_filter: dict[str, Any] | None = None) -> list[Any]:
        """Fetches document counts grouped by status."""
        return await self.make_request("GET", "api/document/status-group", token, params=doc_filter)

    async def retry_bpmn_incident(self, token: str, document_id: UUID | str, incident_id: str) -> None:
        """Retries a BPMN incident."""
        await self.make_request(
            "POST", f"api/document/{document_id}/bpmn/incident/{incident_id}/retry", token, is_json_response=False
        )

    async def skip_bpmn_incident(self, token: str, document_id: UUID | str, incident_id: str) -> None:
        """Skips a BPMN incident."""
        await self.make_request(
            "POST", f"api/document/{document_id}/bpmn/incident/{incident_id}/skip", token, is_json_response=False
        )

    async def update_reg_number(
            self,
            token: str,
            document_id: UUID | str,
            reg_num: str,
            reg_date: datetime,
            journal_number: int | None = None,
    ) -> DocumentDto:
        """Updates document registration number."""
        payload: dict[str, Any] = {
            "regNum": reg_num,
            "regDate": reg_date.isoformat(),
        }
        if journal_number is not None:
            payload["journalNumber"] = journal_number

        return await self._request_dto(
            "PUT",
            f"api/document/{document_id}/reg-number",
            token,
            DocumentDto,
            json_data=payload,
        )

    async def get_document_recipients(
            self, token: str, document_id: UUID | str
    ) -> list[DocumentRecipientDto]:
        """Fetches list of document recipients."""
        try:
            return await self._request_list(
                "GET", f"api/document/{document_id}/recipient", token, DocumentRecipientDto
            )
        except EdmsNotFoundError:
            return []

    async def get_document_versions(
            self, token: str, document_id: UUID | str
    ) -> list[DocumentVersionDto]:
        """Fetches all versions of a document."""
        try:
            return await self._request_list(
                "GET", f"api/document/{document_id}/version", token, DocumentVersionDto
            )
        except EdmsNotFoundError:
            return []

    async def start_document(self, token: str, document_id: UUID | str) -> None:
        """Starts the document routing process. Raises on failure."""
        await self.make_request(
            "POST",
            "api/document/start",
            token,
            json_data={"id": str(document_id)},
            is_json_response=False,
        )

    async def cancel_document(
            self,
            token: str,
            document_id: UUID | str,
            comment: str | None = None,
    ) -> None:
        """Cancels a document. Raises on failure."""
        payload = DocumentCancelAction(id=UUID(str(document_id)), comment=comment.strip() if comment else None)

        await self.make_request(
            "POST",
            "api/document/cancel",
            token,
            json_data=payload.model_dump(by_alias=True),
            is_json_response=False,
        )

    async def execute_document_operations(
            self,
            token: str,
            document_id: UUID | str,
            operations: list[dict[str, Any]],
    ) -> None:
        """Executes a list of operations on a document. Raises on failure."""
        await self.make_request(
            "POST",
            f"api/document/{document_id}/execute",
            token,
            json_data=operations,
            is_json_response=False,
        )

    async def get_stat_user_executor(self, token: str) -> ExecutionDocumentStatCount | None:
        """Fetches document execution statistics for the current user."""
        try:
            return await self._request_dto("GET", "api/document/stat/user-executor", token, ExecutionDocumentStatCount)
        except EdmsNotFoundError:
            return None

    async def get_stat_user_control(self, token: str) -> ExecutionDocumentStatCount | None:
        """Fetches document control statistics for the current user."""
        try:
            return await self._request_dto("GET", "api/document/stat/user-control", token, ExecutionDocumentStatCount)
        except EdmsNotFoundError:
            return None

    async def get_stat_user_author(self, token: str) -> ExecutionDocumentStatCount | None:
        """Fetches document authoring statistics for the current user."""
        try:
            return await self._request_dto("GET", "api/document/stat/user-author", token, ExecutionDocumentStatCount)
        except EdmsNotFoundError:
            return None

    async def get_user_smdo_stat(self, token: str) -> UserSmdoStat | None:
        """Fetches SMDO delivery statistics for the current user."""
        try:
            return await self._request_dto("GET", "api/document/stat/user-smdo", token, UserSmdoStat)
        except EdmsNotFoundError:
            return None

    async def next_process(
            self,
            token: str,
            request: DocumentNextProcessRequest,
    ) -> None:
        """Moves document to the next process stage."""
        await self.make_request(
            "POST",
            "api/document/process/next",
            token,
            json_data=request.model_dump(by_alias=True),
            is_json_response=False,
        )

    async def create_document_answer(
            self,
            token: str,
            doc_id: UUID | str,
            profile_id: UUID | str,
    ) -> DocumentDto:
        """Creates a document answer."""
        return await self._request_dto(
            "POST",
            f"api/document/{doc_id}/answer",
            token,
            DocumentDto,
            json_data={"id": str(profile_id)},
        )

    async def create_document_smdo_answer(
            self,
            token: str,
            doc_id: UUID | str,
            profile_id: UUID | str,
    ) -> DocumentDto:
        """Creates an SMDO document answer."""
        return await self._request_dto(
            "POST",
            f"api/document/{doc_id}/smdo-answer",
            token,
            DocumentDto,
            json_data={"id": str(profile_id)},
        )

    async def recreate_aismv_document(
            self,
            token: str,
            request: DocumentAismvRecreateRequest,
    ) -> UUID:
        """Recreates an AISMV document with a different profile."""
        logger.info(f"Recreating AISMV document {request.id} with profile {request.profile_id}")
        data = await self.make_request(
            "POST",
            "api/document/aismv-recreate",
            token,
            json_data=request.model_dump(by_alias=True),
        )
        return UUID(data["id"])

    async def redo_protocol(
            self,
            token: str,
            document_id: UUID | str,
            profile_id: UUID | str,
    ) -> DocumentDto:
        """Redoes the meeting protocol."""
        return await self._request_dto(
            "POST",
            f"api/document/{document_id}/redo-protocol",
            token,
            DocumentDto,
            json_data={"profileId": str(profile_id)},
        )

    async def notify_meeting_question(self, token: str, document_id: UUID | str) -> None:
        """Sends notification for a meeting question."""
        await self.make_request(
            "POST",
            f"api/document/{document_id}/notify-meeting-question",
            token,
            is_json_response=False,
        )
