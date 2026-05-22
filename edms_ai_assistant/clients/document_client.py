# edms_ai_assistant/clients/document_client.py
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

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
)

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from uuid import UUID
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
        payload: dict[str, Any] = {"id": str(document_id)}
        if comment:
            payload["comment"] = comment.strip()

        await self.make_request(
            "POST",
            "api/document/cancel",
            token,
            json_data=payload,
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
