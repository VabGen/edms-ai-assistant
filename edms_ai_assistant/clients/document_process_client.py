# edms_ai_assistant/clients/document_process_client.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.domain.document import (
    BpmnProcessItemDefinitionDto,
    BpmnStartBeforeRequest,
    CamundaProcessItemDefinitionRequest,
    DocumentProcessDto,
    DocumentProcessExecutorDto,
    DocumentProcessItemDto,
    FreeRegistrationProcessRequest,
    PaperworkProcessAction,
    ProcessActionWithSign,
    ProcessItemExecutorEntry,
    RedirectReviewProcessAction,
    RegistrationProcessRequest,
    ReserveRegnumberRequest,
    SimpleProcessAction,
    SmdoRegistrationReject,
    SwapExecutorRequest,
)

if TYPE_CHECKING:
    from uuid import UUID

    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings
    from edms_ai_assistant.domain.enums import DocumentProcessType

logger = logging.getLogger(__name__)


class DocumentProcessClient(EdmsBaseClient):
    """Client for EDMS Document Process API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def get_process(self, token: str, document_id: UUID) -> DocumentProcessDto:
        """GET api/document/{documentId}/process"""
        logger.info(f"Fetching process for document {document_id}")
        return await self._request_dto(
            "GET", f"api/document/{document_id}/process", token, DocumentProcessDto
        )

    async def get_executors(
        self,
        token: str,
        document_id: UUID,
        item_id: UUID,
        page: bool = False,
        pageable: dict[str, Any] | None = None,
    ) -> list[DocumentProcessExecutorDto] | Any:
        """GET api/document/{documentId}/process/{id}/executors"""
        logger.info(f"Fetching executors for process item {item_id}")
        params = {"page": str(page).lower()}
        if pageable:
            params.update(pageable)

        if not page:
            return await self._request_list(
                "GET",
                f"api/document/{document_id}/process/{item_id}/executors",
                token,
                DocumentProcessExecutorDto,
                params=params,
            )

        return await self.make_request(
            "GET",
            f"api/document/{document_id}/process/{item_id}/executors",
            token,
            params=params,
        )

    async def get_executors_by_type(
        self,
        token: str,
        document_id: UUID,
        item_id: UUID,
        process_type: DocumentProcessType,
    ) -> list[DocumentProcessItemDto]:
        """GET api/document/{documentId}/process/{id}/{type}/executors"""
        logger.info(
            f"Fetching executors for document {document_id} by type {process_type}"
        )
        return await self._request_list(
            "GET",
            f"api/document/{document_id}/process/{item_id}/{process_type}/executors",
            token,
            DocumentProcessItemDto,
        )

    async def statement(
        self, token: str, document_id: UUID, item_id: UUID, body: ProcessActionWithSign
    ) -> None:
        """POST api/document/{documentId}/process/{id}/statement"""
        logger.info(f"Statement for process item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/statement",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def signing(
        self, token: str, document_id: UUID, item_id: UUID, body: ProcessActionWithSign
    ) -> None:
        """POST api/document/{documentId}/process/{id}/signing"""
        logger.info(f"Signing for process item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/signing",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def dispatch(self, token: str, document_id: UUID, item_id: UUID) -> None:
        """POST api/document/{documentId}/process/{id}/dispatch"""
        logger.info(f"Dispatch for document {document_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/dispatch",
            token,
            json_data={},
            is_json_response=False,
        )

    async def agreement(
        self, token: str, document_id: UUID, item_id: UUID, body: SimpleProcessAction
    ) -> None:
        """POST api/document/{documentId}/process/{id}/agreement"""
        logger.info(f"Agreement for process item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/agreement",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def contract_execution(
        self, token: str, document_id: UUID, item_id: UUID, body: SimpleProcessAction
    ) -> None:
        """POST api/document/{documentId}/process/{id}/contract-execution"""
        logger.info(f"Contract execution for process item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/contract-execution",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def review(
        self, token: str, document_id: UUID, item_id: UUID, body: SimpleProcessAction
    ) -> None:
        """POST api/document/{documentId}/process/{id}/review"""
        logger.info(f"Review for process item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/review",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def redirect_review(
        self,
        token: str,
        document_id: UUID,
        item_id: UUID,
        body: RedirectReviewProcessAction,
    ) -> None:
        """POST api/document/{documentId}/process/{id}/redirect-review"""
        logger.info(f"Redirect review for process item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/redirect-review",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def registration(
        self,
        token: str,
        document_id: UUID,
        item_id: UUID,
        body: RegistrationProcessRequest,
    ) -> None:
        """POST api/document/{documentId}/process/{id}/registration"""
        logger.info(f"Registration for document {document_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/registration",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def reserve_regnumber(
        self,
        token: str,
        document_id: UUID,
        item_id: UUID,
        body: ReserveRegnumberRequest,
    ) -> None:
        """POST api/document/{documentId}/process/{id}/reserve-regnumber"""
        logger.info(f"Reserve registration number for document {document_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/reserve-regnumber",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def free_registration(
        self,
        token: str,
        document_id: UUID,
        item_id: UUID,
        body: FreeRegistrationProcessRequest,
    ) -> None:
        """POST api/document/{documentId}/process/{id}/free-registration"""
        logger.info(f"Free registration for document {document_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/free-registration",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def smdo_registration_reject(
        self, token: str, document_id: UUID, item_id: UUID, body: SmdoRegistrationReject
    ) -> None:
        """POST api/document/{documentId}/process/{id}/smdo-registration-reject"""
        logger.info(f"SMDO registration reject for document {document_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/smdo-registration-reject",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def smdo_registration_skip(
        self,
        token: str,
        document_id: UUID,
        item_id: UUID,
        body: RegistrationProcessRequest,
    ) -> None:
        """POST api/document/{documentId}/process/{id}/smdo-registration-skip"""
        logger.info(f"SMDO registration skip for document {document_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/smdo-registration-skip",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def update_bpmn_activity(
        self,
        token: str,
        document_id: UUID,
        process_id: UUID,
        activity_id: str,
        body: CamundaProcessItemDefinitionRequest,
    ) -> None:
        """PUT api/document/{documentId}/process/{processId}/bpmn/activity/{activityId}"""
        logger.info(f"Update BPMN activity {activity_id} for process {process_id}")
        await self.make_request(
            "PUT",
            f"api/document/{document_id}/process/{process_id}/bpmn/activity/{activity_id}",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def get_bpmn_activity(
        self, token: str, document_id: UUID, process_id: UUID, activity_id: str
    ) -> BpmnProcessItemDefinitionDto:
        """GET api/document/{documentId}/process/{processId}/bpmn/activity/{activityId}"""
        logger.info(f"Get BPMN activity {activity_id}")
        return await self._request_dto(
            "GET",
            f"api/document/{document_id}/process/{process_id}/bpmn/activity/{activity_id}",
            token,
            BpmnProcessItemDefinitionDto,
        )

    async def get_bpmn_activity_definition(
        self, token: str, document_id: UUID, process_id: UUID, activity_id: str
    ) -> BpmnProcessItemDefinitionDto:
        """GET api/document/{documentId}/process/{processId}/bpmn/activity/{activityId}/definition"""
        logger.info(f"Get BPMN activity definition {activity_id}")
        return await self._request_dto(
            "GET",
            f"api/document/{document_id}/process/{process_id}/bpmn/activity/{activity_id}/definition",
            token,
            BpmnProcessItemDefinitionDto,
        )

    async def start_before(
        self,
        token: str,
        document_id: UUID,
        process_id: UUID,
        body: BpmnStartBeforeRequest,
    ) -> None:
        """PUT api/document/{documentId}/process/{processId}/bpmn/start-before"""
        logger.info(f"Start before activity {body.id}")
        await self.make_request(
            "PUT",
            f"api/document/{document_id}/process/{process_id}/bpmn/start-before",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def start_after(
        self,
        token: str,
        document_id: UUID,
        process_id: UUID,
        body: BpmnStartBeforeRequest,
    ) -> None:
        """PUT api/document/{documentId}/process/{processId}/bpmn/start-after"""
        logger.info(f"Start after activity {body.id}")
        await self.make_request(
            "PUT",
            f"api/document/{document_id}/process/{process_id}/bpmn/start-after",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def preparation_completed(
        self, token: str, document_id: UUID, item_id: UUID, body: SimpleProcessAction
    ) -> None:
        """POST api/document/{documentId}/process/{id}/preparation-completed"""
        logger.info(f"Preparation completed for process item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/preparation-completed",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def paperwork_completed(
        self, token: str, document_id: UUID, item_id: UUID, body: PaperworkProcessAction
    ) -> Any:
        """POST api/document/{documentId}/process/{id}/paperwork-completed"""
        logger.info(f"Paperwork completed for process item {item_id}")
        return await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/paperwork-completed",
            token,
            json_data=body.model_dump(exclude_none=True),
        )

    async def accepted(
        self, token: str, document_id: UUID, item_id: UUID, body: SimpleProcessAction
    ) -> None:
        """POST api/document/{documentId}/process/{id}/accepted"""
        logger.info(f"Accepted for process item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/accepted",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )

    async def replace_executors(
        self,
        token: str,
        document_id: UUID,
        item_id: UUID,
        executors: list[ProcessItemExecutorEntry],
    ) -> None:
        """POST api/document/{documentId}/process/{itemId}/replace-executors"""
        logger.info(f"Replace executors for item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/replace-executors",
            token,
            json_data=[e.model_dump(exclude_none=True) for e in executors],
            is_json_response=False,
        )

    async def swap_executors(
        self, token: str, document_id: UUID, item_id: UUID, body: SwapExecutorRequest
    ) -> None:
        """POST api/document/{documentId}/process/{itemId}/swap-executors"""
        logger.info(f"Swap executors for item {item_id}")
        await self.make_request(
            "POST",
            f"api/document/{document_id}/process/{item_id}/swap-executors",
            token,
            json_data=body.model_dump(exclude_none=True),
            is_json_response=False,
        )
