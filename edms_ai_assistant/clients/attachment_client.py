# edms_ai_assistant/clients/attachment_client.py
import logging

from uuid import UUID
from typing import Any
from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.config import EdmsSettings
from edms_ai_assistant.domain.document import (
    AttachmentDocumentDto,
    TemporaryAttachmentDto,
    RenameFileRequest,
    ChangeAttachmentDocumentTypeRequest,
    CheckAttachmentSignRequest,
    CreateEmptyFileRequest,
    SimpleCmsDto,
    AttachmentSignature,
)

logger = logging.getLogger(__name__)


class AttachmentClient(EdmsBaseClient):
    """Клиент для работы с контентом и файлами вложений в СЭД."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    # ══════════════════════════════════════════════════════════════════════════════
    # Attachment Operations
    # ══════════════════════════════════════════════════════════════════════════════

    async def upload_attachment(
        self, token: str, document_id: UUID, file_content: bytes, filename: str, is_additional: bool = False
    ) -> AttachmentDocumentDto:
        """
        Загрузить файл в документ.
        POST api/document/{documentId}/attachment
        """
        path = "additional-attachment" if is_additional else "attachment"
        logger.info("Uploading %s for document %s", path, document_id)
        result = await self.make_request(
            "POST",
            f"api/document/{document_id}/{path}",
            token=token,
            files={"file": (filename, file_content)},
        )
        return AttachmentDocumentDto.model_validate(result)

    async def get_attachment_content(
        self, token: str, document_id: UUID, attachment_id: UUID, is_additional: bool = False
    ) -> bytes:
        """
        Скачать содержимое вложения (сырые байты).
        GET api/document/{documentId}/attachment/{id}
        """
        path = "additional-attachment" if is_additional else "attachment"
        logger.info("Downloading content of attachment %s for document %s", attachment_id, document_id)
        result = await self.make_request(
            "GET",
            f"api/document/{document_id}/{path}/{attachment_id}",
            token=token,
            is_json_response=False,
            long_timeout=True,
        )
        return result

    async def get_source_content(self, token: str, document_id: UUID, attachment_id: UUID) -> bytes:
        """GET api/document/{documentId}/attachment/{id}/source"""
        logger.info("Downloading source of attachment %s", attachment_id)
        return await self.make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}/source",
            token=token,
            is_json_response=False,
            long_timeout=True,
        )

    async def get_zip_eds(self, token: str, document_id: UUID, attachment_id: UUID) -> bytes:
        """GET api/document/{documentId}/attachment/{id}/zip-eds"""
        logger.info("Downloading ZIP with EDS for attachment %s", attachment_id)
        return await self.make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}/zip-eds",
            token=token,
            is_json_response=False,
            long_timeout=True,
        )

    async def get_base64_content(self, token: str, document_id: UUID, attachment_id: UUID) -> bytes:
        """GET api/document/{documentId}/attachment/{id}/base64"""
        logger.info("Downloading Base64 content of attachment %s", attachment_id)
        return await self.make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}/base64",
            token=token,
            is_json_response=False,
        )

    async def get_metadata(self, token: str, document_id: UUID, attachment_id: UUID) -> AttachmentDocumentDto:
        """GET api/document/{documentId}/attachment/{id}/metadata"""
        result = await self.make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}/metadata",
            token=token,
        )
        return AttachmentDocumentDto.model_validate(result)

    async def get_all_attachments(self, token: str, document_id: UUID) -> list[AttachmentDocumentDto]:
        """GET api/document/{documentId}/attachment"""
        result = await self.make_request(
            "GET",
            f"api/document/{document_id}/attachment",
            token=token,
        )
        return [AttachmentDocumentDto.model_validate(item) for item in result]

    async def get_attachment_stamp(self, token: str, document_id: UUID, attachment_id: UUID) -> bytes:
        """GET api/document/{documentId}/attachment-stamp/{id}"""
        return await self.make_request(
            "GET",
            f"api/document/{document_id}/attachment-stamp/{attachment_id}",
            token=token,
            is_json_response=False,
        )

    async def rename_attachment(self, token: str, document_id: UUID, attachment_id: UUID, name: str) -> AttachmentDocumentDto:
        """PUT api/document/{documentId}/attachment/{id}/rename"""
        result = await self.make_request(
            "PUT",
            f"api/document/{document_id}/attachment/{attachment_id}/rename",
            token=token,
            json=RenameFileRequest(name=name).model_dump(by_alias=True),
        )
        return AttachmentDocumentDto.model_validate(result)

    async def change_type(
        self, token: str, document_id: UUID, attachment_id: UUID, request: ChangeAttachmentDocumentTypeRequest
    ) -> AttachmentDocumentDto:
        """PUT api/document/{documentId}/attachment/{id}/document-type"""
        result = await self.make_request(
            "PUT",
            f"api/document/{document_id}/attachment/{attachment_id}/document-type",
            token=token,
            json=request.model_dump(by_alias=True),
        )
        return AttachmentDocumentDto.model_validate(result)

    async def delete_attachment(self, token: str, document_id: UUID, attachment_id: UUID, is_additional: bool = False):
        """DELETE api/document/{documentId}/attachment/{id}"""
        path = "additional-attachment" if is_additional else "attachment"
        await self.make_request(
            "DELETE",
            f"api/document/{document_id}/{path}/{attachment_id}",
            token=token,
        )

    # ══════════════════════════════════════════════════════════════════════════════
    # PDF Conversion
    # ══════════════════════════════════════════════════════════════════════════════

    async def convert_to_pdf(self, token: str, document_id: UUID, attachment_id: UUID) -> AttachmentDocumentDto:
        """POST api/document/{documentId}/attachment/{id}/convert-pdf"""
        result = await self.make_request(
            "POST",
            f"api/document/{document_id}/attachment/{attachment_id}/convert-pdf",
            token=token,
        )
        return AttachmentDocumentDto.model_validate(result)

    async def convert_all_to_pdf(self, token: str, document_id: UUID, attachment_ids: list[UUID]):
        """POST api/document/{documentId}/attachment/convert-pdf"""
        await self.make_request(
            "POST",
            f"api/document/{document_id}/attachment/convert-pdf",
            token=token,
            json={"ids": [str(i) for i in attachment_ids]},
        )

    async def upload_and_convert_to_pdf(
        self, token: str, document_id: UUID, file_content: bytes, filename: str
    ) -> AttachmentDocumentDto:
        """POST api/document/{documentId}/attachment/upload-convert-pdf"""
        result = await self.make_request(
            "POST",
            f"api/document/{document_id}/attachment/upload-convert-pdf",
            token=token,
            files={"file": (filename, file_content)},
        )
        return AttachmentDocumentDto.model_validate(result)

    # ══════════════════════════════════════════════════════════════════════════════
    # Specialized Uploads
    # ══════════════════════════════════════════════════════════════════════════════

    async def upload_print_document(self, token: str, document_id: UUID, file_content: bytes, filename: str) -> AttachmentDocumentDto:
        """POST api/document/{documentId}/print-document"""
        result = await self.make_request(
            "POST", f"api/document/{document_id}/print-document", token=token, files={"file": (filename, file_content)}
        )
        return AttachmentDocumentDto.model_validate(result)

    async def upload_project_solution(self, token: str, document_id: UUID, file_content: bytes, filename: str) -> AttachmentDocumentDto:
        """POST api/document/{documentId}/project-solution"""
        result = await self.make_request(
            "POST", f"api/document/{document_id}/project-solution", token=token, files={"file": (filename, file_content)}
        )
        return AttachmentDocumentDto.model_validate(result)

    async def upload_rationale(self, token: str, document_id: UUID, file_content: bytes, filename: str) -> AttachmentDocumentDto:
        """POST api/document/{documentId}/rationale"""
        result = await self.make_request(
            "POST", f"api/document/{document_id}/rationale", token=token, files={"file": (filename, file_content)}
        )
        return AttachmentDocumentDto.model_validate(result)

    async def upload_documents_question(self, token: str, document_id: UUID, file_content: bytes, filename: str) -> AttachmentDocumentDto:
        """POST api/document/{documentId}/documents-question"""
        result = await self.make_request(
            "POST", f"api/document/{document_id}/documents-question", token=token, files={"file": (filename, file_content)}
        )
        return AttachmentDocumentDto.model_validate(result)

    async def upload_decision(self, token: str, document_id: UUID, file_content: bytes, filename: str) -> AttachmentDocumentDto:
        """POST api/document/{documentId}/decision"""
        result = await self.make_request(
            "POST", f"api/document/{document_id}/decision", token=token, files={"file": (filename, file_content)}
        )
        return AttachmentDocumentDto.model_validate(result)

    async def create_empty(self, token: str, document_id: UUID, request: CreateEmptyFileRequest) -> AttachmentDocumentDto:
        """POST api/document/{documentId}/create-empty"""
        result = await self.make_request(
            "POST",
            f"api/document/{document_id}/create-empty",
            token=token,
            json=request.model_dump(by_alias=True),
        )
        return AttachmentDocumentDto.model_validate(result)

    # ══════════════════════════════════════════════════════════════════════════════
    # Signatures
    # ══════════════════════════════════════════════════════════════════════════════

    async def add_sign(self, token: str, document_id: UUID, attachment_id: UUID, cms: str) -> AttachmentSignature:
        """POST api/document/{documentId}/attachment/{id}/sign"""
        result = await self.make_request(
            "POST",
            f"api/document/{document_id}/attachment/{attachment_id}/sign",
            token=token,
            json={"data": cms},
        )
        return AttachmentSignature.model_validate(result)

    async def add_sign_v2(self, token: str, document_id: UUID, attachment_id: UUID, cms: str) -> Any:
        """PUT api/document/{documentId}/attachment/{id}/sign2"""
        return await self.make_request(
            "PUT",
            f"api/document/{document_id}/attachment/{attachment_id}/sign2",
            token=token,
            json=SimpleCmsDto(cms=cms).model_dump(by_alias=True),
        )

    async def remove_sign(self, token: str, document_id: UUID, attachment_id: UUID, sign_id: UUID):
        """DELETE api/document/{documentId}/attachment/{id}/sign/{signId}"""
        await self.make_request(
            "DELETE",
            f"api/document/{document_id}/attachment/{attachment_id}/sign/{sign_id}",
            token=token,
        )

    async def verify_sign(self, token: str, document_id: UUID, attachment_id: UUID, request: CheckAttachmentSignRequest) -> dict[str, Any]:
        """POST api/document/{documentId}/attachment/{id}/verify-sign"""
        return await self.make_request(
            "POST",
            f"api/document/{document_id}/attachment/{attachment_id}/verify-sign",
            token=token,
            json=request.model_dump(by_alias=True),
        )

    # ══════════════════════════════════════════════════════════════════════════════
    # Temporary Attachments
    # ══════════════════════════════════════════════════════════════════════════════

    async def get_temporary(self, token: str, attachment_id: UUID) -> TemporaryAttachmentDto:
        """GET api/temporary-attachment/{id}"""
        result = await self.make_request(
            "GET",
            f"api/temporary-attachment/{attachment_id}",
            token=token,
        )
        return TemporaryAttachmentDto.model_validate(result)

    async def upload_temporary_template(self, token: str, file_content: bytes, filename: str) -> TemporaryAttachmentDto:
        """POST api/temporary-attachment/template"""
        result = await self.make_request(
            "POST",
            "api/temporary-attachment/template",
            token=token,
            files={"file": (filename, file_content)},
        )
        return TemporaryAttachmentDto.model_validate(result)

    async def upload_temporary_for_document(
        self,
        token: str,
        document_id: UUID,
        file_content: bytes,
        filename: str,
        attachment_doc_type: AttachmentDocumentType = AttachmentDocumentType.ATTACHMENT,
        is_additional: bool = False,
    ) -> TemporaryAttachmentDto:
        """
        POST api/temporary-attachment/document/{documentId}[/...]
        Supports all document attachment types (additional, print-document, project-solution, etc.)
        """
        mapping = {
            AttachmentDocumentType.ATTACHMENT: "additional-attachment" if is_additional else "",
            AttachmentDocumentType.PRINT_DOCUMENT: "print-document",
            AttachmentDocumentType.PROJECT_SOLUTION: "project-solution",
            AttachmentDocumentType.RATIONALE: "rationale",
            AttachmentDocumentType.DOCUMENTS_QUESTION: "documents-question",
            AttachmentDocumentType.DECISION: "decision",
        }
        suffix = mapping.get(attachment_doc_type, "")
        url = f"api/temporary-attachment/document/{document_id}"
        if suffix:
            url += f"/{suffix}"

        result = await self.make_request(
            "POST",
            url,
            token=token,
            files={"file": (filename, file_content)},
        )
        return TemporaryAttachmentDto.model_validate(result)

    async def upload_temporary_special(
        self, token: str, file_content: bytes, filename: str, category: str
    ) -> TemporaryAttachmentDto:
        """
        POST api/temporary-attachment/{category}/attachment
        Used for additional-document, control-point, mini-document
        """
        result = await self.make_request(
            "POST",
            f"api/temporary-attachment/{category}/attachment",
            token=token,
            files={"file": (filename, file_content)},
        )
        return TemporaryAttachmentDto.model_validate(result)

    async def upload_temporary_mini_doc(
        self, token: str, file_content: bytes, filename: str, type: AttachmentDocumentType
    ) -> TemporaryAttachmentDto:
        """
        POST api/temporary-attachment/mini-document/{type_path}
        Types: rkk, agreement, introduction
        """
        mapping = {
            AttachmentDocumentType.RKK: "rkk",
            AttachmentDocumentType.AGREEMENT_LIST: "agreement",
            AttachmentDocumentType.INTRODUCTION_LIST: "introduction",
        }
        path = mapping.get(type, "attachment")
        result = await self.make_request(
            "POST",
            f"api/temporary-attachment/mini-document/{path}",
            token=token,
            files={"file": (filename, file_content)},
        )
        return TemporaryAttachmentDto.model_validate(result)

    async def get_temporary_mini_doc_base64(self, token: str, attachment_id: UUID) -> bytes:
        """GET api/temporary-attachment/{id}/mini-document/base64"""
        return await self.make_request(
            "GET",
            f"api/temporary-attachment/{attachment_id}/mini-document/base64",
            token=token,
            is_json_response=False,
        )

    async def sign_temporary_mini_doc(self, token: str, attachment_id: UUID, cms: str) -> Any:
        """PUT api/temporary-attachment/{id}/mini-document/sign"""
        return await self.make_request(
            "PUT",
            f"api/temporary-attachment/{attachment_id}/mini-document/sign",
            token=token,
            json=SimpleCmsDto(cms=cms).model_dump(by_alias=True),
        )

    async def verify_temporary_mini_doc_sign(
        self, token: str, attachment_id: UUID, request: CheckAttachmentSignRequest
    ) -> dict[str, Any]:
        """POST api/temporary-attachment/{id}/mini-document/verify-sign"""
        return await self.make_request(
            "POST",
            f"api/temporary-attachment/{attachment_id}/mini-document/verify-sign",
            token=token,
            json=request.model_dump(by_alias=True),
        )

    async def remove_temporary_mini_doc_sign(self, token: str, attachment_id: UUID, sign_id: UUID):
        """DELETE api/temporary-attachment/{id}/mini-document/sign/{signId}"""
        await self.make_request(
            "DELETE",
            f"api/temporary-attachment/{attachment_id}/mini-document/sign/{sign_id}",
            token=token,
        )

    async def delete_temporary(self, token: str, ids: list[UUID], full: bool = False):
        """DELETE api/temporary-attachment[/full]"""
        path = "full" if full else ""
        url = "api/temporary-attachment"
        if path:
            url += f"/{path}"
        await self.make_request(
            "DELETE",
            url,
            token=token,
            json={"ids": [str(i) for i in ids]},
        )
