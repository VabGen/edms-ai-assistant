import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from edms_ai_assistant.services.appeal_autofill_service import AppealAutofillService
from edms_ai_assistant.domain.document import DocumentDto, AttachmentDocumentDto
from edms_ai_assistant.domain.appeal_fields import AppealFields


@pytest.mark.asyncio
async def test_appeal_autofill_logic():
    mock_doc_client = MagicMock()
    mock_attach_client = MagicMock()
    mock_ref_client = MagicMock()
    mock_extraction = MagicMock()
    mock_llm = MagicMock()

    service = AppealAutofillService(
        doc_client=mock_doc_client,
        attach_client=mock_attach_client,
        ref_client=mock_ref_client,
        extraction_service=mock_extraction,
        chat_model=mock_llm
    )

    doc_id = uuid4()
    att_id = uuid4()
    doc_dto = DocumentDto(
        id=doc_id,
        attachment_document=[
            AttachmentDocumentDto(id=att_id, name="appeal.txt")
        ]
    )

    mock_doc_client.get_document_metadata = AsyncMock(return_value=doc_dto)
    mock_attach_client.get_attachment_content = AsyncMock(
        return_value=b"Sample plain text content longer than fifty characters for successful validation.")

    # Mock extraction result
    extraction_result = AppealFields(
        fioApplicant="Иванов Иван Иванович",
        organizationName="ООО Ромашка",
        shortSummary="Жалоба на отопление"
    )
    mock_extraction.extract_appeal_fields = AsyncMock(return_value=extraction_result)
    mock_doc_client.execute_document_operations = AsyncMock()
    mock_ref_client.find_delivery_method = AsyncMock(return_value="delivery-id")
    mock_ref_client.find_citizen_type = AsyncMock(return_value="citizen-type-id")
    mock_ref_client.find_best_subject = AsyncMock(return_value="subject-id")

    doc_dto = doc_dto.model_copy(update={"doc_category_const": "APPEAL"})
    mock_doc_client.get_document_metadata = AsyncMock(return_value=doc_dto)

    result = await service.process_and_fill("token", str(doc_id), attachment_id=None)

    assert result.status == "success"
    mock_doc_client.get_document_metadata.assert_called_once()
    mock_attach_client.get_attachment_content.assert_called_once()
