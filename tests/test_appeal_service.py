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
            AttachmentDocumentDto(id=att_id, name="appeal.pdf")
        ]
    )

    mock_doc_client.get_document_metadata = AsyncMock(return_value=doc_dto)
    mock_attach_client.get_attachment_content = AsyncMock(return_value=b"fake pdf content")

    # Mock extraction result
    extraction_result = AppealFields(
        fioApplicant="Иванов Иван Иванович",
        organizationName="ООО Ромашка",
        shortSummary="Жалоба на отопление"
    )
    mock_extraction.extract_appeal_fields = AsyncMock(return_value=extraction_result)

    result = await service.autofill_appeal("token", str(doc_id))

    assert result["status"] == "success"
    assert "Иванов" in str(result["updates"])
    mock_doc_client.get_document_metadata.assert_called_once()
    mock_attach_client.get_attachment_content.assert_called_once()
