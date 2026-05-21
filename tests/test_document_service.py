import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from edms_ai_assistant.services.document_service import DocumentService
from edms_ai_assistant.domain.document import DocumentDto

@pytest.mark.asyncio
async def test_document_service_get_analysis():
    mock_doc_client = MagicMock()
    mock_enricher = MagicMock()
    mock_nlp = MagicMock()
    mock_redis = AsyncMock()

    service = DocumentService(
        document_client=mock_doc_client,
        document_enricher=mock_enricher,
        nlp_service=mock_nlp,
        redis=mock_redis,
        cache_ttl=60
    )

    doc_id = uuid4()
    doc_dto = DocumentDto(id=doc_id, name="Test Doc")

    # Mocking behavior:
    # 1. Check cache (returns None)
    # 2. Get metadata from client
    # 3. Enrich document
    # 4. Save to cache
    # 5. NLP processing
    # 6. Save analysis to cache

    mock_doc_client.get_document_metadata = AsyncMock(return_value=doc_dto)
    # Enricher should return a dict that can be validated as DocumentDto
    mock_enricher.enrich = AsyncMock(return_value=doc_dto.model_dump(by_alias=True))
    # NLP processing returns the final analysis dict
    mock_nlp.process_document = MagicMock(return_value={"nlp_result": "ok"})
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock()

    analysis = await service.get_document_analysis("token", str(doc_id))

    assert analysis == {"nlp_result": "ok"}
    mock_doc_client.get_document_metadata.assert_called_once()
    mock_enricher.enrich.assert_called_once()
    mock_nlp.process_document.assert_called_once()
    # Should have saved both doc and analysis to Redis
    assert mock_redis.setex.call_count == 2
