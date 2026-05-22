# edms_ai_assistant/services/document_enricher.py
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from edms_ai_assistant.core.exceptions import EdmsNotFoundError

if TYPE_CHECKING:
    from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)


class DocumentEnricher:
    """Enriches Document DTOs with secondary data (attachments, etc.)."""

    def __init__(self, document_client: DocumentClient):
        self._client = document_client

    async def enrich(self, doc_dict: dict[str, Any], token: str) -> dict[str, Any]:
        """Enriches the document dictionary with additional info."""
        doc_id = doc_dict.get("id")
        if not doc_id:
            return doc_dict

        # Fetch attachments through the client instead of raw transport
        try:
            await self._client.get_document_recipients(token, str(doc_id))
            # Recipients are already part of doc if includes were used,
            # but this illustrates the pattern. For now we just return as is
            # since DocumentClient.get_document_metadata now uses FULL_DOC_INCLUDES.
            pass
        except (EdmsNotFoundError, Exception):
            logger.warning("Failed to enrich document %s", doc_id)

        return doc_dict
