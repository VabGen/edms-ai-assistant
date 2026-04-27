# edms_ai_assistant/agent/repositories.py
"""
Document repository: interface (Protocol) and production implementation.

Applying Dependency Inversion here lets the agent depend on the abstract
IDocumentRepository rather than the concrete HTTP client, which makes
the agent independently testable without a running EDMS server.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto

logger = logging.getLogger(__name__)


@runtime_checkable
class IDocumentRepository(Protocol):
    """Abstract interface for fetching EDMS document metadata."""

    async def get_document(self, token: str, doc_id: str) -> DocumentDto | None:
        """Fetch and validate a DocumentDto by UUID.

        Args:
            token: JWT bearer token.
            doc_id: Document UUID string.

        Returns:
            Validated DocumentDto, or None when not found or on error.
        """
        ...


class DocumentRepository:
    """Production implementation backed by the EDMS REST API.

    Errors are caught and logged so callers receive ``None`` instead of
    an unhandled exception — the agent handles the missing-document case
    gracefully in the prompt context.
    """

    async def get_document(self, token: str, doc_id: str) -> DocumentDto | None:
        """Fetch and validate document metadata from the EDMS REST API.

        Args:
            token: JWT bearer token.
            doc_id: Document UUID string.

        Returns:
            Validated DocumentDto, or None on any error.
        """
        try:
            async with DocumentClient() as client:
                raw = await client.get_document_metadata(token, doc_id)
                doc = DocumentDto.model_validate(raw)
                logger.info("Document fetched", extra={"doc_id": doc_id})
                return doc
        except Exception as exc:
            logger.error(
                "Failed to fetch document",
                exc_info=True,
                extra={"doc_id": doc_id, "error": str(exc)},
            )
            return None