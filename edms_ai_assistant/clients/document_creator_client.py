# edms_ai_assistant/clients/document_creator_client.py
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.domain.document import DocumentWithPermissions

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)


class DocumentCreatorClient(EdmsBaseClient):
    """HTTP client for the document-creation-from-file workflow."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def find_profile_by_category(
            self,
            token: str,
            doc_category: str,
    ) -> dict[str, Any] | None:
        """Find the first active accessible DocumentProfile for the given category."""
        normalized = doc_category.strip().upper()
        params: dict[str, Any] = {
            "docCategoryConst": normalized,
            "active": "true",
            "withAccess": "true",
            "listAttribute": "true",
        }

        result = await self._make_request("GET", "api/doc-profile", token, params=params)

        if isinstance(result, list) and result:
            return cast("dict[str, Any]", result[0])
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and content:
                return cast("dict[str, Any]", content[0])
        return None

    async def create_document(
            self,
            token: str,
            profile_id: str,
    ) -> DocumentWithPermissions | None:
        """Create a new document from the given profile."""
        try:
            return await self._request_dto(
                "POST", "api/document", token, DocumentWithPermissions, json_data={"id": profile_id}
            )
        except Exception:
            logger.error("Document creation failed", exc_info=True)
            return None

    async def upload_attachment(
            self,
            token: str,
            document_id: str,
            file_path: str,
            file_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Upload a local file as MAIN_ATTACHMENT to the document."""
        path = Path(file_path)
        if not path.exists():
            return None

        display_name = (file_name or path.name).strip()
        content_type, _ = mimetypes.guess_type(display_name)
        content_type = content_type or "application/octet-stream"

        file_content = path.read_bytes()
        return await self._upload_file(
            endpoint=f"api/document/{document_id}/attachment",
            token=token,
            file_name=display_name,
            file_content=file_content,
            content_type=content_type,
        )
