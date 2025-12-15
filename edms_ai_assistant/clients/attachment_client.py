# Файл: edms_ai_assistant/clients/attachment_client.py

from typing import Optional, Dict, Any, List
from .base_client import EdmsBaseClient

class AttachmentClient(EdmsBaseClient):
    """Асинхронный клиент для работы с EDMS Attachment API."""

    async def get_document_attachments(self, token: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Получить список вложений документа (предполагая GET /api/document/{documentId}/attachment).
        """
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/attachment",
            token=token
        )
        return result if isinstance(result, list) else []

    async def download_attachment(
        self, token: str, document_id: str, attachment_id: str
    ) -> Optional[bytes]:
        """
        Скачивает вложение документа как байты (GET /api/document/{documentId}/attachment/{id}).
        """
        return await self._make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}",
            token=token,
            is_json_response=False,
            long_timeout=True
        )