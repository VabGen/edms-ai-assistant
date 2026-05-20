# edms_ai_assistant/clients/attachment_client.py
import logging

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)


class AttachmentClient(EdmsBaseClient):
    """Клиент для работы с контентом и файлами вложений в СЭД."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def get_attachment_content(
            self, token: str, document_id: str, attachment_id: str
    ) -> bytes | None:
        """
        Скачать содержимое вложения (сырые байты).
        GET api/document/{documentId}/attachment/{attachmentId}
        """
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}",
            token=token,
            is_json_response=False,
            long_timeout=True,
        )
        return result if isinstance(result, bytes) else None
