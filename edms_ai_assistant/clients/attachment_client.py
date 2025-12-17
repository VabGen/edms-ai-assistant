import re
import urllib.parse
from typing import Optional, Dict, Any, List, Tuple
from abc import abstractmethod
from .base_client import EdmsHttpClient, EdmsBaseClient

# Регулярное выражение для извлечения имени файла из Content-Disposition
# Обработка как кодированного имени, так и прямого
FILENAME_REGEX = re.compile(r'filename\*=UTF-8\'\'(.*)|filename="(.*)"')


class BaseAttachmentClient(EdmsBaseClient):
    """Абстрактный класс для работы с вложениями."""

    @abstractmethod
    async def get_document_attachments(self, token: str, document_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def download_attachment(
            self, token: str, document_id: str, attachment_id: str
    ) -> Optional[Tuple[bytes, str]]:
        """
        Скачивает вложение и возвращает его байты и имя файла (Tuple[bytes, str]).
        """
        raise NotImplementedError


class AttachmentClient(BaseAttachmentClient, EdmsHttpClient):
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
    ) -> Optional[Tuple[bytes, str]]:
        """
        Скачивает вложение документа как байты и извлекает имя файла из заголовка.
        """
        response = await self._make_request_response_object(
            "GET",
            f"api/document/{document_id}/attachment/{attachment_id}",
            token=token,
            long_timeout=True
        )

        if not response or not response.content:
            return None

        # 1. Извлечение имени файла
        content_disposition = response.headers.get("Content-Disposition", "")
        match = FILENAME_REGEX.search(content_disposition)

        file_name = "unknown_attachment.bin"
        if match:
            # Декодируем, если это UTF-8 кодировка (group 1)
            if match.group(1):
                try:
                    file_name = urllib.parse.unquote(match.group(1))
                except Exception:
                    file_name = "encoded_filename"
            # Используем прямое имя (group 2)
            elif match.group(2):
                file_name = match.group(2)

        # 2. Возврат байтов и имени
        return response.content, file_name