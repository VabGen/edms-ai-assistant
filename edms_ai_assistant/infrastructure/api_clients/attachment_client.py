# edms_ai_assistant/infrastructure/api_clients/attachment_client.py

"""
EDMS Attachment Client — асинхронный клиент для взаимодействия с EDMS API (вложения).
"""
import httpx
from typing import Optional, Dict, Any, List, Union
from uuid import UUID
from edms_ai_assistant.config import settings
from edms_ai_assistant.utils.retry_utils import async_retry
from edms_ai_assistant.utils.api_utils import (
    handle_api_error,
    prepare_auth_headers,
)
import logging

logger = logging.getLogger(__name__)


class AttachmentClient:
    """
    Асинхронный клиент для работы с EDMS Attachment API.
    """
    DEFAULT_TIMEOUT = 30

    def __init__(
            self,
            base_url: Optional[str] = None,
            timeout: Optional[int] = None,
            service_token: Optional[str] = None,
    ):
        resolved_base_url = base_url or str(settings.chancellor_next_base_url)
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout or settings.edms_timeout or self.DEFAULT_TIMEOUT
        self.service_token = service_token
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Закрывает HTTP-клиент."""
        await self.client.aclose()

    def _get_headers(self) -> Dict[str, str]:
        """Возвращает заголовки с авторизацией."""
        return prepare_auth_headers(self.service_token)

    @async_retry(
        max_attempts=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(httpx.RequestError, httpx.HTTPStatusError),
    )
    async def _make_request(
            self,
            method: str,
            endpoint: str,
            **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Выполняет HTTP-запрос и возвращает JSON-ответ.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = kwargs.pop("headers", {}) or self._get_headers()

        try:
            response = await self.client.request(method, url, headers=headers, **kwargs)
            await handle_api_error(response, f"{method} {url}")

            if response.status_code == 204 or not response.content:
                return {}

            content_type = response.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                logger.warning(f"Unexpected non-JSON content type for {method} {url}: {content_type}")
                return {}

            return response.json()

        except httpx.HTTPStatusError:
            logger.error(f"HTTP Status Error for {method} {url}: {response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error for {method} {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {method} {url}: {e}", exc_info=True)
            raise

    # === Вложения (бинарные методы возвращают bytes) ===
    @async_retry(
        max_attempts=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(httpx.RequestError, httpx.HTTPStatusError),
    )
    async def download_attachment(
            self, document_id: UUID, attachment_id: UUID
    ) -> Optional[bytes]:
        """
        Скачивает вложение документа как байты.
        """
        endpoint = f"api/document/{document_id}/attachment/{attachment_id}"
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        logger.debug(f"GET (binary) {url}")
        try:
            response = await self.client.get(url, headers=headers)
            await handle_api_error(response, f"GET (binary) {url}")

            if response.status_code == 200:
                return response.content

            logger.warning(f"Download failed with status {response.status_code} for {url}")
            return None

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Ошибка загрузки вложения {attachment_id}: {e}")
            return None

    # === Вложения (JSON) ===
    async def get_document_attachments(
            self, document_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Получить список вложений документа. Возвращает список JSON.
        """
        result = await self._make_request("GET", f"api/document/{document_id}/attachment")
        return result if isinstance(result, list) else []