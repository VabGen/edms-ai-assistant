# edms_ai_assistant/clients/base_client.py

import httpx
import logging
from typing import Dict, Any, Optional, Union, List
from edms_ai_assistant.config import settings
from edms_ai_assistant.utils.retry_utils import async_retry
from edms_ai_assistant.utils.api_utils import handle_api_error, prepare_auth_headers

logger = logging.getLogger(__name__)


class EdmsBaseClient:
    """Универсальный асинхронный клиент для API EDMS Chancellor NEXT."""

    def __init__(
            self,
            base_url: Optional[str] = None,
            timeout: Optional[int] = None,
    ):
        resolved_base_url = base_url or settings.CHANCELLOR_NEXT_BASE_URL
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout or settings.EDMS_TIMEOUT
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Закрывает HTTP-клиент."""
        await self.client.aclose()

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
            token: str,
            is_json_response: bool = True,
            long_timeout: bool = False,
            **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], bytes, None]:
        """
        Выполняет HTTP-запрос с авторизацией, обработкой ошибок и повторными попытками.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = prepare_auth_headers(token)

        if long_timeout:
            kwargs['timeout'] = self.timeout + 30

        kwargs["headers"] = headers

        try:
            response = await self.client.request(method, url, **kwargs)
            await handle_api_error(response, f"{method} {url}")

            if response.status_code == 204 or not response.content:
                return {} if is_json_response else None

            if not is_json_response:
                return response.content

            return response.json()

        except httpx.HTTPStatusError:
            raise
        except httpx.RequestError:
            raise
