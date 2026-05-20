# edms_ai_assistant/clients/transport.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from edms_ai_assistant.core.exceptions import (
    EdmsAuthenticationError,
    EdmsClientError,
    EdmsConnectionError,
    EdmsNotFoundError,
    EdmsServerError,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class IAsyncTransport(Protocol):
    """Контракт для асинхронного HTTP-транспорта."""

    async def request(
            self,
            method: str,
            url: str,
            *,
            token: str,
            params: Optional[Dict[str, Any]] = None,
            json: Any = None,
            files: Optional[Dict[str, Any]] = None,
            timeout: Optional[int] = None,
    ) -> httpx.Response:
        ...


class HttpxTransport(IAsyncTransport):
    """Реализация транспорта на базе httpx с ретраями только для серверных ошибок."""

    def __init__(self, base_url: str, default_timeout: int = 30):
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=default_timeout,
        )

    @staticmethod
    def _get_headers(token: str) -> Dict[str, str]:
        if not token or not token.strip():
            raise ValueError("Authorization token is missing or empty.")
        return {"Authorization": f"Bearer {token}"}

    async def request(
            self,
            method: str,
            url: str,
            *,
            token: str,
            params: Optional[Dict[str, Any]] = None,
            json: Any = None,
            files: Optional[Dict[str, Any]] = None,
            timeout: Optional[int] = None,
    ) -> httpx.Response:
        """Публичный метод, удовлетворяющий контракту IAsyncTransport без декораторов."""
        response: httpx.Response = await self._request_with_retry(
            method,
            url,
            token=token,
            params=params,
            json=json,
            files=files,
            timeout=timeout,
        )
        return response

    @retry(
        retry=retry_if_exception_type((EdmsServerError, EdmsConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request_with_retry(
            self,
            method: str,
            url: str,
            *,
            token: str,
            params: Optional[Dict[str, Any]] = None,
            json: Any = None,
            files: Optional[Dict[str, Any]] = None,
            timeout: Optional[int] = None,
    ) -> httpx.Response:
        """Внутренняя реализация с ретраями"""
        headers = self._get_headers(token)

        if files:
            headers.pop("Content-Type", None)

        try:
            response = await self._client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json,
                files=files,
                timeout=timeout,
            )
        except httpx.RequestError as exc:
            logger.error("Network error during request to %s: %s", url, exc)
            raise EdmsConnectionError(str(exc), context={"url": url}) from exc

        await self._handle_status(response)
        return response

    @staticmethod
    async def _handle_status(response: httpx.Response) -> None:
        """Маппинг HTTP статус-кодов в доменные исключения."""
        status_code = response.status_code

        if 200 <= status_code < 300:
            return

        context = {
            "url": str(response.url),
            "status_code": status_code,
            "response_body": response.text[:500],
        }

        if status_code == 404:
            raise EdmsNotFoundError("Resource not found", status_code=status_code, context=context)
        if status_code in (401, 403):
            raise EdmsAuthenticationError("Authentication/Authorization failed", status_code=status_code,
                                          context=context)
        if 400 <= status_code < 500:
            raise EdmsClientError("Client error", status_code=status_code, context=context)
        if status_code >= 500:
            raise EdmsServerError("Server error", status_code=status_code, context=context)

    async def close(self) -> None:
        await self._client.aclose()
