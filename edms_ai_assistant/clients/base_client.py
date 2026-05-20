# edms_ai_assistant/clients/base_client.py
from __future__ import annotations

import json
import logging
from typing import Any

from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)


class EdmsBaseClient:
    """Базовый клиент для EDMS API, использующий композицию транспорта."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        self._transport = transport
        self._settings = settings

    async def _make_request(
            self,
            method: str,
            endpoint: str,
            token: str,
            *,
            is_json_response: bool = True,
            long_timeout: bool = False,
            **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]] | bytes | None:
        """
        Выполняет запрос через транспорт и извлекает полезную нагрузную.
        """
        timeout = self._settings.long_timeout if long_timeout else self._settings.timeout

        response = await self._transport.request(
            method,
            endpoint,
            token=token,
            timeout=timeout,
            **kwargs,
        )

        if response.status_code == 204 or not response.content:
            return {} if is_json_response else None

        if not is_json_response:
            return response.content

        try:
            return response.json()
        except json.JSONDecodeError:
            logger.error(
                "Failed to decode JSON from %s %s",
                method,
                response.url,
                extra={"response_text": response.text[:300]},
            )
            return {}

    async def _upload_file(
            self,
            endpoint: str,
            token: str,
            file_name: str,
            file_content: bytes,
            content_type: str = "application/octet-stream",
    ) -> dict[str, Any] | None:
        """Хелпер для загрузки файлов (multipart/form-data)."""
        response = await self._transport.request(
            "POST",
            endpoint,
            token=token,
            files={"file": (file_name, file_content, content_type)},
        )

        if response.status_code == 204 or not response.content:
            return {}
        return response.json()
