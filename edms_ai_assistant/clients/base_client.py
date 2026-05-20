# edms_ai_assistant/clients/base_client.py
from __future__ import annotations

import json
import logging
from typing import Any, Type, TypeVar, cast

from pydantic import BaseModel

from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


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
            params: dict[str, Any] | None = None,
            json_data: Any = None,
            is_json_response: bool = True,
            long_timeout: bool = False,
            **kwargs: Any,
    ) -> Any:
        """
        Выполняет запрос через транспорт и извлекает полезную нагрузку.
        """
        timeout = self._settings.long_timeout if long_timeout else self._settings.timeout

        response = await self._transport.request(
            method,
            endpoint,
            token=token,
            params=params,
            json=json_data,
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

    async def _request_dto(
            self,
            method: str,
            endpoint: str,
            token: str,
            response_model: Type[T],
            *,
            params: dict[str, Any] | None = None,
            json_data: Any = None,
            long_timeout: bool = False,
            **kwargs: Any,
    ) -> T:
        """Выполняет запрос и валидирует ответ в Pydantic модель."""
        data = await self._make_request(
            method,
            endpoint,
            token,
            params=params,
            json_data=json_data,
            long_timeout=long_timeout,
            **kwargs
        )
        return response_model.model_validate(data)

    async def _request_list(
            self,
            method: str,
            endpoint: str,
            token: str,
            item_model: Type[T],
            *,
            params: dict[str, Any] | None = None,
            json_data: Any = None,
            long_timeout: bool = False,
            **kwargs: Any,
    ) -> list[T]:
        """Выполняет запрос и валидирует ответ в список Pydantic моделей."""
        data = await self._make_request(
            method,
            endpoint,
            token,
            params=params,
            json_data=json_data,
            long_timeout=long_timeout,
            **kwargs
        )
        if not isinstance(data, list):
            # Обработка Spring Page/Slice если нужно, или просто падение валидации
            if isinstance(data, dict) and "content" in data:
                data = data["content"]
            else:
                logger.warning("Expected list but got %s", type(data))
                return []

        return [item_model.model_validate(item) for item in data]

    async def _upload_file(
            self,
            endpoint: str,
            token: str,
            file_name: str,
            file_content: bytes,
            content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Хелпер для загрузки файлов (multipart/form-data)."""
        response = await self._transport.request(
            "POST",
            endpoint,
            token=token,
            files={"file": (file_name, file_content, content_type)},
        )

        if response.status_code == 204 or not response.content:
            return {}

        return cast(dict[str, Any], response.json())
