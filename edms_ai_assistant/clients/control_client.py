# edms_ai_assistant/clients/control_client.py
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.document import ControlDto

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from uuid import UUID
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)


class ControlClient(EdmsBaseClient):
    """Клиент для управления контролем документов."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def get_control_types(self, token: str) -> list[dict[str, Any]]:
        """Получает типы контроля."""
        result = await self.make_request(
            "GET", "api/control-type", token=token, params={"page": 0, "size": 100}
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "content" in result:
            return result["content"]
        return []

    async def get_control(self, token: str, document_id: UUID | str) -> ControlDto | None:
        """Получает запись о контроле документа."""
        try:
            return await self._request_dto(
                "GET", f"api/document/{document_id}/control", token, ControlDto
            )
        except EdmsNotFoundError:
            return None

    async def set_control(self, token: str, document_id: UUID | str, payload: dict[str, Any]) -> ControlDto:
        """Ставит документ на контроль."""
        return await self._request_dto(
            "POST", f"api/document/{document_id}/control", token, ControlDto, json_data=payload
        )

    async def edit_control(self, token: str, document_id: UUID | str, payload: dict[str, Any]) -> ControlDto:
        """Редактирует запись о контроле."""
        return await self._request_dto(
            "PUT", f"api/document/{document_id}/control", token, ControlDto, json_data=payload
        )

    async def remove_control(self, token: str, document_id: UUID | str) -> None:
        """Снимает с контроля."""
        await self.make_request(
            "PUT", "api/document/control", token, json_data={"id": str(document_id)}, is_json_response=False
        )

    async def delete_control(self, token: str, document_id: UUID | str) -> None:
        """Удаляет запись о контроле."""
        await self.make_request(
            "DELETE", f"api/document/{document_id}/control", token, is_json_response=False
        )
