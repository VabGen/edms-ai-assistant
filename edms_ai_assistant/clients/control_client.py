# edms_ai_assistant/clients/control_client.py
"""
EDMS AI Assistant — Control HTTP Client.
"""
from __future__ import annotations
import logging
from typing import Any
from edms_ai_assistant.clients.transport import IAsyncTransport

logger = logging.getLogger(__name__)


class ControlClient:
    """Клиент для управления контролем документов."""

    def __init__(self, transport: IAsyncTransport):
        self._transport = transport

    async def get_control_types(self, token: str) -> list[dict[str, Any]]:
        result = await self._transport.request("GET", "api/control-type", token=token,
                                               params={"page": "0", "size": "100"})
        if result.status_code == 204: return []
        data = result.json()
        return data if isinstance(data, list) else data.get("content", [])

    async def get_control(self, token: str, document_id: str) -> dict[str, Any] | None:
        resp = await self._transport.request("GET", f"api/document/{document_id}/control", token=token)
        if resp.status_code == 204: return None
        data = resp.json()
        return data if data.get("id") or data.get("controlTypeId") else None

    async def set_control(self, token: str, document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self._transport.request("POST", f"api/document/{document_id}/control", token=token, json=payload)
        return resp.json() if resp.content else {}

    async def edit_control(self, token: str, document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self._transport.request("PUT", f"api/document/{document_id}/control", token=token, json=payload)
        return resp.json() if resp.content else {}

    async def remove_control(self, token: str, document_id: str) -> None:
        await self._transport.request("PUT", "api/document/control", token=token, json={"id": document_id})

    async def delete_control(self, token: str, document_id: str) -> None:
        await self._transport.request("DELETE", f"api/document/{document_id}/control", token=token)
