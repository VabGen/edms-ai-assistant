# edms_ai_assistant/tools/base.py

import httpx
import logging
from typing import Dict, Any, Optional
from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)

class EdmsApiClient:
    """Универсальный асинхронный клиент для API EDMS Chancellor NEXT."""
    def __init__(self, token: str):
        self.base_url = settings.CHANCELLOR_NEXT_BASE_URL.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.timeout = settings.EDMS_TIMEOUT

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET запрос."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/{endpoint}",
                    headers=self.headers,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                if response.status_code == 204:
                    return {"message": "Успешно, но нет контента."}
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"API Error GET {endpoint}: {e.response.text}")
                return {"error": f"API Error: {e.response.status_code}", "details": e.response.text}

    async def post(self,
                   endpoint: str,
                   json: Optional[Dict[str, Any]] = None,
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST запрос."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{endpoint}",
                    headers=self.headers,
                    json=json,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"API Error POST {endpoint}: {e.response.text}")
                return {"error": f"API Error: {e.response.status_code}", "details": e.response.text}

    async def download(self, endpoint: str) -> bytes | None:
        """Для скачивания файлов (вложений)."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/{endpoint}",
                    headers=self.headers,
                    timeout=self.timeout + 30
                )
                response.raise_for_status()
                return response.content
            except Exception as e:
                logger.error(f"Download Error {endpoint}: {e}")
                return None