# edms_ai_assistant/infrastructure/api_clients/employee_client.py

"""
EDMS Employee Client — асинхронный клиент для взаимодействия с EDMS API (сотрудники).
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


class EmployeeClient:
    """
    Асинхронный клиент для работы с EDMS Employee API.
    """

    def __init__(
            self,
            base_url: Optional[str] = None,
            timeout: Optional[int] = None,
            service_token: Optional[str] = None,
    ):
        resolved_base_url = base_url or str(settings.edms.base_url)
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout or settings.edms.timeout
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

            return response.json()

        except httpx.HTTPStatusError:
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error for {method} {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {method} {url}: {e}")
            raise

    # === Сотрудники ===
    async def search_employees(self, query: str) -> List[Dict[str, Any]]:
        """Поиск сотрудников по запросу (ФИО, должность)."""
        params = {"query": query}
        result = await self._make_request("GET", "api/employee", params=params)
        return result if isinstance(result, list) else []

    async def get_employee(self, employee_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Получить сотрудника по ID.
        """
        result = await self._make_request("GET", f"api/employee/{employee_id}")
        return result if result else None