# edms_ai_assistant/clients/employee_client.py

from typing import Optional, Dict, Any, List
from abc import abstractmethod
from .base_client import EdmsHttpClient, EdmsBaseClient


class BaseEmployeeClient(EdmsBaseClient):
    """Абстрактный класс для работы с сотрудниками."""

    @abstractmethod
    async def search_employees(self, token: str, query: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_employee(self, token: str, employee_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class EmployeeClient(BaseEmployeeClient, EdmsHttpClient):
    """Асинхронный клиент для работы с EDMS Employee API."""

    async def search_employees(self, token: str, query: str) -> List[Dict[str, Any]]:
        """
        Поиск сотрудников по запросу (ФИО, должность) через POST api/employee/search.

        NOTE: В EDMS Chancellor Next Search API часто использует `searchQuery` для
        единого текстового поиска. Избыточность полей (lastName, firstName и т.д.)
        может быть нежелательной. Используем только 'searchQuery' для простоты.
        """
        endpoint = "api/employee/search"
        body = {"searchQuery": query}

        result = await self._make_request(
            "POST",
            endpoint,
            token=token,
            json=body
        )

        content = result.get('content', []) if isinstance(result, dict) else []
        return content if isinstance(content, list) else []

    async def get_employee(self, token: str, employee_id: str) -> Optional[Dict[str, Any]]:
        """Получить сотрудника по ID (GET api/employee/{id})."""
        result = await self._make_request(
            "GET",
            f"api/employee/{employee_id}",
            token=token
        )
        return result if result else None
