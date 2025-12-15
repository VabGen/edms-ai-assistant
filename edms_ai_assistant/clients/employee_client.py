# Файл: edms_ai_assistant/clients/employee_client.py

from typing import Optional, Dict, Any, List
from uuid import UUID
from .base_client import EdmsBaseClient


class EmployeeClient(EdmsBaseClient):
    """Асинхронный клиент для работы с EDMS Employee API."""

    async def search_employees(self, token: str, query: str) -> List[Dict[str, Any]]:
        """
        Поиск сотрудников по запросу (ФИО, должность) через POST api/employee/search.

        NOTE: Мы передаем простой 'query' как searchQuery в тело запроса
        и ожидаем ответа в виде SliceDto (который имеет поле 'content').
        """
        # Эндпоинт из реальной спецификации
        endpoint = "api/employee/search"

        # Мы моделируем EmployeeFilter, передавая строку поиска в поле name
        # (или searchQuery, в зависимости от внутренней структуры фильтра).
        # Предположим, что LLM ищет по имени.
        body = {"lastName": query, "firstName": query, "middleName": query, "searchQuery": query}

        result = await self._make_request(
            "POST",  # Используем POST
            endpoint,
            token=token,
            json=body
        )

        # Ожидаем SliceDto<EmployeeDto> и извлекаем список из поля 'content'
        return result.get('content', []) if isinstance(result, dict) else []

    async def get_employee(self, token: str, employee_id: str) -> Optional[Dict[str, Any]]:
        """Получить сотрудника по ID (GET api/employee/{id})."""
        result = await self._make_request(
            "GET",
            f"api/employee/{employee_id}",
            token=token
        )
        return result if result else None