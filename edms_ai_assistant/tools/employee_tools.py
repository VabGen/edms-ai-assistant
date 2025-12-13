# edms_ai_assistant/tools/employee_tools.py

import json
import logging
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Any, Optional
from .base import EdmsApiClient

logger = logging.getLogger(__name__)

class GetEmployeeByIdArgs(BaseModel):
    employee_id: str = Field(description="Уникальный идентификатор (UUID) сотрудника.")

class SearchEmployeeArgs(BaseModel):
    search_query: str = Field(description="Полное или частичное ФИО (например, 'Иванов') или должность.")


class EmployeeTools:
    """Инструменты для работы с данными сотрудников."""

    def __init__(self, api_client: EdmsApiClient):
        self.api_client = api_client

    async def get_by_id(self, employee_id: str) -> str:
        """
        Используется для получения полных метаданных сотрудника по его UUID.
        """
        logger.info(f"Вызов API: employee_tools.get_by_id для {employee_id}")
        endpoint = f"api/employee/{employee_id}"
        data = await self.api_client.get(endpoint)

        if not data or data.get("error"):
            return json.dumps({"error": "Сотрудник не найден или произошла ошибка API.", "details": data})

        # Фильтруем важные поля для LLM
        filtered_data = {
            "id": data.get("id"),
            "ФИО": f"{data.get('lastName', '')} {data.get('firstName', '')} {data.get('middleName', '')}".strip(),
            "Должность": data.get("post", {}).get("postName"),
            "Email": data.get("email"),
            "Телефон": data.get("phone"),
            "Уволен": data.get("fired")
        }
        return json.dumps(filtered_data, ensure_ascii=False)

    async def search(self, search_query: str) -> str:
        """
        Используется для поиска списка сотрудников по частичному совпадению (фамилия, имя, должность).
        """
        logger.info(f"Вызов API: employee_tools.search по запросу: {search_query}")

        employee_filter = {"active": True, "search": search_query, "includes": ["POST"]}
        params = {"page": 0, "size": 10, "sort": "lastName,ASC"}

        endpoint = "api/employee/search"
        data = await self.api_client.post(endpoint, json=employee_filter, params=params)

        if not data or data.get("error"):
            return json.dumps({"error": "Ошибка API или пустой ответ.", "details": data})

        content: List[Dict[str, Any]] = data.get("content")

        if not isinstance(content, list) or not content:
            return json.dumps({"result": f"Сотрудники по запросу '{search_query}' не найдены."})

        # Фильтруем и форматируем список результатов для LLM
        results = []
        for emp in content:
            results.append({
                "id": emp.get("id"),
                "ФИО": f"{emp.get('lastName', '')} {emp.get('firstName', '')} {emp.get('middleName', '')}".strip(),
                "Должность": emp.get("post", {}).get("postName"),
                "Email": emp.get("email"),
            })

        return json.dumps(results, ensure_ascii=False)