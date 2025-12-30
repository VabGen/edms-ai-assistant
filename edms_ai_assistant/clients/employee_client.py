import logging
from typing import Optional, Dict, Any, List
from abc import abstractmethod
from .base_client import EdmsHttpClient, EdmsBaseClient

logger = logging.getLogger(__name__)

class BaseEmployeeClient(EdmsBaseClient):
    """
    Абстрактный интерфейс.
    Используем имя BaseEmployeeClient
    """

    @abstractmethod
    async def search_employees(self, token: str, filter_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_employee(self, token: str, employee_id: str) -> Optional[Dict[str, Any]]:
        pass


class EmployeeClient(BaseEmployeeClient, EdmsHttpClient):
    async def get_employee(self, token: str, employee_id: str) -> Optional[Dict[str, Any]]:
        endpoint = f"api/employee/{employee_id}"
        logger.info(f"Запрос данных сотрудника ID: {employee_id}")

        result = await self._make_request(
            "GET",
            endpoint,
            token=token
        )

        if result and isinstance(result, dict):
            post_info = result.get("post")
            post_name = post_info.get("postName") if isinstance(post_info, dict) else "Не указана"
            logger.info(f"Данные получены. Должность: {post_name}")
            return result

        logger.warning(f"Не удалось получить данные для сотрудника {employee_id}")
        return None

    async def search_employees(self, token: str, filter_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Реализация поиска сотрудников с поддержкой пагинации и корректных фильтров.
        """
        endpoint = "api/employee/search?page=0&size=10&sort=lastName,ASC"

        search_query = filter_data.get("lastName") or filter_data.get("firstName") or ""

        payload = {
            "active": True,
            "search": search_query,
            "firstName": filter_data.get("firstName"),
            "lastName": filter_data.get("lastName"),
            "middleName": filter_data.get("middleName"),
            "includes": filter_data.get("includes", ["POST", "DEPARTMENT"])
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        logger.debug(f"Отправка запроса поиска: {payload}")

        result = await self._make_request(
            "POST",
            endpoint,
            token=token,
            json=payload
        )

        if isinstance(result, dict):
            return result.get('content', [])
        return []

    async def search_with_filter(self, token: str, filter_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Алиас для обратной совместимости."""
        return await self.search_employees(token, filter_data)