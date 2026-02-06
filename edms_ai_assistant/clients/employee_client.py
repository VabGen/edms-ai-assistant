# edms_ai_assistant/clients/employee_client.py
import logging
from typing import Optional, Dict, Any, List
from abc import abstractmethod
from .base_client import EdmsHttpClient, EdmsBaseClient

logger = logging.getLogger(__name__)


class BaseEmployeeClient(EdmsBaseClient):

    @abstractmethod
    async def search_employees(
        self, token: str, filter_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_employee(
        self, token: str, employee_id: str
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def find_by_last_name_fts(
        self, token: str, last_name: str
    ) -> Optional[Dict[str, Any]]:
        pass


class EmployeeClient(BaseEmployeeClient, EdmsHttpClient):

    async def get_employee(
        self, token: str, employee_id: str
    ) -> Optional[Dict[str, Any]]:
        endpoint = f"api/employee/{employee_id}"
        logger.info(f"Fetching employee data for ID: {employee_id}")

        result = await self._make_request("GET", endpoint, token=token)

        if result and isinstance(result, dict):
            post_info = result.get("post")
            post_name = (
                post_info.get("postName")
                if isinstance(post_info, dict)
                else "Не указана"
            )
            logger.info(f"Employee data fetched. Position: {post_name}")
            return result

        logger.warning(f"Employee {employee_id} not found")
        return None

    async def find_by_last_name_fts(
        self, token: str, last_name: str
    ) -> Optional[Dict[str, Any]]:
        endpoint = "api/employee/fts-lastname"
        params = {"fts": last_name}

        try:
            result = await self._make_request(
                "GET", endpoint, token=token, params=params
            )
            if result and isinstance(result, dict):
                logger.info(
                    f"Found employee via FTS: {result.get('lastName', 'Unknown')} "
                    f"{result.get('firstName', '')} (ID: {result.get('id', 'N/A')})"
                )
                return result
            return None
        except Exception as e:
            logger.warning(f"FTS search failed for '{last_name}': {e}")
            return None

    async def search_employees(
        self, token: str, filter_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        endpoint = "api/employee/search?page=0&size=10&sort=lastName,ASC"

        search_query = (
            filter_data.get("lastName") or filter_data.get("firstName") or ""
        )

        payload = {
            "active": True,
            "search": search_query,
            "firstName": filter_data.get("firstName"),
            "lastName": filter_data.get("lastName"),
            "middleName": filter_data.get("middleName"),
            "includes": filter_data.get("includes", ["POST", "DEPARTMENT"]),
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        logger.debug(f"Employee search request: {payload}")

        result = await self._make_request("POST", endpoint, token=token, json=payload)

        if isinstance(result, dict):
            content = result.get("content", [])
            logger.info(f"Found {len(content)} employees matching criteria")
            return content

        return []

    async def search_with_filter(
        self, token: str, filter_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        return await self.search_employees(token, filter_data)