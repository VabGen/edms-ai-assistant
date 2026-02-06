# edms_ai_assistant/clients/group_client.py
import logging
from typing import Optional, Dict, Any, List
from abc import abstractmethod
from uuid import UUID

from .base_client import EdmsHttpClient, EdmsBaseClient

logger = logging.getLogger(__name__)


class BaseGroupClient(EdmsBaseClient):

    @abstractmethod
    async def find_by_name(self, token: str, group_name: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_employees_by_group_ids(
        self, token: str, group_ids: List[UUID]
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class GroupClient(BaseGroupClient, EdmsHttpClient):

    async def find_by_name(
        self, token: str, group_name: str
    ) -> Optional[Dict[str, Any]]:
        endpoint = "api/group/fts-name"
        params = {"fts": group_name}

        try:
            result = await self._make_request(
                "GET", endpoint, token=token, params=params
            )
            if result and isinstance(result, dict):
                logger.info(f"Found group: {result.get('name', 'Unknown')}")
                return result
            return None
        except Exception as e:
            logger.error(f"Error searching group '{group_name}': {e}")
            return None

    async def get_employees_by_group_ids(
        self, token: str, group_ids: List[UUID]
    ) -> List[Dict[str, Any]]:
        if not group_ids:
            return []

        endpoint = f"api/group/employee/all?ids={group_ids[0]}"
        for group_id in group_ids[1:]:
            endpoint += f"&ids={group_id}"

        try:
            result = await self._make_request("GET", endpoint, token=token)

            if isinstance(result, list):
                employees = []
                for item in result:
                    if isinstance(item, dict) and "employee" in item:
                        employees.append(item["employee"])

                logger.info(f"Found {len(employees)} employees in {len(group_ids)} groups")
                return employees
            return []
        except Exception as e:
            logger.error(f"Error fetching employees for groups {group_ids}: {e}")
            return []