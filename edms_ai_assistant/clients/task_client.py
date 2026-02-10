# edms_ai_assistant/clients/task_client.py
import logging
from typing import List
from abc import abstractmethod

from .base_client import EdmsHttpClient, EdmsBaseClient
from edms_ai_assistant.models.task_models import CreateTaskRequest

logger = logging.getLogger(__name__)


class BaseTaskClient(EdmsBaseClient):

    @abstractmethod
    async def create_tasks_batch(
        self, token: str, document_id: str, tasks: List[CreateTaskRequest]
    ) -> bool:
        raise NotImplementedError


class TaskClient(BaseTaskClient, EdmsHttpClient):

    async def create_tasks_batch(
        self, token: str, document_id: str, tasks: List[CreateTaskRequest]
    ) -> bool:
        if not tasks:
            logger.warning("Empty tasks list provided")
            return False

        endpoint = f"api/document/{document_id}/task/batch"

        payload = [task.model_dump(mode="json") for task in tasks]

        try:
            await self._make_request(
                "POST",
                endpoint,
                token=token,
                json=payload,
                is_json_response=False,
            )

            logger.info(
                f"Successfully created {len(tasks)} task(s) for document {document_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error creating tasks for document {document_id}: {e}", exc_info=True
            )
            return False
