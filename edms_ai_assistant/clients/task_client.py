# edms_ai_assistant/clients/task_client.py
from __future__ import annotations

import logging

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.config import EdmsSettings
from edms_ai_assistant.domain.task_models import CreateTaskRequest

logger = logging.getLogger(__name__)


class TaskClient(EdmsBaseClient):
    """Client for EDMS Task API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def create_tasks_batch(
            self,
            token: str,
            document_id: str,
            tasks: list[CreateTaskRequest],
    ) -> bool:
        """Create a batch of tasks for a document."""
        if not tasks:
            return False

        endpoint = f"api/document/{document_id}/task/batch"
        payload = [task.model_dump(mode="json", by_alias=True) for task in tasks]

        await self._make_request(
            "POST",
            endpoint,
            token=token,
            json_data=payload,
            is_json_response=False,
        )
        return True
