# edms_ai_assistant/clients/task_client.py
from __future__ import annotations

import logging

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.domain.task_models import CreateTaskRequest

logger = logging.getLogger(__name__)


class TaskClient:
    """Concrete async HTTP client for EDMS Task API.
    """

    def __init__(self, base_client: EdmsBaseClient):
        self._client = base_client

    async def create_tasks_batch(
            self,
            token: str,
            document_id: str,
            tasks: list[CreateTaskRequest],
    ) -> bool:
        """Create a batch of tasks for a document.

        Args:
            token: Bearer token.
            document_id: Target document UUID.
            tasks: List of task request objects.

        Returns:
            True on success, False if tasks list is empty.

        Raises:
            EdmsClientError: On 4xx responses (e.g., validation/permission errors).
            EdmsServerError: On 5xx responses from EDMS.
        """
        if not tasks:
            logger.warning("Empty tasks list — skipping API call")
            return False

        endpoint = f"api/document/{document_id}/task/batch"

        payload = [task.model_dump(mode="json", by_alias=True) for task in tasks]

        await self._client._make_request(
            "POST",
            endpoint,
            token=token,
            json=payload,
            is_json_response=False,
        )

        logger.info(
            "Successfully created %d task(s) for document %s",
            len(tasks),
            document_id,
        )
        return True
