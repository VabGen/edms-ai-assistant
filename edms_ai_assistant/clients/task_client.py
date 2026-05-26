# edms_ai_assistant/clients/task_client.py
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
from uuid import UUID
from datetime import datetime

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.domain.document import (
    TaskDto,
    TaskExecutorsDto,
    TaskProjectDto,
    ExecutionTaskStatCount,
    TaskExecutionStatByPeriod,
    KanbanBoard,
    ChildTaskInfo,
    TaskExecutionResult,
    OrgKey
)
from edms_ai_assistant.domain.employee import SliceDto
from edms_ai_assistant.domain.system import TaskKanbanColumnDto
from edms_ai_assistant.domain.task_models import (
    CreateTaskRequest,
    UpdateTaskRequest,
    ExecuteTaskRequest,
    TaskRevisionRequest,
    ChangeResponsibleStatus
)

if TYPE_CHECKING:
    from edms_ai_assistant.config import EdmsSettings
    from edms_ai_assistant.clients.transport import IAsyncTransport

logger = logging.getLogger(__name__)


class TaskClient(EdmsBaseClient):
    """Client for EDMS Task API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    # ══════════════════════════════════════════════════════════════════════════
    # General Task Operations (api/task)
    # ══════════════════════════════════════════════════════════════════════════

    async def get_all_tasks(
        self, token: str, page: int = 0, size: int = 20, filter_data: dict[str, Any] | None = None
    ) -> SliceDto[TaskDto]:
        """Fetches all tasks with status 'on execution' for the current user."""
        logger.info("Fetching all active tasks")
        params = {"page": page, "size": size}
        if filter_data:
            params.update(filter_data)
        return await self._request_dto("GET", "api/task", token, SliceDto[TaskDto], params=params)

    async def get_stat_executor(self, token: str) -> ExecutionTaskStatCount:
        """Fetches task statistics for the user as an executor."""
        logger.info("Fetching task executor statistics")
        return await self._request_dto("GET", "api/task/stat/user-executor", token, ExecutionTaskStatCount)

    async def get_stat_control(self, token: str) -> ExecutionTaskStatCount:
        """Fetches task statistics for the user as a controller."""
        logger.info("Fetching task controller statistics")
        return await self._request_dto("GET", "api/task/stat/user-control", token, ExecutionTaskStatCount)

    async def get_stat_author(self, token: str) -> ExecutionTaskStatCount:
        """Fetches task statistics for the user as an author."""
        logger.info("Fetching task author statistics")
        return await self._request_dto("GET", "api/task/stat/user-author", token, ExecutionTaskStatCount)

    async def get_stat_period(self, token: str, start: datetime, end: datetime) -> TaskExecutionStatByPeriod:
        """Fetches task execution statistics for a specific period."""
        logger.info(f"Fetching task statistics for period {start} to {end}")
        params = {"start": start.isoformat(), "end": end.isoformat()}
        return await self._request_dto("GET", "api/task/stat/execution-period", token, TaskExecutionStatByPeriod, params=params)

    # ══════════════════════════════════════════════════════════════════════════
    # Kanban Operations (api/task/kanban)
    # ══════════════════════════════════════════════════════════════════════════

    async def get_kanban(self, token: str) -> KanbanBoard:
        """Fetches the Kanban board for the current user."""
        logger.info("Fetching Kanban board")
        return await self._request_dto("GET", "api/task/kanban", token, KanbanBoard)

    async def create_kanban_column(self, token: str, request: TaskKanbanColumnDto) -> TaskKanbanColumnDto:
        """Creates a new Kanban column."""
        logger.info("Creating Kanban column")
        return await self._request_dto("POST", "api/task/kanban/column", token, TaskKanbanColumnDto, json_data=request.model_dump(by_alias=True))

    async def update_kanban_column(self, token: str, column_id: str | UUID, request: TaskKanbanColumnDto) -> TaskKanbanColumnDto:
        """Updates an existing Kanban column."""
        logger.info(f"Updating Kanban column {column_id}")
        return await self._request_dto("PUT", f"api/task/kanban/column/{column_id}", token, TaskKanbanColumnDto, json_data=request.model_dump(by_alias=True))

    async def delete_kanban_column(self, token: str, column_id: str | UUID) -> None:
        """Deletes a Kanban column."""
        logger.info(f"Deleting Kanban column {column_id}")
        await self.make_request("DELETE", f"api/task/kanban/column/{column_id}", token, json_data={"id": str(column_id)}, is_json_response=False)

    async def change_column_order(self, token: str, column_ids: list[UUID]) -> None:
        """Changes the order of Kanban columns."""
        logger.info("Changing Kanban column order")
        await self.make_request("POST", "api/task/kanban/change-order", token, json_data={"ids": column_ids}, is_json_response=False)

    async def change_task_order_in_column(self, token: str, column_id: str | UUID, task_keys: list[OrgKey]) -> None:
        """Changes the order of tasks within a Kanban column."""
        logger.info(f"Changing task order in column {column_id}")
        await self.make_request(
            "POST",
            f"api/task/kanban/column/{column_id}/tasks/change-order",
            token,
            json_data={"ids": [k.model_dump(by_alias=True) for k in task_keys]},
            is_json_response=False
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Document Task Operations (api/document/{docId}/task)
    # ══════════════════════════════════════════════════════════════════════════

    async def get_document_tasks(self, token: str, document_id: str | UUID) -> list[TaskDto]:
        """Fetches all tasks associated with a document."""
        logger.info(f"Fetching tasks for document {document_id}")
        return await self._request_list("GET", f"api/document/{document_id}/task", token, TaskDto)

    async def create_task(self, token: str, document_id: str | UUID, request: CreateTaskRequest) -> TaskDto:
        """Creates a single task in a document."""
        logger.info(f"Creating task in document {document_id}")
        return await self._request_dto("POST", f"api/document/{document_id}/task", token, TaskDto, json_data=request.model_dump(by_alias=True))

    async def create_tasks_batch(self, token: str, document_id: str | UUID, tasks: list[CreateTaskRequest]) -> list[TaskDto]:
        """Creates a batch of tasks in a document."""
        logger.info(f"Creating batch of tasks for document {document_id}")
        return await self._request_list(
            "POST",
            f"api/document/{document_id}/task/batch",
            token,
            TaskDto,
            json_data=[t.model_dump(by_alias=True) for t in tasks]
        )

    async def update_task(self, token: str, document_id: str | UUID, request: UpdateTaskRequest) -> TaskDto:
        """Updates a task's primary information."""
        logger.info(f"Updating task {request.id} in document {document_id}")
        return await self._request_dto("PUT", f"api/document/{document_id}/task", token, TaskDto, json_data=request.model_dump(by_alias=True))

    async def delete_task(self, token: str, document_id: str | UUID, task_id: str | UUID) -> None:
        """Deletes a task from a document."""
        logger.info(f"Deleting task {task_id} from document {document_id}")
        await self.make_request("DELETE", f"api/document/{document_id}/task", token, json_data={"id": str(task_id)}, is_json_response=False)

    async def get_task(self, token: str, document_id: str | UUID, task_id: str | UUID) -> TaskDto:
        """Fetches a single task by ID."""
        logger.info(f"Fetching task {task_id}")
        return await self._request_dto("GET", f"api/document/{document_id}/task/{task_id}", token, TaskDto)

    async def execute_task(self, token: str, document_id: str | UUID, task_id: str | UUID, executor_id: str | UUID, request: ExecuteTaskRequest) -> TaskExecutionResult:
        """Executes a task as a specific executor."""
        logger.info(f"Executing task {task_id} as {executor_id}")
        return await self._request_dto(
            "PUT",
            f"api/document/{document_id}/task/{task_id}/executor/{executor_id}/execute",
            token,
            TaskExecutionResult,
            json_data=request.model_dump(by_alias=True)
        )

    async def task_revision(self, token: str, document_id: str | UUID, task_id: str | UUID, request: TaskRevisionRequest) -> TaskDto:
        """Sends a task back for revision."""
        logger.info(f"Sending task {task_id} for revision")
        return await self._request_dto(
            "PUT",
            f"api/document/{document_id}/task/{task_id}/revision",
            token,
            TaskDto,
            json_data=request.model_dump(by_alias=True)
        )

    async def change_responsible(self, token: str, document_id: str | UUID, task_id: str | UUID, executor_id: str | UUID, status: bool) -> TaskExecutorsDto:
        """Changes the responsible status of an executor."""
        logger.info(f"Changing responsibility for {executor_id} in task {task_id} to {status}")
        return await self._request_dto(
            "PUT",
            f"api/document/{document_id}/task/{task_id}/executor/{executor_id}/responsible",
            token,
            TaskExecutorsDto,
            json_data={"responsible": status}
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Task Projects (api/task-project)
    # ══════════════════════════════════════════════════════════════════════════

    async def get_task_projects(self, token: str, page: int = 0, size: int = 20, filter_data: dict[str, Any] | None = None) -> SliceDto[TaskProjectDto]:
        """Fetches task projects for the current user."""
        logger.info("Fetching task projects")
        params = {"page": page, "size": size}
        if filter_data:
            params.update(filter_data)
        return await self._request_dto("GET", "api/task-project", token, SliceDto[TaskProjectDto], params=params)
