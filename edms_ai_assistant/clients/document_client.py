# edms_ai_assistant/clients/document_client.py
"""
EDMS AI Assistant — Document HTTP Client.
──────────────────────────────────────────────────────────────────
  ПОИСК И ЧТЕНИЕ ДОКУМЕНТОВ
  GET  /                              → search_documents()
  GET  /{id}                          → get_document_metadata()
  GET  /{id}/all                      → get_document_with_permissions()
  GET  /{id}/permission               → get_document_permissions()
  GET  /{id}/properties               → get_document_properties()

  ИСТОРИЯ И ПРОЦЕССЫ
  GET  /{id}/history                  → get_document_history()
  GET  /{id}/history/v2               → get_document_history_v2()
  GET  /{id}/bpmn                     → get_process_activity()
  GET  /{id}/task-task-project        → get_tasks_and_projects()

  КОНТРОЛЬ
  GET  /{id}/control                  → get_document_control()
  POST /{id}/control                  → set_document_control()
  PUT  /control                       → remove_document_control()
  DELETE /{id}/control                → delete_document_control()

  АДРЕСАТЫ И КОРРЕСПОНДЕНТЫ
  GET  /{id}/recipient                → get_document_recipients()
  GET  /{id}/responsible              → get_contract_responsible()

  НОМЕНКЛАТУРЫ ДЕЛ
  GET  /{id}/nomenclature-affair      → get_nomenclature_affairs()
  GET  /{id}/repeat-identical         → get_repeat_identical_appeals()

  ВЕРСИИ
  GET  /{id}/version                  → get_document_versions()

  ЖИЗНЕННЫЙ ЦИКЛ
  POST /start                         → start_document()
  POST /cancel                        → cancel_document()
  POST /{id}/execute                  → execute_document_operations()

  СТАТИСТИКА
  GET  /stat/user-executor            → get_stat_user_executor()
  GET  /stat/user-control             → get_stat_user_control()
  GET  /stat/user-author              → get_stat_user_author()
──────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Any

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.document import (
    BpmnProcessActivityDto,
    ControlDto,
    DocPermissionContainer,
    DocumentDto,
    DocumentHistoryDto,
    DocumentPropertiesDto,
    DocumentRecipientDto,
    DocumentVersionDto,
    DocumentWithPermissions,
    ExecutionDocumentStatCount,
    TasksAndProjectsDto,
)

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 10

FULL_DOC_INCLUDES: list[str] = [
    "DOCUMENT_TYPE",
    "DELIVERY_METHOD",
    "CORRESPONDENT",
    "RECIPIENT",
    "PRE_NOMENCLATURE_AFFAIRS",
    "CITIZEN_TYPE",
    "REGISTRATION_JOURNAL",
    "CURRENCY",
    "SOLUTION_RESULT",
    "PARENT_SUBJECT",
    "ADDITIONAL_DOCUMENT_AND_TYPE",
]

SEARCH_DOC_INCLUDES: list[str] = [
    "DOCUMENT_TYPE",
    "CORRESPONDENT",
    "REGISTRATION_JOURNAL",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _build_includes_params(includes: list[str]) -> dict[str, list[str]]:
    """Converts a list of Include names to Spring multi-value query params."""
    return {"includes": includes}


# ══════════════════════════════════════════════════════════════════════════════
# Concrete Implementation (Composition)
# ══════════════════════════════════════════════════════════════════════════════


class DocumentClient:
    """Concrete async HTTP client for EDMS Document API.

    Использует композицию: делегирует HTTP-логику базовому клиенту.
    Возвращает строгие Pydantic DTO вместо сырых словарей.

    Контракт:
      - GET-методы (чтение) возвращают None / [] если сущность не найдена (404).
      - Мутации (POST/PUT/DELETE) пробрасывают EdmsNotFoundError наверх,
        так как отсутствие сущности при мутации — это бизнес-ошибка.
      - Сетевые ошибки и 5xx СЭД всегда пробрасываются наверх.
    """

    def __init__(self, base_client: EdmsBaseClient):
        self._client = base_client

    # ── Поиск и чтение ────────────────────────────────────────────────────────

    async def search_documents(
            self,
            token: str,
            doc_filter: dict[str, Any] | None = None,
            pageable: dict[str, Any] | None = None,
            includes: list[str] | None = None,
    ) -> list[DocumentDto]:
        """Searches documents. Returns list of DocumentDto."""
        effective_pageable: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
        if pageable:
            effective_pageable.update(pageable)

        effective_includes = includes if includes is not None else SEARCH_DOC_INCLUDES

        params: dict[str, Any] = {
            **(doc_filter or {}),
            **effective_pageable,
            **_build_includes_params(effective_includes),
        }

        try:
            result = await self._client._make_request(
                "GET", "api/document", token=token, params=params
            )
        except EdmsNotFoundError:
            return []

        items: list[dict[str, Any]] = []
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                items = content
        elif isinstance(result, list):
            items = result

        return [DocumentDto.model_validate(item) for item in items]

    async def get_document_metadata(
            self,
            token: str,
            document_id: str,
            includes: list[str] | None = None,
    ) -> DocumentDto | None:
        """Fetches full document metadata by UUID."""
        effective_includes = includes if includes is not None else FULL_DOC_INCLUDES
        params = _build_includes_params(effective_includes)

        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}", token=token, params=params
            )
            if isinstance(result, dict) and result:
                return DocumentDto.model_validate(result)
        except EdmsNotFoundError:
            logger.info("Document not found: %s", document_id)

        return None

    async def get_document_with_permissions(
            self,
            token: str,
            document_id: str,
            includes: list[str] | None = None,
    ) -> DocumentWithPermissions | None:
        """Fetches document and its permissions in a single request."""
        effective_includes = includes if includes is not None else FULL_DOC_INCLUDES
        params = _build_includes_params(effective_includes)

        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/all", token=token, params=params
            )
            if isinstance(result, dict) and result:
                return DocumentWithPermissions.model_validate(result)
        except EdmsNotFoundError:
            logger.info("Document with permissions not found: %s", document_id)

        return None

    async def get_document_permissions(
            self, token: str, document_id: str
    ) -> DocPermissionContainer | None:
        """Fetches DocPermissionContainer for a document."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/permission", token=token
            )
            if isinstance(result, dict) and result:
                return DocPermissionContainer.model_validate(result)
        except EdmsNotFoundError:
            pass

        return None

    async def get_document_properties(
            self, token: str, document_id: str
    ) -> DocumentPropertiesDto | None:
        """Fetches extended document properties."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/properties", token=token
            )
            if isinstance(result, dict) and result:
                return DocumentPropertiesDto.model_validate(result)
        except EdmsNotFoundError:
            pass

        return None

    # ── История и процессы ────────────────────────────────────────────────────

    async def get_document_history(
            self, token: str, document_id: str
    ) -> list[DocumentHistoryDto]:
        """Fetches document processing protocol (history v1)."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/history", token=token
            )
            items = result if isinstance(result, list) else (
                result.get("content") or result.get("items") if isinstance(result, dict) else [])
            return [DocumentHistoryDto.model_validate(item) for item in items if isinstance(item, dict)]
        except EdmsNotFoundError:
            return []

    async def get_document_history_v2(
            self, token: str, document_id: str
    ) -> list[DocumentHistoryDto]:
        """Fetches document processing protocol (history v2, preferred)."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/history/v2", token=token
            )
            items = result if isinstance(result, list) else (
                result.get("content") or result.get("items") if isinstance(result, dict) else [])
            return [DocumentHistoryDto.model_validate(item) for item in items if isinstance(item, dict)]
        except EdmsNotFoundError:
            return []

    async def get_process_activity(
            self, token: str, document_id: str
    ) -> BpmnProcessActivityDto | None:
        """Fetches current BPMN process with active activities."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/bpmn", token=token
            )
            if isinstance(result, dict) and result:
                return BpmnProcessActivityDto.model_validate(result)
        except EdmsNotFoundError:
            pass

        return None

    async def get_tasks_and_projects(
            self, token: str, document_id: str
    ) -> TasksAndProjectsDto | None:
        """Fetches tasks and task-projects linked to a document."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/task-task-project", token=token
            )
            if isinstance(result, dict) and result:
                return TasksAndProjectsDto.model_validate(result)
        except EdmsNotFoundError:
            pass

        return None

    # ── Контроль ──────────────────────────────────────────────────────────────

    async def get_document_control(
            self, token: str, document_id: str
    ) -> ControlDto | None:
        """Fetches current control record (ControlDto) for a document."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/control", token=token
            )
            if isinstance(result, dict) and result:
                return ControlDto.model_validate(result)
        except EdmsNotFoundError:
            pass

        return None

    async def set_document_control(
            self,
            token: str,
            document_id: str,
            control_request: dict[str, Any],
    ) -> ControlDto | None:
        """Sets a document on control."""
        result = await self._client._make_request(
            "POST",
            f"api/document/{document_id}/control",
            token=token,
            json=control_request,
        )
        if isinstance(result, dict) and result:
            return ControlDto.model_validate(result)
        return None

    async def remove_document_control(self, token: str, document_id: str) -> bool:
        """Removes control mark (снять с контроля). Raises on failure."""
        await self._client._make_request(
            "PUT",
            "api/document/control",
            token=token,
            json={"id": document_id},
            is_json_response=False,
        )
        return True

    async def delete_document_control(self, token: str, document_id: str) -> bool:
        """Deletes control record. Raises on failure."""
        await self._client._make_request(
            "DELETE",
            f"api/document/{document_id}/control",
            token=token,
            is_json_response=False,
        )
        return True

    # ── Адресаты ──────────────────────────────────────────────────────────────

    async def get_document_recipients(
            self, token: str, document_id: str
    ) -> list[DocumentRecipientDto]:
        """Fetches list of document recipients."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/recipient", token=token
            )
            if isinstance(result, list):
                return [DocumentRecipientDto.model_validate(item) for item in result]
        except EdmsNotFoundError:
            pass

        return []

    async def get_contract_responsible(
            self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches responsible employees for a contract document."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/responsible", token=token
            )
            return result if isinstance(result, list) else []
        except EdmsNotFoundError:
            return []

    # ── Номенклатуры дел ──────────────────────────────────────────────────────

    async def get_nomenclature_affairs(
            self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches nomenclature affairs linked to a document."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/nomenclature-affair", token=token
            )
            return result if isinstance(result, list) else []
        except EdmsNotFoundError:
            return []

    async def get_repeat_identical_appeals(
            self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        """Fetches repeat and identical appeals linked to a document."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/repeat-identical", token=token
            )
            return result if isinstance(result, list) else []
        except EdmsNotFoundError:
            return []

    # ── Версии ────────────────────────────────────────────────────────────────

    async def get_document_versions(
            self, token: str, document_id: str
    ) -> list[DocumentVersionDto]:
        """Fetches all versions of a document."""
        try:
            result = await self._client._make_request(
                "GET", f"api/document/{document_id}/version", token=token
            )
            if isinstance(result, list):
                return [DocumentVersionDto.model_validate(item) for item in result]
        except EdmsNotFoundError:
            pass

        return []

    # ── Жизненный цикл ────────────────────────────────────────────────────

    async def start_document(self, token: str, document_id: str) -> bool:
        """Starts the document routing process. Raises on failure."""
        await self._client._make_request(
            "POST",
            "api/document/start",
            token=token,
            json={"id": document_id},
            is_json_response=False,
        )
        return True

    async def cancel_document(
            self,
            token: str,
            document_id: str,
            comment: str | None = None,
    ) -> bool:
        """Cancels (annuls) a document. Raises on failure."""
        payload: dict[str, Any] = {"id": document_id}
        if comment:
            payload["comment"] = comment.strip()

        await self._client._make_request(
            "POST",
            "api/document/cancel",
            token=token,
            json=payload,
            is_json_response=False,
        )
        return True

    async def execute_document_operations(
            self,
            token: str,
            document_id: str,
            operations: list[dict[str, Any]],
    ) -> bool:
        """Executes a list of operations on a document. Raises on failure."""
        if not operations:
            logger.warning("execute_document_operations called with empty list")
            return False

        await self._client._make_request(
            "POST",
            f"api/document/{document_id}/execute",
            token=token,
            json=operations,
            is_json_response=False,
        )
        return True

    # ── Статистика ────────────────────────────────────────────────────────

    async def get_stat_user_executor(self, token: str) -> ExecutionDocumentStatCount | None:
        """Fetches document execution statistics for the current user."""
        try:
            result = await self._client._make_request("GET", "api/document/stat/user-executor", token=token)
            if isinstance(result, dict):
                return ExecutionDocumentStatCount.model_validate(result)
        except EdmsNotFoundError:
            pass
        return None

    async def get_stat_user_control(self, token: str) -> ExecutionDocumentStatCount | None:
        """Fetches document control statistics for the current user."""
        try:
            result = await self._client._make_request("GET", "api/document/stat/user-control", token=token)
            if isinstance(result, dict):
                return ExecutionDocumentStatCount.model_validate(result)
        except EdmsNotFoundError:
            pass
        return None

    async def get_stat_user_author(self, token: str) -> ExecutionDocumentStatCount | None:
        """Fetches document authoring statistics for the current user."""
        try:
            result = await self._client._make_request("GET", "api/document/stat/user-author", token=token)
            if isinstance(result, dict):
                return ExecutionDocumentStatCount.model_validate(result)
        except EdmsNotFoundError:
            pass
        return None
