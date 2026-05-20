# edms_ai_assistant/services/document_service.py
"""
EDMS AI Assistant — Document Service.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import redis.asyncio as aioredis
from pydantic import BaseModel, ConfigDict, Field

from edms_ai_assistant.clients.document_client import (
    FULL_DOC_INCLUDES,
    SEARCH_DOC_INCLUDES,
    DocumentClient,
)
from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.config import EdmsSettings
from edms_ai_assistant.core.exceptions import (
    DocumentNotFoundError,
    DocumentOperationError,
)
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.document_enricher import DocumentEnricher
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)

_CACHE_PREFIX: str = "edms:doc:"
_CACHE_PREFIX_ANALYSIS: str = "edms:doc_analysis:"
_DEFAULT_CACHE_TTL: int = 300


# ══════════════════════════════════════════════════════════════════════════════
# Result / config models
# ══════════════════════════════════════════════════════════════════════════════

class DocumentSearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    items: list[DocumentDto] = Field(default_factory=list)
    total_elements: int = 0
    total_pages: int = 0
    current_page: int = 0
    page_size: int = 10


class DocumentStats(BaseModel):
    model_config = ConfigDict(extra="ignore")
    executor: dict[str, Any] | None = None
    control: dict[str, Any] | None = None
    author: dict[str, Any] | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Redis cache helper (Pydantic-aware)
# ══════════════════════════════════════════════════════════════════════════════

class _DocumentCache:
    """Async Redis cache для DocumentService с поддержкой Pydantic v2."""

    def __init__(self, redis: aioredis.Redis, ttl: int) -> None:
        self._r = redis
        self._ttl = ttl

    async def get_doc(self, doc_id: str) -> DocumentDto | None:
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX}{doc_id}")
            if raw:
                logger.debug("Cache hit: doc", extra={"document_id": doc_id})
                return DocumentDto.model_validate_json(raw)
        except Exception as exc:
            logger.warning("Redis get_doc error", extra={"document_id": doc_id, "error": str(exc)})
        return None

    async def set_doc(self, doc_id: str, doc: DocumentDto) -> None:
        try:
            await self._r.setex(
                f"{_CACHE_PREFIX}{doc_id}",
                self._ttl,
                doc.model_dump_json(by_alias=True),
            )
        except Exception as exc:
            logger.warning("Redis set_doc error", extra={"document_id": doc_id, "error": str(exc)})

    async def get_analysis(self, doc_id: str) -> dict[str, Any] | None:
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX_ANALYSIS}{doc_id}")
            if raw:
                logger.debug("Cache hit: analysis", extra={"document_id": doc_id})
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis get_analysis error", extra={"document_id": doc_id, "error": str(exc)})
        return None

    async def set_analysis(self, doc_id: str, analysis: dict[str, Any]) -> None:
        try:
            await self._r.setex(
                f"{_CACHE_PREFIX_ANALYSIS}{doc_id}",
                self._ttl,
                json.dumps(analysis, default=str),
            )
        except Exception as exc:
            logger.warning("Redis set_analysis error", extra={"document_id": doc_id, "error": str(exc)})

    async def invalidate(self, doc_id: str) -> None:
        try:
            await self._r.delete(
                f"{_CACHE_PREFIX}{doc_id}",
                f"{_CACHE_PREFIX_ANALYSIS}{doc_id}",
            )
        except Exception as exc:
            logger.warning("Redis invalidate error", extra={"document_id": doc_id, "error": str(exc)})


# ══════════════════════════════════════════════════════════════════════════════
# DocumentService
# ══════════════════════════════════════════════════════════════════════════════

class DocumentService:
    """Main application service for all document operations."""

    def __init__(
            self,
            transport: IAsyncTransport,
            settings: EdmsSettings,
            redis: aioredis.Redis,
            enrich_documents: bool = True,
    ) -> None:
        self._client = DocumentClient(transport, settings)
        self._enricher = DocumentEnricher(transport)
        self._nlp = EDMSNaturalLanguageService()
        self._cache = _DocumentCache(redis=redis, ttl=settings.cache_ttl_seconds)
        self._enrich_documents = enrich_documents

    # ── READ ──────────────────────────────────────────────────────────────

    async def get_document(
            self, token: str, document_id: str, force_refresh: bool = False,
    ) -> DocumentDto:
        """Fetch, enrich and return a DocumentDto."""
        if not force_refresh:
            cached = await self._cache.get_doc(document_id)
            if cached is not None:
                return cached

        doc = await self._client.get_document_metadata(token=token, document_id=document_id, includes=FULL_DOC_INCLUDES)
        if not doc:
            raise DocumentNotFoundError(f"Документ {document_id} не найден", document_id=document_id)

        # Enricher пока работает с dict (для совместимости с огромной сгенерированной моделью)
        if self._enrich_documents:
            raw_dict = doc.model_dump(by_alias=True)
            enriched_dict = await self._enricher.enrich(raw_dict, token=token)
            doc = DocumentDto.model_validate(enriched_dict)

        await self._cache.set_doc(document_id, doc)
        return doc

    async def get_document_analysis(
            self, token: str, document_id: str, force_refresh: bool = False,
    ) -> dict[str, Any]:
        if not force_refresh:
            cached = await self._cache.get_analysis(document_id)
            if cached is not None:
                return cached

        doc = await self.get_document(token, document_id, force_refresh)
        analysis = self._nlp.process_document(doc)

        await self._cache.set_analysis(document_id, analysis)
        return analysis

    async def get_document_history(self, token: str, document_id: str) -> list[dict[str, Any]]:
        result = await self._client.get_document_history_v2(token=token, document_id=document_id)
        # Возвращает список DTO, конвертируем в dict для совместимости с NLP
        return [r.model_dump(by_alias=True) for r in result]

    async def get_document_versions(self, token: str, document_id: str) -> list[dict[str, Any]]:
        result = await self._client.get_document_versions(token=token, document_id=document_id)
        return [r.model_dump(by_alias=True) for r in result]

    async def get_document_stats(self, token: str) -> DocumentStats:
        results = await asyncio.gather(
            self._client.get_stat_user_executor(token),
            self._client.get_stat_user_control(token),
            self._client.get_stat_user_author(token),
            return_exceptions=True,
        )
        return DocumentStats(
            executor=results[0].model_dump() if isinstance(results[0], DocumentDto) else None,
            control=results[1].model_dump() if isinstance(results[1], DocumentDto) else None,
            author=results[2].model_dump() if isinstance(results[2], DocumentDto) else None,
        )

    # ── SEARCH ────────────────────────────────────────────────────────────

    async def search_documents(
            self, token: str, doc_filter: dict[str, Any] | None = None,
            page: int = 0, size: int | None = None, sort: str | None = None,
            includes: list[str] | None = None,
    ) -> DocumentSearchResult:
        effective_size = size or 20
        pageable = {"page": page, "size": effective_size}
        if sort:
            pageable["sort"] = sort

        items = await self._client.search_documents(
            token=token, doc_filter=doc_filter, pageable=pageable, includes=includes or SEARCH_DOC_INCLUDES
        )
        return DocumentSearchResult(items=items, page_size=effective_size)

    async def search_for_agent(
            self, token: str, status: str | None = None, category: str | None = None,
            reg_number: str | None = None, short_summary: str | None = None,
            author_id: str | None = None, responsible_executor_id: str | None = None,
            reg_date_from: str | None = None, reg_date_to: str | None = None,
            on_control: bool | None = None, page: int = 0, size: int = 10,
            sort: str = "regDate,desc",
    ) -> DocumentSearchResult:
        doc_filter: dict[str, Any] = {}
        if status: doc_filter["status"] = status
        if category: doc_filter["docCategoryConstant"] = category
        if reg_number: doc_filter["regNumber"] = reg_number
        if short_summary: doc_filter["shortSummary"] = short_summary
        if author_id: doc_filter["authorId"] = author_id
        if responsible_executor_id: doc_filter["responsibleExecutorId"] = responsible_executor_id
        if reg_date_from: doc_filter["regDateFrom"] = reg_date_from
        if reg_date_to: doc_filter["regDateTo"] = reg_date_to
        if on_control is not None: doc_filter["controlFlag"] = on_control

        return await self.search_documents(token=token, doc_filter=doc_filter or None, page=page, size=size, sort=sort)

    # ── LIFECYCLE & CONTROL ───────────────────────────────────────────────

    async def start_document(self, token: str, document_id: str) -> bool:
        success = await self._client.start_document(token=token, document_id=document_id)
        if not success:
            raise DocumentOperationError(f"Не удалось запустить документ {document_id}", document_id=document_id)
        await self._cache.invalidate(document_id)
        return True

    async def cancel_document(self, token: str, document_id: str, comment: str | None = None) -> bool:
        success = await self._client.cancel_document(token=token, document_id=document_id, comment=comment)
        if not success:
            raise DocumentOperationError(f"Не удалось аннулировать документ {document_id}", document_id=document_id)
        await self._cache.invalidate(document_id)
        return True

    async def execute_operations(self, token: str, document_id: str, operations: list[dict[str, Any]]) -> bool:
        if not operations:
            raise ValueError("operations list cannot be empty")
        success = await self._client.execute_document_operations(token=token, document_id=document_id,
                                                                 operations=operations)
        if not success:
            raise DocumentOperationError(f"Не удалось выполнить операции над документом {document_id}",
                                         document_id=document_id)
        await self._cache.invalidate(document_id)
        return True

    async def set_control(self, token: str, document_id: str, control_type_id: str, date_control_end: str,
                          control_employee_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"controlTypeId": control_type_id, "dateControlEnd": date_control_end}
        if control_employee_id:
            payload["controlEmployeeId"] = control_employee_id
        result = await self._client.set_document_control(token=token, document_id=document_id, control_request=payload)
        if not result:
            raise DocumentOperationError(f"Не удалось поставить документ {document_id} на контроль",
                                         document_id=document_id)
        await self._cache.invalidate(document_id)
        return result.model_dump(by_alias=True)

    async def remove_control(self, token: str, document_id: str) -> bool:
        success = await self._client.remove_document_control(token=token, document_id=document_id)
        if not success:
            raise DocumentOperationError(f"Не удалось снять документ {document_id} с контроля", document_id=document_id)
        await self._cache.invalidate(document_id)
        return True

    async def delete_control(self, token: str, document_id: str) -> bool:
        success = await self._client.delete_document_control(token=token, document_id=document_id)
        if not success:
            raise DocumentOperationError(f"Не удалось удалить контроль документа {document_id}",
                                         document_id=document_id)
        await self._cache.invalidate(document_id)
        return True
