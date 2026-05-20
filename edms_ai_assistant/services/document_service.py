# edms_ai_assistant/services/document_service.py
from __future__ import annotations

import asyncio
import logging
from typing import Any

import redis.asyncio as aioredis
from pydantic import BaseModel, ConfigDict, Field

from edms_ai_assistant.clients.document_client import (
    FULL_DOC_INCLUDES,
    SEARCH_DOC_INCLUDES,
    DocumentClient,
)
from edms_ai_assistant.core.exceptions import (
    DocumentNotFoundError,
    DocumentOperationError,
)
from edms_ai_assistant.domain.document import DocumentDto
from edms_ai_assistant.services.document_enricher import DocumentEnricher
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)

_CACHE_PREFIX: str = "edms:doc:"
_CACHE_PREFIX_ANALYSIS: str = "edms:doc_analysis:"


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


class _DocumentCache:
    def __init__(self, redis: aioredis.Redis, ttl: int) -> None:
        self._r = redis
        self._ttl = ttl

    async def get_doc(self, doc_id: str) -> DocumentDto | None:
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX}{doc_id}")
            if raw:
                return DocumentDto.model_validate_json(raw)
        except Exception:
            logger.warning("Redis get_doc error", exc_info=True)
        return None

    async def set_doc(self, doc_id: str, doc: DocumentDto) -> None:
        try:
            await self._r.setex(
                f"{_CACHE_PREFIX}{doc_id}",
                self._ttl,
                doc.model_dump_json(by_alias=True),
            )
        except Exception:
            logger.warning("Redis set_doc error", exc_info=True)

    async def get_analysis(self, doc_id: str) -> dict[str, Any] | None:
        import json
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX_ANALYSIS}{doc_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            logger.warning("Redis get_analysis error", exc_info=True)
        return None

    async def set_analysis(self, doc_id: str, analysis: dict[str, Any]) -> None:
        import json
        try:
            await self._r.setex(
                f"{_CACHE_PREFIX_ANALYSIS}{doc_id}",
                self._ttl,
                json.dumps(analysis, default=str),
            )
        except Exception:
            logger.warning("Redis set_analysis error", exc_info=True)

    async def invalidate(self, doc_id: str) -> None:
        try:
            await self._r.delete(f"{_CACHE_PREFIX}{doc_id}", f"{_CACHE_PREFIX_ANALYSIS}{doc_id}")
        except Exception:
            logger.warning("Redis invalidate error", exc_info=True)


class DocumentService:
    """Main application service for all document operations."""

    def __init__(
            self,
            document_client: DocumentClient,
            document_enricher: DocumentEnricher,
            nlp_service: EDMSNaturalLanguageService,
            redis: aioredis.Redis,
            cache_ttl: int = 300,
            enrich_documents: bool = True,
    ) -> None:
        self._client = document_client
        self._enricher = document_enricher
        self._nlp = nlp_service
        self._cache = _DocumentCache(redis=redis, ttl=cache_ttl)
        self._enrich_documents = enrich_documents

    async def get_document(
            self, token: str, document_id: str, force_refresh: bool = False,
    ) -> DocumentDto:
        if not force_refresh:
            cached = await self._cache.get_doc(document_id)
            if cached is not None:
                return cached

        doc = await self._client.get_document_metadata(token=token, document_id=document_id, includes=FULL_DOC_INCLUDES)
        if not doc:
            raise DocumentNotFoundError(f"Документ {document_id} не найден", document_id=document_id)

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
            executor=results[0].model_dump() if hasattr(results[0], "model_dump") else None,
            control=results[1].model_dump() if hasattr(results[1], "model_dump") else None,
            author=results[2].model_dump() if hasattr(results[2], "model_dump") else None,
        )

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

    async def start_document(self, token: str, document_id: str) -> None:
        await self._client.start_document(token=token, document_id=document_id)
        await self._cache.invalidate(document_id)

    async def cancel_document(self, token: str, document_id: str, comment: str | None = None) -> None:
        await self._client.cancel_document(token=token, document_id=document_id, comment=comment)
        await self._cache.invalidate(document_id)

    async def execute_operations(self, token: str, document_id: str, operations: list[dict[str, Any]]) -> None:
        await self._client.execute_document_operations(token=token, document_id=document_id, operations=operations)
        await self._cache.invalidate(document_id)

    async def set_control(self, token: str, document_id: str, control_type_id: str, date_control_end: str,
                          control_employee_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"controlTypeId": control_type_id, "dateControlEnd": date_control_end}
        if control_employee_id:
            payload["controlEmployeeId"] = control_employee_id
        result = await self._client.set_document_control(token=token, document_id=document_id, control_request=payload)
        await self._cache.invalidate(document_id)
        return result.model_dump(by_alias=True)

    async def remove_control(self, token: str, document_id: str) -> None:
        await self._client.remove_document_control(token=token, document_id=document_id)
        await self._cache.invalidate(document_id)

    async def delete_control(self, token: str, document_id: str) -> None:
        await self._client.delete_document_control(token=token, document_id=document_id)
        await self._cache.invalidate(document_id)
