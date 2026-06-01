# edms_ai_assistant/services/document_service.py
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from edms_ai_assistant.clients.document_client import (
    FULL_DOC_INCLUDES,
    SEARCH_DOC_INCLUDES,
    DocumentClient,
)
from edms_ai_assistant.core.exceptions import (
    DocumentNotFoundError,
)
from edms_ai_assistant.core.redis_circuit import CircuitBreakerRedisCache, RedisCircuitBreaker
from edms_ai_assistant.domain.document import DocumentDto

if TYPE_CHECKING:
    import redis.asyncio as aioredis

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
    def __init__(self, redis: aioredis.Redis, ttl: int, enable_circuit_breaker: bool = True) -> None:
        self._r = redis
        self._ttl = ttl
        
        # Initialize circuit breaker if enabled
        if enable_circuit_breaker:
            circuit_breaker = RedisCircuitBreaker(redis)
            self._cache = CircuitBreakerRedisCache(circuit_breaker)
            logger.info("Redis circuit breaker enabled for document cache")
        else:
            self._cache = None  # Fallback to direct Redis access
            logger.warning("Redis circuit breaker disabled - using direct access")

    async def get_doc(self, doc_id: str) -> DocumentDto | None:
        """Get document from cache with circuit breaker protection."""
        if self._cache:
            try:
                raw = await self._cache.get(f"{_CACHE_PREFIX}{doc_id}")
                if raw:
                    return DocumentDto.model_validate_json(raw)
            except Exception as exc:
                logger.warning("Circuit breaker cache get failed, falling back to direct access: %s", exc)
                # Fallback to direct access on circuit breaker failure
                return await self._get_doc_direct(doc_id)
        else:
            return await self._get_doc_direct(doc_id)
        
        return None
    
    async def _get_doc_direct(self, doc_id: str) -> DocumentDto | None:
        """Direct Redis access fallback without circuit breaker."""
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX}{doc_id}")
            if raw:
                return DocumentDto.model_validate_json(raw)
        except Exception as exc:
            logger.warning("Direct Redis get_doc failed: %s", exc)
        return None

    async def set_doc(self, doc_id: str, doc: DocumentDto) -> None:
        """Set document in cache with circuit breaker protection."""
        if self._cache:
            try:
                await self._cache.set(
                    f"{_CACHE_PREFIX}{doc_id}",
                    doc.model_dump_json(by_alias=True),
                    ex=self._ttl,
                )
                return
            except Exception as exc:
                logger.warning("Circuit breaker cache set failed, falling back to direct access: %s", exc)
        
        # Fallback to direct access
        await self._set_doc_direct(doc_id, doc)
    
    async def _set_doc_direct(self, doc_id: str, doc: DocumentDto) -> None:
        """Direct Redis access fallback without circuit breaker."""
        try:
            await self._r.setex(
                f"{_CACHE_PREFIX}{doc_id}",
                self._ttl,
                doc.model_dump_json(by_alias=True),
            )
        except Exception as exc:
            logger.warning("Direct Redis set_doc failed: %s", exc)

    async def get_analysis(self, doc_id: str) -> dict[str, Any] | None:
        """Get document analysis from cache with circuit breaker protection."""
        import json

        if self._cache:
            try:
                raw = await self._cache.get(f"{_CACHE_PREFIX_ANALYSIS}{doc_id}")
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("Circuit breaker cache get_analysis failed, falling back: %s", exc)
                return await self._get_analysis_direct(doc_id)
        else:
            return await self._get_analysis_direct(doc_id)
        
        return None
    
    async def _get_analysis_direct(self, doc_id: str) -> dict[str, Any] | None:
        """Direct Redis access fallback without circuit breaker."""
        import json
        try:
            raw = await self._r.get(f"{_CACHE_PREFIX_ANALYSIS}{doc_id}")
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Direct Redis get_analysis failed: %s", exc)
        return None

    async def set_analysis(self, doc_id: str, analysis: dict[str, Any]) -> None:
        """Set document analysis in cache with circuit breaker protection."""
        import json

        if self._cache:
            try:
                await self._cache.set(
                    f"{_CACHE_PREFIX_ANALYSIS}{doc_id}",
                    json.dumps(analysis, default=str),
                    ex=self._ttl,
                )
                return
            except Exception as exc:
                logger.warning("Circuit breaker cache set_analysis failed, falling back: %s", exc)
        
        await self._set_analysis_direct(doc_id, analysis)
    
    async def _set_analysis_direct(self, doc_id: str, analysis: dict[str, Any]) -> None:
        """Direct Redis access fallback without circuit breaker."""
        import json
        try:
            await self._r.setex(
                f"{_CACHE_PREFIX_ANALYSIS}{doc_id}",
                self._ttl,
                json.dumps(analysis, default=str),
            )
        except Exception as exc:
            logger.warning("Direct Redis set_analysis failed: %s", exc)

    async def invalidate(self, doc_id: str) -> None:
        """Invalidate document cache with circuit breaker protection."""
        if self._cache:
            try:
                await self._cache.delete(
                    f"{_CACHE_PREFIX}{doc_id}", f"{_CACHE_PREFIX_ANALYSIS}{doc_id}"
                )
                return
            except Exception as exc:
                logger.warning("Circuit breaker cache invalidate failed, falling back: %s", exc)
        
        await self._invalidate_direct(doc_id)
    
    async def _invalidate_direct(self, doc_id: str) -> None:
        """Direct Redis access fallback without circuit breaker."""
        try:
            await self._r.delete(
                f"{_CACHE_PREFIX}{doc_id}", f"{_CACHE_PREFIX_ANALYSIS}{doc_id}"
            )
        except Exception as exc:
            logger.warning("Direct Redis invalidate failed: %s", exc)
    
    async def get_cache_health(self) -> dict[str, Any]:
        """Get cache health status for monitoring."""
        if self._cache:
            return await self._cache.get_health_status()
        return {"status": "unknown", "circuit_breaker": "disabled"}


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
        enable_circuit_breaker: bool = True,
    ) -> None:
        from edms_ai_assistant.config import settings
        
        self._client = document_client
        self._enricher = document_enricher
        self._nlp = nlp_service
        # Use settings if circuit breaker enabled is not explicitly provided
        circuit_enabled = enable_circuit_breaker if enable_circuit_breaker is not None else settings.REDIS_CIRCUIT_BREAKER_ENABLED
        self._cache = _DocumentCache(redis=redis, ttl=cache_ttl, enable_circuit_breaker=circuit_enabled)
        self._enrich_documents = enrich_documents

    async def get_document(
        self,
        token: str,
        document_id: str,
        force_refresh: bool = False,
    ) -> DocumentDto:
        if not force_refresh:
            cached = await self._cache.get_doc(document_id)
            if cached is not None:
                return cached

        doc = await self._client.get_document_metadata(
            token=token, document_id=document_id, includes=FULL_DOC_INCLUDES
        )
        if not doc:
            raise DocumentNotFoundError(
                f"Документ {document_id} не найден", document_id=document_id
            )

        if self._enrich_documents:
            raw_dict = doc.model_dump(by_alias=True)
            enriched_dict = await self._enricher.enrich(raw_dict, token=token)
            doc = DocumentDto.model_validate(enriched_dict)

        await self._cache.set_doc(document_id, doc)
        return doc

    async def get_document_analysis(
        self,
        token: str,
        document_id: str,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        if not force_refresh:
            cached = await self._cache.get_analysis(document_id)
            if cached is not None:
                return cached

        doc = await self.get_document(token, document_id, force_refresh)
        analysis = self._nlp.process_document(doc)

        await self._cache.set_analysis(document_id, analysis)
        return analysis

    async def get_document_history(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        result = await self._client.get_document_history_v2(
            token=token, document_id=document_id
        )
        return [r.model_dump(by_alias=True) for r in result]

    async def get_document_versions(
        self, token: str, document_id: str
    ) -> list[dict[str, Any]]:
        result = await self._client.get_document_versions(
            token=token, document_id=document_id
        )
        return [r.model_dump(by_alias=True) for r in result]

    async def get_document_stats(self, token: str) -> DocumentStats:
        results = await asyncio.gather(
            self._client.get_stat_user_executor(token),
            self._client.get_stat_user_control(token),
            self._client.get_stat_user_author(token),
            return_exceptions=True,
        )
        return DocumentStats(
            executor=(
                results[0].model_dump() if hasattr(results[0], "model_dump") else None
            ),
            control=(
                results[1].model_dump() if hasattr(results[1], "model_dump") else None
            ),
            author=(
                results[2].model_dump() if hasattr(results[2], "model_dump") else None
            ),
        )

    async def search_documents(
        self,
        token: str,
        doc_filter: dict[str, Any] | None = None,
        page: int = 0,
        size: int | None = None,
        sort: str | None = None,
        includes: list[str] | None = None,
    ) -> DocumentSearchResult:
        effective_size = size or 20
        pageable = {"page": page, "size": effective_size}
        if sort:
            pageable["sort"] = sort

        items = await self._client.search_documents(
            token=token,
            doc_filter=doc_filter,
            pageable=pageable,
            includes=includes or SEARCH_DOC_INCLUDES,
        )
        return DocumentSearchResult(items=items, page_size=effective_size)

    async def start_document(self, token: str, document_id: str) -> None:
        await self._client.start_document(token=token, document_id=document_id)
        await self._cache.invalidate(document_id)

    async def cancel_document(
        self, token: str, document_id: str, comment: str | None = None
    ) -> None:
        await self._client.cancel_document(
            token=token, document_id=document_id, comment=comment
        )
        await self._cache.invalidate(document_id)

    async def execute_operations(
        self, token: str, document_id: str, operations: list[dict[str, Any]]
    ) -> None:
        await self._client.execute_document_operations(
            token=token, document_id=document_id, operations=operations
        )
        await self._cache.invalidate(document_id)

    async def set_control(
        self,
        token: str,
        document_id: str,
        control_type_id: str,
        date_control_end: str,
        control_employee_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "controlTypeId": control_type_id,
            "dateControlEnd": date_control_end,
        }
        if control_employee_id:
            payload["controlEmployeeId"] = control_employee_id
        result = await self._client.set_document_control(
            token=token, document_id=document_id, control_request=payload
        )
        await self._cache.invalidate(document_id)
        return result.model_dump(by_alias=True)

    async def remove_control(self, token: str, document_id: str) -> None:
        await self._client.remove_document_control(token=token, document_id=document_id)
        await self._cache.invalidate(document_id)

    async def delete_control(self, token: str, document_id: str) -> None:
        await self._client.delete_document_control(token=token, document_id=document_id)
        await self._cache.invalidate(document_id)
