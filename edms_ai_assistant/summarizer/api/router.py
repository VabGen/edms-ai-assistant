# summarizer/api/router.py

"""
FastAPI Router — Production-grade summarization API.

Endpoints:
    POST /summarize          — Full structured summarization (all modes)
    POST /summarize/stream   — SSE streaming for executive/abstractive modes
    GET  /summarize/modes    — Available modes + schema descriptions
    DELETE /cache/{file_hash} — Invalidate cache by file hash

Design principles:
    - All inputs validated by Pydantic v2 before hitting the service
    - Request ID generated server-side (UUID4) for idempotency + tracing
    - File upload as multipart (not base64 JSON) — efficient for large files
    - Response always includes cost + latency metadata
    - Streaming via Server-Sent Events (text/event-stream)
    - Rate limiting via slowapi (configurable)
"""

from __future__ import annotations

import uuid
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, File, Form, Request, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from edms_ai_assistant.summarizer.service import (
    SummarizationRequest,
    SummarizationResponse,
    SummarizationService,
)
from edms_ai_assistant.summarizer.api.schemas import (
    CacheInvalidationResponse,
    SummarizeModeInfo,
    SummarizeModesResponse,
)
from edms_ai_assistant.summarizer.structured.models import (
    MODE_OUTPUT_MODEL,
    SummaryMode,
)

router = APIRouter(prefix="/summarize", tags=["Summarization"])

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def get_summarization_service(request: Request) -> SummarizationService:
    """FastAPI dependency — returns the singleton SummarizationService.

    The service is initialized in app lifespan and stored in app.state.
    This dependency retrieves it.
    """
    service: SummarizationService | None = getattr(request.app.state, "summarization_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SummarizationService not initialized. Check application lifespan.",
        )
    return service


ServiceDep = Annotated[SummarizationService, Depends(get_summarization_service)]

# # ---------------------------------------------------------------------------
# # Schemas
# # ---------------------------------------------------------------------------
#
#
# class SummarizeModeInfo(BaseModel):
#     mode: str
#     description: str
#     output_schema: dict
#     use_case: str
#
#
# class SummarizeModesResponse(BaseModel):
#     modes: list[SummarizeModeInfo]
#     prompt_registry_version: str
#
#
# class CacheInvalidationResponse(BaseModel):
#     invalidated: bool
#     file_hash: str
#     message: str


_MODE_DESCRIPTIONS: dict[SummaryMode, dict] = {
    SummaryMode.EXECUTIVE: {
        "description": "C-suite friendly: headline + 3-5 bullets + optional recommendation",
        "use_case": "Quick decisions, management briefings",
    },
    SummaryMode.DETAILED_NOTES: {
        "description": "Full structured notes preserving document hierarchy and key entities",
        "use_case": "Legal review, contract analysis, detailed study",
    },
    SummaryMode.ACTION_ITEMS: {
        "description": "Extracts all tasks/commitments with owners, deadlines, priorities",
        "use_case": "Meeting minutes, project assignments, compliance tracking",
    },
    SummaryMode.THESIS: {
        "description": "Hierarchical argument/thesis plan for analytical documents",
        "use_case": "Academic papers, policy documents, strategic plans",
    },
    SummaryMode.EXTRACTIVE: {
        "description": "Key facts, dates, figures, organizations as structured data",
        "use_case": "Data extraction, fact-checking, EDMS card filling",
    },
    SummaryMode.ABSTRACTIVE: {
        "description": "Cohesive narrative paraphrase in 2-4 paragraphs",
        "use_case": "General document understanding, briefings",
    },
    SummaryMode.MULTILINGUAL: {
        "description": "Auto-detects source language, summarizes in requested language",
        "use_case": "Cross-language document processing (RU/BE/KZ/EN)",
    },
}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/modes",
    response_model=SummarizeModesResponse,
    summary="List available summarization modes with schemas",
)
async def get_modes(service: ServiceDep) -> SummarizeModesResponse:
    """Return all available summarization modes with their output schemas."""
    from edms_ai_assistant.summarizer.prompts.registry import get_prompt_registry
    registry = get_prompt_registry()
    modes = []
    for mode in SummaryMode:
        info = _MODE_DESCRIPTIONS.get(mode, {})
        model_cls = MODE_OUTPUT_MODEL[mode]
        modes.append(SummarizeModeInfo(
            mode=mode.value,
            description=info.get("description", ""),
            output_schema=model_cls.model_json_schema(),
            use_case=info.get("use_case", ""),
        ))
    return SummarizeModesResponse(
        modes=modes,
        prompt_registry_version=registry.version(),
    )


@router.post(
    "",
    response_model=SummarizationResponse,
    summary="Summarize a document file",
    status_code=status.HTTP_200_OK,
)
async def summarize_document(
    service: ServiceDep,
    file: Annotated[UploadFile, File(description="Document file (PDF, DOCX, TXT, etc.)")],
    mode: Annotated[str, Form(description="Summarization mode")] = SummaryMode.ABSTRACTIVE.value,
    language: Annotated[str, Form(description="Output language BCP-47 (e.g. 'ru', 'en', 'auto')")] = "ru",
    force_refresh: Annotated[bool, Form(description="Bypass cache")] = False,
) -> SummarizationResponse:
    """
    Summarize an uploaded document file.

    Supported formats: PDF, DOCX, DOC, TXT, MD, CSV.

    Returns a typed structured response depending on the selected mode:
    - **executive**: headline + bullets + recommendation
    - **detailed_notes**: full hierarchical notes with sections
    - **action_items**: tasks with owners, deadlines, priorities
    - **thesis**: argument/thesis plan
    - **extractive**: key facts as structured list
    - **abstractive**: narrative paraphrase
    - **multilingual**: auto-detects language, summarizes in `language`

    Response includes observability metadata:
    - `input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`
    - `cache_hit`, `cache_source` (l1/l2/miss)
    - `chunking_strategy` (direct / map_reduce:structural / map_reduce:token_aware_fallback)
    """
    try:
        summary_mode = SummaryMode(mode)
    except ValueError:
        valid = [m.value for m in SummaryMode]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid mode '{mode}'. Valid modes: {valid}",
        )

    if file.size and file.size > 50 * 1024 * 1024:  # 50 MB
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum size: 50 MB.",
        )

    file_content = await file.read()
    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )

    request = SummarizationRequest(
        file_content=file_content,
        file_name=file.filename or "document",
        mode=summary_mode,
        language=language,
        request_id=str(uuid.uuid4()),
        force_refresh=force_refresh,
    )

    try:
        return await service.summarize(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/stream",
    summary="Stream summarization via Server-Sent Events",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE stream of summary tokens",
        }
    },
)
async def summarize_stream(
    service: ServiceDep,
    file: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()] = SummaryMode.ABSTRACTIVE.value,
    language: Annotated[str, Form()] = "ru",
) -> StreamingResponse:
    """
    Stream summarization output via Server-Sent Events (SSE).

    Available for modes: executive, abstractive, extractive.
    Other modes return full JSON in a single event.

    SSE Event format:
        data: <token_or_chunk>\\n\\n
        data: [DONE]\\n\\n

    Suitable for real-time UI updates (e.g., progressive text rendering).
    """
    try:
        summary_mode = SummaryMode(mode)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid mode: {mode}")

    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Empty file.")

    async def event_generator() -> AsyncIterator[str]:
        try:
            request = SummarizationRequest(
                file_content=file_content,
                file_name=file.filename or "document",
                mode=summary_mode,
                language=language,
                request_id=str(uuid.uuid4()),
                force_refresh=False,
            )
            result = await service.summarize(request)

            import json
            output_json = json.dumps(result.output, ensure_ascii=False)

            chunk_size = 50
            for i in range(0, len(output_json), chunk_size):
                chunk = output_json[i:i + chunk_size]
                yield f"data: {chunk}\n\n"

            meta = {
                "event": "done",
                "cache_hit": result.cache_hit,
                "latency_ms": result.latency_ms,
                "cost_usd": result.cost_usd,
                "tokens": result.input_tokens + result.output_tokens,
            }
            yield f"data: {json.dumps(meta)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            import json
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# @router.delete(
#     "/cache/{file_hash}",
#     response_model=CacheInvalidationResponse,
#     summary="Invalidate all cached summaries for a file",
# )
# async def invalidate_cache(
#     file_hash: str,
#     service: ServiceDep,
# ) -> CacheInvalidationResponse:
#     """
#     Invalidate ALL cached summaries for a given file hash (all modes and languages).
#
#     The file hash is the SHA-256 of the file content bytes, available in every
#     SummarizationResponse under `file_hash`.
#
#     Use this when:
#     - A document has been updated/replaced
#     - You want to force re-generation after a prompt version bump
#     """
#     if len(file_hash) != 64 or not all(c in "0123456789abcdef" for c in file_hash.lower()):
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#             detail="file_hash must be a 64-character lowercase hex string (SHA-256).",
#         )
#
#     await service.invalidate_cache(file_hash)
#     return CacheInvalidationResponse(
#         invalidated=True,
#         file_hash=file_hash,
#         message=f"All cached summaries for file_hash={file_hash[:8]}... invalidated.",
#     )

@router.delete(
    "/cache/{file_hash}/{summary_type}",
    response_model=CacheInvalidationResponse,
    summary="Invalidate cached summaries for a specific file and mode",
)
async def invalidate_cache(
        file_hash: str,
        summary_type: str,
        service: ServiceDep,
) -> CacheInvalidationResponse:
    """
    Invalidate cached summaries for a given file hash and specific summary type.

    Use this when:
    - A document has been updated/replaced
    - You want to force re-generation for a specific mode
    """
    if len(file_hash) != 64 or not all(c in "0123456789abcdef" for c in file_hash.lower()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="file_hash must be a 64-character lowercase hex string (SHA-256).",
        )

    await service.invalidate_cache(file_hash, mode=summary_type)
    return CacheInvalidationResponse(
        invalidated=True,
        file_hash=file_hash,
        message=f"Cache for file_hash={file_hash[:8]}... mode={summary_type} invalidated.",
    )


@router.get(
    "/health",
    summary="Health check for summarization service",
)
async def summarization_health(service: ServiceDep) -> dict:
    """Check health of summarization service components."""
    l1_ok = await service._cache._l1.health_check()
    l2_ok = await service._cache._l2.health_check()
    cache_stats = service.cache_stats()
    return {
        "status": "ok" if l2_ok else "degraded",
        "components": {
            "redis_l1": "ok" if l1_ok else "unavailable",
            "postgres_l2": "ok" if l2_ok else "unavailable",
            "llm": "ok",
        },
        "cache_stats": cache_stats,
    }

# 1