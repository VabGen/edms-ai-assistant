# edms_ai_assistant/api/routes/summarize.py
"""Streaming summarization endpoint.

Emits Server-Sent Events (text/event-stream) for incremental LLM output.

SSE event protocol
------------------
data: {"type": "status",    "status": "checking_cache"}
data: {"type": "status",    "status": "extracting_text"}
data: {"type": "status",    "status": "streaming"}
data: {"type": "metadata",  "file_identifier": "...", "summary_type": "...", "text_length": N}
data: {"type": "cache_hit", "content": "...", "metadata": {...}}
data: {"type": "progress",  "stage": "map", "done": 2, "total": 5}
data: {"type": "progress",  "stage": "reduce", "total": 5}
data: {"type": "error",     "error": "..."}
data: <raw LLM token>           ← newlines escaped as \\n
data: [DONE]
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.core.dependencies import get_rate_limiter
from edms_ai_assistant.core.exceptions import AppException
from edms_ai_assistant.model import UserInput
from edms_ai_assistant.services.rate_limiter import RateLimiter
from edms_ai_assistant.services.summarization_orchestrator import (
    get_orchestrator,
    stream_summarize,
)
from edms_ai_assistant.tools.attachment import doc_get_file_content
from edms_ai_assistant.utils.hash_utils import get_file_hash

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Summarization"])

_SSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_PROGRESS_PREFIX = "__progress__:"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _is_system_attachment(file_path: str | None) -> bool:
    return bool(file_path and _UUID_RE.match(str(file_path)))


def _client_key(request: Request) -> str:
    if request.client:
        return f"summarize:{request.client.host}"
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return f"summarize:{forwarded.split(',')[0].strip()}"
    return "summarize:unknown"


def _norm(s: str) -> str:
    return re.sub(r"[^a-zA-Zа-яА-Я0-9]", "", s.lower()) if s else ""


def _sse(payload: object) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _att_field(att: object, field: str) -> str:
    return str(
        att.get(field, "") if isinstance(att, dict) else getattr(att, field, "")
    ) or ""


# ── File resolution ────────────────────────────────────────────────────────────


async def _resolve_file_identifier(body: UserInput) -> tuple[str | None, str, bool]:
    """Resolve a stable cache key without extracting text yet.

    This runs BEFORE the cache check — if we get a cache hit we never need
    to call the EDMS API for the document text at all.

    Returns:
        (file_identifier, current_path, is_uuid)
    """
    current_path = (body.file_path or "").strip()
    is_uuid = _is_system_attachment(current_path)

    if is_uuid:
        return current_path, current_path, True

    if current_path and Path(current_path).exists():
        try:
            return get_file_hash(current_path), current_path, False
        except Exception as exc:
            logger.warning("Could not hash local file: %s", exc)
            return None, current_path, False

    if body.context_ui_id:
        try:
            async with DocumentClient() as doc_client:
                doc_dto = await doc_client.get_document_metadata(
                    body.user_token, body.context_ui_id
                )
            attachments: list = (
                doc_dto.attachmentDocument
                if hasattr(doc_dto, "attachmentDocument")
                else (
                    doc_dto.get("attachmentDocument") or []
                    if isinstance(doc_dto, dict)
                    else []
                )
            )

            clean_input = _norm(current_path)
            file_identifier: str | None = None

            if clean_input:
                for att in attachments:
                    if clean_input in _norm(_att_field(att, "name")):
                        file_identifier = _att_field(att, "id") or None
                        break

            if not file_identifier and attachments:
                file_identifier = _att_field(attachments[0], "id") or None

            if file_identifier:
                return file_identifier, file_identifier, True

        except Exception as exc:
            logger.error("Error resolving EDMS attachments: %s", exc)

    return None, current_path, False


async def _extract_text(
    body: UserInput, current_path: str, is_uuid: bool
) -> str:
    """Extract raw document text. Only called on cache MISS."""
    raw_text = ""
    try:
        if is_uuid and body.context_ui_id:
            result = await doc_get_file_content.ainvoke({
                "document_id": body.context_ui_id,
                "token": body.user_token,
                "context_ui_id": body.context_ui_id,
                "attachment_id": current_path,
            })
            raw_text = str(result)
        elif current_path and Path(current_path).exists():
            from edms_ai_assistant.services.file_processor import extract_text_from_file
            raw_text = await extract_text_from_file(current_path)
    except Exception as exc:
        logger.warning("Text extraction failed: %s", exc)
    return raw_text


# ── SSE event stream ───────────────────────────────────────────────────────────


async def _event_stream(
    request: Request,
    body: UserInput,
) -> AsyncGenerator[str, None]:
    """Generate SSE events for the full summarization lifecycle.

    Fast path (cache hit):
        status:checking_cache → metadata → cache_hit → [DONE]

    Slow path (cache miss):
        status:checking_cache → metadata →
        status:extracting_text →
        status:streaming → tokens / progress → [DONE]

    Progress events (map-reduce only):
        data: {"type":"progress","stage":"map","done":N,"total":M}
        data: {"type":"progress","stage":"reduce","total":M}
    """
    summary_type = body.human_choice or "extractive"
    orchestrator = get_orchestrator()

    try:
        # ── Step 1: resolve identifier (cheap, no text extraction yet) ───────
        yield _sse({"type": "status", "status": "checking_cache"})
        file_identifier, current_path, is_uuid = await _resolve_file_identifier(body)

        yield _sse({
            "type": "metadata",
            "file_identifier": file_identifier,
            "summary_type": summary_type,
            "context_ui_id": body.context_ui_id,
        })

        # ── Step 2: cache check BEFORE text extraction ────────────────────────
        if file_identifier:
            cached = await orchestrator.check_cache(file_identifier, summary_type)
            if cached is not None:
                logger.info("Cache HIT for %s…", file_identifier[:8])
                yield _sse({
                    "type": "cache_hit",
                    "content": cached.content,
                    "metadata": {
                        "cache_file_identifier": file_identifier,
                        "cache_summary_type": summary_type,
                        "cache_context_ui_id": body.context_ui_id,
                        "from_cache": True,
                        "pipeline": cached.pipeline,
                        "quality_score": cached.quality_score,
                    },
                })
                return  # ← skip all text extraction and LLM calls

        # ── Step 3: extract text only on cache MISS ───────────────────────────
        yield _sse({"type": "status", "status": "extracting_text"})
        raw_text = await _extract_text(body, current_path, is_uuid)

        if not raw_text or len(raw_text.strip()) < 30:
            yield _sse({
                "type": "error",
                "error": "Не удалось извлечь текст из файла для анализа.",
            })
            return

        # ── Step 4: stream LLM tokens ─────────────────────────────────────────
        yield _sse({"type": "status", "status": "streaming"})

        async for token in stream_summarize(
            text=raw_text,
            fmt=summary_type,
            file_identifier=file_identifier,
        ):
            if await request.is_disconnected():
                logger.info("Client disconnected — aborting stream")
                return

            # Intercept progress markers from orchestrator
            if token.startswith(_PROGRESS_PREFIX):
                try:
                    progress_data = json.loads(token[len(_PROGRESS_PREFIX):])
                    yield _sse({"type": "progress", **progress_data})
                except Exception:
                    pass  # Malformed progress marker — silently skip
                continue

            # Regular token — escape newlines for SSE framing
            yield f"data: {token.replace(chr(10), chr(92) + 'n')}\n\n"

    except Exception as exc:
        logger.error("SSE stream error", exc_info=True)
        yield _sse({"type": "error", "error": str(exc)})

    finally:
        yield "data: [DONE]\n\n"


# ── Endpoint ───────────────────────────────────────────────────────────────────


@router.post(
    "/actions/summarize/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream summarization output as Server-Sent Events",
    responses={
        200: {"content": {"text/event-stream": {}}},
        429: {"description": "Rate limit exceeded"},
    },
)
async def stream_summarize_endpoint(
    request: Request,
    body: UserInput,
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
) -> StreamingResponse:
    """Stream LLM-generated summary tokens as SSE.

    Rate-limited per client IP; Redis failure is handled gracefully (skipped,
    not 500) so summarization is never blocked by Redis downtime.
    """
    key = _client_key(request)

    try:
        if await rate_limiter.is_rate_limited(key):
            raise AppException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests — please slow down.",
                error_code="RATE_LIMIT",
            )
    except AppException:
        raise
    except RuntimeError:
        logger.warning("Rate limiter skipped: Redis not initialised")
    except Exception as exc:
        logger.warning("Rate limiter error (%s) — skipping", exc)

    return StreamingResponse(
        _event_stream(request=request, body=body),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )