"""
FastAPI router для суммаризации.

Эндпоинты:
  GET  /summarize/modes                 — список режимов суммаризации
  POST /summarize                       — основная суммаризация (multipart upload)
  POST /summarize/stream                — SSE-стриминг финального вывода
  DELETE /summarize/cache/{hash}/{mode} — инвалидация кэша
  GET  /summarize/health                — проверка работоспособности

Схемы вынесены в `summarizer/api/schemas.py`. Доменные модели запроса/ответа
живут в `summarizer/service.py`.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Annotated, AsyncIterator

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse

from edms_ai_assistant.summarizer.api.schemas import (
    CacheInvalidationResponse,
    SummarizeModeInfo,
    SummarizeModesResponse,
)
from edms_ai_assistant.summarizer.errors import (
    LLMClientError,
    LLMRateLimitedError,
    LLMServerError,
    LLMTransportError,
    PipelineError,
    SummarizerError,
    TextExtractionError,
)
from edms_ai_assistant.summarizer.errors import ValidationError as SummarizerValidationError
from edms_ai_assistant.summarizer.pipeline.direct import StreamEvent
from edms_ai_assistant.summarizer.prompts.registry import get_prompt_registry
from edms_ai_assistant.summarizer.service import (
    SummarizationRequest,
    SummarizationResponse,
    SummarizationService,
    format_output_as_markdown,
)
from edms_ai_assistant.summarizer.structured.models import SummaryMode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summarize", tags=["Summarization"])

# Hex-валидация SHA-256 ключа кэша
_HEX_CHARS = frozenset("0123456789abcdef")
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_STREAM_CHUNK_BYTES = 256


# ---------------------------------------------------------------------------
# Описания режимов
# ---------------------------------------------------------------------------

_MODE_DESCRIPTIONS: dict[SummaryMode, dict[str, str]] = {
    SummaryMode.EXECUTIVE: {
        "description": "Краткая выжимка для руководителя: заголовок + 3-5 тезисов",
        "use_case": "Быстрые решения, управленческие брифинги",
    },
    SummaryMode.DETAILED_NOTES: {
        "description": "Полный структурированный конспект с сохранением иерархии",
        "use_case": "Юридический анализ, изучение договоров",
    },
    SummaryMode.ACTION_ITEMS: {
        "description": "Извлечение задач с ответственными, сроками и приоритетами",
        "use_case": "Протоколы совещаний, контроль исполнения",
    },
    SummaryMode.THESIS: {
        "description": "Иерархический тезисный план для аналитических документов",
        "use_case": "Научные документы, стратегические планы",
    },
    SummaryMode.EXTRACTIVE: {
        "description": "Ключевые факты, даты, суммы как структурированные данные",
        "use_case": "Извлечение данных, заполнение карточки СЭД",
    },
    SummaryMode.ABSTRACTIVE: {
        "description": "Связный пересказ своими словами в 2-4 абзацах",
        "use_case": "Общее понимание документа, брифинги",
    },
    SummaryMode.MULTILINGUAL: {
        "description": "Авто-определение языка источника, изложение на нужном языке",
        "use_case": "Многоязычная обработка (RU/BE/KZ/EN)",
    },
}


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_summarization_service(request: Request) -> SummarizationService:
    service: SummarizationService | None = getattr(
        request.app.state, "summarization_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SummarizationService не инициализирован.",
        )
    return service


ServiceDep = Annotated[SummarizationService, Depends(get_summarization_service)]


def _parse_mode(mode: str) -> SummaryMode:
    try:
        return SummaryMode(mode)
    except ValueError:
        valid = sorted(m.value for m in SummaryMode)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Неверный режим '{mode}'. Допустимые: {valid}",
        )


def _http_status_for(exc: SummarizerError) -> int:
    """Маппинг типизированных исключений на HTTP-коды."""
    if isinstance(exc, (TextExtractionError, SummarizerValidationError)):
        return status.HTTP_422_UNPROCESSABLE_ENTITY
    if isinstance(exc, LLMRateLimitedError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if isinstance(exc, LLMClientError):
        return status.HTTP_502_BAD_GATEWAY
    if isinstance(exc, (LLMServerError, LLMTransportError)):
        return status.HTTP_504_GATEWAY_TIMEOUT
    return status.HTTP_500_INTERNAL_SERVER_ERROR


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/modes",
    response_model=SummarizeModesResponse,
    summary="Список доступных режимов суммаризации",
)
async def get_modes(service: ServiceDep) -> SummarizeModesResponse:
    registry = get_prompt_registry()
    modes = [
        SummarizeModeInfo(
            mode=mode.value,
            description=_MODE_DESCRIPTIONS.get(mode, {}).get("description", ""),
            use_case=_MODE_DESCRIPTIONS.get(mode, {}).get("use_case", ""),
        )
        for mode in SummaryMode
    ]
    return SummarizeModesResponse(
        modes=modes,
        prompt_registry_version=registry.version(),
    )


@router.post(
    "",
    response_model=SummarizationResponse,
    summary="Суммаризация документа",
    status_code=status.HTTP_200_OK,
)
async def summarize_document(
    service: ServiceDep,
    file: Annotated[
        UploadFile, File(description="Файл документа (PDF, DOCX, TXT и др.)")
    ],
    mode: Annotated[
        str, Form(description="Режим суммаризации")
    ] = SummaryMode.ABSTRACTIVE.value,
    language: Annotated[
        str, Form(description="Язык вывода BCP-47 (ru, en, auto)")
    ] = "ru",
    force_refresh: Annotated[bool, Form(description="Игнорировать кэш")] = False,
    enable_quality_score: Annotated[
        bool,
        Form(description="Запустить LLM-as-judge для оценки качества (≈ +1 LLM-вызов)"),
    ] = False,
) -> SummarizationResponse:
    summary_mode = _parse_mode(mode)

    if file.size and file.size > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл слишком большой. Максимум: {_MAX_UPLOAD_BYTES // (1024 * 1024)} МБ.",
        )

    file_content = await file.read()
    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загружен пустой файл.",
        )

    req = SummarizationRequest(
        file_content=file_content,
        file_name=file.filename or "document",
        mode=summary_mode,
        language=language or "ru",
        request_id=str(uuid.uuid4()),
        force_refresh=force_refresh,
        enable_quality_score=enable_quality_score,
    )

    try:
        return await service.summarize(req)
    except SummarizerError as exc:
        code = _http_status_for(exc)
        if code < 500:
            logger.warning("Summarization rejected: %s", exc)
        else:
            logger.error("Summarization failed: %s", exc, exc_info=True)
        detail = str(exc) if code < 500 else "Ошибка суммаризации. См. логи сервиса."
        raise HTTPException(status_code=code, detail=detail)


@router.post(
    "/stream",
    summary="Стриминг финального вывода через Server-Sent Events",
    response_class=StreamingResponse,
)
async def summarize_stream(
    service: ServiceDep,
    file: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()] = SummaryMode.ABSTRACTIVE.value,
    language: Annotated[str, Form()] = "ru",
) -> StreamingResponse:
    summary_mode = _parse_mode(mode)

    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Пустой файл.")

    req = SummarizationRequest(
        file_content=file_content,
        file_name=file.filename or "document",
        mode=summary_mode,
        language=language or "ru",
        request_id=str(uuid.uuid4()),
        force_refresh=False,
    )

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in service.summarize_stream(req):
                if isinstance(event, StreamEvent):
                    if event.kind == "delta" and event.text:
                        yield _sse({"event": "delta", "text": event.text})
                    elif event.kind == "error":
                        yield _sse({"event": "error", "message": "LLM streaming error"})
                elif isinstance(event, SummarizationResponse):
                    formatted = format_output_as_markdown(event)
                    yield _sse(
                        {
                            "event": "result",
                            "formatted": formatted,
                            "cache_hit": event.cache_hit,
                            "cache_source": event.cache_source,
                            "latency_ms": event.latency_ms,
                            "cost_usd": event.cost_usd,
                            "input_tokens": event.input_tokens,
                            "output_tokens": event.output_tokens,
                            "chunking_strategy": event.chunking_strategy,
                            "request_id": event.request_id,
                        }
                    )
            yield "data: [DONE]\n\n"

        except SummarizerError as exc:
            code = _http_status_for(exc)
            if code < 500:
                logger.warning("Stream rejected: %s", exc)
                yield _sse({"event": "error", "message": str(exc)})
            else:
                logger.error("Stream summarization failed: %s", exc, exc_info=True)
                yield _sse({"event": "error", "message": "Ошибка суммаризации"})
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("Stream summarization failed: %s", exc, exc_info=True)
            yield _sse({"event": "error", "message": "Ошибка суммаризации"})
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


@router.delete(
    "/cache/{file_hash}/{summary_type}",
    response_model=CacheInvalidationResponse,
    summary="Инвалидация кэша для конкретного файла и режима",
)
async def invalidate_cache(
    file_hash: str,
    summary_type: str,
    service: ServiceDep,
) -> CacheInvalidationResponse:
    if len(file_hash) != 64 or any(c not in _HEX_CHARS for c in file_hash.lower()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="file_hash должен быть 64-символьной hex-строкой (SHA-256).",
        )
    _parse_mode(summary_type)  # validate

    await service.invalidate_cache(file_hash, mode=summary_type)
    return CacheInvalidationResponse(
        invalidated=True,
        file_hash=file_hash,
        message=f"Кэш для file_hash={file_hash[:8]}... mode={summary_type} инвалидирован.",
    )


@router.get("/health", summary="Проверка работоспособности сервиса суммаризации")
async def summarization_health(service: ServiceDep) -> dict:
    info = await service.health()
    cache_status = info["cache"]
    return {
        "status": cache_status.get("overall", "ok"),
        "components": {
            "redis_l1": cache_status.get("redis_l1"),
            "postgres_l2": cache_status.get("postgres_l2"),
            "llm": "ok",
        },
        "cache_stats": info["cache_stats"],
    }
