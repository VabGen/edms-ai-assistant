from __future__ import annotations

import asyncio
import logging
import re
import shutil
import signal
import time
import uuid
import tempfile
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Annotated, Callable, Dict, Optional

import aiofiles
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage

from edms_ai_assistant.agent import EdmsDocumentAgent
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.config import settings
from edms_ai_assistant.model import AssistantResponse, FileUploadResponse, NewChatRequest, UserInput
from edms_ai_assistant.security import extract_user_id_from_token


def safe_extra(**kwargs: object) -> dict[str, object]:
    """Prefix reserved LogRecord keys to avoid log-record conflicts.

    Args:
        **kwargs: Arbitrary key-value pairs.

    Returns:
        Safe dict for use as ``extra`` in logger calls.
    """
    reserved = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "lineno", "funcName", "created",
        "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "exc_info",
        "exc_text", "stack_info",
    }
    return {f"ctx_{k}" if k in reserved else k: v for k, v in kwargs.items()}


logging.basicConfig(
    level=getattr(settings, "logging_level", "INFO").upper(),
    format=getattr(settings, "logging_format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"
_agent: Optional[EdmsDocumentAgent] = None
_shutdown_event = asyncio.Event()

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

API_PORT: int = getattr(settings, "api_port", 8000)
DEBUG: bool = getattr(settings, "debug", False)
LOGGING_LEVEL: str = getattr(settings, "logging_level", "INFO")
MAX_FILE_SIZE_MB: int = getattr(settings, "MAX_FILE_SIZE_MB", 50)
ALLOWED_EXTENSIONS: str = getattr(settings, "ALLOWED_FILE_EXTENSIONS", ".docx,.doc,.pdf,.txt,.rtf,.xlsx,.xls")
RATE_LIMIT_MAX_REQUESTS: int = getattr(settings, "RATE_LIMIT_MAX_REQUESTS", 10)
RATE_LIMIT_WINDOW_SECONDS: int = getattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60)


ALLOWED_EXTENSIONS_SET: set[str] = {
    ext.strip().lower() for ext in ALLOWED_EXTENSIONS.split(",")
}


class RateLimiter:
    """Sliding-window in-memory rate limiter.

    Args:
        max_requests: Maximum allowed requests per window.
        window_seconds: Window duration in seconds.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._store: Dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_ip: str) -> bool:
        """Check whether *client_ip* is within its request quota.

        Args:
            client_ip: Remote IP address.

        Returns:
            True if the request is permitted.
        """
        async with self._lock:
            now = time.time()
            self._store[client_ip] = [
                t for t in self._store[client_ip] if now - t < self.window_seconds
            ]
            if len(self._store[client_ip]) >= self.max_requests:
                return False
            self._store[client_ip].append(now)
            return True

    def get_remaining(self, client_ip: str) -> int:
        """Return remaining request quota for *client_ip*.

        Args:
            client_ip: Remote IP address.

        Returns:
            Non-negative integer quota.
        """
        now = time.time()
        valid = [t for t in self._store.get(client_ip, []) if now - t < self.window_seconds]
        return max(0, self.max_requests - len(valid))


_rate_limiter = RateLimiter(
    max_requests=RATE_LIMIT_MAX_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)


def rate_limit(
    max_requests: Optional[int] = None,
    window_seconds: Optional[int] = None,
) -> Callable[[Callable], Callable]:
    """Decorator factory that applies per-IP rate limiting.

    Args:
        max_requests: Override for the limiter max requests.
        window_seconds: Override for the limiter window.

    Returns:
        Async decorator.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args: object, **kwargs: object) -> object:
            client_ip = request.client.host if request.client else "unknown"
            prev_max = _rate_limiter.max_requests
            prev_window = _rate_limiter.window_seconds
            if max_requests is not None:
                _rate_limiter.max_requests = max_requests
            if window_seconds is not None:
                _rate_limiter.window_seconds = window_seconds
            try:
                if not await _rate_limiter.is_allowed(client_ip):
                    remaining = _rate_limiter.get_remaining(client_ip)
                    logger.warning(
                        "Rate limit exceeded",
                        extra=safe_extra(
                            trace_id=getattr(request.state, "trace_id", "unknown"),
                            remaining=remaining,
                            client_ip=client_ip,
                        ),
                    )
                    raise HTTPException(
                        status_code=429,
                        detail="Слишком много запросов. Попробуйте позже.",
                        headers={"X-RateLimit-Remaining": str(remaining)},
                    )
                return await func(request, *args, **kwargs)
            finally:
                _rate_limiter.max_requests = prev_max
                _rate_limiter.window_seconds = prev_window

        return wrapper
    return decorator


def get_agent() -> EdmsDocumentAgent:
    """FastAPI dependency: resolve the singleton agent or raise 503.

    Returns:
        Initialised EdmsDocumentAgent.

    Raises:
        HTTPException: 503 if the agent is not initialised.
    """
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="ИИ-Агент не инициализирован. Проверьте логи запуска.",
        )
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialise agent on startup, clean up on shutdown."""
    global _agent
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Upload directory ready", extra=safe_extra(upload_dir=str(UPLOAD_DIR)))
    try:
        _agent = EdmsDocumentAgent()
        logger.info("EDMS Assistant started", extra=safe_extra(version="2.1.2"))
    except Exception as exc:
        logger.critical(
            "Agent initialisation failed",
            extra=safe_extra(error=str(exc)),
            exc_info=True,
        )
        _agent = None
        raise

    yield

    logger.info("Graceful shutdown initiated")
    _shutdown_event.set()
    try:
        await asyncio.wait_for(_shutdown_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Shutdown timeout – forcing cleanup")
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("Upload directory cleaned")
    logger.info("Application shutdown complete")


app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.1.2",
    description="Production-ready AI assistant for EDMS document management",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

_allowed_origins_raw = getattr(settings, "ALLOWED_ORIGINS", "*")
if isinstance(_allowed_origins_raw, list):
    _cors_origins = _allowed_origins_raw
elif _allowed_origins_raw == "*":
    _cors_origins = ["*"]
else:
    _cors_origins = [o.strip() for o in str(_allowed_origins_raw).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Trace-ID", "X-RateLimit-Remaining"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Callable) -> Response:
    """Inject standard security response headers."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.middleware("http")
async def add_trace_id_and_timing(request: Request, call_next: Callable) -> Response:
    """Assign trace/request IDs and log response timing."""
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    request_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    request.state.request_id = request_id
    request.state.start_time = time.time()
    try:
        response = await call_next(request)
        duration_ms = (time.time() - request.state.start_time) * 1000
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        logger.info(
            "Request completed",
            extra=safe_extra(
                trace_id=trace_id,
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            ),
        )
        return response
    except Exception as exc:
        duration_ms = (time.time() - request.state.start_time) * 1000
        logger.error(
            "Request failed",
            extra=safe_extra(
                trace_id=trace_id,
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(exc),
                duration_ms=round(duration_ms, 2),
            ),
            exc_info=True,
        )
        raise


@app.middleware("http")
async def validate_request(request: Request, call_next: Callable) -> Response:
    """Block path traversal and oversized request bodies."""
    if ".." in request.url.path:
        logger.warning(
            "Path traversal attempt blocked",
            extra=safe_extra(
                trace_id=getattr(request.state, "trace_id", "unknown"),
                path=request.url.path,
            ),
        )
        raise HTTPException(status_code=400, detail="Invalid path")
    if request.method in ("POST", "PUT"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Request too large")
    return await call_next(request)


def _cleanup_file(file_path: str) -> None:
    """Safely delete a file only if it resides within UPLOAD_DIR.

    Args:
        file_path: Absolute path string to the target file.
    """
    try:
        p = Path(file_path)
        resolved = p.resolve()
        upload_resolved = UPLOAD_DIR.resolve()
        if p.exists() and p.is_file() and (
            upload_resolved in resolved.parents or resolved.parent == upload_resolved
        ):
            p.unlink()
            logger.debug("File cleaned", extra=safe_extra(file_path=file_path))
        else:
            logger.warning(
                "Attempted to delete file outside upload dir",
                extra=safe_extra(file_path=file_path),
            )
    except Exception as exc:
        logger.warning(
            "Failed to cleanup file",
            extra=safe_extra(file_path=file_path, error=str(exc)),
        )


@app.get("/health", tags=["health"], summary="Basic health check")
async def health_check() -> dict:
    """Return basic liveness information."""
    return {"status": "healthy", "timestamp": time.time(), "version": "2.1.2"}


@app.get("/health/ready", tags=["health"], summary="Readiness check")
async def readiness_check(
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    """Return readiness state including agent health indicators.

    Args:
        agent: Injected agent instance.

    Returns:
        Health dict.

    Raises:
        HTTPException: 503 if any critical indicator is False.
    """
    health = agent.health_check()
    is_ready = all([
        health.get("model", False),
        health.get("tools", False),
        health.get("graph", False),
        health.get("state_manager", False),
    ])
    if not is_ready:
        logger.warning("Agent not ready", extra=safe_extra(health=health))
        raise HTTPException(
            status_code=503,
            detail={"reason": "Agent not ready", "health": health},
        )
    return {"status": "ready", "health": health, "timestamp": time.time()}


@app.get("/health/live", tags=["health"], summary="Liveness check")
async def liveness_check() -> dict:
    """Return simple liveness confirmation."""
    return {"status": "alive", "timestamp": time.time()}


@app.post(
    "/chat",
    response_model=AssistantResponse,
    tags=["chat"],
    summary="Отправить сообщение агенту",
)
@rate_limit(max_requests=10, window_seconds=60)
async def chat_endpoint(
    request: Request,
    user_input: UserInput,
    background_tasks: BackgroundTasks,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    """Process a chat message through the EDMS agent.

    Args:
        request: FastAPI request (injected by rate_limit decorator).
        user_input: Validated user payload.
        background_tasks: FastAPI background task queue.
        agent: Injected agent instance.

    Returns:
        Standardised AssistantResponse.

    Raises:
        HTTPException: 401 on invalid token.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.info(
        "Chat request received",
        extra=safe_extra(
            trace_id=trace_id,
            message_length=len(user_input.message),
            context_ui_id=user_input.context_ui_id,
            file_path=str(user_input.file_path)[:80] if user_input.file_path else None,
        ),
    )

    try:
        user_id = extract_user_id_from_token(user_input.user_token)
    except Exception as exc:
        logger.error(
            "Token extraction failed",
            extra=safe_extra(trace_id=trace_id, error=str(exc)),
        )
        raise HTTPException(status_code=401, detail="Неверный токен")

    thread_id = (
        user_input.thread_id
        or f"user_{user_id}_doc_{user_input.context_ui_id or 'general'}"
    )

    user_context = user_input.context.model_dump() if user_input.context else None
    if not user_context:
        try:
            async with EmployeeClient() as emp_client:
                user_context = await emp_client.get_employee(user_input.user_token, user_id)
        except Exception:
            user_context = {"firstName": "Коллега"}
            logger.warning(
                "Failed to fetch user context – using default",
                extra=safe_extra(trace_id=trace_id),
            )

    result = await agent.chat(
        message=user_input.message,
        user_token=user_input.user_token,
        context_ui_id=user_input.context_ui_id,
        thread_id=thread_id,
        user_context=user_context,
        file_path=user_input.file_path,
        human_choice=user_input.human_choice,
    )

    if user_input.file_path:
        is_system_attachment = bool(UUID_PATTERN.match(str(user_input.file_path)))
        if not is_system_attachment and result.get("status") not in (
            "requires_action",
            "requires_choice",
        ):
            background_tasks.add_task(_cleanup_file, user_input.file_path)

    logger.info(
        "Chat request completed",
        extra=safe_extra(
            trace_id=trace_id,
            status=result.get("status"),
            content_length=len(result.get("content") or ""),
        ),
    )

    return AssistantResponse(
        status=result.get("status", "success"),
        response=result.get("content"),
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id,
    )


@app.post(
    "/actions/summarize",
    response_model=AssistantResponse,
    tags=["actions"],
    summary="Суммаризация документа",
)
@rate_limit(max_requests=5, window_seconds=60)
async def api_direct_summarize(
    request: Request,
    user_input: UserInput,
    background_tasks: BackgroundTasks,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> AssistantResponse:
    """Trigger document summarisation via the agent.

    Args:
        request: FastAPI request.
        user_input: Validated user payload.
        background_tasks: FastAPI background task queue.
        agent: Injected agent instance.

    Returns:
        AssistantResponse with summarised content.

    Raises:
        HTTPException: 500 on unexpected errors.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.info(
        "Summarize request received",
        extra=safe_extra(
            trace_id=trace_id,
            context_ui_id=user_input.context_ui_id,
            file_path=str(user_input.file_path)[:50] if user_input.file_path else None,
        ),
    )

    try:
        user_id = extract_user_id_from_token(user_input.user_token)
        new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:6]}"
        current_path = user_input.file_path

        if current_path and not UUID_PATTERN.match(str(current_path)):
            if not Path(current_path).exists():
                logger.warning(
                    "Local file not found",
                    extra=safe_extra(trace_id=trace_id, file_path=current_path),
                )
                current_path = None

        agent_result = await agent.chat(
            message=f"Проанализируй вложение: {user_input.message}",
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
            thread_id=new_thread_id,
            file_path=current_path,
            human_choice=user_input.human_choice,
        )

        if current_path and not UUID_PATTERN.match(str(current_path)):
            background_tasks.add_task(_cleanup_file, current_path)

        logger.info(
            "Summarize request completed",
            extra=safe_extra(trace_id=trace_id, status=agent_result.get("status")),
        )

        return AssistantResponse(
            status="success",
            response=agent_result.get("content") or "Анализ готов.",
            thread_id=new_thread_id,
        )

    except Exception as exc:
        logger.error(
            "Summarize error",
            exc_info=True,
            extra=safe_extra(trace_id=trace_id, error=str(exc)),
        )
        if user_input.file_path and not UUID_PATTERN.match(str(user_input.file_path)):
            _cleanup_file(user_input.file_path)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chat/history/{thread_id}", tags=["chat"], summary="История сообщений треда")
async def get_history(
    thread_id: str,
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
) -> dict:
    """Return filtered message history for a conversation thread.

    Args:
        thread_id: LangGraph thread identifier.
        agent: Injected agent instance.

    Returns:
        Dict with ``messages`` list and ``count``.
    """
    try:
        state = await agent.state_manager.get_state(thread_id)
        messages = state.values.get("messages", [])
        filtered = []
        for m in messages:
            if not isinstance(m, (HumanMessage, AIMessage)):
                continue
            if isinstance(m, AIMessage) and not m.content:
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                continue
            filtered.append(
                {
                    "type": "human" if isinstance(m, HumanMessage) else "ai",
                    "content": m.content,
                }
            )
        return {"messages": filtered, "count": len(filtered)}
    except Exception as exc:
        logger.error("History error", extra=safe_extra(error=str(exc)))
        return {"messages": [], "count": 0, "error": str(exc)}


@app.post("/chat/new", tags=["chat"], summary="Создать новый чат-тред")
async def create_new_thread(req: NewChatRequest) -> dict:
    """Create a fresh conversation thread for the authenticated user.

    Args:
        req: Request containing the bearer token.

    Returns:
        Dict with ``status`` and ``thread_id``.

    Raises:
        HTTPException: 401 on invalid token.
    """
    try:
        user_id = extract_user_id_from_token(req.user_token)
        new_thread_id = f"chat_{user_id}_{uuid.uuid4().hex[:6]}"
        return {"status": "success", "thread_id": new_thread_id}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post(
    "/upload-file",
    response_model=FileUploadResponse,
    tags=["files"],
    summary="Загрузить файл",
)
@rate_limit(max_requests=20, window_seconds=60)
async def upload_file(
    request: Request,
    user_token: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    """Upload a file to the temporary upload directory.

    Args:
        request: FastAPI request.
        user_token: Bearer token submitted as form field.
        file: Uploaded file.

    Returns:
        FileUploadResponse with path, name and size.

    Raises:
        HTTPException: 400 on invalid extension or oversized file; 500 on IO error.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    try:
        user_id = extract_user_id_from_token(user_token)

        if not file.filename:
            raise HTTPException(status_code=400, detail="Имя файла не указано")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS_SET:
            logger.warning(
                "Invalid file extension",
                extra=safe_extra(
                    trace_id=trace_id,
                    file_name=file.filename,
                    extension=suffix,
                ),
            )
            raise HTTPException(
                status_code=400,
                detail=f"Недопустимый тип файла. Разрешены: {ALLOWED_EXTENSIONS}",
            )

        file_id = f"{user_id}_{uuid.uuid4().hex}{suffix}"
        dest_path = UPLOAD_DIR / file_id
        max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        file_size = 0

        async with aiofiles.open(dest_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)
                file_size += len(chunk)
                if file_size > max_size_bytes:
                    dest_path.unlink(missing_ok=True)
                    logger.warning(
                        "File too large",
                        extra=safe_extra(
                            trace_id=trace_id,
                            file_name=file.filename,
                            size_bytes=file_size,
                        ),
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Файл слишком большой (макс. {MAX_FILE_SIZE_MB}MB)",
                    )

        logger.info(
            "File uploaded",
            extra=safe_extra(
                trace_id=trace_id,
                file_id=file_id,
                file_name=file.filename,
                size_bytes=file_size,
                size_mb=round(file_size / 1024 / 1024, 2),
            ),
        )
        return FileUploadResponse(
            file_path=str(dest_path),
            file_name=file.filename,
            size_bytes=file_size,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Upload error",
            exc_info=True,
            extra=safe_extra(trace_id=trace_id, error=str(exc)),
        )
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return a structured JSON payload for all HTTP exceptions."""
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.warning(
        "HTTP error",
        extra=safe_extra(
            trace_id=trace_id,
            status_code=exc.status_code,
            path=request.url.path,
            detail=exc.detail,
        ),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "trace_id": trace_id,
            "status_code": exc.status_code,
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 500 JSON payload for all unhandled exceptions."""
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.error(
        "Unhandled error",
        extra=safe_extra(
            trace_id=trace_id,
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
        ),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Внутренняя ошибка сервера",
            "trace_id": trace_id,
            "type": type(exc).__name__,
        },
    )


def _handle_signal(signum: int, frame: object) -> None:
    logger.info("Received shutdown signal", extra=safe_extra(signal=signum))
    _shutdown_event.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


if __name__ == "__main__":
    logger.info(
        "Starting EDMS AI Assistant API",
        extra=safe_extra(
            port=API_PORT,
            debug=DEBUG,
            log_level=LOGGING_LEVEL,
            version="2.1.2",
        ),
    )
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=DEBUG,
        log_level=LOGGING_LEVEL.lower(),
        access_log=True,
        timeout_keep_alive=30,
    )