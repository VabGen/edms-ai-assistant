from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse

from edms_ai_assistant.schemas.summarization import SummarizeRequest
from edms_ai_assistant.services.summarization_orchestrator import stream_summarize
from edms_ai_assistant.core.dependencies import get_rate_limiter
from edms_ai_assistant.services.rate_limiter import RateLimiter
from edms_ai_assistant.core.exceptions import AppException

router = APIRouter()

@router.post("/actions/summarize/stream", responses={200: {"content": {"text/event-stream": {}}}})
async def stream_summarize_endpoint(
    request: Request,
    body: SummarizeRequest,
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    client_host = request.client.host if request.client else "unknown"
    if await rate_limiter.is_rate_limited(f"summarize:{client_host}"):
        raise AppException(status_code=429, detail="Too many requests", error_code="RATE_LIMIT")

    async def _event_generator():
        try:
            async for token in stream_summarize(
                text=body.text, fmt=body.summary_type, file_identifier=body.file_identifier
            ):
                clean_token = token.replace("\n", "\\n")
                yield f"data: {clean_token}\n\n"
                if await request.is_disconnected():
                    break
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"}
    )