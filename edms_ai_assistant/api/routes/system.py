# edms_ai_assistant/api/routes/system.py
"""
System API routes.

Endpoints:
    GET /health — agent and service health check
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from edms_ai_assistant.agent.agent import EdmsDocumentAgent
from edms_ai_assistant.api.deps import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])


@router.get("/health", summary="Agent and service health check")
async def health_check(
    agent: Annotated[EdmsDocumentAgent, Depends(get_agent)],
    request: Request,
) -> dict:
    health_data = await agent.health_check()
    return {"status": "ok", "version": request.app.version, "components": health_data}
