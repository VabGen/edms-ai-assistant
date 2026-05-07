# edms_ai_assistant/api/deps.py
"""
Shared FastAPI dependencies and application-level constants.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from edms_ai_assistant.agent.agent import EdmsDocumentAgent

UPLOAD_DIR: Path = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"


def get_agent(request: Request) -> EdmsDocumentAgent:
    """FastAPI dependency: return the agent from app.state.

    Raises:
        HTTPException 503: Agent not yet initialised.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="ИИ-Агент не инициализирован.")
    return agent


AgentDep = Annotated[EdmsDocumentAgent, Depends(get_agent)]
