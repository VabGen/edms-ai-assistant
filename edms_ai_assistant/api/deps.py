# edms_ai_assistant/api/deps.py
"""
Shared FastAPI dependencies and application-level constants.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from edms_ai_assistant.agent.agent import EdmsDocumentAgent
from edms_ai_assistant.core.deps import AppDeps
from edms_ai_assistant.core.di_container import get_container
from edms_ai_assistant.security import decode_token, logger

UPLOAD_DIR: Path = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

# ── Auth Security Scheme ─────────────────────────────────────────────────────
_bearer_scheme = HTTPBearer(auto_error=False)


# ── Agent Dependencies ───────────────────────────────────────────────────────


def get_agent(request: Request) -> EdmsDocumentAgent:
    """FastAPI dependency: return the agent from DI container.

    Raises:
        HTTPException 503: Agent or container not initialised.
    """
    try:
        container = get_container()
        return container.agent()
    except RuntimeError:
        agent = getattr(request.app.state, "agent", None)
        if agent is None:
            raise HTTPException(status_code=503, detail="ИИ-Агент не инициализирован.")
        return agent


AgentDep = Annotated[EdmsDocumentAgent, Depends(get_agent)]


def get_deps(request: Request) -> AppDeps:
    """FastAPI dependency: return the AppDeps container from DI container.

    Raises:
        HTTPException 503: Dependencies or container not initialised.
    """
    try:
        container = get_container()
        return container.app_deps()
    except RuntimeError:
        # Fallback to app.state for backward compatibility during migration
        deps = getattr(request.app.state, "deps", None)
        if deps is None:
            raise HTTPException(
                status_code=503, detail="Зависимости приложения не инициализированы."
            )
        return deps


DepsDep = Annotated[AppDeps, Depends(get_deps)]


# ── Auth Dependencies ────────────────────────────────────────────────────────


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
    ] = None,
) -> dict[str, Any]:
    """FastAPI dependency: extract and validate current user from JWT.
    """
    if credentials is None:
        logger.warning("Authentication failed: No credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not credentials.credentials or not credentials.credentials.strip():
        logger.warning("Authentication failed: Empty credentials in Bearer token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: empty credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_data = decode_token(credentials.credentials)
        return user_data
    except Exception as e:
        logger.warning(
            "Authentication failed: Invalid token",
            extra={"error": str(e), "token_prefix": credentials.credentials[:20] if credentials.credentials else "empty"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_admin_user(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    """FastAPI dependency: ensure the current user has admin privileges.

    Зависит от get_current_user, поэтому токен проверяется только один раз.

    Raises:
        HTTPException 403: Пользователь авторизован, но не имеет роли 'admin'.
    """
    user_role = (
        user.get("role") if isinstance(user, dict) else getattr(user, "role", None)
    )

    if user_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user does not have enough privileges (admin required)",
        )

    return user


CurrentUserDep = Annotated[dict[str, Any], Depends(get_current_user)]
AdminUserDep = Annotated[dict[str, Any], Depends(get_admin_user)]
