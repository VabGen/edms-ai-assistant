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

UPLOAD_DIR: Path = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

# ── Auth Security Scheme ─────────────────────────────────────────────────────
# auto_error=False позволяет обрабатывать отсутствие токена вручную
_bearer_scheme = HTTPBearer(auto_error=False)


# ── Agent Dependencies ───────────────────────────────────────────────────────

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


def get_deps(request: Request) -> AppDeps:
    """FastAPI dependency: return the AppDeps container from app.state."""
    deps = getattr(request.app.state, "deps", None)
    if deps is None:
        raise HTTPException(status_code=503, detail="Зависимости приложения не инициализированы.")
    return deps


DepsDep = Annotated[AppDeps, Depends(get_deps)]


# ── Auth Dependencies ────────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, Any]:
    """FastAPI dependency: extract and validate current user from JWT.

    NOTE: Это структурный каркас. Замените логику декодирования токена
    на вашу реальную реализацию (например, python-jose или PyJWT).

    Returns:
        Словарь с данными пользователя (например, {'id': '...', 'role': 'admin'}).

    Raises:
        HTTPException 401: Токен отсутствует или невалиден.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )


    # TODO: Заменить на реальную логику декодирования JWT
    # from edms_ai_assistant.core.security import decode_access_token
    # user_data = decode_access_token(token)
    # if not user_data:
    #     raise HTTPException(status_code=401, detail="Invalid or expired token")
    # return user_data

    # ── Временный заглушка для компиляции (УДАЛИТЬ ПРИ РЕАЛИЗАЦИИ) ──
    raise NotImplementedError(
        "JWT token decoding is not implemented in edms_ai_assistant.api.deps"
    )


async def get_admin_user(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """FastAPI dependency: ensure the current user has admin privileges.

    Зависит от get_current_user, поэтому токен проверяется только один раз.

    Raises:
        HTTPException 403: Пользователь авторизован, но не имеет роли 'admin'.
    """
    user_role = user.get("role") if isinstance(user, dict) else getattr(user, "role", None)

    if user_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user does not have enough privileges (admin required)",
        )

    return user


CurrentUserDep = Annotated[dict[str, Any], Depends(get_current_user)]
AdminUserDep = Annotated[dict[str, Any], Depends(get_admin_user)]
