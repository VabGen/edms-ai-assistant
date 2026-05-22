# edms_ai_assistant/api/helpers.py
"""
Shared helper functions used across multiple API route modules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.utils.regex_utils import UUID_RE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edms_ai_assistant.model import UserInput

logger = logging.getLogger(__name__)


def is_system_attachment(file_path: str | None) -> bool:
    """Return True if file_path is an EDMS attachment UUID (not a local path)."""
    return bool(file_path and UUID_RE.match(str(file_path)))


def cleanup_file(file_path: str) -> None:
    """Remove a temporary local file, logging any failure."""
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            logger.debug("Temporary file removed", extra={"path": file_path})
    except Exception as exc:
        logger.warning(
            "Failed to remove temporary file",
            extra={"path": file_path, "error": str(exc)},
        )


async def resolve_user_context(user_input: UserInput, user_id: str) -> dict:
    """Resolve user context dict from request or EDMS employee API."""
    if user_input.context:
        return user_input.context.model_dump(exclude_none=True)

    try:
        async with EmployeeClient() as emp_client:
            ctx = await emp_client.get_employee(user_input.user_token, user_id)
            if ctx:
                return ctx
    except Exception as exc:
        logger.warning(
            "Failed to fetch employee context",
            extra={"user_id": user_id, "error": str(exc)},
        )

    return {"firstName": "Коллега"}


def unwrap_text_from_agent_result(content: str) -> str:
    """Extract plain text from agent result that may be a JSON envelope."""
    if not content or not content.strip().startswith("{"):
        return content
    try:
        payload = json.loads(content)
        for key in ("content", "text", "document_info"):
            val = payload.get(key)
            if val and isinstance(val, str) and len(val) > 50:
                return val
    except (json.JSONDecodeError, AttributeError):
        pass
    return content
