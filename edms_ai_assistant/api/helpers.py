# edms_ai_assistant/api/helpers.py
"""
Shared helper functions used across multiple API route modules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.model import UserInput
from edms_ai_assistant.summarizer.structured.models import SummaryMode
from edms_ai_assistant.utils.regex_utils import UUID_RE

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


def format_output_as_text(resp: object) -> str:
    """Format a structured summarization response as human-readable Markdown."""
    output = getattr(resp, "output", {})
    mode = getattr(resp, "mode", None)

    if mode == SummaryMode.EXECUTIVE:
        lines = [f"**{output.get('headline', '')}**", ""]
        for bullet in output.get("bullets", []):
            lines.append(f"• {bullet}")
        rec = output.get("recommendation")
        if rec:
            lines.extend(["", f"💡 **Рекомендация:** {rec}"])
        return "\n".join(lines)

    if mode == SummaryMode.ACTION_ITEMS:
        items = output.get("action_items", [])
        if not items:
            return "Задачи и поручения не найдены."
        lines = [f"**Найдено задач: {len(items)}**", ""]
        for i, item in enumerate(items, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                item.get("priority", "medium"), "⚪"
            )
            lines.append(f"{i}. {priority_emoji} {item.get('task', '')}")
            if item.get("owner"):
                lines.append(f"   Ответственный: {item['owner']}")
            if item.get("deadline"):
                lines.append(f"   Срок: {item['deadline']}")
        return "\n".join(lines)

    if mode == SummaryMode.THESIS:
        sections = output.get("sections", [])
        lines = [f"**{output.get('main_argument', 'Анализ документа')}**", ""]
        for sec in sections:
            lines.append(f"## {sec.get('title', '')}")
            lines.append(sec.get("thesis", ""))
            for pt in sec.get("points", []):
                lines.append(f"- {pt.get('claim', '')}")
            lines.append("")
        return "\n".join(lines)

    if mode == SummaryMode.EXTRACTIVE:
        facts = output.get("facts", [])
        lines = [output.get("document_summary", ""), ""]
        for fact in facts:
            lines.append(f"- **{fact.get('label', '')}**: {fact.get('value', '')}")
        return "\n".join(lines)

    return (
        output.get("summary", "")
        or output.get("content", "")
        or json.dumps(output, ensure_ascii=False, indent=2)
    )
