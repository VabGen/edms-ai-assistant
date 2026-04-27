# edms_ai_assistant/agent/context.py
"""
Value objects and Pydantic models for the agent request/response boundary.

Responsibilities:
- ContextParams: immutable execution context for one agent turn.
- AgentRequest:  validated HTTP-layer input (Pydantic v2).
- AgentResponse: typed output returned to the HTTP layer.
- AgentStatus / ActionType: domain enumerations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.utils.datetime_utils import (
    format_date_for_display,
    now_local,
    today_local,
)
from edms_ai_assistant.utils.regex_utils import UUID_RE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_valid_uuid(value: str) -> bool:
    """Return True if *value* is a canonical UUID4 string."""
    return bool(UUID_RE.match(value.strip()))


# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------


class AgentStatus(str, Enum):
    """Agent execution result statuses."""

    SUCCESS = "success"
    ERROR = "error"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"


class ActionType(str, Enum):
    """Types of interactive actions that require user participation."""

    SUMMARIZE_SELECTION = "summarize_selection"
    DISAMBIGUATION = "requires_disambiguation"
    CONFIRMATION = "requires_confirmation"


# ---------------------------------------------------------------------------
# ContextParams
# ---------------------------------------------------------------------------


def _current_date() -> str:
    local = today_local()
    return format_date_for_display(local, "%d.%m.%Y") or datetime.now().strftime("%d.%m.%Y")


def _current_year() -> str:
    local = today_local()
    return str(local.year) if local else str(datetime.now().year)


def _current_time() -> str:
    local = now_local()
    return local.strftime("%H:%M") if local else datetime.now().strftime("%H:%M")


def _current_datetime_iso() -> str:
    local = now_local()
    return local.isoformat() if local else datetime.now().isoformat()


def _timezone_offset() -> str:
    local = now_local()
    if local and local.utcoffset() is not None:
        total = local.utcoffset().total_seconds()  # type: ignore[union-attr]
        return f"{total / 3600:+.1f}h"
    return "+0.0h"


@dataclass
class ContextParams:
    """Immutable execution context threaded through the entire agent lifecycle.

    Constructed once per ``chat()`` call and never mutated after that —
    except for ``intent``, which is set by the agent after semantic analysis.

    Attributes:
        user_token: JWT authorization token (never logged).
        document_id: UUID of the active EDMS document in the UI.
        file_path: UUID of an EDMS attachment or an absolute local path.
        thread_id: LangGraph conversation thread identifier.
        user_name: Display name (first name or fallback) for greetings.
        user_first_name: First name extracted from user context.
        user_last_name: Last name used for self-reference commands.
        user_full_name: Full name constructed from available parts.
        user_id: Employee UUID from the user context dict.
        current_date: Formatted date string injected into the prompt.
        current_year: Four-digit year string for deadline calculations.
        current_time: HH:MM local time string.
        current_datetime_iso: Full ISO-8601 datetime for tool arguments.
        timezone_offset: UTC offset string, e.g. "+3.0h".
        uploaded_file_name: Human-readable filename shown to the user.
        user_context: Raw user profile dict from the HTTP request.
        intent: Primary user intent — set after semantic analysis.
    """

    user_token: str
    document_id: str | None = None
    file_path: str | None = None
    thread_id: str = "default"
    user_name: str = "пользователь"
    user_first_name: str | None = None
    user_last_name: str | None = None
    user_full_name: str | None = None
    user_id: str | None = None

    current_date: str = field(default_factory=_current_date)
    current_year: str = field(default_factory=_current_year)
    current_time: str = field(default_factory=_current_time)
    current_datetime_iso: str = field(default_factory=_current_datetime_iso)
    timezone_offset: str = field(default_factory=_timezone_offset)

    uploaded_file_name: str | None = None
    user_context: dict[str, Any] = field(default_factory=dict)

    # Mutable post-construction — set by the agent after semantic analysis.
    intent: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")

        # Auto-derive display filename from local path when not supplied.
        if self.file_path and not self.uploaded_file_name:
            fp = self.file_path.strip()
            if not is_valid_uuid(fp):
                self.uploaded_file_name = Path(fp).name

        # Auto-derive full_name from parts when not supplied.
        if not self.user_full_name:
            parts = [p for p in (self.user_last_name, self.user_first_name) if p]
            if parts:
                self.user_full_name = " ".join(parts)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def time_context_for_prompt(self) -> str:
        """Return the time context block injected into the LEAN system prompt."""
        now = now_local()
        date_str = format_date_for_display(now, "%d.%m.%Y") or self.current_date
        return (
            f"Текущее время: {now.strftime('%H:%M')} (UTC{self.timezone_offset}), {date_str}\n"
            f"Часовой пояс сервера: UTC{self.timezone_offset}\n"
            "⚠️ Важно: используй указанное выше время. Не выдумывай текущее время."
        )


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class AgentRequest(BaseModel):
    """Validated incoming request at the Service Layer boundary.

    All field validation happens here so that the agent never receives
    malformed or unsafe input.
    """

    message: str = Field(default="", max_length=8_000)
    user_token: str = Field(..., min_length=10)
    context_ui_id: str | None = Field(
        None,
        pattern=(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^$"
        ),
    )
    thread_id: str | None = Field(None, max_length=255)
    user_context: dict[str, Any] = Field(default_factory=dict)
    file_path: str | None = Field(None, max_length=500)
    file_name: str | None = Field(None, max_length=260)
    human_choice: str | None = Field(None, max_length=200)

    @field_validator("message")
    @classmethod
    def _strip_message(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _validate_message_or_choice(self) -> AgentRequest:
        """Require either a non-empty message or a human_choice.

        Human-in-the-Loop flows (summarize type selection, disambiguation)
        send human_choice as the primary payload — message may be empty.
        """
        has_message = bool(self.message and self.message.strip())
        has_choice = bool(self.human_choice and self.human_choice.strip())
        if not has_message and not has_choice:
            raise ValueError("Either message or human_choice must be provided")
        if not has_message and has_choice:
            # Treat the choice itself as the message for history purposes.
            self.message = self.human_choice  # type: ignore[assignment]
        return self

    @field_validator("file_path")
    @classmethod
    def _validate_file_path(cls, v: str | None) -> str | None:
        """Accept UUID strings, Unix absolute paths, and Windows absolute paths."""
        if not v:
            return None
        stripped = v.strip()
        if is_valid_uuid(stripped):
            return stripped
        if len(stripped) < 500:
            if stripped.startswith("/"):
                return stripped
            if re.match(r"^[A-Za-z]:\\", stripped):
                return stripped
            if re.match(r"^[^/\\]+[\\/]", stripped):
                return stripped
        raise ValueError(f"Invalid file_path format: {v!r}")


class AgentResponse(BaseModel):
    """Standardised agent execution result (internal — not exposed via HTTP directly).

    The FastAPI layer maps this to ``AssistantResponse`` defined in ``model.py``.
    """

    status: AgentStatus
    content: str | None = None
    message: str | None = None
    action_type: ActionType | None = None
    requires_reload: bool = False
    navigate_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)