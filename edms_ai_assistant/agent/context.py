"""
ContextParams — иммутабельный контекст выполнения одного turn агента.

- frozen=True (dataclass не мутируется после создания)
- метод with_intent() для создания нового контекста с intent
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
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
    """Return True if *value* is a canonical UUID string."""
    return bool(UUID_RE.match(value.strip()))


# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------


class AgentStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"


class ActionType(str, Enum):
    SUMMARIZE_SELECTION = "summarize_selection"
    DISAMBIGUATION = "requires_disambiguation"
    CONFIRMATION = "requires_confirmation"


# ---------------------------------------------------------------------------
# ContextParams — ИММУТАБЕЛЬНЫЙ dataclass
# ---------------------------------------------------------------------------


def _current_date() -> str:
    local = today_local()
    return format_date_for_display(local, "%d.%m.%Y") or datetime.now().strftime(
        "%d.%m.%Y"
    )


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


@dataclass(frozen=True)
class ContextParams:
    """
    Иммутабельный контекст выполнения одного turn агента.

    ВАЖНО: frozen=True — объект нельзя изменить после создания.
    Для установки intent используйте with_intent():
        context = context.with_intent(UserIntent.SEARCH)
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

    intent: Any = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        # Валидация через object.__setattr__ т.к. frozen=True
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")

        # Auto-derive uploaded_file_name из локального пути
        if self.file_path and not self.uploaded_file_name:
            fp = self.file_path.strip()
            if not is_valid_uuid(fp):
                object.__setattr__(self, "uploaded_file_name", Path(fp).name)

        # Auto-derive full_name
        if not self.user_full_name:
            parts = [p for p in (self.user_last_name, self.user_first_name) if p]
            if parts:
                object.__setattr__(self, "user_full_name", " ".join(parts))

    def with_intent(self, intent: Any) -> "ContextParams":
        """
        Создаёт новый иммутабельный ContextParams с установленным intent.

        Использование:
            context = context.with_intent(UserIntent.SEARCH)

        Не мутирует исходный объект.
        """
        return replace(self, intent=intent)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def time_context_for_prompt(self) -> str:
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
    """Валидированный входящий запрос на границе Service Layer."""

    message: str = Field(default="", max_length=8_000)
    user_token: str = Field(..., min_length=10)
    context_ui_id: str | None = Field(
        None,
        pattern=(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^$"),
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
    def _validate_message_or_choice(self) -> "AgentRequest":
        has_message = bool(self.message and self.message.strip())
        has_choice = bool(self.human_choice and self.human_choice.strip())
        if not has_message and not has_choice:
            raise ValueError("Either message or human_choice must be provided")
        if not has_message and has_choice:
            object.__setattr__(self, "message", self.human_choice)
        return self

    @field_validator("file_path")
    @classmethod
    def _validate_file_path(cls, v: str | None) -> str | None:
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
    """Стандартизированный результат выполнения агента."""

    status: AgentStatus
    content: str | None = None
    message: str | None = None
    action_type: str | None = None
    requires_reload: bool = False
    navigate_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
