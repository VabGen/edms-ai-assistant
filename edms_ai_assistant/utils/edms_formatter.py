# edms_ai_assistant/utils/edms_formatter.py
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from edms_ai_assistant.utils.datetime_utils import to_local_timezone

logger = logging.getLogger(__name__)


def _camel_to_snake(name: str) -> str:
    """Конвертирует camelCase в snake_case для совместимости с Pydantic V2."""
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


class EdmsFormatter:
    """Утилиты для форматирования и безопасного извлечения атрибутов из доменных DTO."""

    @staticmethod
    def format_user(user: Any) -> str | None:
        if not user:
            return None
        try:
            ln = getattr(user, "lastName", "") or ""
            fn = getattr(user, "firstName", "") or ""
            mn = getattr(user, "middleName", "") or ""
            post = getattr(user, "authorPost", "") or getattr(
                getattr(user, "post", None), "name", ""
            )
            name = f"{ln} {fn} {mn}".strip()
            return f"{name} ({post})" if post else name or None
        except Exception:
            return None

    @staticmethod
    def format_date(instant: Any) -> str | None:
        if not instant:
            return None
        try:
            local_str = to_local_timezone(instant)
            if local_str and "T" in local_str:
                return datetime.fromisoformat(local_str).strftime("%d.%m.%Y")
            if hasattr(instant, "strftime"):
                return instant.strftime("%d.%m.%Y")
            return str(instant)[:10]
        except Exception:
            return None

    @staticmethod
    def format_datetime(instant: Any) -> str | None:
        if not instant:
            return None
        try:
            local_str = to_local_timezone(instant)
            if local_str and "T" in local_str:
                return datetime.fromisoformat(local_str).strftime("%d.%m.%Y %H:%M")
            if hasattr(instant, "strftime"):
                return instant.strftime("%d.%m.%Y %H:%M")
            return str(instant)[:16]
        except Exception:
            return None

    @staticmethod
    def format_date_iso(instant: Any) -> str | None:
        return to_local_timezone(instant)

    @staticmethod
    def get_safe(obj: Any, path: str, default: Any = None) -> Any:
        """Safely traverse a dot-separated attribute path.
        Автоматически поддерживает совместимость camelCase (из API) и snake_case (из Pydantic V2 DTO).
        """
        if obj is None:
            return default
        val = obj
        try:
            for part in path.split("."):
                if val is None:
                    return default

                if isinstance(val, BaseModel):
                    if hasattr(val, part):
                        val = getattr(val, part)
                    else:
                        snake_part = _camel_to_snake(part)
                        val = getattr(val, snake_part, default)
                elif isinstance(val, dict):
                    val = val.get(part, default)
                else:
                    val = getattr(val, part, default)

                # Развертывание Enum значений
                if hasattr(val, "value"):
                    val = val.value

            return val if val is not None else default
        except (AttributeError, KeyError, TypeError):
            return default
