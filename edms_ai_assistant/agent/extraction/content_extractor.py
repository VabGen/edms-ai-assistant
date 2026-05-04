# edms_ai_assistant/agent/extraction/content_extractor.py
"""
ContentExtractor — оптимизированное извлечение контента из message chain.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Final

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger(__name__)

_JSON_PRIORITY_FIELDS: Final[tuple[str, ...]] = (
    "content",
    "message",
    "text",
    "text_preview",
    "result",
)

# Паттерны технических маркеров в AI-сообщениях
_TECHNICAL_PATTERNS: Final[tuple[str, ...]] = (
    "вызвал инструмент",
    "tool call",
    '"name"',
    '"id"',
    '"tool_calls"',
)

_MIN_CONTENT_LENGTH: Final[int] = 30

# Statuses, которые не являются ошибками — нужно показать пользователю
_INFORMATIONAL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "already_exists",
        "already_added",
        "duplicate",
    }
)


@dataclass(frozen=True)
class ExtractionResult:
    """Типизированный результат извлечения контента."""

    content: str | None
    source: str  # "ai_message" | "tool_json" | "informational" | "fallback" | "none"
    confidence: float  # 0.0 — 1.0


def _is_technical_content(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _TECHNICAL_PATTERNS)


def _extract_from_tool_json(raw: str) -> str | None:
    """Извлекает текст из JSON-содержимого ToolMessage."""
    if not raw.startswith("{"):
        return None
    try:
        data: dict[str, Any] = json.loads(raw)
        if data.get("status") == "error":
            return None
        for key in _JSON_PRIORITY_FIELDS:
            val = data.get(key)
            if val and isinstance(val, str) and len(val) >= _MIN_CONTENT_LENGTH:
                return val
    except json.JSONDecodeError:
        pass
    return None


def _extract_informational(raw: str) -> str | None:
    """Извлекает текст из информационных статусов (already_exists и т.п.)."""
    if not raw.startswith("{"):
        return None
    try:
        data: dict[str, Any] = json.loads(raw)
        if data.get("status") in _INFORMATIONAL_STATUSES:
            return data.get("message") or data.get("detail")
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


class ContentExtractor:
    """Извлекает контент из цепочки сообщений LangGraph за ОДИН проход."""

    @classmethod
    def extract_final_content(cls, messages: list[BaseMessage]) -> str | None:
        """Извлекает финальный пользовательский контент из цепочки.

        Один обратный проход с явными приоритетами:
        1. Последний AIMessage с нетехническим контентом
        2. Последний ToolMessage с информационным статусом
        3. Последний ToolMessage с парсируемым JSON
        4. Любой AIMessage (fallback)

        Args:
            messages: Цепочка сообщений LangGraph.

        Returns:
            Строка контента или None.
        """
        result = cls._extract_with_priority(messages)
        return result.content

    @classmethod
    def _extract_with_priority(cls, messages: list[BaseMessage]) -> ExtractionResult:
        """Извлекает контент с явными приоритетами за один проход."""
        best_ai: str | None = None
        best_ai_fallback: str | None = None
        best_tool_json: str | None = None
        best_informational: str | None = None

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                text = str(msg.content).strip()
                if not text:
                    continue
                if not _is_technical_content(text) and len(text) >= _MIN_CONTENT_LENGTH:
                    if best_ai is None:
                        best_ai = text
                elif best_ai_fallback is None:
                    best_ai_fallback = text

            elif isinstance(msg, ToolMessage):
                raw = str(msg.content).strip()
                if not raw:
                    continue

                if best_informational is None:
                    info = _extract_informational(raw)
                    if info:
                        best_informational = info

                if best_tool_json is None:
                    extracted = _extract_from_tool_json(raw)
                    if extracted:
                        best_tool_json = extracted

        # Применяем приоритеты
        if best_ai:
            return ExtractionResult(best_ai, "ai_message", 1.0)
        if best_informational:
            return ExtractionResult(best_informational, "informational", 0.9)
        if best_tool_json:
            return ExtractionResult(best_tool_json, "tool_json", 0.7)
        if best_ai_fallback:
            return ExtractionResult(best_ai_fallback, "fallback", 0.4)

        return ExtractionResult(None, "none", 0.0)

    @classmethod
    def extract_last_tool_text(cls, messages: list[BaseMessage]) -> str | None:
        """Извлекает текст из последнего ToolMessage для передачи в следующий инструмент."""
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                continue
            try:
                raw = str(msg.content).strip()
                if raw.startswith("{"):
                    data: dict[str, Any] = json.loads(raw)
                    for key in _JSON_PRIORITY_FIELDS:
                        val = data.get(key)
                        if val and len(str(val)) > 100:
                            return str(val)
                if len(raw) > 100:
                    return raw
            except json.JSONDecodeError:
                raw = str(msg.content)
                if len(raw) > 100:
                    return raw
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """Удаляет JSON-обёртки из финального контента."""
        stripped = content.strip()

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data: dict[str, Any] = json.loads(stripped)
                for key in _JSON_PRIORITY_FIELDS:
                    val = data.get(key)
                    if val and isinstance(val, str) and len(val) >= _MIN_CONTENT_LENGTH:
                        return val.replace("\\n", "\n").replace('\\"', '"').strip()
            except (json.JSONDecodeError, ValueError):
                pass

        # Удаляем частичные JSON-префиксы/суффиксы
        content = re.sub(
            r'\{"status"\s*:\s*"[^"]*",\s*"(?:content|message|text)"\s*:\s*"',
            "",
            content,
        )
        content = re.sub(r'",\s*"meta"\s*:\s*\{[^}]*\}\s*\}', "", content)
        content = re.sub(
            r'",?\s*"[a-z_]+"\s*:\s*(?:true|false|null|\d+)\s*\}?\s*$', "", content
        )
        content = re.sub(r'"\s*\}$', "", content)
        return content.replace('\\"', '"').replace("\\n", "\n").strip()

    @classmethod
    def extract_navigate_url(cls, messages: list[BaseMessage]) -> str | None:
        """Извлекает navigate_url из последних ToolMessages."""
        for msg in reversed(messages[-8:]):
            if not isinstance(msg, ToolMessage):
                continue
            try:
                data: dict[str, Any] = json.loads(str(msg.content))
                url = data.get("navigate_url")
                if url and isinstance(url, str) and url.startswith("/"):
                    return url
            except (json.JSONDecodeError, AttributeError):
                pass
        return None

    @classmethod
    def extract_compliance_data(
        cls, messages: list[BaseMessage]
    ) -> dict[str, Any] | None:
        """Извлекает результат compliance check."""
        for msg in reversed(messages[-10:]):
            if not isinstance(msg, ToolMessage):
                continue
            try:
                data: dict[str, Any] = json.loads(str(msg.content))
                if (
                    data.get("status") == "success"
                    and isinstance(data.get("fields"), list)
                    and "overall" in data
                ):
                    return data
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        return None
