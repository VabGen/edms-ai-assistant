"""
ContentExtractor — pulls final human-readable content from a LangGraph message chain.

All methods are pure class-methods: the class carries no instance state and
acts as a stateless namespace.

Priority chain for extract_final_content:
  1. Last AIMessage with non-trivial text (not a tool-call marker).
  2. Last ToolMessage parsed as JSON (content / message / text fields).
  2.5. ToolMessage with already_exists / already_added / duplicate status.
  3. Fallback: any AIMessage with content.
  4. Last resort: raw ToolMessage content.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger(__name__)

# Fields tried in priority order when extracting text from a JSON tool result.
_JSON_PRIORITY_FIELDS: tuple[str, ...] = (
    "content",
    "message",
    "text",
    "text_preview",
    "result",
)

# Substrings that identify a message as a technical LLM marker rather than
# a user-facing response.
_SKIP_PATTERNS: tuple[str, ...] = (
    "вызвал инструмент",
    "tool call",
    '"name"',
    '"id"',
    '"tool_calls"',
)

_MIN_CONTENT_LENGTH: int = 30


def _is_technical(content: str) -> bool:
    lower = content.lower()
    return any(p in lower for p in _SKIP_PATTERNS)


def _parse_tool_message_json(message: ToolMessage) -> str | None:
    """Extract a human-readable string from a ToolMessage that contains JSON."""
    try:
        raw = str(message.content).strip()
        if not raw.startswith("{"):
            return None
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


class ContentExtractor:
    """Stateless helper for extracting content from LangGraph message chains."""

    @classmethod
    def extract_final_content(cls, messages: list[BaseMessage]) -> str | None:
        """Extract the final user-visible content from the message chain.

        Args:
            messages: Complete LangGraph message chain.

        Returns:
            Cleaned content string, or ``None`` when nothing suitable is found.
        """
        # Pass 1: last substantial AIMessage.
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                text = str(m.content).strip()
                if not _is_technical(text) and len(text) >= _MIN_CONTENT_LENGTH:
                    logger.debug("Extracted AIMessage", extra={"chars": len(text)})
                    return text

        # Pass 2: last ToolMessage with parseable JSON.
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = _parse_tool_message_json(m)
                if extracted:
                    logger.debug("Extracted ToolMessage JSON", extra={"chars": len(extracted)})
                    return extracted

        # Pass 2.5: ToolMessage with already_exists / already_added / duplicate status.
        # These are informational, not errors — the tool message contains a
        # human-readable explanation that must reach the user.
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                try:
                    raw = str(m.content).strip()
                    if raw.startswith("{"):
                        data: dict[str, Any] = json.loads(raw)
                        status = data.get("status", "")
                        if status in ("already_exists", "already_added", "duplicate"):
                            msg = data.get("message") or data.get("detail") or ""
                            if msg:
                                logger.debug(
                                    "Extracted already_exists ToolMessage",
                                    extra={"status": status, "chars": len(msg)},
                                )
                                return msg
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Pass 3: any AIMessage at all.
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                text = str(m.content).strip()
                if text:
                    logger.debug("Fallback AIMessage", extra={"chars": len(text)})
                    return text

        # Pass 4: raw ToolMessage content.
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = _parse_tool_message_json(m)
                if extracted:
                    return extracted

        return None

    @classmethod
    def extract_last_tool_text(cls, messages: list[BaseMessage]) -> str | None:
        """Extract substantial text content from the most recent ToolMessage.

        Used to feed extracted file content into ``doc_summarize_text`` on the
        next orchestration iteration without requiring the LLM to repeat the
        full text in its tool_call arguments.

        Args:
            messages: Complete LangGraph message chain.

        Returns:
            Text string of 100+ characters, or ``None``.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                raw = str(m.content).strip()
                if raw.startswith("{"):
                    data: dict[str, Any] = json.loads(raw)
                    for key in _JSON_PRIORITY_FIELDS:
                        val = data.get(key)
                        if val and len(str(val)) > 100:
                            return str(val)
                if len(raw) > 100:
                    return raw
            except json.JSONDecodeError:
                raw = str(m.content)
                if len(raw) > 100:
                    return raw
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """Strip residual JSON wrappers from final content.

        Handles both a fully-JSON response and mixed text with an embedded
        JSON envelope prepended or appended by the LLM.

        Args:
            content: Raw content string that may contain JSON fragments.

        Returns:
            Clean, human-readable text.
        """
        stripped = content.strip()

        # Case 1: entire content is a JSON object — extract the text field.
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data: dict[str, Any] = json.loads(stripped)
                for key in _JSON_PRIORITY_FIELDS:
                    val = data.get(key)
                    if val and isinstance(val, str) and len(val) >= _MIN_CONTENT_LENGTH:
                        return val.replace("\\n", "\n").replace('\\"', '"').strip()
            except (json.JSONDecodeError, ValueError):
                pass

        # Case 2: partial JSON prefix/suffix — strip with regex.
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
        """Scan recent ToolMessages for a ``navigate_url`` field.

        ``create_document_from_file`` returns a ``navigate_url`` in its result.
        The LLM reformulates it as prose, but the raw URL must still reach the
        frontend so it can perform client-side navigation.

        Args:
            messages: Complete LangGraph message chain.

        Returns:
            URL string starting with ``"/"`` (e.g. ``"/document-form/uuid"``),
            or ``None``.
        """
        for m in reversed(messages[-8:]):
            if not isinstance(m, ToolMessage):
                continue
            try:
                data: dict[str, Any] = json.loads(str(m.content))
                url = data.get("navigate_url")
                if url and isinstance(url, str) and url.startswith("/"):
                    return url
            except (json.JSONDecodeError, AttributeError):
                pass
        return None

    @classmethod
    def extract_compliance_data(cls, messages: list[BaseMessage]) -> dict[str, Any] | None:
        """Extract a compliance check result from recent ToolMessages.

        Detection is structural (``status=success`` + ``fields`` list +
        ``overall`` key) rather than by tool name, because
        ``ToolMessage.name`` can be ``None`` in some LangGraph versions.

        Args:
            messages: Complete LangGraph message chain.

        Returns:
            Compliance result dict, or ``None``.
        """
        for m in reversed(messages[-10:]):
            if not isinstance(m, ToolMessage):
                continue
            try:
                data: dict[str, Any] = json.loads(str(m.content))
                if (
                    data.get("status") == "success"
                    and isinstance(data.get("fields"), list)
                    and "overall" in data
                ):
                    return data
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        return None