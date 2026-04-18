# edms_ai_assistant/response_assembler.py
"""
ResponseAssembler — assembles the final AgentResponse from a LangGraph message chain.

Critical behaviours preserved from the original agent:
1. Compliance data attached as metadata["compliance"] so frontend renders cards
2. navigate_url extracted for create_document_from_file navigation
3. Interactive status detection delegated to InteractiveStatusDetector
4. Technical content sanitization (UUIDs, file paths)
5. Correct content extraction priority: AIMessage > ToolMessage JSON > fallback
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from edms_ai_assistant.interactive_status_detector import InteractiveStatusDetector
from edms_ai_assistant.model import (
    AgentResponse,
    AgentStatus,
    ContextParams,
    _is_mutation_response,
)
from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

_MIN_CONTENT = 30
_JSON_PRIORITY_FIELDS = ("content", "message", "text", "text_preview", "result")
_SKIP_PATTERNS = ('"name"', '"id"', '"tool_calls"', "вызвал инструмент", "tool call")


def _is_valid_uuid(value: str) -> bool:
    return bool(UUID_RE.match(value.strip()))


# ─────────────────────────────────────────────────────────────────────────────
# Content extraction helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_final_content(messages: list[BaseMessage]) -> str | None:
    """
    Priority order:
    1. Last AIMessage with substantial non-technical content
    2. Last ToolMessage with content/message/text JSON field
    3. Any AIMessage (fallback)
    4. Last ToolMessage raw text (last resort)
    """
    # Pass 1: last substantive AIMessage
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            text = str(m.content).strip()
            lower = text.lower()
            if len(text) >= _MIN_CONTENT and not any(p in lower for p in _SKIP_PATTERNS):
                return text

    # Pass 2: last ToolMessage JSON priority fields
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            extracted = _parse_tool_content(m)
            if extracted:
                return extracted

    # Pass 3: any AIMessage
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            text = str(m.content).strip()
            if text:
                return text

    # Pass 4: raw ToolMessage
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            raw = str(m.content).strip()
            if len(raw) >= _MIN_CONTENT:
                return raw

    return None


def _parse_tool_content(message: ToolMessage) -> str | None:
    raw = str(message.content).strip()
    if not raw.startswith("{"):
        return None
    try:
        data = json.loads(raw)
        if data.get("status") == "error":
            return None
        for key in _JSON_PRIORITY_FIELDS:
            val = data.get(key)
            if val and isinstance(val, str) and len(val) >= _MIN_CONTENT:
                return val
    except json.JSONDecodeError:
        pass
    return None


def _clean_json_artifacts(content: str) -> str:
    """Strip JSON envelope wrappers that leaked into the final content string."""
    stripped = content.strip()

    # Whole content is a JSON object — extract text field
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            for key in _JSON_PRIORITY_FIELDS:
                val = data.get(key)
                if val and isinstance(val, str) and len(val) >= _MIN_CONTENT:
                    return val.replace("\\n", "\n").strip()
        except (json.JSONDecodeError, ValueError):
            pass

    # Trailing JSON fragments
    content = re.sub(
        r'",?\s*"[a-z_]+"\s*:\s*(?:true|false|null|\d+)\s*\}?\s*$',
        "",
        content,
    )
    return content.replace('\\"', '"').replace("\\n", "\n").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Compliance extractor
# ─────────────────────────────────────────────────────────────────────────────


def _extract_compliance_data(messages: list[BaseMessage]) -> dict | None:
    """
    Extracts compliance check result from recent ToolMessages.
    Matches by structure (status=success + fields list + overall) — more reliable
    than matching by tool name since ToolMessage.name can be None.
    """
    for m in reversed(messages[-10:]):
        if isinstance(m, HumanMessage):
            break
        if not isinstance(m, ToolMessage):
            continue
        try:
            data = json.loads(str(m.content))
            if (
                data.get("status") == "success"
                and isinstance(data.get("fields"), list)
                and "overall" in data
            ):
                return data
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Navigate URL extractor
# ─────────────────────────────────────────────────────────────────────────────


def _extract_navigate_url(messages: list[BaseMessage]) -> str | None:
    """
    Finds navigate_url in the last 8 ToolMessages.
    Returned by create_document_from_file — frontend uses it for navigation.
    """
    for m in reversed(messages[-8:]):
        if not isinstance(m, ToolMessage):
            continue
        try:
            data = json.loads(str(m.content))
            url = data.get("navigate_url")
            if url and isinstance(url, str) and url.startswith("/"):
                return url
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Technical content sanitizer
# ─────────────────────────────────────────────────────────────────────────────


def _sanitize_content(content: str, context: ContextParams) -> str:
    """
    Removes technical artifacts from user-visible content:
    - Filesystem paths → file label
    - UUID + extension filenames → file label
    - Raw document/attachment UUIDs → human-readable labels
    """
    file_label = (
        f"«{context.uploaded_file_name}»"
        if context.uploaded_file_name
        else "«загруженный файл»"
    )

    lines = content.split("\n")
    result = []
    for line in lines:
        # Preserve markdown table rows verbatim — they may contain IDs
        # intentionally shown to the user (doc_search_tool table with id column)
        if line.strip().startswith("|"):
            result.append(line)
            continue
        result.append(_sanitize_line(line, context, file_label))
    return "\n".join(result)


def _sanitize_line(line: str, context: ContextParams, file_label: str) -> str:
    # Absolute filesystem paths
    line = re.sub(
        r"[A-Za-z]:\\[^\s,;)'\"]{3,}|/(?:tmp|var|home|uploads)/[^\s,;)'\"]{3,}",
        file_label,
        line,
    )

    # UUID_hex.ext filenames (e.g. uuid_md5hash.docx)
    line = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        r"_[0-9a-f]{32}\.[a-zA-Z]{2,5}",
        file_label,
        line,
        flags=re.I,
    )

    # _md5hash.ext filenames
    line = re.sub(
        r"_?[0-9a-f]{32}\.[a-zA-Z]{2,5}\b",
        file_label,
        line,
        flags=re.I,
    )

    # Known UUIDs from context
    if context.file_path and _is_valid_uuid(str(context.file_path).strip()):
        line = line.replace(str(context.file_path).strip(), file_label)

    if context.document_id and _is_valid_uuid(context.document_id):
        line = line.replace(context.document_id, "«текущего документа»")

    # Cleanup artefacts
    line = re.sub(r"«документ»\s*(?=«)", "", line)
    line = re.sub(r"«документ»_\s*", "", line)

    return line


# ─────────────────────────────────────────────────────────────────────────────
# ResponseAssembler
# ─────────────────────────────────────────────────────────────────────────────


class ResponseAssembler:
    """
    Assembles final AgentResponse from a LangGraph message chain.

    Call sequence:
      1. Check for interactive status (disambiguation / choice)
      2. Extract compliance data → metadata["compliance"] for frontend cards
      3. Extract navigate_url
      4. Extract and clean final text content
      5. Sanitize technical artifacts
      6. Apply guardrails (optional)
      7. Return AgentResponse dict
    """

    def __init__(
        self,
        guardrail_pipeline: Any | None = None,
        enable_guardrails: bool = False,
    ) -> None:
        self._guardrails = guardrail_pipeline
        self._enable_guardrails = enable_guardrails
        self._interactive_detector = InteractiveStatusDetector()

    def assemble(
        self,
        messages: list[BaseMessage],
        context: ContextParams,
    ) -> dict[str, Any]:
        """Main entry point — returns serialised AgentResponse dict."""

        # ── Step 1: Interactive status check ──────────────────────────────────
        interactive = self._interactive_detector.detect(messages)
        if interactive:
            logger.info(
                "Interactive status detected: %s",
                interactive.get("action_type") or interactive.get("status"),
            )
            return interactive

        # ── Step 2: Compliance data (for frontend card rendering) ─────────────
        compliance_data = _extract_compliance_data(messages)

        # ── Step 3: Navigate URL ──────────────────────────────────────────────
        navigate_url = _extract_navigate_url(messages)

        # ── Step 4: Extract final content ─────────────────────────────────────
        final_content = _extract_final_content(messages)

        if not final_content:
            logger.warning(
                "No final content found in %d messages (thread=%s)",
                len(messages),
                context.thread_id,
            )
            metadata: dict[str, Any] = {}
            if compliance_data:
                metadata["compliance"] = compliance_data
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                content="Операция завершена.",
                navigate_url=navigate_url,
                metadata=metadata,
            ).model_dump()

        # ── Step 5: Clean and sanitize ────────────────────────────────────────
        final_content = _clean_json_artifacts(final_content)
        final_content = _sanitize_content(final_content, context)

        # ── Step 6: Guardrails ────────────────────────────────────────────────
        if self._enable_guardrails and self._guardrails:
            try:
                result = self._guardrails.run(final_content)
                if result.blocked:
                    logger.warning("Guardrail BLOCKED: %s", result.block_reason)
                    return AgentResponse(
                        status=AgentStatus.ERROR,
                        message="Ответ заблокирован политикой безопасности.",
                    ).model_dump()
                final_content = result.content
            except Exception as exc:
                logger.error("Guardrail error (non-fatal): %s", exc)

        # ── Step 7: Build response ────────────────────────────────────────────
        requires_reload = _is_mutation_response(final_content)
        response_metadata: dict[str, Any] = {}

        if compliance_data:
            response_metadata["compliance"] = compliance_data
            logger.info(
                "Compliance attached to metadata: overall=%s fields=%d",
                compliance_data.get("overall"),
                len(compliance_data.get("fields", [])),
            )

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            content=final_content,
            requires_reload=requires_reload,
            navigate_url=navigate_url,
            metadata=response_metadata,
        ).model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_response_assembler(
    config: Any | None = None,
    guardrail_pipeline: Any | None = None,
) -> ResponseAssembler:
    """Factory used in agent.py to create a configured ResponseAssembler."""
    enable_guardrails = getattr(config, "enable_guardrails", False)
    return ResponseAssembler(
        guardrail_pipeline=guardrail_pipeline,
        enable_guardrails=enable_guardrails,
    )