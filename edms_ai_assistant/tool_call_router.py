# edms_ai_assistant/tool_call_router.py
"""
ToolCallRouter — Single Responsibility: decide tool name redirections
based on conversation history context (post-disambiguation state,
versions-already-compared state).

This is separate from ToolArgsPatcher because:
- Router decides WHICH tool to use (name change)
- Patcher decides HOW to call it (args mutation)

Extracted from EdmsDocumentAgent._orchestrate (the second half of
the for-loop, the "blocking" section).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)


def _is_valid_uuid(value: str) -> bool:
    return bool(UUID_RE.match(value.strip()))


class ToolCallRouter:
    """
    Decides tool name redirections based on conversation state.

    Handles two cases that require looking back into message history:
    1. After compare-disambiguation → block doc_compare_documents,
       redirect to doc_compare_attachment_with_local.
    2. After doc_get_versions returned comparison_complete=True →
       block redundant doc_compare_documents, substitute with no-op.

    Single Responsibility: name-level routing decisions based on history.
    Does NOT mutate args — that is ToolArgsPatcher's job.
    Does NOT enforce call limits — that is ToolCallGuard's job.
    """

    def __init__(
        self,
        document_id: str | None,
        file_path: str | None,
        uploaded_file_name: str | None,
        user_token: str,
        is_choice_active: bool,
    ) -> None:
        self._document_id = document_id
        self._uploaded_file_name = uploaded_file_name
        self._user_token = user_token
        self._is_choice_active = is_choice_active

        fp = str(file_path or "").strip()
        self._clean_path = fp
        self._path_is_uuid = _is_valid_uuid(fp) if fp else False
        self._path_is_local = bool(fp) and not self._path_is_uuid

    def route(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: list[BaseMessage],
    ) -> tuple[str, dict[str, Any]]:
        """
        Apply history-based routing rules.

        Args:
            tool_name: Tool name (already processed by ToolArgsPatcher).
            tool_args: Tool args (already processed by ToolArgsPatcher).
            messages: Full message history.

        Returns:
            Tuple of (final_tool_name, final_args).
        """
        if tool_name != "doc_compare_documents":
            return tool_name, tool_args

        # Check if we're in a post-disambiguation compare flow
        after_compare_disambiguation = self._detect_compare_disambiguation(messages)

        if after_compare_disambiguation or (self._is_choice_active and self._path_is_local):
            return self._redirect_compare_to_local(tool_args)

        # Check if versions were already compared
        if self._detect_versions_complete(messages):
            return self._redirect_to_noop()

        # No-op: pass through unchanged (doc_compare_documents allowed)
        return tool_name, tool_args

    # ── Private detection methods ─────────────────────────────────────────────

    @staticmethod
    def _detect_compare_disambiguation(messages: list[BaseMessage]) -> bool:
        """Scan recent messages for a disambiguation result from compare tool."""
        for prev_msg in reversed(messages[-15:]):
            if isinstance(prev_msg, ToolMessage):
                try:
                    data = json.loads(str(prev_msg.content))
                    if (
                        data.get("status") == "requires_disambiguation"
                        and prev_msg.name == "doc_compare_attachment_with_local"
                    ):
                        return True
                except (json.JSONDecodeError, AttributeError):
                    continue
            if isinstance(prev_msg, HumanMessage):
                break
        return False

    @staticmethod
    def _detect_versions_complete(messages: list[BaseMessage]) -> bool:
        """Check if doc_get_versions already returned all comparisons."""
        for prev_msg in reversed(messages):
            if isinstance(prev_msg, ToolMessage):
                try:
                    data = json.loads(str(prev_msg.content))
                    if data.get("comparison_complete") and data.get("comparisons"):
                        return True
                except (json.JSONDecodeError, AttributeError):
                    continue
            if isinstance(prev_msg, HumanMessage):
                break
        return False

    def _redirect_compare_to_local(
        self,
        tool_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        logger.warning(
            "GUARD: doc_compare_documents blocked → doc_compare_attachment_with_local"
        )
        new_args: dict[str, Any] = {
            "token": self._user_token,
            "document_id": self._document_id,
            "local_file_path": self._clean_path if self._path_is_local else tool_args.get("local_file_path"),
            "attachment_id": tool_args.get("document_id_2") or tool_args.get("attachment_id"),
            "original_filename": self._uploaded_file_name,
        }
        return "doc_compare_attachment_with_local", new_args

    def _redirect_to_noop(self) -> tuple[str, dict[str, Any]]:
        logger.warning(
            "GUARD: doc_compare_documents blocked — doc_get_versions already complete"
        )
        noop_args: dict[str, Any] = {}
        if self._document_id:
            noop_args["document_id"] = self._document_id
        return "doc_get_details", noop_args