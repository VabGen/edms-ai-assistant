# edms_ai_assistant/tool_args_patcher.py
"""
ToolArgsPatcher — Single Responsibility: patch/mutate tool call arguments.

Extracted from EdmsDocumentAgent._orchestrate (the giant for-loop).
Handles:
1. Token injection
2. document_id injection
3. file_path routing (local file vs UUID attachment)
4. Placeholder replacement for read_local_file_content
5. Text injection for doc_summarize_text
6. create_document_from_file arg injection
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

_TOOLS_REQUIRING_DOCUMENT_ID: frozenset[str] = frozenset({
    "doc_get_details",
    "doc_get_versions",
    "doc_compare_documents",
    "doc_get_file_content",
    "doc_compare_attachment_with_local",
    "doc_summarize_text",
    "doc_search_tool",
    "introduction_create_tool",
    "task_create_tool",
})

_COMPARE_LOCAL_PLACEHOLDERS: frozenset[str] = frozenset({
    "",
    "local_file",
    "local_file_path",
    "/path/to/file",
    "path/to/file",
    "none",
    "null",
    "<local_file_path>",
    "<path>",
})

_LOCAL_FILE_PLACEHOLDERS: frozenset[str] = frozenset({
    "local_file",
    "file_path",
    "none",
    "null",
    "",
})


def _is_valid_uuid(value: str) -> bool:
    return bool(UUID_RE.match(value.strip()))


class ToolArgsPatcher:
    """
    Patches tool call arguments before execution.

    Immutable per-call context (user_token, document_id, file_path, etc.)
    is injected here — the LLM never sees or controls these values.

    Single Responsibility: mutate tool call args dict based on runtime context.
    Does NOT decide which tool to call — that is ToolCallRouter's job.
    Does NOT enforce call limits — that is ToolCallGuard's job.
    """

    def __init__(
        self,
        user_token: str,
        document_id: str | None,
        file_path: str | None,
        uploaded_file_name: str | None,
        user_context: dict,
        is_choice_active: bool,
    ) -> None:
        self._token = user_token
        self._document_id = document_id
        self._file_path = file_path
        self._uploaded_file_name = uploaded_file_name
        self._user_context = user_context
        self._is_choice_active = is_choice_active

        fp = str(file_path or "").strip()
        self._clean_path = fp
        self._path_is_uuid = _is_valid_uuid(fp) if fp else False
        self._path_is_local = bool(fp) and not self._path_is_uuid

    def patch(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: list[BaseMessage],
        last_tool_text: str | None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Apply all patches to tool_name and tool_args.

        Args:
            tool_name: Original tool name from LLM.
            tool_args: Original args from LLM.
            messages: Full message history (needed for some patches).
            last_tool_text: Text extracted from last ToolMessage (for summarize).

        Returns:
            Tuple of (patched_tool_name, patched_args).
        """
        t_name = tool_name
        t_args = dict(tool_args)

        t_args["token"] = self._token

        t_name, t_args = self._patch_create_document(t_name, t_args, messages)
        t_name, t_args = self._patch_document_id(t_name, t_args)
        t_name, t_args = self._patch_file_routing(t_name, t_args)
        t_name, t_args = self._patch_summarize(t_name, t_args, last_tool_text)

        return t_name, t_args

    # ── Private patch methods ─────────────────────────────────────────────────

    def _patch_create_document(
        self,
        t_name: str,
        t_args: dict[str, Any],
        messages: list[BaseMessage],
    ) -> tuple[str, dict[str, Any]]:
        if t_name != "create_document_from_file":
            return t_name, t_args

        if t_args.get("doc_category") is None:
            from edms_ai_assistant.tools.create_document_from_file import (
                _extract_category_from_message,
            )
            last_human_text = next(
                (str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)),
                "",
            )
            detected = _extract_category_from_message(last_human_text)
            if detected:
                t_args["doc_category"] = detected
                logger.info("Injected doc_category=%s for create_document_from_file", detected)

        if t_args.get("file_path") is None and self._path_is_local:
            t_args["file_path"] = self._clean_path
            logger.info("Injected file_path for create_document_from_file: %s...", self._clean_path[:32])

        if t_args.get("file_name") is None and self._uploaded_file_name:
            t_args["file_name"] = self._uploaded_file_name

        return t_name, t_args

    def _patch_document_id(
        self,
        t_name: str,
        t_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if not self._document_id:
            return t_name, t_args
        if t_name not in _TOOLS_REQUIRING_DOCUMENT_ID:
            return t_name, t_args

        cur = str(t_args.get("document_id", "")).strip()
        if not cur or not _is_valid_uuid(cur):
            t_args["document_id"] = self._document_id
            logger.debug("Injected document_id for tool '%s'", t_name)

        return t_name, t_args

    def _patch_file_routing(
        self,
        t_name: str,
        t_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if self._path_is_local:
            t_name, t_args = self._route_local_file(t_name, t_args)
        elif self._path_is_uuid:
            t_name, t_args = self._route_uuid_attachment(t_name, t_args)

        # inject local_file_path for compare tool
        if t_name == "doc_compare_attachment_with_local" and self._path_is_local:
            t_name, t_args = self._inject_local_file_path(t_name, t_args)

        return t_name, t_args

    def _route_local_file(
        self,
        t_name: str,
        t_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Redirect tool calls when a local file is present in context."""
        if t_name == "doc_get_versions":
            logger.warning(
                "GUARD: doc_get_versions blocked (local file present) → doc_compare_attachment_with_local"
            )
            new_args: dict[str, Any] = {"local_file_path": self._clean_path}
            if self._document_id:
                new_args["document_id"] = self._document_id
            return "doc_compare_attachment_with_local", new_args

        if t_name == "doc_compare_documents":
            logger.warning(
                "GUARD: doc_compare_documents blocked (local file present) → doc_compare_attachment_with_local"
            )
            t_args["local_file_path"] = self._clean_path
            t_args.pop("document_id_1", None)
            t_args.pop("document_id_2", None)
            return "doc_compare_attachment_with_local", t_args

        if t_name == "doc_get_file_content" and not t_args.get("attachment_id"):
            logger.info(
                "AUTO-PRIORITY: doc_get_file_content → read_local_file_content"
            )
            t_args["file_path"] = self._clean_path
            t_args.pop("attachment_id", None)
            return "read_local_file_content", t_args

        # placeholder replacement for read_local_file_content
        if t_name == "read_local_file_content":
            cur_fp = str(t_args.get("file_path", "")).strip()
            if cur_fp.lower() in _LOCAL_FILE_PLACEHOLDERS:
                t_args["file_path"] = self._clean_path
                logger.info("Injected local file_path (placeholder replaced)")

        return t_name, t_args

    def _route_uuid_attachment(
        self,
        t_name: str,
        t_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Redirect tool calls when context file_path is an EDMS attachment UUID."""
        if t_name == "read_local_file_content":
            logger.info("Routed read_local_file_content → doc_get_file_content")
            t_args["attachment_id"] = self._clean_path
            t_args.pop("file_path", None)
            return "doc_get_file_content", t_args

        if t_name == "doc_get_file_content":
            cur_att = str(t_args.get("attachment_id", "")).strip()
            if not cur_att or not _is_valid_uuid(cur_att):
                t_args["attachment_id"] = self._clean_path
                logger.info("Injected attachment_id from context")

        return t_name, t_args

    def _inject_local_file_path(
        self,
        t_name: str,
        t_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Ensure local_file_path and original_filename are set for compare tool."""
        cur_local = str(t_args.get("local_file_path", "")).strip()
        if not cur_local or cur_local.lower() in _COMPARE_LOCAL_PLACEHOLDERS or not Path(cur_local).exists():
            t_args["local_file_path"] = self._clean_path
            logger.info("Force-injected local_file_path for doc_compare_attachment_with_local")

        if self._uploaded_file_name and not t_args.get("original_filename"):
            t_args["original_filename"] = self._uploaded_file_name

        return t_name, t_args

    def _patch_summarize(
        self,
        t_name: str,
        t_args: dict[str, Any],
        last_tool_text: str | None,
    ) -> tuple[str, dict[str, Any]]:
        if t_name != "doc_summarize_text":
            return t_name, t_args

        if last_tool_text:
            t_args["text"] = last_tool_text

        if not t_args.get("summary_type"):
            if self._is_choice_active:
                t_args["summary_type"] = "extractive"
                logger.warning("safety-net: summary_type=extractive (is_choice_active but type not set)")
            else:
                preferred = self._user_context.get("preferred_summary_format")
                if preferred and preferred != "ask":
                    t_args["summary_type"] = preferred
                    logger.info("Using preferred_summary_format from user settings: %s", preferred)

        return t_name, t_args