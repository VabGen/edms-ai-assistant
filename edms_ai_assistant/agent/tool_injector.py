# edms_ai_assistant/agent/tool_injector.py
"""
ToolCallInjector — patches tool_call arguments before the graph resumes.

Responsibilities:
  - Auto-inject token, document_id, file_path into every tool_call.
  - Route tool names based on file context (local path vs UUID vs absent).
  - Prevent redundant calls (doc_get_versions after comparison is complete).
  - Auto-inject control_employee_id from a preceding single-result search.

This module contains no business logic — only argument transformation.
All helpers are pure functions or private static methods.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from edms_ai_assistant.agent.context import ContextParams, is_valid_uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Tools that require document_id to be present in their arguments.
TOOLS_REQUIRING_DOCUMENT_ID: frozenset[str] = frozenset(
    {
        "doc_get_details",
        "doc_get_versions",
        "doc_compare_documents",
        "doc_get_file_content",
        "doc_compare_attachment_with_local",
        "doc_summarize_text",
        "doc_search_tool",
        "introduction_create_tool",
        "task_create_tool",
    }
)

# Placeholder values that must be replaced with the real local path.
COMPARE_LOCAL_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "",
        "local_file",
        "local_file_path",
        "/path/to/file",
        "path/to/file",
        "none",
        "null",
        "<local_file_path>",
        "<path>",
    }
)

# Tools that participate in the disambiguation flow.
DISAMBIGUATION_TOOLS: frozenset[str] = frozenset(
    {
        "introduction_create_tool",
        "task_create_tool",
        "doc_control",
    }
)

# Maps each disambiguation tool to the argument that receives the resolved IDs.
TOOL_DISAMBIG_ID_FIELD: dict[str, str] = {
    "introduction_create_tool": "selected_employee_ids",
    "task_create_tool": "selected_employee_ids",
    "doc_control": "control_employee_id",
}

# Error signals that indicate a broken thread.
BROKEN_THREAD_SIGNALS: tuple[str, ...] = (
    "tool_calls must be followed by tool messages",
    "tool_call_ids did not have response messages",
    "invalid_request_error",
    "messages.[",
)


# ---------------------------------------------------------------------------
# Pure helper functions (message chain inspection)
# ---------------------------------------------------------------------------


def last_human_text(messages: list[BaseMessage]) -> str:
    """Return the text of the most recent HumanMessage, or empty string."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def was_compare_disambiguation(messages: list[BaseMessage]) -> bool:
    """Return True if the last ToolMessage before the current HumanMessage
    was a ``requires_disambiguation`` from ``doc_compare_attachment_with_local``."""
    for m in reversed(messages[-15:]):
        if isinstance(m, HumanMessage):
            return False
        if isinstance(m, ToolMessage):
            try:
                data: dict[str, Any] = json.loads(str(m.content))
                if (
                    data.get("status") == "requires_disambiguation"
                    and m.name == "doc_compare_attachment_with_local"
                ):
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass
    return False


def versions_comparison_complete(messages: list[BaseMessage]) -> bool:
    """Return True if ``doc_get_versions`` already returned ``comparison_complete=True``."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return False
        if isinstance(m, ToolMessage):
            try:
                data: dict[str, Any] = json.loads(str(m.content))
                if data.get("comparison_complete") and data.get("comparisons"):
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass
    return False


def find_single_employee_id(messages: list[BaseMessage]) -> str | None:
    """Return the employee UUID from the most recent single-result employee search,
    or ``None`` if no such result exists within the last 8 messages."""
    for m in reversed(messages[-8:]):
        if isinstance(m, HumanMessage):
            return None
        if not isinstance(m, ToolMessage):
            continue
        try:
            data: dict[str, Any] = json.loads(str(m.content))
            if data.get("status") == "found" and data.get("total") == 1:
                emp_card: dict[str, Any] = data.get("employee_card") or {}
                raw_id = (
                    data.get("id")
                    or emp_card.get("id")
                    or ((data.get("choices") or [{}])[0]).get("id")
                )
                if raw_id and is_valid_uuid(str(raw_id)):
                    return str(raw_id)
        except (json.JSONDecodeError, AttributeError, IndexError):
            pass
    return None


# ---------------------------------------------------------------------------
# ToolCallInjector
# ---------------------------------------------------------------------------


class ToolCallInjector:
    """Patches tool_call argument dicts before the LangGraph graph resumes.

    Each ``patch()`` call applies all injection rules in a single pass and
    returns a new list — the original ``raw_calls`` is never mutated.

    The injector is stateless: all context comes from the ``ContextParams``
    and the message chain passed to ``patch()``.
    """

    def patch(
        self,
        raw_calls: list[dict[str, Any]],
        context: ContextParams,
        messages: list[BaseMessage],
        last_tool_text: str | None,
        is_choice_active: bool,
    ) -> list[dict[str, Any]]:
        """Apply all injection rules to *raw_calls*.

        Args:
            raw_calls: The ``tool_calls`` list from the last ``AIMessage``.
            context: Current execution context (read-only).
            messages: Full message chain (read-only, for history lookups).
            last_tool_text: Text extracted from the last ``ToolMessage``
                (injected into ``doc_summarize_text.text``).
            is_choice_active: ``True`` when resuming from a HITL choice.

        Returns:
            New list of patched tool_call dicts.
        """
        clean_path = (context.file_path or "").strip()
        path_is_uuid = is_valid_uuid(clean_path) if clean_path else False
        path_is_local = bool(clean_path) and not path_is_uuid
        after_compare_disambig = was_compare_disambiguation(messages)

        patched: list[dict[str, Any]] = []
        for tc in raw_calls:
            t_name: str = tc["name"]
            t_args: dict[str, Any] = dict(tc["args"])
            t_id: str = tc["id"]

            # Rules are applied in dependency order:
            # token first, then document_id, then file routing, then specifics.
            t_name, t_args = _inject_token(t_name, t_args, context)
            t_name, t_args = _inject_document_id(t_name, t_args, context)
            t_name, t_args = _route_by_file_path(
                t_name,
                t_args,
                context,
                clean_path,
                path_is_uuid,
                path_is_local,
            )
            t_name, t_args = _inject_compare_local(
                t_name,
                t_args,
                context,
                clean_path,
                path_is_local,
                after_compare_disambig,
                is_choice_active,
            )
            t_name, t_args = _guard_compare_documents(
                t_name,
                t_args,
                context,
                messages,
                clean_path,
                path_is_local,
                after_compare_disambig,
                is_choice_active,
            )
            t_name, t_args = _inject_summarize(
                t_name,
                t_args,
                context,
                last_tool_text,
                is_choice_active,
            )
            t_name, t_args = _inject_create_document(
                t_name,
                t_args,
                context,
                messages,
            )
            t_name, t_args = _inject_control_employee(t_name, t_args, messages)
            t_name, t_args = _inject_local_file_placeholder(
                t_name,
                t_args,
                clean_path,
                path_is_local,
            )

            patched.append({"name": t_name, "args": t_args, "id": t_id})

        return patched


# ---------------------------------------------------------------------------
# Private injection rules
# ---------------------------------------------------------------------------


def _inject_token(
    t_name: str,
    t_args: dict[str, Any],
    context: ContextParams,
) -> tuple[str, dict[str, Any]]:
    """Inject the user JWT token into every tool_call."""
    t_args["token"] = context.user_token
    return t_name, t_args


def _inject_document_id(
    t_name: str,
    t_args: dict[str, Any],
    context: ContextParams,
) -> tuple[str, dict[str, Any]]:
    """Inject document_id when the tool requires it and the LLM omitted it."""
    if not context.document_id or t_name not in TOOLS_REQUIRING_DOCUMENT_ID:
        return t_name, t_args
    cur = str(t_args.get("document_id", "")).strip()
    if not cur or not is_valid_uuid(cur):
        t_args["document_id"] = context.document_id
        logger.debug("Injected document_id for '%s'", t_name)
    return t_name, t_args


def _route_by_file_path(
    t_name: str,
    t_args: dict[str, Any],
    context: ContextParams,
    clean_path: str,
    path_is_uuid: bool,
    path_is_local: bool,
) -> tuple[str, dict[str, Any]]:
    """Route tool name based on the type of file in context.

    Local file path → prefer local file tools.
    UUID attachment → prefer EDMS attachment tools.
    """
    if path_is_local:
        if t_name == "doc_get_versions":
            logger.warning(
                "GUARD: doc_get_versions blocked (local file present) "
                "→ doc_compare_attachment_with_local",
            )
            t_args = {"local_file_path": clean_path}
            if context.document_id:
                t_args["document_id"] = context.document_id
            return "doc_compare_attachment_with_local", t_args

        if t_name == "doc_compare_documents":
            logger.warning(
                "GUARD: doc_compare_documents blocked (local file present) "
                "→ doc_compare_attachment_with_local",
            )
            t_args["local_file_path"] = clean_path
            t_args.pop("document_id_1", None)
            t_args.pop("document_id_2", None)
            return "doc_compare_attachment_with_local", t_args

        if t_name == "doc_get_file_content" and not t_args.get("attachment_id"):
            logger.info(
                "AUTO-PRIORITY: doc_get_file_content → read_local_file_content "
                "(local file present, no explicit attachment_id)",
            )
            t_args["file_path"] = clean_path
            t_args.pop("attachment_id", None)
            return "read_local_file_content", t_args

    elif path_is_uuid:
        if t_name == "read_local_file_content":
            logger.info("Routed read_local_file_content → doc_get_file_content")
            t_args["attachment_id"] = clean_path
            t_args.pop("file_path", None)
            return "doc_get_file_content", t_args

        if t_name == "doc_get_file_content":
            cur = str(t_args.get("attachment_id", "")).strip()
            if not cur or not is_valid_uuid(cur):
                t_args["attachment_id"] = clean_path

    return t_name, t_args


def _inject_compare_local(
    t_name: str,
    t_args: dict[str, Any],
    context: ContextParams,
    clean_path: str,
    path_is_local: bool,
    after_compare_disambig: bool,
    is_choice_active: bool,
) -> tuple[str, dict[str, Any]]:
    """Ensure local_file_path and original_filename are set for compare_with_local.

    Also handles the fallback: when no file is present and the tool was called
    anyway without a prior disambiguation, route to version compare instead.
    """
    if t_name != "doc_compare_attachment_with_local":
        return t_name, t_args

    if (
        not clean_path
        and not after_compare_disambig
        and not (is_choice_active and t_args.get("attachment_id"))
    ):
        logger.info(
            "Routed doc_compare_attachment_with_local → doc_compare_documents "
            "(no file, no disambiguation context)",
        )
        t_args.pop("local_file_path", None)
        t_args.pop("attachment_id", None)
        return "doc_compare_documents", t_args

    # Inject local_file_path when placeholder or non-existent.
    if path_is_local:
        cur_local = str(t_args.get("local_file_path", "")).strip()
        if (
            not cur_local
            or cur_local.lower() in COMPARE_LOCAL_PLACEHOLDERS
            or not Path(cur_local).exists()
        ):
            t_args["local_file_path"] = clean_path
            logger.info(
                "Force-injected local_file_path for doc_compare_attachment_with_local"
            )

    # Inject original_filename for user-facing display in the tool.
    if context.uploaded_file_name and not t_args.get("original_filename"):
        t_args["original_filename"] = context.uploaded_file_name

    return t_name, t_args


def _guard_compare_documents(
    t_name: str,
    t_args: dict[str, Any],
    context: ContextParams,
    messages: list[BaseMessage],
    clean_path: str,
    path_is_local: bool,
    after_compare_disambig: bool,
    is_choice_active: bool,
) -> tuple[str, dict[str, Any]]:
    """Block redundant ``doc_compare_documents`` calls in two scenarios:

    1. After a ``requires_disambiguation`` from ``doc_compare_attachment_with_local``
       (the user has just resolved the ambiguity — we should compare, not version-diff).
    2. After ``doc_get_versions`` already completed all comparisons.
    """
    if t_name != "doc_compare_documents":
        return t_name, t_args

    if after_compare_disambig or (is_choice_active and path_is_local):
        logger.warning(
            "GUARD: doc_compare_documents blocked → doc_compare_attachment_with_local "
            "(reason: %s)",
            (
                "disambiguation_flow"
                if after_compare_disambig
                else "choice_active_with_local_file"
            ),
        )
        new_args: dict[str, Any] = {
            "token": t_args.get("token", ""),
            "document_id": context.document_id,
            "local_file_path": (
                clean_path if path_is_local else t_args.get("local_file_path")
            ),
            "attachment_id": t_args.get("document_id_2") or t_args.get("attachment_id"),
            "original_filename": context.uploaded_file_name,
        }
        for key in ("document_id_1", "document_id_2", "comparison_focus"):
            new_args.pop(key, None)
        return "doc_compare_attachment_with_local", new_args

    if versions_comparison_complete(messages):
        logger.warning(
            "GUARD: doc_compare_documents blocked — doc_get_versions already completed "
            "(comparison_complete=True). Replacing with doc_get_details (no-op).",
        )
        no_op: dict[str, Any] = {}
        if context.document_id:
            no_op["document_id"] = context.document_id
        return "doc_get_details", no_op

    return t_name, t_args


def _inject_summarize(
    t_name: str,
    t_args: dict[str, Any],
    context: ContextParams,
    last_tool_text: str | None,
    is_choice_active: bool,
) -> tuple[str, dict[str, Any]]:
    """Inject extracted text and summary_type into ``doc_summarize_text``."""
    if t_name != "doc_summarize_text":
        return t_name, t_args

    if last_tool_text:
        t_args["text"] = last_tool_text

    if not t_args.get("summary_type"):
        preferred: str | None = context.user_context.get("preferred_summary_format")
        if is_choice_active:
            t_args["summary_type"] = "extractive"
            logger.warning(
                "safety-net: summary_type=extractive "
                "(is_choice_active=True but type was not set)",
            )
        elif preferred and preferred != "ask":
            t_args["summary_type"] = preferred
            logger.info(
                "Using preferred_summary_format from user settings: %s", preferred
            )

    return t_name, t_args


def _inject_create_document(
    t_name: str,
    t_args: dict[str, Any],
    context: ContextParams,
    messages: list[BaseMessage],
) -> tuple[str, dict[str, Any]]:
    """Inject file_path, file_name, and doc_category into ``create_document_from_file``."""
    if t_name != "create_document_from_file":
        return t_name, t_args

    if t_args.get("doc_category") is None:
        from edms_ai_assistant.tools.create_document_from_file import (  # noqa: PLC0415
            _extract_category_from_message,
        )

        detected = _extract_category_from_message(last_human_text(messages))
        if detected:
            t_args["doc_category"] = detected
            logger.info(
                "Injected doc_category=%s for create_document_from_file", detected
            )

    if t_args.get("file_path") is None and context.file_path:
        cp = context.file_path.strip()
        if not is_valid_uuid(cp):
            t_args["file_path"] = cp

    if t_args.get("file_name") is None and context.uploaded_file_name:
        t_args["file_name"] = context.uploaded_file_name

    return t_name, t_args


def _inject_control_employee(
    t_name: str,
    t_args: dict[str, Any],
    messages: list[BaseMessage],
) -> tuple[str, dict[str, Any]]:
    """Auto-inject ``control_employee_id`` when a single employee was just found."""
    if t_name != "doc_control" or t_args.get("control_employee_id"):
        return t_name, t_args

    emp_id = find_single_employee_id(messages)
    if emp_id:
        t_args["control_employee_id"] = emp_id
        logger.info(
            "AUTO-INJECT: control_employee_id=%s from employee_search (single result)",
            emp_id[:8],
        )

    return t_name, t_args


def _inject_local_file_placeholder(
    t_name: str,
    t_args: dict[str, Any],
    clean_path: str,
    path_is_local: bool,
) -> tuple[str, dict[str, Any]]:
    """Replace placeholder file_path values in ``read_local_file_content``."""
    if not path_is_local or t_name != "read_local_file_content":
        return t_name, t_args

    cur_fp = str(t_args.get("file_path", "")).strip()
    if not cur_fp or cur_fp.lower() in ("local_file", "file_path", "none", "null", ""):
        t_args["file_path"] = clean_path
        logger.info("Injected local file_path (placeholder replaced)")

    return t_name, t_args
