# edms_ai_assistant/tool_call_patch_pipeline.py
"""
ToolCallPatchPipeline — composes ToolArgsPatcher + ToolCallRouter + ToolCallGuard
into a single processing step for one tool call.

Single Responsibility: orchestrate the three-stage patch pipeline.
Called once per tool call inside EdmsDocumentAgent._orchestrate.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage

from edms_ai_assistant.tool_args_patcher import ToolArgsPatcher
from edms_ai_assistant.tool_call_guard import GuardResult, ToolCallGuard
from edms_ai_assistant.tool_call_router import ToolCallRouter

logger = logging.getLogger(__name__)


class PatchedToolCall:
    """Result of processing a single tool call through the pipeline."""

    __slots__ = ("name", "args", "id", "guard_result")

    def __init__(
        self,
        name: str,
        args: dict[str, Any],
        call_id: str,
        guard_result: GuardResult,
    ) -> None:
        self.name = name
        self.args = args
        self.id = call_id
        self.guard_result = guard_result

    @property
    def allowed(self) -> bool:
        return self.guard_result.allowed

    @property
    def skip_silently(self) -> bool:
        return not self.guard_result.allowed and self.guard_result.skip_silently


class ToolCallPatchPipeline:
    """
    Three-stage pipeline for a single tool call:

    Stage 1 — Patch  (ToolArgsPatcher): inject token, document_id, file routing
    Stage 2 — Route  (ToolCallRouter):  history-based name redirections
    Stage 3 — Guard  (ToolCallGuard):   enforce call limits and ordering rules

    Returns PatchedToolCall with final name/args and whether the call is allowed.
    """

    def __init__(
        self,
        patcher: ToolArgsPatcher,
        router: ToolCallRouter,
        guard: ToolCallGuard,
    ) -> None:
        self._patcher = patcher
        self._router = router
        self._guard = guard

    def process(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_id: str,
        messages: list[BaseMessage],
        last_tool_text: str | None,
    ) -> PatchedToolCall:
        """
        Process one tool call through all three stages.

        Args:
            tool_name: Raw tool name from LLM.
            tool_args: Raw args from LLM.
            call_id: Tool call ID from LLM.
            messages: Full message history.
            last_tool_text: Text from last ToolMessage (for summarize injection).

        Returns:
            PatchedToolCall with final name, args, and guard decision.
        """
        # Stage 1: Patch args
        patched_name, patched_args = self._patcher.patch(
            tool_name, tool_args, messages, last_tool_text
        )

        # Stage 2: Route based on history
        routed_name, routed_args = self._router.route(
            patched_name, patched_args, messages
        )

        # Stage 3: Guard
        guard_result = self._guard.check(routed_name, routed_args)

        if not guard_result.allowed:
            logger.warning(
                "ToolCallGuard BLOCKED: %s — %s (silent=%s)",
                routed_name,
                guard_result.reason,
                guard_result.skip_silently,
            )

        return PatchedToolCall(
            name=routed_name,
            args=routed_args,
            call_id=call_id,
            guard_result=guard_result,
        )