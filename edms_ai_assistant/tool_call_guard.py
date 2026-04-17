# edms_ai_assistant/tool_call_guard.py
"""
Tool Call Guard - prevents invalid/duplicate tool calls.

Fixes a real bug: doc_compliance_check can theoretically be called twice
in one request (the prompt says not to, but LLMs sometimes violate it).
This guard makes it a code-level guarantee, not just a prompt instruction.

Also blocks doc_compare_documents after doc_get_versions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Result of a tool call guard check."""
    allowed: bool
    reason: str = ""
    skip_silently: bool = False


@dataclass
class ToolCallGuard:
    """
    Per-request guard that tracks which tools have been called and
    enforces ordering/exclusion rules.

    Rules enforced:
    1. doc_compliance_check: MAX once per request
    2. doc_compare_documents: BLOCKED after doc_get_versions
    3. doc_get_versions: MAX once per request
    4. doc_get_details: MAX 3 times per request (prevent loops)
    """

    _called: dict[str, int] = field(default_factory=dict)

    MAX_CALLS: dict[str, int] = field(default_factory=lambda: {
        "doc_compliance_check": 1,
        "doc_get_versions": 1,
        "doc_get_details": 3,
    })

    BLOCKED_AFTER: dict[str, set[str]] = field(default_factory=lambda: {
        "doc_compare_documents": {"doc_get_versions"},
    })

    def check(self, tool_name: str, args: dict | None = None) -> GuardResult:
        """
        Check whether a tool call is allowed.

        Args:
            tool_name: Name of the tool being called.
            args: Tool call arguments (reserved for future arg-based rules).

        Returns:
            GuardResult with allowed=True if the call is permitted.
        """
        max_calls = self.MAX_CALLS.get(tool_name)
        if max_calls is not None:
            current = self._called.get(tool_name, 0)
            if current >= max_calls:
                return GuardResult(
                    allowed=False,
                    reason=f"{tool_name} already called {current} time(s), max is {max_calls}",
                    skip_silently=False,
                )

        blocked_after = self.BLOCKED_AFTER.get(tool_name)
        if blocked_after:
            for blocker in blocked_after:
                if blocker in self._called:
                    return GuardResult(
                        allowed=False,
                        reason=f"{tool_name} is blocked after {blocker} was called",
                        skip_silently=True,
                    )

        self._called[tool_name] = self._called.get(tool_name, 0) + 1
        return GuardResult(allowed=True)

    @property
    def called_tools(self) -> dict[str, int]:
        """Return a copy of the called tools dict."""
        return dict(self._called)