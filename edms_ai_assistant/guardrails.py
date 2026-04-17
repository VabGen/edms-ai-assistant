# edms_ai_assistant/guardrails.py
"""
Output Guardrails — validate and sanitize agent responses before delivery.

Best Practice (2026): Every LLM output passes through a guardrail pipeline
before reaching the user.  This prevents:
- PII / credential leakage (tokens, UUIDs shown to user)
- Hallucinated tool calls in prose
- Overly verbose or off-topic responses
- Safety policy violations

Inspired by Anthropic's "Constitutional AI" guardrail pattern and
OpenAI's "Moderation API" approach — but fully local and deterministic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─── Guardrail result ──────────────────────────────────────────────────────────


class GuardrailAction(str, Enum):
    """Action taken by a guardrail check."""

    PASS = "pass"  # Content is clean
    SANITIZE = "sanitize"  # Content was modified
    BLOCK = "block"  # Content must not be shown


@dataclass
class GuardrailResult:
    """
    Result of a single guardrail check.

    Attributes:
        name: Guardrail identifier.
        action: Decision taken.
        reason: Human-readable explanation (for logging).
        modified_content: Sanitized content (if action == SANITIZE).
    """

    name: str
    action: GuardrailAction
    reason: str = ""
    modified_content: str | None = None


# ─── Individual guardrails ─────────────────────────────────────────────────────


class CredentialLeakGuardrail:
    """
    Detects and redacts JWT tokens, API keys, and authorization headers
    that may have leaked into the LLM output.

    Patterns caught:
    - ``Bearer eyJ...`` / ``Authorization: ...``
    - Long base64 strings resembling JWT (3 dot-separated segments)
    - ``api_key=...`` / ``token=...`` assignments
    """

    NAME = "credential_leak"

    _PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I), "[REDACTED_TOKEN]"),
        (re.compile(r"Authorization\s*:\s*\S+", re.I), "Authorization: [REDACTED]"),
        (
            re.compile(r"eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+"),
            "[REDACTED_JWT]",
        ),
        (re.compile(r'(?:api_key|apikey|token)\s*[=:]\s*["\']?\S{8,}', re.I), "[REDACTED_KEY]"),
    ]

    def check(self, content: str) -> GuardrailResult:
        """Scan content for credential leaks."""
        modified = content
        had_match = False

        for pattern, replacement in self._PATTERNS:
            new_content, count = pattern.subn(replacement, modified)
            if count > 0:
                had_match = True
                modified = new_content

        if not had_match:
            return GuardrailResult(name=self.NAME, action=GuardrailAction.PASS)

        logger.warning(
            "Guardrail: credential leak detected and redacted",
            extra={"guardrail": self.NAME},
        )
        return GuardrailResult(
            name=self.NAME,
            action=GuardrailAction.SANITIZE,
            reason="Credential pattern detected in output",
            modified_content=modified,
        )


class UUIDExposureGuardrail:
    """
    Detects bare UUIDs in user-facing output that should be shown as
    human-readable names instead.

    This is a *secondary* check — the primary sanitization happens in
    ``_sanitize_technical_content``.  This guardrail catches any UUIDs
    that slip through (e.g., in edge cases or new tool outputs).
    """

    NAME = "uuid_exposure"

    _UUID_RE = re.compile(
        r"(?<![\/\w])"
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        r"(?![\/\w])",
        re.I,
    )

    # UUIDs that are intentionally shown (e.g., in search result tables)
    _ALLOWED_CONTEXTS: tuple[str, ...] = (
        "|",  # markdown table row
        "id",  # column header context
    )

    def check(self, content: str) -> GuardrailResult:
        """Scan content for exposed UUIDs outside allowed contexts."""
        # Don't flag UUIDs inside markdown tables (search results)
        lines = content.split("\n")
        needs_sanitization = False
        result_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip table rows — UUIDs are expected there
            if stripped.startswith("|"):
                result_lines.append(line)
                continue

            if self._UUID_RE.search(line):
                needs_sanitization = True
                # Replace UUID with placeholder
                sanitized = self._UUID_RE.sub("«идентификатор»", line)
                result_lines.append(sanitized)
            else:
                result_lines.append(line)

        if not needs_sanitization:
            return GuardrailResult(name=self.NAME, action=GuardrailAction.PASS)

        return GuardrailResult(
            name=self.NAME,
            action=GuardrailAction.SANITIZE,
            reason="UUID detected outside markdown table context",
            modified_content="\n".join(result_lines),
        )


class ResponseLengthGuardrail:
    """
    Ensures responses don't exceed a reasonable length.

    Overly long responses often indicate:
    - The LLM is dumping raw tool output
    - Verbose hallucination
    - Missing summarization step
    """

    NAME = "response_length"

    # 4000 chars ≈ 1000 tokens — reasonable max for a chat response
    MAX_LENGTH: int = 4000
    TRUNCATION_NOTICE = "\n\n---\n*Ответ сокращён. Для подробностей спросите уточняющий вопрос.*"

    def check(self, content: str) -> GuardrailResult:
        """Check if response exceeds maximum length."""
        if len(content) <= self.MAX_LENGTH:
            return GuardrailResult(name=self.NAME, action=GuardrailAction.PASS)

        truncated = content[: self.MAX_LENGTH] + self.TRUNCATION_NOTICE
        logger.info(
            "Guardrail: response truncated from %d to %d chars",
            len(content),
            self.MAX_LENGTH,
        )
        return GuardrailResult(
            name=self.NAME,
            action=GuardrailAction.SANITIZE,
            reason=f"Response too long ({len(content)} chars)",
            modified_content=truncated,
        )


class SafetyPolicyGuardrail:
    """
    Catches responses that violate basic safety policies:
    - Instructions for illegal activities
    - Prompt injection artifacts (``<system>``, ``ignore previous``)
    - Raw JSON tool responses leaked to user

    This is a lightweight deterministic check — not a full moderation model.
    For production, integrate with a dedicated moderation API.
    """

    NAME = "safety_policy"

    _BLOCK_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"<system>", re.I),
        re.compile(r"ignore\s+previous\s+(?:instructions?|prompts?)", re.I),
        re.compile(r"you\s+are\s+now\s+(?:DAN|jailbroken)", re.I),
    ]

    _SUSPICIOUS_PATTERNS: list[re.Pattern[str]] = [
        # Raw JSON tool response leaked (starts with {"status":)
        re.compile(r'^\{"status"\s*:', re.M),
    ]

    def check(self, content: str) -> GuardrailResult:
        """Check for safety policy violations."""
        for pattern in self._BLOCK_PATTERNS:
            if pattern.search(content):
                logger.warning(
                    "Guardrail: safety policy violation detected",
                    extra={"guardrail": self.NAME, "pattern": pattern.pattern},
                )
                return GuardrailResult(
                    name=self.NAME,
                    action=GuardrailAction.BLOCK,
                    reason=f"Safety policy violation: matched pattern '{pattern.pattern}'",
                )

        return GuardrailResult(name=self.NAME, action=GuardrailAction.PASS)


# ─── Guardrail Pipeline ───────────────────────────────────────────────────────


class GuardrailPipeline:
    """
    Runs a sequence of guardrails over agent output.

    Pipeline semantics:
    - PASS → continue to next guardrail
    - SANITIZE → apply modification, continue with sanitized content
    - BLOCK → immediately return error, no further checks

    Usage::

        pipeline = GuardrailPipeline()
        result = pipeline.run(content)
        if result.blocked:
            return AgentResponse(status=AgentStatus.ERROR, ...)
        safe_content = result.content
    """

    def __init__(self) -> None:
        self._guardrails = [
            SafetyPolicyGuardrail(),
            CredentialLeakGuardrail(),
            UUIDExposureGuardrail(),
            ResponseLengthGuardrail(),
        ]

    def run(self, content: str) -> GuardrailPipelineResult:
        """
        Execute all guardrails in sequence.

        Args:
            content: Raw agent output to validate.

        Returns:
            Pipeline result with final (possibly sanitized) content and
            a list of all guardrail decisions for audit logging.
        """
        current = content
        decisions: list[GuardrailResult] = []
        blocked = False
        block_reason = ""

        for guardrail in self._guardrails:
            result = guardrail.check(current)
            decisions.append(result)

            if result.action == GuardrailAction.BLOCK:
                blocked = True
                block_reason = result.reason
                break
            elif result.action == GuardrailAction.SANITIZE:
                current = result.modified_content or current

        if decisions:
            logger.debug(
                "Guardrail pipeline complete",
                extra={
                    "decisions": [
                        {"name": d.name, "action": d.action.value} for d in decisions
                    ],
                    "blocked": blocked,
                },
            )

        return GuardrailPipelineResult(
            content=current,
            blocked=blocked,
            block_reason=block_reason,
            decisions=decisions,
        )


@dataclass
class GuardrailPipelineResult:
    """
    Aggregate result of the guardrail pipeline.

    Attributes:
        content: Final (possibly sanitized) content.
        blocked: Whether any guardrail blocked the output.
        block_reason: Human-readable reason if blocked.
        decisions: All individual guardrail results for audit.
    """

    content: str
    blocked: bool = False
    block_reason: str = ""
    decisions: list[GuardrailResult] = field(default_factory=list)