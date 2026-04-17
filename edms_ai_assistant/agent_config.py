# edms_ai_assistant/agent_config.py
"""
Agent configuration — centralized, typed, environment-overridable.

Best Practice (2026): All magic numbers and tunables live in one immutable
dataclass.  No scattered constants — every knob is documented, typed,
and can be overridden via environment variables for A/B testing or hot-fixes.

Inspired by Anthropic's "Constitutional AI Config" pattern and Google's
"Config as Data" approach.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentConfig:
    """
    Immutable agent configuration.

    Every field has a sensible default but can be overridden via the
    ``EDMS_AGENT_`` environment variable prefix (e.g. ``EDMS_AGENT_MAX_ITERATIONS=15``).

    Attributes:
        max_iterations: Maximum orchestration loop iterations before abort.
        execution_timeout: Wall-clock timeout (seconds) for a single graph invocation.
        max_history_messages: Truncation window for non-system messages sent to LLM.
        min_content_length: Minimum characters for a "substantial" AIMessage.
        token_budget_per_request: Soft token budget per request (for observability).
        retry_max_attempts: Number of retry attempts for transient LLM errors.
        retry_base_delay: Base delay (seconds) for exponential backoff.
        retry_max_delay: Maximum delay (seconds) between retries.
        enable_guardrails: Whether to run output guardrails before returning responses.
        enable_tracing: Whether to emit OpenTelemetry-compatible trace spans.
        enable_token_tracking: Whether to accumulate token usage in metadata.
        compliance_max_calls: Maximum allowed doc_compliance_check calls per request.
        parallel_tool_calls_allowed: Whether to allow parallel tool calls from LLM.
    """

    max_iterations: int = 10
    execution_timeout: float = 120.0
    max_history_messages: int = 40
    min_content_length: int = 30
    token_budget_per_request: int = 8000
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    enable_guardrails: bool = True
    enable_tracing: bool = True
    enable_token_tracking: bool = True
    compliance_max_calls: int = 1
    parallel_tool_calls_allowed: bool = False

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Build config from environment variables with ``EDMS_AGENT_`` prefix.

        Example::

            EDMS_AGENT_MAX_ITERATIONS=15 EDMS_AGENT_ENABLE_GUARDRAILS=true python -m ...

        Returns:
            AgentConfig with environment overrides applied.
        """
        overrides: dict[str, object] = {}
        prefix = "EDMS_AGENT_"

        int_fields = {
            "max_iterations",
            "min_content_length",
            "token_budget_per_request",
            "retry_max_attempts",
            "max_history_messages",
            "compliance_max_calls",
        }
        float_fields = {
            "execution_timeout",
            "retry_base_delay",
            "retry_max_delay",
        }
        bool_fields = {
            "enable_guardrails",
            "enable_tracing",
            "enable_token_tracking",
            "parallel_tool_calls_allowed",
        }

        for key in int_fields:
            val = os.environ.get(f"{prefix}{key.upper()}")
            if val is not None:
                overrides[key] = int(val)

        for key in float_fields:
            val = os.environ.get(f"{prefix}{key.upper()}")
            if val is not None:
                overrides[key] = float(val)

        for key in bool_fields:
            val = os.environ.get(f"{prefix}{key.upper()}")
            if val is not None:
                overrides[key] = val.lower() in ("true", "1", "yes")

        return cls(**overrides)


# Module-level singleton — created once at import time.
# Tests can override by creating their own AgentConfig instance.
DEFAULT_CONFIG: AgentConfig = AgentConfig.from_env()