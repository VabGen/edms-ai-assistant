# edms_ai_assistant/agent_components.py
"""
AgentComponents — factory that wires all SRP-extracted collaborators.

Single place where concrete implementations are assembled.
Tests can provide their own implementations without touching EdmsDocumentAgent.
"""
from __future__ import annotations

from edms_ai_assistant.agent_config import AgentConfig
from edms_ai_assistant.compliance_extractor import ComplianceExtractor
from edms_ai_assistant.content_sanitizer import ContentSanitizer
from edms_ai_assistant.guardrails import GuardrailPipeline
from edms_ai_assistant.interactive_status_detector import InteractiveStatusDetector
from edms_ai_assistant.navigate_url_extractor import NavigateUrlExtractor
from edms_ai_assistant.response_assembler import ResponseAssembler
from edms_ai_assistant.tool_call_guard import ToolCallGuard


def build_response_assembler(
    config: AgentConfig,
    guardrail_pipeline: GuardrailPipeline | None = None,
) -> ResponseAssembler:
    """Create a fully wired ResponseAssembler."""
    return ResponseAssembler(
        sanitizer=ContentSanitizer(),
        interactive_detector=InteractiveStatusDetector(),
        navigate_extractor=NavigateUrlExtractor(),
        compliance_extractor=ComplianceExtractor(),
        guardrail_pipeline=guardrail_pipeline,
        config=config,
    )


def build_tool_call_guard() -> ToolCallGuard:
    """Create a fresh per-request ToolCallGuard."""
    return ToolCallGuard()