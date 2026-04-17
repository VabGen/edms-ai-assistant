# edms_ai_assistant/response_assembler.py
"""
Response Assembler - extracted from EdmsDocumentAgent._build_final_response.

Single Responsibility: takes raw messages + context and produces
a clean AgentResponse with all post-processing applied.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from edms_ai_assistant.agent import (
    AgentResponse,
    AgentStatus,
    ActionType,
    ContentExtractor,
    ContextParams,
    _is_mutation_response,
)

if TYPE_CHECKING:
    from edms_ai_assistant.guardrails import GuardrailPipeline

logger = logging.getLogger(__name__)


class ResponseAssembler:
    """
    Assembles the final AgentResponse from a LangGraph message chain.

    Pipeline:
    1. Extract final content from messages
    2. Clean JSON artifacts
    3. Sanitize technical content (UUIDs, paths, tokens)
    4. Run guardrails (if enabled)
    5. Detect interactive status (disambiguation, confirmation)
    6. Build AgentResponse with requires_reload, navigate_url, etc.
    """

    def __init__(
        self,
        guardrail_pipeline: "GuardrailPipeline | None" = None,
        enable_guardrails: bool = True,
    ) -> None:
        self._guardrail_pipeline = guardrail_pipeline
        self._enable_guardrails = enable_guardrails

    def assemble(
        self,
        messages: list[BaseMessage],
        context: ContextParams,
        *,
        sanitize_fn: Any = None,
        detect_interactive_fn: Any = None,
        extract_navigate_url_fn: Any = None,
    ) -> AgentResponse:
        """Build the final AgentResponse from the message chain."""
        # 1. Extract final content
        final_content = ContentExtractor.extract_final_content(messages)
        if not final_content:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="No content could be extracted from the agent response.",
            )

        # 2. Clean JSON artifacts
        final_content = ContentExtractor.clean_json_artifacts(final_content)

        # 3. Sanitize technical content (UUIDs, paths, tokens)
        if sanitize_fn is not None:
            final_content = sanitize_fn(final_content, context)

        # 4. Guardrails
        if self._enable_guardrails and self._guardrail_pipeline is not None:
            guardrail_result = self._guardrail_pipeline.run(final_content)
            if guardrail_result and guardrail_result.get("blocked"):
                logger.warning(
                    "Guardrail blocked response: %s",
                    guardrail_result.get("reason", "unknown"),
                )
                final_content = guardrail_result.get(
                    "replacement",
                    "Response blocked by safety policy.",
                )

        # 5. Detect interactive status
        action_type: ActionType | None = None
        interactive_data: dict | None = None
        if detect_interactive_fn is not None:
            action_type, interactive_data = detect_interactive_fn(messages)

        # 6. Extract navigate_url
        navigate_url: str | None = None
        if extract_navigate_url_fn is not None:
            navigate_url = extract_navigate_url_fn(messages)

        # 7. Determine requires_reload
        requires_reload = _is_mutation_response(final_content)

        # 8. Build response
        metadata: dict[str, Any] = {}
        if interactive_data:
            metadata["interactive_data"] = interactive_data

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            content=final_content,
            action_type=action_type,
            requires_reload=requires_reload,
            navigate_url=navigate_url,
            metadata=metadata,
        )