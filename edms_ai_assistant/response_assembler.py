# edms_ai_assistant/response_assembler.py  (ПЕРЕРАБОТАН)
"""
ResponseAssembler — Single Responsibility: assemble the final AgentResponse
from a message chain, using injected collaborators for each sub-task.

Depends on abstractions (Protocols), not on EdmsDocumentAgent concretions.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage

from edms_ai_assistant.agent import (
    AgentResponse,
    AgentStatus,
    ContextParams,
    _is_mutation_response,
)
from edms_ai_assistant.agent_config import AgentConfig
from edms_ai_assistant.response_protocols import (
    IComplianceExtractor,
    IInteractiveStatusDetector,
    INavigateUrlExtractor,
    ISanitizer,
)

logger = logging.getLogger(__name__)


class ResponseAssembler:
    """
    Assembles final AgentResponse from a LangGraph message chain.

    All sub-tasks are delegated to injected collaborators:
    - ISanitizer: remove UUIDs / paths from text
    - IInteractiveStatusDetector: detect disambiguation / choice prompts
    - INavigateUrlExtractor: find navigate_url in tool results
    - IComplianceExtractor: find compliance check data
    - GuardrailPipeline: validate content before delivery

    Single Responsibility: orchestrate the response assembly pipeline.
    Does NOT implement any sub-task logic itself.
    """

    def __init__(
        self,
        sanitizer: ISanitizer,
        interactive_detector: IInteractiveStatusDetector,
        navigate_extractor: INavigateUrlExtractor,
        compliance_extractor: IComplianceExtractor,
        guardrail_pipeline: Any | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self._sanitizer = sanitizer
        self._interactive_detector = interactive_detector
        self._navigate_extractor = navigate_extractor
        self._compliance_extractor = compliance_extractor
        self._guardrails = guardrail_pipeline
        self._config = config

    def assemble(
        self,
        messages: list[BaseMessage],
        context: ContextParams,
    ) -> dict[str, Any]:
        """
        Build the final response dict from the message chain.

        Returns a serialized AgentResponse dict (compatible with HTTP layer).
        """
        # 1. Interactive status takes priority (disambiguation, choice selection)
        interactive = self._interactive_detector.detect(messages)
        if interactive:
            logger.info(
                "Interactive status detected",
                extra={"status": interactive.get("status")},
            )
            return interactive

        # 2. Extract compliance data (attached as metadata)
        compliance_data = self._compliance_extractor.extract(messages)

        # 3. Extract final text content
        from edms_ai_assistant.agent import ContentExtractor
        final_content = ContentExtractor.extract_final_content(messages)
        navigate_url = self._navigate_extractor.extract(messages)

        if not final_content:
            logger.warning("No final content found in message chain")
            meta: dict[str, Any] = {}
            if compliance_data:
                meta["compliance"] = compliance_data
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                content="Операция завершена.",
                navigate_url=navigate_url,
                metadata=meta,
            ).model_dump()

        # 4. Clean JSON wrappers
        final_content = ContentExtractor.clean_json_artifacts(final_content)

        # 5. Sanitize technical content
        final_content = self._sanitizer.sanitize(final_content, context)

        # 6. Guardrails
        if self._config and self._config.enable_guardrails and self._guardrails:
            result = self._guardrails.run(final_content)
            if result.blocked:
                logger.warning("Guardrail BLOCKED: %s", result.block_reason)
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Ответ заблокирован политикой безопасности.",
                ).model_dump()
            final_content = result.content

        # 7. Build response
        requires_reload = _is_mutation_response(final_content)
        metadata: dict[str, Any] = {}
        if compliance_data:
            metadata["compliance"] = compliance_data
            logger.info(
                "Compliance added to metadata: overall=%s fields=%d",
                compliance_data.get("overall"),
                len(compliance_data.get("fields", [])),
            )

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            content=final_content,
            requires_reload=requires_reload,
            navigate_url=navigate_url,
            metadata=metadata,
        ).model_dump()