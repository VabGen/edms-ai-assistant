# edms_ai_assistant/response_protocols.py
"""
Protocols for ResponseAssembler dependencies.

Dependency Inversion: assembler depends on abstractions, not on
EdmsDocumentAgent concrete methods.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage

from edms_ai_assistant.agent import ContextParams


@runtime_checkable
class ISanitizer(Protocol):
    """Removes technical artifacts (UUIDs, paths) from user-visible text."""

    def sanitize(self, content: str, context: ContextParams) -> str:
        ...


@runtime_checkable
class IInteractiveStatusDetector(Protocol):
    """
    Detects whether the last ToolMessage requires user interaction.
    Returns serialized AgentResponse dict or None.
    """

    def detect(self, messages: list[BaseMessage]) -> dict[str, Any] | None:
        ...


@runtime_checkable
class INavigateUrlExtractor(Protocol):
    """Extracts navigate_url from the last ToolMessage if present."""

    def extract(self, messages: list[BaseMessage]) -> str | None:
        ...


@runtime_checkable
class IComplianceExtractor(Protocol):
    """Extracts compliance check result from ToolMessages."""

    def extract(self, messages: list[BaseMessage]) -> dict[str, Any] | None:
        ...