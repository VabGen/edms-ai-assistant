# edms_ai_assistant/compliance_extractor.py
"""
ComplianceExtractor — Single Responsibility: find compliance check
results in ToolMessages.

Extracted from EdmsDocumentAgent._extract_compliance_data.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage


class ComplianceExtractor:
    """Implements IComplianceExtractor."""

    def extract(self, messages: list[BaseMessage]) -> dict[str, Any] | None:
        for m in reversed(messages[-10:]):
            if not isinstance(m, ToolMessage):
                continue
            try:
                data = json.loads(str(m.content))
                if (
                    data.get("status") == "success"
                    and isinstance(data.get("fields"), list)
                    and "overall" in data
                ):
                    return data
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        return None