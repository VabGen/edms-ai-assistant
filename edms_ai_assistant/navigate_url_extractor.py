# edms_ai_assistant/navigate_url_extractor.py
"""
NavigateUrlExtractor — Single Responsibility: scan ToolMessages for
navigate_url (produced by create_document_from_file).

Extracted from EdmsDocumentAgent._extract_navigate_url.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage


class NavigateUrlExtractor:
    """Implements INavigateUrlExtractor."""

    def extract(self, messages: list[BaseMessage]) -> str | None:
        for m in reversed(messages[-8:]):
            if not isinstance(m, ToolMessage):
                continue
            try:
                data: dict[str, Any] = json.loads(str(m.content))
                url = data.get("navigate_url")
                if url and isinstance(url, str) and url.startswith("/"):
                    return url
            except (json.JSONDecodeError, AttributeError):
                pass
        return None