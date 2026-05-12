# edms_ai_assistant/agent/escaping.py
"""
XML escaping utilities for prompt construction.

Used by both ``agent.py`` (semantic XML) and ``prompts.py`` (system prompt
template rendering) to sanitise user-controlled values before embedding
them inside XML markup.
"""

from __future__ import annotations

import html


def xml_escape_text(value: str | None) -> str:
    """Escape XML special characters in user-controlled values.

    Prevents prompt injection via user-supplied text (names, file paths, etc.)
    that is embedded inside XML markup in system prompts.

    This is **not** HTML escaping for browser output — it is specifically
    for XML-tag contexts within LLM prompts.

    Args:
        value: Raw string that originates from user input.

    Returns:
        String safe to embed inside an XML element.
    """
    if not value:
        return ""
    return html.escape(value, quote=False)