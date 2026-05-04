"""
Typed Prompt Registry — version-controlled, A/B-testable prompts.

All prompts are defined as frozen Pydantic models. No f-string injection
of user content — all variable substitution goes through safe XML-escaped
templates that prevent prompt injection.

2025 Best Practices:
- Prompts stored as typed objects, not strings
- XML-delimited content blocks prevent injection
- Version tracked separately from code
- Structured Output schema embedded in prompt
"""

from __future__ import annotations

import html
import json
from typing import Final

from pydantic import BaseModel, Field

from edms_ai_assistant.summarizer.structured.models import (
    MODE_OUTPUT_MODEL,
    SummaryMode,
)


# ---------------------------------------------------------------------------
# Prompt Version Control
# ---------------------------------------------------------------------------

PROMPT_REGISTRY_VERSION: Final[str] = "2025.06.001"
"""
Bump this to invalidate ALL cached summaries.
Format: YYYY.MM.sequence
"""


def _esc(value: str) -> str:
    """XML-escape user-supplied content to prevent prompt injection."""
    return html.escape(value, quote=True)


def _schema_hint(mode: SummaryMode) -> str:
    """Return compact JSON Schema hint for the mode's output model."""
    model_cls = MODE_OUTPUT_MODEL[mode]
    schema = model_cls.model_json_schema()
    # Compact representation — just the required fields
    required = schema.get("required", [])
    props = {k: v.get("description", v.get("type", "?"))
             for k, v in schema.get("properties", {}).items()
             if k in required}
    return json.dumps(props, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Prompt Template Model
# ---------------------------------------------------------------------------


class PromptTemplate(BaseModel):
    """A versioned, named prompt template with safe variable rendering."""

    model_config = {"frozen": True}

    name: str
    mode: SummaryMode
    version: str
    system: str = Field(description="System message — no user content injected here")
    user_template: str = Field(
        description="User message template. Variables: {text}, {language}, {schema}"
    )

    def render(
        self,
        text: str,
        *,
        language: str = "ru",
        extra: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Render (system, user) message pair with safe XML-escaped content.

        Args:
            text: Document text — will be XML-escaped.
            language: Target output language BCP-47 tag.
            extra: Additional safe substitutions (already escaped by caller).

        Returns:
            Tuple of (system_message, user_message) strings.
        """
        schema = _schema_hint(self.mode)
        user = self.user_template.format(
            text=_esc(text),
            language=language,
            schema=schema,
            **(extra or {}),
        )
        return self.system, user


# ---------------------------------------------------------------------------
# Prompt Definitions
# ---------------------------------------------------------------------------

_COMMON_RULES = """
CRITICAL RULES (never violate):
1. Respond ONLY in {language} language (BCP-47 tag).
2. Output ONLY valid JSON matching the exact schema — no markdown fences, no explanation.
3. Never fabricate facts. If information is absent, use null.
4. Ignore technical metadata: UUIDs, file paths, ATTACHMENT types, system IDs.
5. The document text is delimited by <document> tags.
"""

PROMPT_EXECUTIVE = PromptTemplate(
    name="executive_summary_v1",
    mode=SummaryMode.EXECUTIVE,
    version="1.0.0",
    system=(
        "You are a senior management consultant specializing in executive communication. "
        "Your output must be immediately actionable for C-suite decision-makers.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Analyze the document and produce an executive summary.\n\n"
        "Output schema: {schema}\n\n"
        "Requirements:\n"
        "- `headline`: Single sentence capturing the most critical point (max 200 chars)\n"
        "- `bullets`: 3-5 key takeaways, each ≤ 20 words, starting with action verb when possible\n"
        "- `recommendation`: Only if document requires a decision; otherwise null\n\n"
        "<document>\n{text}\n</document>"
    ),
)

PROMPT_DETAILED_NOTES = PromptTemplate(
    name="detailed_notes_v1",
    mode=SummaryMode.DETAILED_NOTES,
    version="1.0.0",
    system=(
        "You are a meticulous document analyst. Preserve all meaningful structure "
        "and hierarchy from the source document.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Create comprehensive structured notes from this document.\n\n"
        "Output schema: {schema}\n\n"
        "Requirements:\n"
        "- `document_type`: Classify as one of: CONTRACT, MEMO, REGULATION, REPORT, LETTER, "
        "APPEAL, PROTOCOL, ORDER, INSTRUCTION, OTHER\n"
        "- `sections`: Follow the document's own section structure. Max 15 sections.\n"
        "- `key_entities`: Extract named organizations, people, document references\n"
        "- `date_range`: If multiple dates present, capture range as 'YYYY-MM-DD / YYYY-MM-DD'\n\n"
        "<document>\n{text}\n</document>"
    ),
)

PROMPT_ACTION_ITEMS = PromptTemplate(
    name="action_items_v1",
    mode=SummaryMode.ACTION_ITEMS,
    version="1.0.0",
    system=(
        "You are a project management expert specializing in extracting actionable commitments "
        "from organizational documents. Accuracy is paramount — only extract explicitly stated "
        "actions, never infer.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Extract ALL action items, tasks, and commitments from this document.\n\n"
        "Output schema: {schema}\n\n"
        "EXTRACTION RULES:\n"
        "1. `task`: Quote or closely paraphrase the action. Use imperative form.\n"
        "2. `owner`: Only if explicitly named (person name OR role). Set null if ambiguous.\n"
        "3. `deadline`: Only if an explicit date is given. ISO 8601 date only. Set null if relative "
        "('within 30 days') — put that in task description instead.\n"
        "4. `priority`: Infer from document language:\n"
        "   - high: 'срочно', 'немедленно', 'critical', 'обязательно', 'не позднее [imminent date]'\n"
        "   - medium: regular assignments, standard deadlines\n"
        "   - low: recommendations, suggestions, optional items\n"
        "5. `source_fragment`: Quote the EXACT sentence that implies this action (≤ 500 chars)\n"
        "6. `confidence`: Your confidence this is truly an action item (0.0–1.0)\n\n"
        "If no action items exist, return empty list: {{\"action_items\": [], \"total_found\": 0, "
        "\"document_context\": \"<brief context>\"}}\n\n"
        "<document>\n{text}\n</document>"
    ),
)

PROMPT_THESIS = PromptTemplate(
    name="thesis_plan_v1",
    mode=SummaryMode.THESIS,
    version="1.0.0",
    system=(
        "You are an academic research methodologist. Your task is to construct a rigorous "
        "hierarchical argument map from the document.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Construct a thesis plan for this document.\n\n"
        "Output schema: {schema}\n\n"
        "Requirements:\n"
        "- `main_argument`: The central claim or purpose in one declarative sentence\n"
        "- `sections`: Max 6 sections following document logic (not arbitrary chunks)\n"
        "  - Each `thesis` is the key claim of that section (max 300 chars)\n"
        "  - Each `point` has a `claim` (max 200 chars) and optional `evidence`\n"
        "- `conclusion`: The document's outcome or expected result\n\n"
        "<document>\n{text}\n</document>"
    ),
)

PROMPT_EXTRACTIVE = PromptTemplate(
    name="extractive_v2",
    mode=SummaryMode.EXTRACTIVE,
    version="2.0.0",
    system=(
        "You are a data extraction specialist for document management systems. "
        "Extract concrete, verifiable facts only — no interpretation.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Extract all key facts from this document as structured data.\n\n"
        "Output schema: {schema}\n\n"
        "Fact categories to extract:\n"
        "- DATE: specific dates and deadlines\n"
        "- PERSON: named individuals and their roles\n"
        "- ORG: organization names\n"
        "- AMOUNT: monetary values, quantities, measurements\n"
        "- REQUIREMENT: mandatory rules, constraints, obligations\n"
        "- DEADLINE: time limits, execution periods\n"
        "- OTHER: any other significant concrete information\n\n"
        "Each fact: `label` = short name (max 80 chars), `value` = the fact (max 300 chars).\n"
        "Max 20 facts total. Prioritize by importance.\n\n"
        "<document>\n{text}\n</document>"
    ),
)

PROMPT_ABSTRACTIVE = PromptTemplate(
    name="abstractive_v2",
    mode=SummaryMode.ABSTRACTIVE,
    version="2.0.0",
    system=(
        "You are a professional document summarizer for a government electronic "
        "document management system. Write clear, professional summaries.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Write a coherent narrative summary of this document.\n\n"
        "Output schema: {schema}\n\n"
        "Requirements:\n"
        "- `summary`: 2-4 paragraphs, paraphrased in your own words. "
        "Professional tone. Max 2000 chars.\n"
        "- `key_themes`: 2-5 main topics covered (single words or short phrases)\n\n"
        "<document>\n{text}\n</document>"
    ),
)

PROMPT_MULTILINGUAL = PromptTemplate(
    name="multilingual_v1",
    mode=SummaryMode.MULTILINGUAL,
    version="1.0.0",
    system=(
        "You are a multilingual document analyst with expertise in Russian, "
        "Belarusian, Kazakh, and English document processing.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Analyze and summarize this document.\n\n"
        "Output schema: {schema}\n\n"
        "Requirements:\n"
        "- `detected_language`: Auto-detect the source language (BCP-47 tag)\n"
        "- `summary_language`: Use '{language}' as requested output language\n"
        "- `summary`: Full narrative summary in `summary_language`. Max 2000 chars.\n"
        "- `translation_notes`: Note significant terms that lose meaning in translation; null otherwise\n\n"
        "<document>\n{text}\n</document>"
    ),
)

# --- Map/Reduce variants (used for large document chunked processing) ---

PROMPT_MAP_EXTRACTIVE = PromptTemplate(
    name="map_extractive_v1",
    mode=SummaryMode.EXTRACTIVE,
    version="1.0.0",
    system=(
        "You are a fact extractor processing a CHUNK of a larger document. "
        "Extract only the most important facts from this chunk.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Extract key facts from this document chunk.\n\n"
        "Output schema: {schema}\n\n"
        "CHUNK CONTEXT: This is one part of a larger document. "
        "Extract self-contained facts only. Max 10 facts.\n\n"
        "<chunk>\n{text}\n</chunk>"
    ),
)

PROMPT_MAP_ABSTRACTIVE = PromptTemplate(
    name="map_abstractive_v1",
    mode=SummaryMode.ABSTRACTIVE,
    version="1.0.0",
    system=(
        "You are summarizing a CHUNK of a larger document. "
        "Be concise — this partial summary will be combined with others.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Summarize this document chunk in 1-2 paragraphs.\n\n"
        "Output schema: {schema}\n\n"
        "<chunk>\n{text}\n</chunk>"
    ),
)

PROMPT_MAP_GENERIC = PromptTemplate(
    name="map_generic_v1",
    mode=SummaryMode.ABSTRACTIVE,
    version="1.0.0",
    system=(
        "You are processing a CHUNK of a larger document. Extract the essential "
        "information as a concise partial summary.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Summarize the key content of this chunk in 2-4 sentences max. "
        "Output only plain text, no JSON.\n\n"
        "<chunk>\n{text}\n</chunk>"
    ),
)

PROMPT_REDUCE_EXECUTIVE = PromptTemplate(
    name="reduce_executive_v1",
    mode=SummaryMode.EXECUTIVE,
    version="1.0.0",
    system=(
        "You are combining partial summaries into a final executive summary. "
        "Eliminate duplicates and synthesize into the most important insights.\n\n"
        + _COMMON_RULES
    ),
    user_template=(
        "Combine these partial summaries into a final executive summary.\n\n"
        "Output schema: {schema}\n\n"
        "Partial summaries from document chunks:\n"
        "<partial_summaries>\n{text}\n</partial_summaries>"
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """Central registry for all prompt templates.

    Usage:
        registry = PromptRegistry()
        template = registry.get(SummaryMode.EXECUTIVE)
        system, user = template.render(text=doc_text)
    """

    _DIRECT: dict[SummaryMode, PromptTemplate] = {
        SummaryMode.EXECUTIVE: PROMPT_EXECUTIVE,
        SummaryMode.DETAILED_NOTES: PROMPT_DETAILED_NOTES,
        SummaryMode.ACTION_ITEMS: PROMPT_ACTION_ITEMS,
        SummaryMode.THESIS: PROMPT_THESIS,
        SummaryMode.EXTRACTIVE: PROMPT_EXTRACTIVE,
        SummaryMode.ABSTRACTIVE: PROMPT_ABSTRACTIVE,
        SummaryMode.MULTILINGUAL: PROMPT_MULTILINGUAL,
    }

    _MAP: dict[SummaryMode, PromptTemplate] = {
        SummaryMode.EXTRACTIVE: PROMPT_MAP_EXTRACTIVE,
        SummaryMode.ABSTRACTIVE: PROMPT_MAP_ABSTRACTIVE,
        SummaryMode.EXECUTIVE: PROMPT_MAP_ABSTRACTIVE,
        SummaryMode.DETAILED_NOTES: PROMPT_MAP_ABSTRACTIVE,
        SummaryMode.ACTION_ITEMS: PROMPT_MAP_ABSTRACTIVE,
        SummaryMode.THESIS: PROMPT_MAP_ABSTRACTIVE,
        SummaryMode.MULTILINGUAL: PROMPT_MAP_ABSTRACTIVE,
    }

    _REDUCE: dict[SummaryMode, PromptTemplate] = {
        SummaryMode.EXECUTIVE: PROMPT_REDUCE_EXECUTIVE,
        # All others fall back to their direct prompt with combined summaries as input
    }

    def get(self, mode: SummaryMode) -> PromptTemplate:
        """Get direct summarization prompt for given mode."""
        return self._DIRECT[mode]

    def get_map(self, mode: SummaryMode) -> PromptTemplate:
        """Get Map-stage prompt (for chunked processing)."""
        return self._MAP.get(mode, PROMPT_MAP_GENERIC)

    def get_reduce(self, mode: SummaryMode) -> PromptTemplate:
        """Get Reduce-stage prompt (for combining chunk summaries)."""
        return self._REDUCE.get(mode, self._DIRECT[mode])

    def version(self) -> str:
        return PROMPT_REGISTRY_VERSION

    def cache_version_tag(self) -> str:
        """Version tag embedded in cache keys — bump PROMPT_REGISTRY_VERSION to invalidate."""
        return f"prompts:{PROMPT_REGISTRY_VERSION}"


# Singleton
_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry