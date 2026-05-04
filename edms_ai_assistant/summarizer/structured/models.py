"""
Typed Structured Output models for all summarization modes.

All LLM responses go through these Pydantic v2 models — no raw string parsing.
The JSON Schema is passed directly to the LLM's `response_format` parameter
(OpenAI-compatible Structured Outputs), ensuring type safety at the API boundary.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SummaryMode(StrEnum):
    """Available summarization modes."""
    EXECUTIVE = "executive"          # Short C-suite friendly, 3-5 bullets
    DETAILED_NOTES = "detailed_notes"  # Full structured notes with sections
    ACTION_ITEMS = "action_items"    # Tasks, owners, deadlines — Structured Output
    THESIS = "thesis"                # Academic/analytical thesis plan
    EXTRACTIVE = "extractive"        # Key facts, dates, figures
    ABSTRACTIVE = "abstractive"      # Paraphrased narrative summary
    MULTILINGUAL = "multilingual"    # Summary in detected or forced language


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceLevel(StrEnum):
    HIGH = "high"       # > 0.85
    MEDIUM = "medium"   # 0.60 - 0.85
    LOW = "low"         # < 0.60


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class LLMBaseModel(BaseModel):
    """Base for all LLM structured outputs — strict mode, no extras."""

    model_config = {
        "strict": True,
        "extra": "ignore",
        "frozen": True,
    }


# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------


class ExecutiveSummaryOutput(LLMBaseModel):
    """C-suite friendly 3-5 bullet summary.

    Used with: SummaryMode.EXECUTIVE
    """
    headline: Annotated[str, Field(
        description="One sentence capturing the single most important point",
        max_length=200,
    )]
    bullets: Annotated[list[str], Field(
        description="3-5 key takeaways, each max 20 words",
        min_length=0,
        max_length=5,
    )]
    recommendation: Annotated[str | None, Field(
        description="Optional: single recommended action if document requires decision",
        default=None,
        max_length=300,
    )]

    @field_validator("bullets")
    @classmethod
    def validate_bullet_length(cls, bullets: list[str]) -> list[str]:
        return [b.strip() for b in bullets if b.strip()]


# ---------------------------------------------------------------------------
# Detailed Notes
# ---------------------------------------------------------------------------


class NoteSection(LLMBaseModel):
    title: Annotated[str, Field(max_length=100)]
    content: Annotated[str, Field(max_length=2000)]
    subsections: Annotated[list[str], Field(
        default=[],
        description="Optional bullet sub-points",
        max_length=10,
    )]


class DetailedNotesOutput(LLMBaseModel):
    """Full structured notes preserving document hierarchy.

    Used with: SummaryMode.DETAILED_NOTES
    """
    document_type: Annotated[str, Field(
        description="Detected document type (e.g. CONTRACT, MEMO, REGULATION)",
        default="UNKNOWN",
        max_length=50,
    )]
    sections: Annotated[list[NoteSection], Field(
        min_length=1,
        max_length=15,
    )]
    key_entities: Annotated[list[str], Field(
        description="Named entities: organizations, people, document numbers",
        default=[],
        max_length=20,
    )]
    date_range: Annotated[str | None, Field(
        description="Relevant date range mentioned in document (ISO format range)",
        default=None,
    )]


# ---------------------------------------------------------------------------
# Action Items (Fully Structured Output — zero hallucination tolerance)
# ---------------------------------------------------------------------------


class ActionItem(LLMBaseModel):
    """A single extracted action item with structured metadata."""

    task: Annotated[str, Field(
        description="Clear description of what needs to be done",
        max_length=300,
    )]
    owner: Annotated[str | None, Field(
        description="Person or role responsible. null if not explicitly stated.",
        default=None,
        max_length=100,
    )]
    deadline: Annotated[date | None, Field(
        description="Deadline in ISO 8601 date format. null if not explicitly stated.",
        default=None,
    )]
    priority: Annotated[Priority, Field(
        default=Priority.MEDIUM,
        description="Priority inferred from document language and context",
    )]
    source_fragment: Annotated[str, Field(
        description="Exact quoted sentence from source that contains this action item",
        default="",
        max_length=500,
    )]
    confidence: Annotated[float, Field(
        description="Extraction confidence 0.0-1.0",
        default=0.8,
        ge=0.0,
        le=1.0,
    )]


class ActionItemsOutput(LLMBaseModel):
    """Structured extraction of all action items from document.

    Used with: SummaryMode.ACTION_ITEMS
    This is the highest-confidence structured output — uses JSON Schema enforcement.
    """
    action_items: Annotated[list[ActionItem], Field(
        description="All action items found. Empty list if none found.",
        default=[],
        max_length=50,
    )]
    total_found: Annotated[int, Field(
        description="Total count of action items",
        default=0,
        ge=0,
    )]
    document_context: Annotated[str, Field(
        description="Brief document context for action items interpretation",
        default="",
        max_length=200,
    )]

    @model_validator(mode="after")
    def sync_total(self) -> ActionItemsOutput:
        # Allow model to be frozen after validation
        object.__setattr__(self, "total_found", len(self.action_items))
        return self


# ---------------------------------------------------------------------------
# Thesis Plan
# ---------------------------------------------------------------------------


class ThesisPoint(LLMBaseModel):
    claim: Annotated[str, Field(max_length=200)]
    evidence: Annotated[str | None, Field(default=None, max_length=300)]
    sub_points: Annotated[list[str], Field(default=[], max_length=5)]


class ThesisSection(LLMBaseModel):
    title: Annotated[str, Field(
        default="",
        max_length=100
    )]
    thesis: Annotated[str, Field(max_length=300)]
    points: Annotated[list[ThesisPoint], Field(default=[], max_length=5)]


class ThesisPlanOutput(LLMBaseModel):
    """Hierarchical thesis plan for analytical documents.

    Used with: SummaryMode.THESIS
    """
    main_argument: Annotated[str, Field(
        description="Central thesis of the document in one sentence",
        max_length=250,
    )]
    sections: Annotated[list[ThesisSection], Field(
        min_length=0,
        max_length=6,
    )]
    conclusion: Annotated[str, Field(
        default="",
        max_length=300,
    )]


# ---------------------------------------------------------------------------
# Extractive
# ---------------------------------------------------------------------------


class ExtractedFact(LLMBaseModel):
    category: Annotated[str, Field(
        description="Category: DATE, PERSON, ORG, AMOUNT, REQUIREMENT, DEADLINE, OTHER",
        default="OTHER",
        max_length=20,
    )]
    label: Annotated[str, Field(max_length=80)]
    value: Annotated[str, Field(max_length=300)]


class ExtractiveOutput(LLMBaseModel):
    """Key facts extracted from document.

    Used with: SummaryMode.EXTRACTIVE
    """
    facts: Annotated[list[ExtractedFact], Field(
        min_length=0,
        max_length=20,
    )]
    document_summary: Annotated[str, Field(
        description="One-sentence document summary",
        default="",
        max_length=200,
    )]


# ---------------------------------------------------------------------------
# Abstractive
# ---------------------------------------------------------------------------


class AbstractiveOutput(LLMBaseModel):
    """Paraphrased narrative summary.

    Used with: SummaryMode.ABSTRACTIVE
    """
    summary: Annotated[str, Field(
        description="Cohesive paraphrased summary in 2-4 paragraphs",
        max_length=2000,
    )]
    key_themes: Annotated[list[str], Field(
        description="2-5 main themes covered",
        default=[],
        min_length=1,
        max_length=5,
    )]


# ---------------------------------------------------------------------------
# Multilingual
# ---------------------------------------------------------------------------


class MultilingualOutput(LLMBaseModel):
    """Summary preserving or translating document language.

    Used with: SummaryMode.MULTILINGUAL
    """
    detected_language: Annotated[str, Field(
        description="BCP-47 language tag of source document (e.g. 'ru', 'en', 'be')",
        default="ru",
        max_length=10,
    )]
    summary_language: Annotated[str, Field(
        description="BCP-47 language tag of output summary",
        default="ru",
        max_length=10,
    )]
    summary: Annotated[str, Field(max_length=2000)]
    translation_notes: Annotated[str | None, Field(
        description="Notes on significant translation choices if applicable",
        default=None,
        max_length=300,
    )]


# ---------------------------------------------------------------------------
# Quality Score (internal, not from LLM)
# ---------------------------------------------------------------------------


class QualityScore(BaseModel):
    """Quality assessment computed post-generation."""

    model_config = {"frozen": True}

    score: Annotated[float, Field(ge=0.0, le=1.0)]
    confidence: ConfidenceLevel
    critique: str | None = None
    scored_at_ms: int = 0  # Unix timestamp ms

    @classmethod
    def from_score(cls, score: float, critique: str | None = None) -> "QualityScore":
        import time
        if score > 0.85:
            confidence = ConfidenceLevel.HIGH
        elif score > 0.60:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
        return cls(
            score=round(score, 3),
            confidence=confidence,
            critique=critique,
            scored_at_ms=int(time.time() * 1000),
        )


# ---------------------------------------------------------------------------
# Union type for mode dispatch
# ---------------------------------------------------------------------------

type SummarizationOutput = (
    ExecutiveSummaryOutput
    | DetailedNotesOutput
    | ActionItemsOutput
    | ThesisPlanOutput
    | ExtractiveOutput
    | AbstractiveOutput
    | MultilingualOutput
)

MODE_OUTPUT_MODEL: dict[SummaryMode, type[LLMBaseModel]] = {
    SummaryMode.EXECUTIVE: ExecutiveSummaryOutput,
    SummaryMode.DETAILED_NOTES: DetailedNotesOutput,
    SummaryMode.ACTION_ITEMS: ActionItemsOutput,
    SummaryMode.THESIS: ThesisPlanOutput,
    SummaryMode.EXTRACTIVE: ExtractiveOutput,
    SummaryMode.ABSTRACTIVE: AbstractiveOutput,
    SummaryMode.MULTILINGUAL: MultilingualOutput,
}