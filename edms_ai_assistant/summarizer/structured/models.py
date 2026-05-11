"""
Typed Structured Output models for all summarization modes.

All LLM responses go through these Pydantic v2 models.
JSON Schema is passed to LLM response_format for type safety at API boundary.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Auto-truncation infrastructure
# ---------------------------------------------------------------------------

_TRUNCATE_CACHE: dict[type, dict[str, int]] = {}
"""Cache: model class → {field_name: max_length} to avoid re-computing schemas."""


def _extract_max_length(schema: dict) -> int | None:
    """Recursively extract maxLength from a JSON schema fragment.

    Handles ``anyOf`` / ``oneOf`` / ``allOf`` wrappers that appear for
    ``str | None`` and other union types.
    """
    if "maxLength" in schema:
        return schema["maxLength"]
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            ml = _extract_max_length(sub)
            if ml is not None:
                return ml
    return None


def _max_lengths_for(cls: type) -> dict[str, int]:
    """Return ``{field_name: maxLength}`` for *cls*, cached after first call."""
    if cls not in _TRUNCATE_CACHE:
        try:
            props = cls.model_json_schema().get("properties", {})
            result: dict[str, int] = {}
            for name, schema in props.items():
                ml = _extract_max_length(schema)
                if ml is not None:
                    result[name] = ml
            _TRUNCATE_CACHE[cls] = result
        except Exception:
            _TRUNCATE_CACHE[cls] = {}
    return _TRUNCATE_CACHE[cls]


class SummaryMode(StrEnum):
    EXECUTIVE = "executive"
    DETAILED_NOTES = "detailed_notes"
    ACTION_ITEMS = "action_items"
    THESIS = "thesis"
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    MULTILINGUAL = "multilingual"


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LLMBaseModel(BaseModel):
    model_config = {
        "strict": False,
        "extra": "ignore",
        "frozen": True,
    }

    @model_validator(mode="before")
    @classmethod
    def _truncate_long_strings(cls, data: Any) -> Any:
        """Auto-truncate strings that exceed their ``Field(max_length=…)``.

        LLMs routinely ignore ``maxLength`` in ``response_format`` schemas.
        Instead of crashing with a ``ValidationError``, we silently truncate
        and add an ellipsis — preserving the rest of the response.

        This validator is inherited by **every** sub-model, including nested
        ones (``ThesisSection`` → ``ThesisPoint``), because Pydantic calls
        ``model_validator`` on each model class independently during
        recursive validation.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)  # shallow copy — don't mutate caller's dict
        for field_name, max_len in _max_lengths_for(cls).items():
            value = data.get(field_name)
            if isinstance(value, str) and len(value) > max_len:
                data[field_name] = value[: max_len - 1].rstrip() + "…"
        return data


# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------


class ExecutiveSummaryOutput(LLMBaseModel):
    headline: Annotated[
        str,
        Field(
            description="Одно предложение — главная мысль документа",
            max_length=200,
        ),
    ]
    bullets: Annotated[
        list[str],
        Field(
            description="3-5 ключевых тезисов, каждый не более 20 слов",
            default_factory=list,
            min_length=0,
            max_length=5,
        ),
    ]
    recommendation: Annotated[
        str | None,
        Field(
            description="Рекомендуемое действие, если документ требует решения",
            default=None,
            max_length=300,
        ),
    ]

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
    subsections: Annotated[list[str], Field(default_factory=list, max_length=10)]


class DetailedNotesOutput(LLMBaseModel):
    document_type: Annotated[
        str,
        Field(
            description="Тип документа: ДОГОВОР, ПИСЬМО, РЕГЛАМЕНТ, ПРОТОКОЛ и т.д.",
            default="ДОКУМЕНТ",
            max_length=50,
        ),
    ]
    sections: Annotated[list[NoteSection], Field(min_length=1, max_length=15)]
    key_entities: Annotated[
        list[str],
        Field(
            description="Организации, люди, номера документов",
            default_factory=list,
            max_length=20,
        ),
    ]
    date_range: Annotated[
        str | None,
        Field(
            description="Диапазон дат в документе (ISO формат)",
            default=None,
        ),
    ]


# ---------------------------------------------------------------------------
# Action Items
# ---------------------------------------------------------------------------


class ActionItem(LLMBaseModel):
    task: Annotated[
        str,
        Field(
            description="Описание задачи",
            max_length=300,
        ),
    ]
    owner: Annotated[
        str | None,
        Field(
            description="Ответственный. null если не указан явно.",
            default=None,
            max_length=100,
        ),
    ]
    deadline: Annotated[
        date | None,
        Field(
            description="Срок в ISO 8601. null если не указан явно.",
            default=None,
        ),
    ]
    priority: Annotated[Priority, Field(default=Priority.MEDIUM)]
    source_fragment: Annotated[
        str,
        Field(
            description="Цитата из источника",
            default="",
            max_length=500,
        ),
    ]
    confidence: Annotated[float, Field(default=0.8, ge=0.0, le=1.0)]


class ActionItemsOutput(LLMBaseModel):
    action_items: Annotated[
        list[ActionItem], Field(default_factory=list, max_length=50)
    ]
    document_context: Annotated[str, Field(default="", max_length=200)]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_found(self) -> int:
        return len(self.action_items)


# ---------------------------------------------------------------------------
# Thesis Plan
# ---------------------------------------------------------------------------


class ThesisPoint(LLMBaseModel):
    claim: Annotated[str, Field(max_length=200)]
    evidence: Annotated[str | None, Field(default=None, max_length=300)]
    sub_points: Annotated[list[str], Field(default_factory=list, max_length=5)]


class ThesisSection(LLMBaseModel):
    title: Annotated[str, Field(default="", max_length=100)]
    thesis: Annotated[str, Field(max_length=300)]
    points: Annotated[list[ThesisPoint], Field(default_factory=list, max_length=5)]


class ThesisPlanOutput(LLMBaseModel):
    main_argument: Annotated[
        str,
        Field(
            description="Центральный тезис документа",
            max_length=250,
        ),
    ]
    sections: Annotated[list[ThesisSection], Field(default_factory=list, max_length=6)]
    conclusion: Annotated[str, Field(default="", max_length=300)]


# ---------------------------------------------------------------------------
# Extractive
# ---------------------------------------------------------------------------


class ExtractedFact(LLMBaseModel):
    category: Annotated[
        str,
        Field(
            description="Категория: ДАТА, ПЕРСОНА, ОРГАНИЗАЦИЯ, СУММА, ТРЕБОВАНИЕ, СРОК, ПРОЧЕЕ",
            default="ПРОЧЕЕ",
            max_length=20,
        ),
    ]
    label: Annotated[str, Field(max_length=80)]
    value: Annotated[str, Field(max_length=300)]


class ExtractiveOutput(LLMBaseModel):
    facts: Annotated[list[ExtractedFact], Field(default_factory=list, max_length=20)]
    document_summary: Annotated[
        str,
        Field(
            description="Одно предложение — суть документа",
            default="",
            max_length=200,
        ),
    ]


# ---------------------------------------------------------------------------
# Abstractive
# ---------------------------------------------------------------------------


class AbstractiveOutput(LLMBaseModel):
    summary: Annotated[
        str,
        Field(
            description="Связный пересказ в 2-4 абзацах",
            max_length=3000,
        ),
    ]
    key_themes: Annotated[
        list[str],
        Field(
            description="2-5 главных тем",
            default_factory=list,
            min_length=0,
            max_length=5,
        ),
    ]


# ---------------------------------------------------------------------------
# Multilingual
# ---------------------------------------------------------------------------


class MultilingualOutput(LLMBaseModel):
    detected_language: Annotated[
        str,
        Field(
            description="BCP-47 язык источника (ru, en, be и т.д.)",
            default="ru",
            max_length=10,
        ),
    ]
    summary_language: Annotated[
        str,
        Field(
            description="BCP-47 язык вывода",
            default="ru",
            max_length=10,
        ),
    ]
    summary: Annotated[str, Field(max_length=2000)]
    translation_notes: Annotated[
        str | None,
        Field(
            description="Примечания по переводу",
            default=None,
            max_length=300,
        ),
    ]


# ---------------------------------------------------------------------------
# Quality Score
# ---------------------------------------------------------------------------


class QualityScore(BaseModel):
    model_config = {"frozen": True}

    score: Annotated[float, Field(ge=0.0, le=1.0)]
    confidence: ConfidenceLevel
    critique: str | None = None
    scored_at_ms: int = 0

    @classmethod
    def from_score(cls, score: float, critique: str | None = None) -> QualityScore:
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
# Union + dispatch table
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
