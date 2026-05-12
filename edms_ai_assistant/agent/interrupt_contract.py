# edms_ai_assistant/agent/interrupt_contract.py
"""
Universal Human-in-the-Loop interruption contract for LangGraph.

Design:
- Outbound (server → frontend): ``InterruptPayload`` — discriminated union by
  ``kind`` describing what the tool needs from the user.
- Inbound (frontend → server): ``ResumeValue`` — discriminated union by ``kind``
  carrying the user's answer.  The matching ``kind`` on both sides guarantees
  static, parse-free routing.

LLM never sees these objects.  Tools call ``ask_human(payload)`` and receive a
typed ``ResumeValue``.  The graph engine (LangGraph ``interrupt()`` / ``Command``)
takes care of suspension, persistence and resume.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

INTERRUPT_SCHEMA_VERSION: int = 1


# ── Shared building blocks ────────────────────────────────────────────────


class InterruptOption(BaseModel):
    """Single selectable option presented to the user."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier (UUID, slug, ...).")
    label: str = Field(description="Primary human-readable text.")
    description: str | None = Field(
        default=None,
        description="Secondary context shown under the label (dept, type, ...).",
    )
    metadata: dict | None = Field(
        default=None,
        description="Tool-specific extras (position, mime, file_size, ...).",
    )


class InterruptCard(BaseModel):
    """Rich card for ``CardSelectInterrupt`` — extends InterruptOption.

    Adds visual attributes (image, badges, tabular data) that the frontend
    renders as a structured card component instead of a plain list item.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier (UUID, slug, ...).")
    label: str = Field(description="Primary human-readable text (e.g. full name).")
    description: str | None = Field(
        default=None,
        description="Subtitle (e.g. department).",
    )
    image_url: str | None = Field(
        default=None,
        description="Avatar / preview image URL.",
    )
    badges: list[str] = Field(
        default_factory=list,
        description="Short labels (e.g. 'Руководитель', 'В отпуске').",
    )
    primary_attrs: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs rendered as a compact table (e.g. Должность, Телефон).",
    )
    metadata: dict | None = Field(
        default=None,
        description="Tool-specific extras not shown directly.",
    )


class FileRef(BaseModel):
    """Reference to an uploaded file returned by ``FilePickerResume``."""

    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(description="Server-side file identifier / path.")
    file_name: str = Field(description="Original filename.")
    mime_type: str | None = None
    size_bytes: int | None = None


# ── Outbound payloads (tool → frontend) ───────────────────────────────────


class _BasePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = INTERRUPT_SCHEMA_VERSION


class DisambiguationInterrupt(_BasePayload):
    """User must pick one (or several) entities from a list."""

    kind: Literal["disambiguation"] = "disambiguation"
    entity_type: str = Field(
        description="employee | document | department | attachment | ..."
    )
    prompt: str = Field(description="Human-readable question.")
    options: list[InterruptOption]
    multiple: bool = False
    search_term: str | None = Field(
        default=None,
        description="Original query used to fetch candidates (for UI grouping).",
    )


class ConfirmationInterrupt(_BasePayload):
    """Yes / No confirmation. ``danger=True`` signals destructive ops."""

    kind: Literal["confirmation"] = "confirmation"
    prompt: str
    danger: bool = False
    confirm_label: str = "Подтвердить"
    cancel_label: str = "Отмена"


class TextInputInterrupt(_BasePayload):
    """Free-form text. ``secret=True`` for passwords / tokens."""

    kind: Literal["text_input"] = "text_input"
    prompt: str
    placeholder: str | None = None
    secret: bool = False
    validator_regex: str | None = None


class SelectInterrupt(_BasePayload):
    """Single-choice select (radio / dropdown)."""

    kind: Literal["select"] = "select"
    prompt: str
    options: list[InterruptOption]
    default: str | None = None


class CardSelectInterrupt(_BasePayload):
    """Rich clickable cards — rendered as ``<CardSelector/>`` on the frontend.

    Unlike ``DisambiguationInterrupt`` (plain list), this payload carries
    ``InterruptCard`` objects with image, badges, and tabular attributes.
    The frontend dispatches to a different renderer based on ``kind``.
    """

    kind: Literal["card_select"] = "card_select"
    prompt: str
    cards: list[InterruptCard]
    multiple: bool = False
    layout: Literal["grid", "list"] = "list"


class FilePickerInterrupt(_BasePayload):
    """File upload request — rendered as ``<FileDropzone/>`` on the frontend."""

    kind: Literal["file_picker"] = "file_picker"
    prompt: str
    accept_mime: list[str] | None = Field(
        default=None,
        description="Allowed MIME types, e.g. ['application/pdf', 'image/*'].",
    )
    max_size_bytes: int | None = None
    multiple: bool = False


InterruptPayload = Annotated[
    Union[
        DisambiguationInterrupt,
        ConfirmationInterrupt,
        TextInputInterrupt,
        SelectInterrupt,
        CardSelectInterrupt,
        FilePickerInterrupt,
    ],
    Field(discriminator="kind"),
]
InterruptPayloadAdapter: TypeAdapter[InterruptPayload] = TypeAdapter(InterruptPayload)


# ── Inbound resume values (frontend → tool) ───────────────────────────────


class _BaseResume(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DisambiguationResume(_BaseResume):
    kind: Literal["disambiguation"] = "disambiguation"
    selected_ids: list[str] = Field(min_length=1)


class ConfirmationResume(_BaseResume):
    kind: Literal["confirmation"] = "confirmation"
    confirmed: bool


class TextInputResume(_BaseResume):
    kind: Literal["text_input"] = "text_input"
    value: str


class SelectResume(_BaseResume):
    kind: Literal["select"] = "select"
    selected_id: str


class CardSelectResume(_BaseResume):
    kind: Literal["card_select"] = "card_select"
    selected_ids: list[str] = Field(min_length=1)


class FilePickerResume(_BaseResume):
    kind: Literal["file_picker"] = "file_picker"
    file_refs: list[FileRef] = Field(min_length=1)


class AbortResume(_BaseResume):
    """Universal cancellation signal — any tool may handle this."""

    kind: Literal["__abort__"] = "__abort__"
    reason: str | None = None


ResumeValue = Annotated[
    Union[
        DisambiguationResume,
        ConfirmationResume,
        TextInputResume,
        SelectResume,
        CardSelectResume,
        FilePickerResume,
        AbortResume,
    ],
    Field(discriminator="kind"),
]
ResumeValueAdapter: TypeAdapter[ResumeValue] = TypeAdapter(ResumeValue)


__all__ = [
    "INTERRUPT_SCHEMA_VERSION",
    "AbortResume",
    "CardSelectInterrupt",
    "CardSelectResume",
    "ConfirmationInterrupt",
    "ConfirmationResume",
    "DisambiguationInterrupt",
    "DisambiguationResume",
    "FilePickerInterrupt",
    "FilePickerResume",
    "FileRef",
    "InterruptCard",
    "InterruptOption",
    "InterruptPayload",
    "InterruptPayloadAdapter",
    "ResumeValue",
    "ResumeValueAdapter",
    "SelectInterrupt",
    "SelectResume",
    "TextInputInterrupt",
    "TextInputResume",
]