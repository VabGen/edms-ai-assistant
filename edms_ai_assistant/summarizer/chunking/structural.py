"""
StructuralChunker — header-aware document splitting.

2025 Best Practices:
- Parse document structure (headers, sections) before splitting
- Preserve section context across chunk boundaries
- Never split mid-sentence within a section
- Token-accurate boundaries via tiktoken

Why better than RecursiveCharacterTextSplitter:
  Old: splits by char count → breaks sections mid-thought → LLM loses context
  New: splits at section boundaries → each chunk is semantically complete

Hierarchy:
  1. StructuralChunker: detects headers, splits at section level
  2. TokenAwareFallbackChunker: paragraph/sentence split for unstructured text
  3. Both respect max_tokens hard limit and overlap budget
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from edms_ai_assistant.summarizer.chunking.token_aware import count_tokens

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextChunk:
    """A single document chunk with metadata."""

    text: str
    token_count: int
    index: int
    section_title: str | None = None
    is_header: bool = False
    char_start: int = 0
    char_end: int = 0

    @classmethod
    def from_text(
        cls,
        text: str,
        index: int,
        section_title: str | None = None,
        char_start: int = 0,
    ) -> "TextChunk":
        stripped = text.strip()
        return cls(
            text=stripped,
            token_count=count_tokens(stripped),
            index=index,
            section_title=section_title,
            char_start=char_start,
            char_end=char_start + len(stripped),
        )


@dataclass
class Section:
    """A document section with its header and body text."""

    title: str | None
    body: str
    depth: int = 0  # Header depth: 0=body, 1=H1, 2=H2, 3=H3


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------


class ChunkingStrategy(ABC):
    """Abstract base for all chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        *,
        max_tokens: int = 1500,
        overlap_tokens: int = 100,
    ) -> list[TextChunk]:
        """Split text into chunks respecting max_tokens limit.

        Args:
            text: Input document text.
            max_tokens: Hard upper limit per chunk.
            overlap_tokens: Token overlap between consecutive chunks for context continuity.

        Returns:
            Ordered list of TextChunk objects.
        """
        ...

    @abstractmethod
    def can_handle(self, text: str) -> bool:
        """Return True if this strategy is appropriate for the given text."""
        ...


# ---------------------------------------------------------------------------
# Structural Chunker
# ---------------------------------------------------------------------------

# Patterns for detecting document headers in Russian/English EDMS documents
_HEADER_PATTERNS: list[tuple[int, re.Pattern[str]]] = [
    # Numbered sections: "1. Название", "1.1 Название", "Статья 1."
    (1, re.compile(r"^(?:Статья|Article|Раздел|Section|Глава|Chapter)\s+\d+", re.MULTILINE)),
    (1, re.compile(r"^\d+\.\s+[А-ЯЁA-Z][^\n]{3,60}$", re.MULTILINE)),
    (2, re.compile(r"^\d+\.\d+\.\s+[А-ЯЁA-Z][^\n]{3,60}$", re.MULTILINE)),
    (3, re.compile(r"^\d+\.\d+\.\d+\.\s+[^\n]{3,60}$", re.MULTILINE)),
    # All-caps headers (common in CIS official documents)
    (1, re.compile(r"^[А-ЯЁ\s]{10,80}$", re.MULTILINE)),
    # Markdown-style headers
    (1, re.compile(r"^#{1}\s+.+$", re.MULTILINE)),
    (2, re.compile(r"^#{2}\s+.+$", re.MULTILINE)),
    (3, re.compile(r"^#{3}\s+.+$", re.MULTILINE)),
]

# Minimum chars for structural detection to be worth using
_STRUCTURAL_MIN_CHARS = 100
_STRUCTURAL_MIN_HEADERS = 2


class StructuralChunker(ChunkingStrategy):
    """Header-aware chunker that splits at section boundaries.

    Algorithm:
    1. Detect all header lines using regex patterns
    2. Split document into sections at header boundaries
    3. Merge small sections that fit together within max_tokens
    4. Split oversized sections using sentence-boundary fallback

    Context preservation:
    - Each chunk includes its section title in metadata
    - Cross-boundary overlap added at sentence level (not char level)
    """

    def can_handle(self, text: str) -> bool:
        """Return True if document has enough structure for header-based splitting."""
        if len(text) < _STRUCTURAL_MIN_CHARS:
            return False
        header_count = sum(
            len(pattern.findall(text)) for _, pattern in _HEADER_PATTERNS
        )
        return header_count >= _STRUCTURAL_MIN_HEADERS

    def chunk(
        self,
        text: str,
        *,
        max_tokens: int = 1500,
        overlap_tokens: int = 100,
    ) -> list[TextChunk]:
        sections = self._parse_sections(text)
        return self._sections_to_chunks(sections, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    def _parse_sections(self, text: str) -> list[Section]:
        """Parse document into sections using detected header positions."""
        # Build a unified header position map
        header_positions: list[tuple[int, int, int, str]] = []  # (start, end, depth, title)

        for depth, pattern in _HEADER_PATTERNS:
            for match in pattern.finditer(text):
                title = match.group(0).strip()
                if len(title) > 3:
                    header_positions.append((match.start(), match.end(), depth, title))

        if not header_positions:
            return [Section(title=None, body=text, depth=0)]

        # Sort by position, deduplicate overlapping matches
        header_positions.sort(key=lambda x: x[0])
        deduped: list[tuple[int, int, int, str]] = []
        last_end = -1
        for start, end, depth, title in header_positions:
            if start >= last_end:
                deduped.append((start, end, depth, title))
                last_end = end

        sections: list[Section] = []

        # Text before first header
        if deduped and deduped[0][0] > 0:
            preamble = text[: deduped[0][0]].strip()
            if preamble:
                sections.append(Section(title=None, body=preamble, depth=0))

        for i, (start, end, depth, title) in enumerate(deduped):
            next_start = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
            body = text[end:next_start].strip()
            sections.append(Section(title=title, body=body, depth=depth))

        return sections if sections else [Section(title=None, body=text, depth=0)]

    def _sections_to_chunks(
        self,
        sections: list[Section],
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[TextChunk]:
        """Convert sections to chunks, merging small ones and splitting large ones."""
        chunks: list[TextChunk] = []
        chunk_index = 0
        pending_text = ""
        pending_title: str | None = None
        pending_tokens = 0
        char_cursor = 0

        def flush(title: str | None, text: str, tokens: int, start: int) -> None:
            nonlocal chunk_index
            if text.strip():
                chunks.append(TextChunk.from_text(
                    text=text,
                    index=chunk_index,
                    section_title=title,
                    char_start=start,
                ))
                chunk_index += 1

        for section in sections:
            section_text = f"{section.title}\n{section.body}" if section.title else section.body
            section_tokens = count_tokens(section_text)

            if section_tokens > max_tokens:
                # Flush pending first
                if pending_text:
                    flush(pending_title, pending_text, pending_tokens, char_cursor)
                    pending_text = ""
                    pending_tokens = 0

                # Split oversized section by sentences
                sub_chunks = self._split_by_sentences(
                    section_text,
                    section.title,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    start_index=chunk_index,
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

            elif pending_tokens + section_tokens <= max_tokens:
                # Merge with pending
                if pending_text:
                    pending_text += "\n\n" + section_text
                else:
                    pending_text = section_text
                    pending_title = section.title
                pending_tokens += section_tokens

            else:
                # Flush pending, start new
                flush(pending_title, pending_text, pending_tokens, char_cursor)
                char_cursor += len(pending_text)
                pending_text = section_text
                pending_title = section.title
                pending_tokens = section_tokens

        # Flush remainder
        if pending_text:
            flush(pending_title, pending_text, pending_tokens, char_cursor)

        return chunks if chunks else [TextChunk.from_text(text="", index=0)]

    @staticmethod
    def _split_by_sentences(
        text: str,
        section_title: str | None,
        max_tokens: int,
        overlap_tokens: int,
        start_index: int,
    ) -> list[TextChunk]:
        """Split oversized text at sentence boundaries."""
        sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])")
        sentences = sentence_pattern.split(text)

        chunks: list[TextChunk] = []
        current: list[str] = []
        current_tokens = 0
        overlap_buffer: list[str] = []
        idx = start_index

        for sentence in sentences:
            s_tokens = count_tokens(sentence)

            if current_tokens + s_tokens > max_tokens and current:
                chunk_text = " ".join(current)
                chunks.append(TextChunk.from_text(
                    text=chunk_text,
                    index=idx,
                    section_title=section_title,
                ))
                idx += 1

                # Build overlap from end of current chunk
                overlap_buffer = []
                overlap_t = 0
                for s in reversed(current):
                    t = count_tokens(s)
                    if overlap_t + t > overlap_tokens:
                        break
                    overlap_buffer.insert(0, s)
                    overlap_t += t

                current = overlap_buffer.copy()
                current_tokens = overlap_t

            current.append(sentence)
            current_tokens += s_tokens

        if current:
            chunks.append(TextChunk.from_text(
                text=" ".join(current),
                index=idx,
                section_title=section_title,
            ))

        return chunks


# ---------------------------------------------------------------------------
# Token-Aware Fallback Chunker (for unstructured text)
# ---------------------------------------------------------------------------


class TokenAwareFallbackChunker(ChunkingStrategy):
    """Paragraph → sentence fallback chunker for unstructured documents.

    Used when StructuralChunker.can_handle() returns False.
    Splits at paragraph boundaries first, sentences second.
    Respects max_tokens using tiktoken counting.
    """

    def can_handle(self, text: str) -> bool:
        return True  # Always applicable as fallback

    def chunk(
        self,
        text: str,
        *,
        max_tokens: int = 1500,
        overlap_tokens: int = 100,
    ) -> list[TextChunk]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not paragraphs:
            return [TextChunk.from_text(text=text, index=0)]

        chunks: list[TextChunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        idx = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)

            if para_tokens > max_tokens:
                # Split paragraph by sentences
                if current_parts:
                    chunks.append(TextChunk.from_text(
                        text="\n\n".join(current_parts), index=idx
                    ))
                    idx += 1
                    current_parts = []
                    current_tokens = 0
                # Sentence-level split
                sentences = re.split(r"(?<=[.!?])\s+", para)
                acc: list[str] = []
                acc_tokens = 0
                for sent in sentences:
                    st = count_tokens(sent)
                    if acc_tokens + st > max_tokens and acc:
                        chunks.append(TextChunk.from_text(text=" ".join(acc), index=idx))
                        idx += 1
                        # overlap
                        acc = acc[-2:] if len(acc) >= 2 else acc
                        acc_tokens = count_tokens(" ".join(acc))
                    acc.append(sent)
                    acc_tokens += st
                if acc:
                    chunks.append(TextChunk.from_text(text=" ".join(acc), index=idx))
                    idx += 1

            elif current_tokens + para_tokens > max_tokens:
                chunks.append(TextChunk.from_text(
                    text="\n\n".join(current_parts), index=idx
                ))
                idx += 1
                # Overlap: keep last paragraph
                if current_parts and count_tokens(current_parts[-1]) <= overlap_tokens:
                    current_parts = [current_parts[-1], para]
                    current_tokens = count_tokens(current_parts[-1]) + para_tokens
                else:
                    current_parts = [para]
                    current_tokens = para_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        if current_parts:
            chunks.append(TextChunk.from_text(
                text="\n\n".join(current_parts), index=idx
            ))

        return chunks if chunks else [TextChunk.from_text(text=text, index=0)]


# ---------------------------------------------------------------------------
# Smart Chunker (auto-selects strategy)
# ---------------------------------------------------------------------------


class SmartChunker:
    """Automatically selects the best chunking strategy for a given document."""

    def __init__(self) -> None:
        self._structural = StructuralChunker()
        self._fallback = TokenAwareFallbackChunker()

    def chunk(
        self,
        text: str,
        *,
        max_tokens: int = 1500,
        overlap_tokens: int = 100,
    ) -> tuple[list[TextChunk], str]:
        """Chunk text, returning chunks and the strategy name used.

        Returns:
            (chunks, strategy_name) — strategy_name for observability.
        """
        if self._structural.can_handle(text):
            return (
                self._structural.chunk(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens),
                "structural",
            )
        return (
            self._fallback.chunk(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens),
            "token_aware_fallback",
        )

    def needs_map_reduce(self, text: str, *, context_window: int = 4096) -> bool:
        """Return True if document exceeds direct summarization context window."""
        # Safety margin: leave 30% for system prompt + output
        effective_window = int(context_window * 0.70)
        return count_tokens(text) > effective_window