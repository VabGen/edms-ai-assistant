"""
Reexports for chunking abstractions.

Концепции (TextChunk, Section, ChunkingStrategy) определены в `structural.py`,
этот модуль предоставляет стабильный публичный путь импорта.
"""

from __future__ import annotations

from edms_ai_assistant.summarizer.chunking.structural import (
    ChunkingStrategy,
    Section,
    SmartChunker,
    StructuralChunker,
    TextChunk,
    TokenAwareFallbackChunker,
)

__all__ = [
    "ChunkingStrategy",
    "Section",
    "SmartChunker",
    "StructuralChunker",
    "TextChunk",
    "TokenAwareFallbackChunker",
]
