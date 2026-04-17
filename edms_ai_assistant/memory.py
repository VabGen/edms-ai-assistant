# edms_ai_assistant/memory.py
"""
Agent Memory — L0/L1/L2 architecture.

L0 — System Prompt    : Static instructions, never changes within a session.
L1 — Working Memory   : Sliding window of last N messages (context buffer).
L2 — Summarization    : Background compression of long conversations into
                         a rolling summary injected back into the system prompt.
L3 — Episodic/RAG     : TODO — vector DB (Qdrant/Pinecone) for cross-session
                         retrieval. Stub is left in place; wire up embed_fn
                         and a real VectorStore to activate.

Design principle: NEVER load full history into the prompt.
The LLM should receive at most MAX_WORKING_MESSAGES recent turns plus
a compressed summary of everything older.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ── L1 tunables ───────────────────────────────────────────────────────────────
# Keep only this many recent non-system messages in the context window.
# Older messages are compressed into the L2 rolling summary.
MAX_WORKING_MESSAGES: int = 20

# Compress when history exceeds this threshold.
SUMMARIZE_AFTER: int = 30


@dataclass
class MemoryState:
    """
    Holds all memory levels for one conversation thread.

    Attributes:
        rolling_summary: L2 compressed text of older turns. Injected into
                         the system prompt on every LLM call.
        working_messages: L1 sliding window of recent messages.
        total_turns: Total number of human turns processed (for metrics).
    """

    rolling_summary: str = ""
    working_messages: list[BaseMessage] = field(default_factory=list)
    total_turns: int = 0

    def add(self, message: BaseMessage) -> None:
        self.working_messages.append(message)
        if isinstance(message, HumanMessage):
            self.total_turns += 1

    def trim_to_window(self) -> None:
        """Keep only the most recent MAX_WORKING_MESSAGES non-system messages."""
        non_sys = [m for m in self.working_messages if not isinstance(m, SystemMessage)]
        if len(non_sys) > MAX_WORKING_MESSAGES:
            self.working_messages = non_sys[-MAX_WORKING_MESSAGES:]
            logger.debug(
                "L1 trimmed to %d messages (total turns: %d)",
                MAX_WORKING_MESSAGES,
                self.total_turns,
            )

    @property
    def needs_compression(self) -> bool:
        non_sys = [m for m in self.working_messages if not isinstance(m, SystemMessage)]
        return len(non_sys) > SUMMARIZE_AFTER


class ConversationMemoryManager:
    """
    Manages the L0/L1/L2 memory lifecycle for one agent session.

    Usage pattern (inside agent graph node)::

        mgr = ConversationMemoryManager(llm=llm)

        # On each turn, call prepare() to get messages to send to LLM.
        messages_for_llm = await mgr.prepare(state_messages, system_prompt)

        # After turn, call record() to update working memory.
        mgr.record(human_msg, ai_msg)

    L3 (RAG) stub:
        Override ``retrieve_long_term`` with a real vector search.
        Currently returns empty list — zero-latency passthrough.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._state: MemoryState = MemoryState()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def prepare(
        self,
        all_messages: list[BaseMessage],
        system_prompt: str,
    ) -> list[BaseMessage]:
        """
        Build the message list to send to the LLM.

        Steps:
        1. Separate system messages from conversation history.
        2. If history is too long, trigger L2 compression.
        3. Optionally prepend L3 context (stub).
        4. Return: [enriched_system_msg] + L1_window.
        """
        system_msgs = [m for m in all_messages if isinstance(m, SystemMessage)]
        non_sys = [m for m in all_messages if not isinstance(m, SystemMessage)]

        # Sync internal state
        self._state.working_messages = non_sys

        # L2: compress if needed
        if self._state.needs_compression:
            await self._compress(non_sys[:-MAX_WORKING_MESSAGES])
            self._state.working_messages = non_sys[-MAX_WORKING_MESSAGES:]

        # Build enriched system prompt (L0 + L2 injection)
        enriched_system = self._enrich_system_prompt(system_prompt)

        # L3 RAG stub — replace with real retrieval
        rag_context = await self.retrieve_long_term(non_sys)
        if rag_context:
            enriched_system += f"\n\n<long_term_context>\n{rag_context}\n</long_term_context>"

        result_system = [SystemMessage(content=enriched_system)] if system_msgs else []
        return result_system + self._state.working_messages

    def record(self, *messages: BaseMessage) -> None:
        """Add new messages to working memory after a turn."""
        for m in messages:
            self._state.add(m)
        self._state.trim_to_window()

    @property
    def summary(self) -> str:
        """Current L2 rolling summary (for debugging/observability)."""
        return self._state.rolling_summary

    # ── L3 stub ────────────────────────────────────────────────────────────────

    async def retrieve_long_term(
        self,
        recent_messages: list[BaseMessage],
    ) -> str:
        """
        L3 — Episodic/Long-term memory via RAG.

        TODO: Implement with real vector search:
            1. Embed last human message.
            2. Search Qdrant/Pinecone for similar past conversations.
            3. Return top-k snippets as context string.

        Currently returns empty string (passthrough).
        """
        # Example implementation (commented out):
        # query = next(
        #     (m.content for m in reversed(recent_messages) if isinstance(m, HumanMessage)),
        #     ""
        # )
        # if not query:
        #     return ""
        # results = await self._vector_store.similarity_search(query, k=3)
        # return "\n".join(r.page_content for r in results)
        return ""

    # ── Private ────────────────────────────────────────────────────────────────

    async def _compress(self, old_messages: list[BaseMessage]) -> None:
        """
        L2: Summarize old messages and append to rolling_summary.

        The summary is injected into the system prompt on next LLM call.
        This prevents context overflow while preserving semantic continuity.
        """
        if not old_messages:
            return

        conversation_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {str(m.content)[:300]}"
            for m in old_messages
            if not isinstance(m, SystemMessage)
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Ты — система сжатия диалога. Сделай краткое резюме переписки "
                "(не более 200 слов). Сохрани ключевые факты, решения и контекст. "
                "Отвечай только резюме, без предисловий.",
            ),
            ("user", "Диалог:\n{conversation}"),
        ])

        try:
            chain = prompt | self._llm | StrOutputParser()
            new_summary = await chain.ainvoke({"conversation": conversation_text})
            existing = self._state.rolling_summary
            self._state.rolling_summary = (
                f"{existing}\n\n[Новый блок]\n{new_summary.strip()}"
                if existing
                else new_summary.strip()
            )
            logger.info(
                "L2 compression complete: %d messages → %d chars summary",
                len(old_messages),
                len(self._state.rolling_summary),
            )
        except Exception as exc:
            logger.error("L2 compression failed: %s", exc)

    def _enrich_system_prompt(self, base_prompt: str) -> str:
        """Inject L2 rolling summary into the system prompt if present."""
        if not self._state.rolling_summary:
            return base_prompt
        return (
            base_prompt
            + f"\n\n<conversation_history_summary>\n"
            f"{self._state.rolling_summary}\n"
            f"</conversation_history_summary>\n"
            f"(Это сжатая история предыдущих сообщений. "
            f"Учитывай этот контекст при ответе.)"
        )