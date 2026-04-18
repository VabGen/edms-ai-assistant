# edms_ai_assistant/router_planner.py
"""
Router + Planner — decide HOW to handle incoming requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from edms_ai_assistant.services.nlp_service import QueryComplexity, UserIntent

logger = logging.getLogger(__name__)


# ── Routing decision ──────────────────────────────────────────────────────────


class RouteDecision(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    USE_TOOL = "use_tool"
    NEEDS_PLAN = "needs_plan"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass
class RouterResult:
    decision: RouteDecision
    intent: UserIntent
    rationale: str = ""
    suggested_tools: list[str] = field(default_factory=list)


# ── Intent classification ─────────────────────────────────────────────────────

_KNOWLEDGE_ONLY_INTENTS: frozenset[UserIntent] = frozenset({
    UserIntent.UNKNOWN,
})

_TOOL_INTENTS: frozenset[UserIntent] = frozenset({
    UserIntent.SUMMARIZE,
    UserIntent.COMPARE,
    UserIntent.ANALYZE,
    UserIntent.SEARCH,
    UserIntent.CREATE_TASK,
    UserIntent.CREATE_INTRODUCTION,
    UserIntent.CREATE_DOCUMENT,
    UserIntent.COMPLIANCE_CHECK,
    UserIntent.FILE_ANALYSIS,
    UserIntent.EXTRACT,
})

_CONTEXT_DEPENDENT: frozenset[UserIntent] = frozenset({
    UserIntent.QUESTION,
    UserIntent.UPDATE,
})

# FIX: QueryComplexity.HIGH does not exist — use VERY_COMPLEX
_COMPLEX_INTENTS: frozenset[UserIntent] = frozenset({
    UserIntent.ANALYZE,
    UserIntent.COMPLIANCE_CHECK,
    UserIntent.COMPARE,
})


class AgentRouter:
    """Routes incoming requests to the correct processing path."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def route(
        self,
        intent: UserIntent,
        complexity: QueryComplexity,
        has_document_context: bool,
        message: str,
    ) -> RouterResult:
        # Rule 1: Greetings/meta → direct answer
        if self._is_greeting_or_meta(message):
            return RouterResult(
                decision=RouteDecision.DIRECT_ANSWER,
                intent=intent,
                rationale="Greeting or meta-question",
            )

        # Rule 2: Tool intents
        if intent in _TOOL_INTENTS:
            # FIX: use VERY_COMPLEX instead of HIGH (QueryComplexity has no HIGH)
            is_complex = (
                intent in _COMPLEX_INTENTS
                and complexity in (QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX)
            )
            decision = RouteDecision.NEEDS_PLAN if is_complex else RouteDecision.USE_TOOL
            return RouterResult(
                decision=decision,
                intent=intent,
                rationale=f"Intent {intent.value} requires tools (complexity={complexity.value})",
                suggested_tools=self._suggest_tools(intent),
            )

        # Rule 3: Context-dependent
        if intent in _CONTEXT_DEPENDENT:
            if has_document_context:
                return RouterResult(
                    decision=RouteDecision.USE_TOOL,
                    intent=intent,
                    rationale="Question about document → needs doc_get_details",
                    suggested_tools=["doc_get_details"],
                )
            return RouterResult(
                decision=RouteDecision.DIRECT_ANSWER,
                intent=intent,
                rationale="General question without document context",
            )

        # Rule 4: Unknown → direct answer
        return RouterResult(
            decision=RouteDecision.DIRECT_ANSWER,
            intent=intent,
            rationale="Unknown intent — defaulting to direct answer",
        )

    @staticmethod
    def _is_greeting_or_meta(message: str) -> bool:
        lower = message.lower().strip()
        greetings = (
            "привет", "здравствуй", "добрый", "помоги", "что ты умеешь",
            "кто ты", "как дела", "hello", "hi ",
        )
        return any(lower.startswith(g) for g in greetings) and len(message) < 60

    @staticmethod
    def _suggest_tools(intent: UserIntent) -> list[str]:
        from edms_ai_assistant.tools.router import _INTENT_TOOL_NAMES
        return list(_INTENT_TOOL_NAMES.get(intent, []))


# ── Planner ───────────────────────────────────────────────────────────────────


@dataclass
class ExecutionPlan:
    steps: list[str]
    required_tools: list[str]
    rationale: str = ""
    is_sequential: bool = True

    def to_prompt_hint(self) -> str:
        steps_text = "\n".join(f"  Шаг {i+1}: {s}" for i, s in enumerate(self.steps))
        tools_text = ", ".join(self.required_tools)
        return (
            f"\n<execution_plan>\n"
            f"Задача требует {len(self.steps)} шагов:\n"
            f"{steps_text}\n"
            f"Инструменты: {tools_text}\n"
            f"Выполняй строго последовательно.\n"
            f"</execution_plan>"
        )


class AgentPlanner:
    """Breaks complex tasks into ordered sub-tasks."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def plan(
        self,
        intent: UserIntent,
        has_file: bool,
        has_document: bool,
        message: str,
    ) -> ExecutionPlan | None:
        if intent == UserIntent.ANALYZE:
            return self._plan_analysis(has_file, has_document)
        if intent == UserIntent.COMPLIANCE_CHECK:
            return self._plan_compliance(has_file)
        if intent == UserIntent.COMPARE:
            return self._plan_comparison(has_file)
        if intent == UserIntent.SUMMARIZE:
            return self._plan_summarize(has_file)
        return None

    @staticmethod
    def _plan_analysis(has_file: bool, has_document: bool) -> ExecutionPlan:
        if has_file:
            return ExecutionPlan(
                steps=[
                    "Прочитать файл (read_local_file_content)",
                    "Суммаризировать (doc_summarize_text)",
                ],
                required_tools=["read_local_file_content", "doc_summarize_text"],
                rationale="File analysis: read → summarize",
            )
        return ExecutionPlan(
            steps=[
                "Получить метаданные (doc_get_details)",
                "Получить текст вложения (doc_get_file_content)",
                "Суммаризировать (doc_summarize_text)",
            ],
            required_tools=["doc_get_details", "doc_get_file_content", "doc_summarize_text"],
            rationale="Document analysis: metadata → content → summarize",
        )

    @staticmethod
    def _plan_compliance(has_file: bool) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[
                "Вызвать doc_compliance_check ОДИН РАЗ с check_all=True",
                "Сформулировать ответ пользователю",
            ],
            required_tools=["doc_compliance_check"],
            rationale="Compliance: single call then respond",
        )

    @staticmethod
    def _plan_comparison(has_file: bool) -> ExecutionPlan:
        if has_file:
            return ExecutionPlan(
                steps=["doc_compare_attachment_with_local"],
                required_tools=["doc_compare_attachment_with_local"],
                rationale="File vs attachment comparison",
            )
        return ExecutionPlan(
            steps=[
                "doc_get_versions (сравнивает все версии автоматически)",
                "Представить все изменения пользователю",
            ],
            required_tools=["doc_get_versions"],
            rationale="Version comparison: single call",
        )

    @staticmethod
    def _plan_summarize(has_file: bool) -> ExecutionPlan:
        if has_file:
            return ExecutionPlan(
                steps=["read_local_file_content", "doc_summarize_text"],
                required_tools=["read_local_file_content", "doc_summarize_text"],
                rationale="Local file summarization",
            )
        return ExecutionPlan(
            steps=["doc_get_file_content", "doc_summarize_text"],
            required_tools=["doc_get_file_content", "doc_summarize_text"],
            rationale="EDMS attachment summarization",
        )