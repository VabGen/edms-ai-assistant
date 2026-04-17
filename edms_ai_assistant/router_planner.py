# edms_ai_assistant/router_planner.py
"""
Router + Planner — decide HOW to handle incoming requests.

Router decision tree:
    Incoming query
        ↓
    ┌─────────────┐
    │   ROUTER    │  ← "Does this need a tool, or can we answer from context?"
    └──────┬──────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
  Direct        Tool(s)
  Answer        Needed
              ↓
        ┌─────────────┐
        │   PLANNER   │  ← "Is this complex? Break into sub-tasks."
        └─────────────┘

The Router is a LangGraph node — it runs BEFORE the LLM agent node.
The Planner is invoked when complexity == HIGH and task requires multiple tools.
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
    """What the Router decides to do with the incoming request."""

    DIRECT_ANSWER = "direct_answer"   # Answer from LLM knowledge / system prompt
    USE_TOOL = "use_tool"             # Route to tool-using agent
    NEEDS_PLAN = "needs_plan"         # Complex multi-step → invoke Planner first
    NEEDS_CLARIFICATION = "needs_clarification"  # Ambiguous request


@dataclass
class RouterResult:
    """
    Output of the Router node.

    Attributes:
        decision: What to do next.
        intent: Classified user intent.
        rationale: Why this decision was made (for logging/observability).
        suggested_tools: Tool names the Router thinks are needed (hint for Planner).
    """

    decision: RouteDecision
    intent: UserIntent
    rationale: str = ""
    suggested_tools: list[str] = field(default_factory=list)


# ── Intents that never need a tool ────────────────────────────────────────────
_KNOWLEDGE_ONLY_INTENTS: frozenset[UserIntent] = frozenset({
    UserIntent.UNKNOWN,
})

# Intents that always need tools
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

# Intents that might need tools depending on context
_CONTEXT_DEPENDENT: frozenset[UserIntent] = frozenset({
    UserIntent.QUESTION,
    UserIntent.UPDATE,
})

# Intents requiring multi-step planning
_COMPLEX_INTENTS: frozenset[UserIntent] = frozenset({
    UserIntent.ANALYZE,
    UserIntent.COMPLIANCE_CHECK,
    UserIntent.COMPARE,
})


class AgentRouter:
    """
    Routes incoming requests to the correct processing path.

    This is a LangGraph node that runs BEFORE the main agent.
    Decision is deterministic (rule-based) for known intents,
    falls back to LLM classification for ambiguous cases.

    Single Responsibility: decide WHERE to route, not HOW to execute.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm  # Optional: used for fallback classification

    def route(
        self,
        intent: UserIntent,
        complexity: QueryComplexity,
        has_document_context: bool,
        message: str,
    ) -> RouterResult:
        """
        Make a synchronous routing decision.

        Args:
            intent: Pre-classified user intent from SemanticDispatcher.
            complexity: Query complexity level.
            has_document_context: Whether a document_id is available.
            message: Original user message (for fallback analysis).

        Returns:
            RouterResult with the routing decision.
        """
        # Rule 1: Simple greetings / meta-questions → direct answer
        if self._is_greeting_or_meta(message):
            return RouterResult(
                decision=RouteDecision.DIRECT_ANSWER,
                intent=intent,
                rationale="Greeting or meta-question — no tool needed",
            )

        # Rule 2: Known tool intents → USE_TOOL
        if intent in _TOOL_INTENTS:
            decision = (
                RouteDecision.NEEDS_PLAN
                if intent in _COMPLEX_INTENTS and complexity == QueryComplexity.HIGH
                else RouteDecision.USE_TOOL
            )
            return RouterResult(
                decision=decision,
                intent=intent,
                rationale=f"Intent {intent.value} requires tools (complexity={complexity.value})",
                suggested_tools=self._suggest_tools(intent),
            )

        # Rule 3: Context-dependent intents
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

        # Rule 4: Unknown → direct answer (safer than hallucinated tool calls)
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
        """Provide tool hints to the Planner based on intent."""
        from edms_ai_assistant.tools.router import _INTENT_TOOL_NAMES
        return list(_INTENT_TOOL_NAMES.get(intent, []))


# ── Planner ───────────────────────────────────────────────────────────────────


@dataclass
class ExecutionPlan:
    """
    A multi-step execution plan for complex tasks.

    Attributes:
        steps: Ordered list of step descriptions.
        required_tools: Tools needed (in order).
        rationale: Why this plan was created.
        is_sequential: Whether steps must execute in order (vs parallel).
    """

    steps: list[str]
    required_tools: list[str]
    rationale: str = ""
    is_sequential: bool = True

    def to_prompt_hint(self) -> str:
        """Convert plan to a system-prompt-injectable hint for the LLM."""
        steps_text = "\n".join(f"  Шаг {i+1}: {s}" for i, s in enumerate(self.steps))
        tools_text = ", ".join(self.required_tools)
        return (
            f"\n<execution_plan>\n"
            f"Задача требует {len(self.steps)} шагов:\n"
            f"{steps_text}\n"
            f"Инструменты: {tools_text}\n"
            f"Выполняй строго последовательно. "
            f"НЕ пропускай шаги и НЕ вызывай несколько инструментов одновременно.\n"
            f"</execution_plan>"
        )


class AgentPlanner:
    """
    Breaks complex multi-step tasks into ordered sub-tasks.

    Invoked by the Router when RouteDecision == NEEDS_PLAN.
    Creates an ExecutionPlan that is injected into the system prompt
    as a hint to guide the LLM through the required steps.

    For most tasks, plans are built deterministically (rule-based).
    For truly novel tasks, falls back to LLM-assisted planning.

    Single Responsibility: create ordered execution plans.
    Does NOT execute plans — that is the agent's job.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def plan(
        self,
        intent: UserIntent,
        has_file: bool,
        has_document: bool,
        message: str,
    ) -> ExecutionPlan | None:
        """
        Build an execution plan for a complex intent.

        Returns None for simple single-tool intents.
        """
        if intent == UserIntent.ANALYZE:
            return self._plan_analysis(has_file, has_document)

        if intent == UserIntent.COMPLIANCE_CHECK:
            return self._plan_compliance(has_file)

        if intent == UserIntent.COMPARE:
            return self._plan_comparison(has_file)

        if intent == UserIntent.SUMMARIZE:
            return self._plan_summarize(has_file)

        return None  # No plan needed for simple intents

    @staticmethod
    def _plan_analysis(has_file: bool, has_document: bool) -> ExecutionPlan:
        if has_file:
            return ExecutionPlan(
                steps=[
                    "Прочитать содержимое загруженного файла (read_local_file_content)",
                    "Выполнить тезисную суммаризацию полученного текста (doc_summarize_text)",
                ],
                required_tools=["read_local_file_content", "doc_summarize_text"],
                rationale="File analysis requires read then summarize",
            )
        return ExecutionPlan(
            steps=[
                "Получить метаданные документа (doc_get_details)",
                "Получить текст основного вложения (doc_get_file_content)",
                "Выполнить тезисную суммаризацию (doc_summarize_text)",
            ],
            required_tools=["doc_get_details", "doc_get_file_content", "doc_summarize_text"],
            rationale="Document analysis: metadata → content → summarize",
        )

    @staticmethod
    def _plan_compliance(has_file: bool) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[
                "Вызвать doc_compliance_check ОДИН РАЗ с check_all=True",
                "Сформулировать ответ пользователю на основе полученных данных",
                "НЕ вызывать doc_compliance_check повторно",
            ],
            required_tools=["doc_compliance_check"],
            rationale="Compliance: single call then respond",
        )

    @staticmethod
    def _plan_comparison(has_file: bool) -> ExecutionPlan:
        if has_file:
            return ExecutionPlan(
                steps=[
                    "Сравнить загруженный файл с вложением документа "
                    "(doc_compare_attachment_with_local)",
                    "Представить результат сравнения пользователю",
                ],
                required_tools=["doc_compare_attachment_with_local"],
                rationale="File vs attachment comparison",
            )
        return ExecutionPlan(
            steps=[
                "Получить все версии документа и их сравнение (doc_get_versions)",
                "Представить все найденные изменения пользователю",
                "НЕ вызывать doc_compare_documents — данные уже есть",
            ],
            required_tools=["doc_get_versions"],
            rationale="Version comparison: single call returns all pairs",
        )

    @staticmethod
    def _plan_summarize(has_file: bool) -> ExecutionPlan:
        if has_file:
            return ExecutionPlan(
                steps=[
                    "Прочитать файл (read_local_file_content)",
                    "Вызвать суммаризацию (doc_summarize_text)",
                ],
                required_tools=["read_local_file_content", "doc_summarize_text"],
                rationale="Local file summarization pipeline",
            )
        return ExecutionPlan(
            steps=[
                "Получить текст вложения (doc_get_file_content)",
                "Вызвать суммаризацию с полученным текстом (doc_summarize_text)",
            ],
            required_tools=["doc_get_file_content", "doc_summarize_text"],
            rationale="EDMS attachment summarization pipeline",
        )