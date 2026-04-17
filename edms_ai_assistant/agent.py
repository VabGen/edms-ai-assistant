# edms_ai_assistant/agent_v2.py
"""
EDMS AI Assistant — Refactored Agent (Production-Ready).

Architecture (5-Part Standard):

PART 1 — MEMORY (L0/L1/L2/L3):
    L0: System prompt (static instructions)
    L1: Working memory — last MAX_WORKING_MESSAGES messages only
    L2: Auto-summarization of older turns → injected back into system prompt
    L3: RAG stub (wire up Qdrant/Pinecone to activate)

PART 2 — ROUTER + PLANNER:
    Router node runs FIRST — decides: direct answer vs tool vs multi-step plan.
    Planner creates ordered ExecutionPlan for complex intents (ANALYZE, COMPLIANCE, etc.)
    Plan is injected into the system prompt as a step-by-step hint.

PART 3 — SUPERVISOR PATTERN:
    EdmsDocumentAgent  = MANAGER (delegates, never executes directly)
    DocumentSubAgent   = reads/analyzes documents
    WorkflowSubAgent   = creates tasks, introductions, notifications
    SearchSubAgent     = searches documents and employees
    FileSubAgent       = handles local file operations
    The manager receives intent + plan, dispatches to the right sub-agent.

PART 4 — TOOLS:
    Each sub-agent gets only the minimal tool subset for its domain.
    Tool errors are caught and handled: retry → graceful degradation.

PART 5 — SAFETY:
    Streaming: SSE via FastAPI StreamingResponse.
    Human-in-the-Loop: destructive operations require explicit confirmation.
    Idempotency: guard against duplicate tool calls (ToolCallGuard).
    Pydantic validation on all inputs.
    GuardrailPipeline on all outputs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.agent_components import build_response_assembler, build_tool_call_guard
from edms_ai_assistant.agent_config import AgentConfig, DEFAULT_CONFIG
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.guardrails import GuardrailPipeline
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.memory import ConversationMemoryManager
from edms_ai_assistant.model import AgentState, ContextParams
from edms_ai_assistant.observability import SpanKind, Trace
from edms_ai_assistant.resilience import CircuitBreaker, RetryConfig, retry_with_backoff
from edms_ai_assistant.router_planner import AgentPlanner, AgentRouter, RouteDecision
from edms_ai_assistant.semantic_cache import SemanticCache
from edms_ai_assistant.services.nlp_service import SemanticDispatcher, UserIntent
from edms_ai_assistant.tool_args_patcher import ToolArgsPatcher
from edms_ai_assistant.tool_call_guard import ToolCallGuard
from edms_ai_assistant.tool_call_patch_pipeline import ToolCallPatchPipeline
from edms_ai_assistant.tool_call_router import ToolCallRouter
from edms_ai_assistant.tools import all_tools
from edms_ai_assistant.tools.router import get_tools_for_intent
from edms_ai_assistant.utils.regex_utils import UUID_RE
from edms_ai_assistant.model import (
    AgentRequest,
    AgentResponse,
    AgentStatus,
    ActionType,
    AgentState,
    ContextParams,
)

logger = logging.getLogger(__name__)

# ─── Destructive operations requiring HITL confirmation ──────────────────────
_DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({
    "introduction_create_tool",
    "task_create_tool",
    "doc_update_field",
    "doc_send_notification",
    "create_document_from_file",
})


def _is_valid_uuid(value: str) -> bool:
    return bool(UUID_RE.match(value.strip()))


# ─────────────────────────────────────────────────────────────────────────────
# Sub-Agents (Supervisor pattern — each handles ONE domain)
# ─────────────────────────────────────────────────────────────────────────────


class SubAgentBase:
    """
    Base class for domain-specific sub-agents.

    Each sub-agent has its OWN minimal tool set and its OWN LLM binding.
    The manager (EdmsDocumentAgent) dispatches to sub-agents — it never
    calls tools directly.

    Single Responsibility: execute ONE category of EDMS operations.
    """

    domain: str = "base"

    def __init__(self, llm: Any, tool_names: list[str], all_tools_list: list[Any]) -> None:
        self._llm = llm
        self._tools = [t for t in all_tools_list if getattr(t, "name", None) in set(tool_names)]
        self._model = llm.bind_tools(self._tools) if self._tools else llm
        logger.info(
            "SubAgent[%s] initialized with %d tools: %s",
            self.domain,
            len(self._tools),
            [getattr(t, "name", "?") for t in self._tools],
        )

    @property
    def tools(self) -> list[Any]:
        return self._tools

    @property
    def model_with_tools(self) -> Any:
        return self._model


class DocumentSubAgent(SubAgentBase):
    """
    Handles: read, analyze, summarize, compare documents.
    Tools: doc_get_details, doc_get_file_content, doc_get_versions,
           doc_compare_documents, doc_compare_attachment_with_local,
           doc_summarize_text, read_local_file_content, doc_compliance_check.
    """
    domain = "document"

    def __init__(self, llm: Any, all_tools_list: list[Any]) -> None:
        super().__init__(llm, [
            "doc_get_details", "doc_get_file_content", "doc_get_versions",
            "doc_compare_documents", "doc_compare_attachment_with_local",
            "doc_summarize_text", "read_local_file_content",
            "doc_compliance_check", "doc_update_field",
        ], all_tools_list)


class WorkflowSubAgent(SubAgentBase):
    """
    Handles: creating tasks, introductions, autofill, document creation.
    Tools: task_create_tool, introduction_create_tool, autofill_appeal_document,
           create_document_from_file, doc_send_notification.
    """
    domain = "workflow"

    def __init__(self, llm: Any, all_tools_list: list[Any]) -> None:
        super().__init__(llm, [
            "task_create_tool", "introduction_create_tool",
            "autofill_appeal_document", "create_document_from_file",
            "doc_send_notification",
        ], all_tools_list)


class SearchSubAgent(SubAgentBase):
    """
    Handles: searching documents and employees.
    Tools: doc_search_tool, employee_search_tool.
    """
    domain = "search"

    def __init__(self, llm: Any, all_tools_list: list[Any]) -> None:
        super().__init__(llm, [
            "doc_search_tool", "employee_search_tool", "doc_get_details",
        ], all_tools_list)


# ─────────────────────────────────────────────────────────────────────────────
# AgentStateManager (unchanged from v1 — already solid)
# ─────────────────────────────────────────────────────────────────────────────


class AgentStateManager:
    """Manages LangGraph graph state: invocation, inspection, repair."""

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
        self.graph = graph
        self.checkpointer = checkpointer

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def get_state(self, thread_id: str) -> Any:
        return await self.graph.aget_state(self._config(thread_id))

    async def update_state(
            self,
            thread_id: str,
            messages: list[BaseMessage],
            as_node: str = "agent",
    ) -> None:
        await self.graph.aupdate_state(
            self._config(thread_id),
            {"messages": messages},
            as_node=as_node,
        )

    async def invoke(self, inputs: dict[str, Any], thread_id: str, timeout: float) -> None:
        await asyncio.wait_for(
            self.graph.ainvoke(inputs, config=self._config(thread_id)),
            timeout=timeout,
        )

    async def is_thread_broken(self, thread_id: str) -> bool:
        try:
            state = await self.get_state(thread_id)
            messages = state.values.get("messages", [])
            if not messages:
                return False
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                if len(messages) < 2 or not isinstance(messages[-2], ToolMessage):
                    return True
            return False
        except Exception:
            return False

    async def repair_thread(self, thread_id: str) -> bool:
        try:
            state = await self.get_state(thread_id)
            messages = state.values.get("messages", [])
            if not messages:
                return False
            last = messages[-1]
            if not isinstance(last, AIMessage):
                return False
            tool_calls = getattr(last, "tool_calls", []) or []
            if not tool_calls:
                return False
            synthetic = [
                ToolMessage(
                    content=json.dumps({
                        "status": "error",
                        "message": "Запрос прерван. Повторите вопрос.",
                    }),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in tool_calls
            ]
            await self.update_state(thread_id, synthetic, as_node="tools")
            logger.warning("Thread %s repaired: injected %d synthetic ToolMessages",
                           thread_id, len(synthetic))
            return True
        except Exception as exc:
            logger.error("Thread repair failed: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder (streamlined)
# ─────────────────────────────────────────────────────────────────────────────


class PromptBuilder:
    """
    Strategy: build the L0 system prompt with injected context.
    L2 rolling summary is injected separately by ConversationMemoryManager.
    """

    CORE_TEMPLATE = """<role>
Ты — экспертный ИИ-помощник системы электронного документооборота (EDMS/СЭД).
Специализация: анализ документов, управление персоналом, автоматизация задач.
</role>

<context>
- Пользователь: {user_name} ({user_last_name})
- Дата: {current_date} (год: {current_year})
- Активный документ: {context_ui_id}
- Файл в контексте: {local_file}
<local_file_path>{local_file}</local_file_path>
</context>

<current_user_rules>
Когда пользователь говорит "добавь меня", "я", "моя фамилия":
- Его фамилия: {user_last_name}
- Его полное имя: {user_full_name}
- Передавай эти данные напрямую в инструменты.
</current_user_rules>

<critical_rules>
1. token и document_id добавляются АВТОМАТИЧЕСКИ. Не указывай их в вызовах.
2. Один инструмент за раз. Дождись результата перед следующим вызовом.
3. При requires_disambiguation → покажи список, жди выбора пользователя.
4. Ответ ТОЛЬКО на русском языке. UUID в ответах — ЗАПРЕЩЕНЫ.
5. Используй ФИО вместо UUID в ответах пользователю.
</critical_rules>

<response_format>
✅ Структурировано, кратко, информативно
✅ Маркированные списки для перечислений
❌ JSON-структуры в ответе
❌ UUID в тексте ответа
</response_format>"""

    @classmethod
    def build(cls, context: ContextParams, semantic_xml: str, plan_hint: str = "") -> str:
        base = cls.CORE_TEMPLATE.format(
            user_name=context.user_first_name or context.user_name,
            user_last_name=context.user_last_name or "Не указана",
            user_full_name=context.user_full_name or context.user_name,
            current_date=context.current_date,
            current_year=context.current_year,
            context_ui_id=context.document_id or "Не указан",
            local_file=context.file_path or "Не загружен",
        )
        return base + plan_hint + semantic_xml


# ─────────────────────────────────────────────────────────────────────────────
# Content Extractor (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────


class ContentExtractor:
    """Extracts final human-readable content from a LangGraph message chain."""

    MIN_CONTENT_LENGTH = 30
    _SKIP_PATTERNS: tuple[str, ...] = ('"name"', '"id"', '"tool_calls"')
    _JSON_PRIORITY_FIELDS: tuple[str, ...] = ("content", "message", "text", "result")

    @classmethod
    def extract_final_content(cls, messages: list[BaseMessage]) -> str | None:
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                text = str(m.content).strip()
                if not cls._is_technical(text) and len(text) >= cls.MIN_CONTENT_LENGTH:
                    return text
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._parse_tool_message(m)
                if extracted:
                    return extracted
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                return str(m.content).strip()
        return None

    @classmethod
    def extract_last_tool_text(cls, messages: list[BaseMessage]) -> str | None:
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                raw = str(m.content).strip()
                if raw.startswith("{"):
                    data = json.loads(raw)
                    for key in cls._JSON_PRIORITY_FIELDS:
                        val = data.get(key)
                        if val and len(str(val)) > 100:
                            return str(val)
                if len(raw) > 100:
                    return raw
            except json.JSONDecodeError:
                if len(str(m.content)) > 100:
                    return str(m.content)
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        import re
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                for key in cls._JSON_PRIORITY_FIELDS:
                    val = data.get(key)
                    if val and isinstance(val, str) and len(val) >= cls.MIN_CONTENT_LENGTH:
                        return val.replace("\\n", "\n").strip()
            except (json.JSONDecodeError, ValueError):
                pass
        content = re.sub(r'",?\s*"[a-z_]+"\s*:\s*(?:true|false|null|\d+)\s*\}?\s*$', "", content)
        return content.replace('\\"', '"').replace("\\n", "\n").strip()

    @classmethod
    def _is_technical(cls, content: str) -> bool:
        lower = content.lower()
        return any(p in lower for p in cls._SKIP_PATTERNS)

    @classmethod
    def _parse_tool_message(cls, message: ToolMessage) -> str | None:
        try:
            raw = str(message.content).strip()
            if raw.startswith("{"):
                data = json.loads(raw)
                if data.get("status") == "error":
                    return None
                for key in cls._JSON_PRIORITY_FIELDS:
                    val = data.get(key)
                    if val and isinstance(val, str) and len(val) >= cls.MIN_CONTENT_LENGTH:
                        return val
        except json.JSONDecodeError:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR — EdmsDocumentAgent
# ─────────────────────────────────────────────────────────────────────────────


class EdmsDocumentAgent:
    """
    EDMS AI Agent — SUPERVISOR / MANAGER.

    This class is a MANAGER, not an executor.
    It delegates to sub-agents based on intent:
      - DocumentSubAgent    → read, analyze, summarize, compare
      - WorkflowSubAgent    → create tasks, introductions, notifications
      - SearchSubAgent      → search documents, employees

    Flow:
        chat()
          → _build_context()
          → Router.route()            (decide: direct / tool / plan)
          → Planner.plan()            (optional: build ExecutionPlan)
          → _check_hitl_confirmation() (safety: require confirm for destructive ops)
          → dispatch to sub-agent     (based on intent)
          → _orchestrate()            (LangGraph loop via sub-agent's model)
          → ResponseAssembler.assemble()

    Memory:
        Each thread has its own ConversationMemoryManager.
        History is compressed (L2) before being sent to the LLM.
    """

    def __init__(
            self,
            config: AgentConfig | None = None,
    ):
        self._config = config or DEFAULT_CONFIG

        try:
            self._llm = get_chat_model()
            self._all_tools = all_tools
            self._checkpointer = MemorySaver()

            # ── Sub-agents (Supervisor pattern) ───────────────────────────────
            self._document_agent = DocumentSubAgent(self._llm, self._all_tools)
            self._workflow_agent = WorkflowSubAgent(self._llm, self._all_tools)
            self._search_agent = SearchSubAgent(self._llm, self._all_tools)

            # ── Active model (swapped by dispatcher) ──────────────────────────
            self._active_model = self._llm.bind_tools(self._all_tools)
            self._active_tools = self._all_tools

            # ── Router + Planner ───────────────────────────────────────────────
            self._router = AgentRouter(llm=self._llm)
            self._planner = AgentPlanner(llm=self._llm)

            # ── Semantic dispatcher ────────────────────────────────────────────
            self._dispatcher = SemanticDispatcher()

            # ── Memory managers per thread ─────────────────────────────────────
            self._memory_managers: dict[str, ConversationMemoryManager] = {}

            # ── Cross-cutting concerns ─────────────────────────────────────────
            self._guardrail_pipeline = GuardrailPipeline()
            self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
            self._semantic_cache = SemanticCache()

            # ── Response assembler ─────────────────────────────────────────────
            self._response_assembler = build_response_assembler(
                config=self._config,
                guardrail_pipeline=self._guardrail_pipeline,
            )

            # ── Graph (compiled once) ──────────────────────────────────────────
            self._compiled_graph = self._build_graph()
            self.state_manager = AgentStateManager(self._compiled_graph, self._checkpointer)

            # ── Per-request guard (reset in chat()) ───────────────────────────
            self._tool_call_guard: ToolCallGuard | None = None

            logger.info("EdmsDocumentAgent initialized (Supervisor architecture)")

        except Exception as exc:
            logger.error("Agent initialization failed", exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {exc}") from exc

    # ── Public API ─────────────────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        return {
            "model": self._llm is not None,
            "tools": len(self._all_tools),
            "sub_agents": {
                "document": len(self._document_agent.tools),
                "workflow": len(self._workflow_agent.tools),
                "search": len(self._search_agent.tools),
            },
            "graph": self._compiled_graph is not None,
            "circuit_breaker": self._circuit_breaker.state,
            "semantic_cache": self._semantic_cache.stats(),
            "guardrails_enabled": self._config.enable_guardrails,
        }

    async def chat(
            self,
            message: str,
            user_token: str,
            context_ui_id: str | None = None,
            thread_id: str | None = None,
            user_context: dict[str, Any] | None = None,
            file_path: str | None = None,
            file_name: str | None = None,
            human_choice: str | None = None,
            confirmed: bool = False,
    ) -> dict[str, Any]:
        """
        Main entry point. MANAGER role: orchestrate, delegate, assemble.

        SAFETY: For destructive operations (tasks, introductions, doc creation),
        requires confirmed=True on the second call. First call returns
        requires_confirmation with a prompt for the user.
        """
        # Reset per-request guard
        self._tool_call_guard = build_tool_call_guard()

        try:
            request = AgentRequest(
                message=message,
                user_token=user_token,
                context_ui_id=context_ui_id,
                thread_id=thread_id,
                user_context=user_context or {},
                file_path=file_path,
                file_name=file_name,
                human_choice=human_choice,
                confirmed=confirmed,
            )
            context = await self._build_context(request)

            # Repair broken thread
            if await self.state_manager.is_thread_broken(context.thread_id):
                repaired = await self.state_manager.repair_thread(context.thread_id)
                logger.warning("Thread %s broken → repaired=%s", context.thread_id, repaired)

            state = await self.state_manager.get_state(context.thread_id)

            # Human-in-the-Loop: resume from interrupted state
            if human_choice and state.next:
                return await self._handle_human_choice(context, human_choice)

            # Semantic analysis
            document: DocumentDto | None = None
            if context.document_id:
                document = await self._fetch_document(context)

            semantic_ctx = self._dispatcher.build_context(
                request.message, document, context.file_path
            )
            context.intent = semantic_ctx.query.intent

            # ── PART 2: Router decision ────────────────────────────────────────
            route_result = self._router.route(
                intent=semantic_ctx.query.intent,
                complexity=semantic_ctx.query.complexity,
                has_document_context=bool(context.document_id),
                message=request.message,
            )
            logger.info(
                "Router: intent=%s decision=%s tools=%s",
                route_result.intent.value,
                route_result.decision.value,
                route_result.suggested_tools,
            )

            # ── PART 5: HITL — require confirmation for destructive ops ───────
            if route_result.decision != RouteDecision.DIRECT_ANSWER and not confirmed:
                hitl = self._check_hitl_confirmation(
                    intent=semantic_ctx.query.intent,
                    message=request.message,
                )
                if hitl:
                    return hitl

            # Direct answer — no tools needed
            if route_result.decision == RouteDecision.DIRECT_ANSWER:
                return await self._direct_answer(context, request.message)

            # ── PART 2: Planner ────────────────────────────────────────────────
            plan = self._planner.plan(
                intent=semantic_ctx.query.intent,
                has_file=bool(context.file_path),
                has_document=bool(context.document_id),
                message=request.message,
            )
            plan_hint = plan.to_prompt_hint() if plan else ""

            # ── PART 3: Dispatch to sub-agent ─────────────────────────────────
            self._dispatch_to_sub_agent(semantic_ctx.query.intent, context)

            # Build system prompt (L0) — L2 injected by memory manager
            system_prompt = PromptBuilder.build(
                context,
                self._build_semantic_xml(semantic_ctx),
                plan_hint=plan_hint,
            )

            # ── L1/L2: get memory-aware messages ──────────────────────────────
            memory_mgr = self._get_memory_manager(context.thread_id)
            raw_inputs: list[BaseMessage] = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=semantic_ctx.query.refined),
            ]
            prepared_messages = await memory_mgr.prepare(raw_inputs, system_prompt)

            inputs = {"messages": prepared_messages}

            # Forced tool call for CREATE_DOCUMENT (bypass LLM)
            forced = await self._try_forced_tool_call(context, inputs, request.message)

            return await self._orchestrate(
                context=context,
                inputs=None if forced else inputs,
                is_choice_active=bool(human_choice),
                iteration=0,
            )

        except Exception as exc:
            logger.error("Chat error: %s", exc, exc_info=True)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {exc}",
            ).model_dump()

    # ── Streaming API (PART 5) ─────────────────────────────────────────────────

    async def chat_stream(
            self,
            message: str,
            user_token: str,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Streaming response via Server-Sent Events.

        Yields chunks as they arrive from the LLM.
        Tool call results are yielded as metadata events.

        Usage in FastAPI::
            @app.get("/chat/stream")
            async def stream_endpoint(req: UserInput):
                return StreamingResponse(
                    agent.chat_stream(req.message, req.user_token, ...),
                    media_type="text/event-stream",
                )
        """
        try:
            context = await self._build_context(AgentRequest(
                message=message, user_token=user_token, **kwargs
            ))
            semantic_ctx = self._dispatcher.build_context(message, None, context.file_path)
            system_prompt = PromptBuilder.build(context, self._build_semantic_xml(semantic_ctx))

            yield f"data: {json.dumps({'type': 'start', 'intent': semantic_ctx.query.intent.value})}\n\n"

            memory_mgr = self._get_memory_manager(context.thread_id)
            messages = await memory_mgr.prepare(
                [SystemMessage(content=system_prompt), HumanMessage(content=message)],
                system_prompt,
            )

            self._dispatch_to_sub_agent(semantic_ctx.query.intent, context)

            # Stream from LLM
            async for chunk in self._active_model.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk.content})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    # ── HITL Confirmation (PART 5) ─────────────────────────────────────────────

    def _check_hitl_confirmation(
            self,
            intent: UserIntent,
            message: str,
    ) -> dict[str, Any] | None:
        """
        Returns a requires_confirmation response for destructive operations
        when not yet confirmed by the user.

        Destructive intents: CREATE_TASK, CREATE_INTRODUCTION, CREATE_DOCUMENT, UPDATE.
        The frontend should display a confirmation dialog and re-send with confirmed=True.
        """
        destructive_intents = {
            UserIntent.CREATE_TASK,
            UserIntent.CREATE_INTRODUCTION,
            UserIntent.CREATE_DOCUMENT,
            UserIntent.UPDATE,
        }
        if intent not in destructive_intents:
            return None

        labels = {
            UserIntent.CREATE_TASK: "создать поручение",
            UserIntent.CREATE_INTRODUCTION: "добавить сотрудников в список ознакомления",
            UserIntent.CREATE_DOCUMENT: "создать новый документ",
            UserIntent.UPDATE: "обновить поле документа",
        }
        label = labels.get(intent, "выполнить операцию")

        return AgentResponse(
            status=AgentStatus.REQUIRES_CONFIRMATION,
            action_type=ActionType.CONFIRMATION,
            requires_confirmation=True,
            confirmation_prompt=(
                f"Вы уверены, что хотите {label}? "
                f"Эта операция изменит данные в системе. "
                f"Нажмите «Подтвердить» для продолжения."
            ),
            message=f"Требуется подтверждение для операции: {label}",
        ).model_dump()

    # ── Sub-agent dispatch (PART 3 — Supervisor) ──────────────────────────────

    def _dispatch_to_sub_agent(
            self,
            intent: UserIntent,
            context: ContextParams,
    ) -> None:
        """
        MANAGER decision: which sub-agent handles this intent?

        Updates self._active_model and self._active_tools.
        The LangGraph call_model node always uses self._active_model.
        """
        workflow_intents = {
            UserIntent.CREATE_TASK,
            UserIntent.CREATE_INTRODUCTION,
            UserIntent.CREATE_DOCUMENT,
            UserIntent.EXTRACT,
        }
        search_intents = {
            UserIntent.SEARCH,
        }

        if intent in workflow_intents:
            sub = self._workflow_agent
            logger.info("Dispatching to WorkflowSubAgent (intent=%s)", intent.value)
        elif intent in search_intents:
            sub = self._search_agent
            logger.info("Dispatching to SearchSubAgent (intent=%s)", intent.value)
        else:
            sub = self._document_agent
            logger.info("Dispatching to DocumentSubAgent (intent=%s)", intent.value)

        # Check if appeal autofill should be included
        is_appeal = context.user_context.get("doc_category", "") == "APPEAL"
        if is_appeal and intent in {UserIntent.ANALYZE, UserIntent.SUMMARIZE}:
            tools = get_tools_for_intent(intent, self._all_tools, include_appeal=True)
            self._active_model = self._llm.bind_tools(tools)
            self._active_tools = tools
            return

        self._active_model = sub.model_with_tools
        self._active_tools = sub.tools

    # ── Orchestration loop ─────────────────────────────────────────────────────

    async def _orchestrate(
            self,
            context: ContextParams,
            inputs: dict[str, Any] | None,
            is_choice_active: bool,
            iteration: int,
    ) -> dict[str, Any]:
        """
        Core LangGraph orchestration loop.

        Delegates all tool patching to ToolCallPatchPipeline.
        Delegates response assembly to ResponseAssembler.
        This method only manages the loop logic.
        """
        if iteration > self._config.max_iterations:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышен лимит итераций.",
            ).model_dump()

        try:
            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self._config.execution_timeout,
            )

            state = await self.state_manager.get_state(context.thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])

            if not messages:
                return AgentResponse(status=AgentStatus.ERROR, message="Пустое состояние.").model_dump()

            last_msg = messages[-1]
            last_is_tool = isinstance(last_msg, ToolMessage)
            last_has_calls = isinstance(last_msg, AIMessage) and bool(getattr(last_msg, "tool_calls", None))
            is_finished = not state.next and not last_is_tool and not last_has_calls

            if is_finished:
                # Update memory after successful turn
                memory_mgr = self._get_memory_manager(context.thread_id)
                memory_mgr.record(*messages[-4:])  # Record recent messages
                return self._response_assembler.assemble(messages, context)

            # Process tool calls through patch pipeline
            raw_calls = list(getattr(last_msg, "tool_calls", []))
            if len(raw_calls) > 1:
                logger.warning("Parallel calls blocked: keeping only %s", raw_calls[0]["name"])
                raw_calls = raw_calls[:1]

            patcher = ToolArgsPatcher(
                user_token=context.user_token,
                document_id=context.document_id,
                file_path=context.file_path,
                uploaded_file_name=context.uploaded_file_name,
                user_context=context.user_context,
                is_choice_active=is_choice_active,
            )
            tool_router = ToolCallRouter(
                document_id=context.document_id,
                file_path=context.file_path,
                uploaded_file_name=context.uploaded_file_name,
                user_token=context.user_token,
                is_choice_active=is_choice_active,
            )
            pipeline = ToolCallPatchPipeline(
                patcher=patcher,
                router=tool_router,
                guard=self._tool_call_guard,
            )

            last_tool_text = ContentExtractor.extract_last_tool_text(messages)
            patched_calls = []

            for tc in raw_calls:
                result = pipeline.process(
                    tool_name=tc["name"],
                    tool_args=tc["args"],
                    call_id=tc["id"],
                    messages=messages,
                    last_tool_text=last_tool_text,
                )
                if result.allowed:
                    patched_calls.append({"name": result.name, "args": result.args, "id": result.id})
                else:
                    logger.warning("Tool %s blocked: %s", result.name, result.guard_result.reason)

            await self.state_manager.update_state(
                context.thread_id,
                [AIMessage(content=last_msg.content or "", tool_calls=patched_calls, id=last_msg.id)],
                as_node="agent",
            )

            return await self._orchestrate(
                context=context,
                inputs=None,
                is_choice_active=is_choice_active,
                iteration=iteration + 1,
            )

        except TimeoutError:
            return AgentResponse(status=AgentStatus.ERROR, message="Превышено время ожидания.").model_dump()

        except Exception as exc:
            err_str = str(exc)
            logger.error("Orchestration error iter=%d: %s", iteration, exc, exc_info=True)

            broken_signals = (
                "tool_calls must be followed by tool messages",
                "tool_call_ids did not have response messages",
                "invalid_request_error",
            )
            if any(s in err_str for s in broken_signals) and iteration == 0:
                repaired = await self.state_manager.repair_thread(context.thread_id)
                if repaired:
                    return AgentResponse(
                        status=AgentStatus.ERROR,
                        message="Предыдущий запрос завершился некорректно. Повторите вопрос.",
                    ).model_dump()

            return AgentResponse(status=AgentStatus.ERROR, message=f"Ошибка: {exc}").model_dump()

    # ── Direct answer (no tools) ───────────────────────────────────────────────

    async def _direct_answer(
            self,
            context: ContextParams,
            message: str,
    ) -> dict[str, Any]:
        """
        Answer directly from LLM knowledge without any tool calls.
        Used for greetings, meta-questions, general knowledge.
        """
        try:
            memory_mgr = self._get_memory_manager(context.thread_id)
            system_prompt = (
                f"Ты — ИИ-помощник системы EDMS. Пользователь: {context.user_name}. "
                f"Отвечай кратко и по-русски."
            )
            messages = await memory_mgr.prepare(
                [SystemMessage(content=system_prompt), HumanMessage(content=message)],
                system_prompt,
            )
            response = await self._llm.ainvoke(messages)
            content = str(response.content).strip()
            return AgentResponse(status=AgentStatus.SUCCESS, content=content).model_dump()
        except Exception as exc:
            return AgentResponse(status=AgentStatus.ERROR, message=str(exc)).model_dump()

    # ── Human-in-the-Loop ─────────────────────────────────────────────────────

    async def _handle_human_choice(
            self,
            context: ContextParams,
            human_choice: str,
    ) -> dict[str, Any]:
        """Resume from an interrupted graph state with user's choice."""
        if human_choice.startswith("fix_field:"):
            parts = human_choice.split(":", 2)
            if len(parts) == 3:
                _, update_field, correct_value = parts
                return await self._orchestrate(
                    context=context,
                    inputs={"messages": [HumanMessage(
                        content=f'Исправь поле "{update_field}" на значение "{correct_value}"'
                    )]},
                    is_choice_active=True,
                    iteration=0,
                )

        state = await self.state_manager.get_state(context.thread_id)
        if not state.next:
            return await self._orchestrate(
                context=context,
                inputs={"messages": [HumanMessage(content=human_choice)]},
                is_choice_active=False,
                iteration=0,
            )

        last_msg: AIMessage = state.values["messages"][-1]
        raw_calls = getattr(last_msg, "tool_calls", [])
        patched_calls = self._patch_human_choice_calls(raw_calls, human_choice, context)

        await self.state_manager.update_state(
            context.thread_id,
            [AIMessage(content=last_msg.content or "", tool_calls=patched_calls, id=last_msg.id)],
            as_node="agent",
        )
        return await self._orchestrate(context=context, inputs=None, is_choice_active=True, iteration=0)

    def _patch_human_choice_calls(
            self,
            raw_calls: list[dict],
            human_choice: str,
            context: ContextParams,
    ) -> list[dict]:
        """Apply human_choice to pending tool calls (disambiguation resolution)."""
        patched = []
        _DISAMBIGUATION_TOOLS = frozenset({"introduction_create_tool", "task_create_tool"})

        for tc in raw_calls:
            t_args = dict(tc["args"])
            t_name = tc["name"]

            if t_name == "doc_summarize_text":
                t_args["summary_type"] = human_choice.strip()

            elif t_name in _DISAMBIGUATION_TOOLS:
                valid_ids = [x.strip() for x in human_choice.split(",")
                             if x.strip()]
                if valid_ids:
                    t_args["selected_employee_ids"] = valid_ids
                    t_args.pop("last_names", None)
                    t_args.pop("executor_last_names", None)

            elif t_name == "doc_compare_attachment_with_local":
                t_args["attachment_id"] = human_choice.strip()
                if context.document_id:
                    t_args["document_id"] = context.document_id
                if context.uploaded_file_name:
                    t_args["original_filename"] = context.uploaded_file_name

            patched.append({"name": t_name, "args": t_args, "id": tc["id"]})

        return patched

    # ── Graph compilation ─────────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        """Compile the LangGraph ReAct workflow with Router node."""
        workflow = StateGraph(AgentState)

        # ── Router node (runs before agent) ───────────────────────────────────
        async def router_node(state: AgentState) -> dict[str, Any]:
            """
            PART 2: Router node.
            Determines if the last message needs tools or can be answered directly.
            For now, always passes through — routing happens in chat() before graph invocation.
            This node is a hook for future graph-level routing logic.
            """
            return {"messages": []}

        # ── Agent node ─────────────────────────────────────────────────────────
        async def call_model(state: AgentState) -> dict[str, Any]:
            """Invoke the LLM with bound tools. Uses active_model set by dispatcher."""
            from edms_ai_assistant.memory import MAX_WORKING_MESSAGES

            sys_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
            non_sys = [m for m in state["messages"] if not isinstance(m, SystemMessage)]

            # L1: apply working memory window
            if len(non_sys) > MAX_WORKING_MESSAGES:
                non_sys = non_sys[-MAX_WORKING_MESSAGES:]

            # Sanitize dangling AIMessages
            final_msgs = []
            for i, msg in enumerate(sys_msgs + non_sys):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    next_msg = (sys_msgs + non_sys)[i + 1] if i + 1 < len(sys_msgs + non_sys) else None
                    if not isinstance(next_msg, ToolMessage):
                        final_msgs.append(AIMessage(content=msg.content or "", id=msg.id))
                        continue
                final_msgs.append(msg)

            response = await self._active_model.ainvoke(final_msgs)
            return {"messages": [response]}

        # ── Validator node ─────────────────────────────────────────────────────
        async def validator(state: AgentState) -> dict[str, Any]:
            """Post-tool validation: detect interactive statuses, errors, empty results."""
            last = state["messages"][-1]
            if not isinstance(last, ToolMessage):
                return {"messages": []}

            raw = str(last.content).strip()

            if not raw or raw in ("None", "{}", "null"):
                return {"messages": [AIMessage(content=(
                    "⚠️ Инструмент вернул пустой результат. Попробуй другой подход."
                ))]}

            if raw.startswith("{"):
                try:
                    tool_data = json.loads(raw)
                    interactive_status = tool_data.get("status", "")
                    if interactive_status == "success" and "fields" in tool_data:
                        return {"messages": [AIMessage(content="Анализ завершен.")]}
                    if interactive_status in ("requires_choice", "requires_disambiguation", "requires_action"):
                        return {"messages": [AIMessage(content="")]}
                except json.JSONDecodeError:
                    pass

            return {"messages": []}

        # ── Tool error handler ─────────────────────────────────────────────────
        async def handle_tool_error(state: AgentState) -> dict[str, Any]:
            """
            PART 4: Tool error handling with graceful degradation.

            If the last ToolMessage contains an error status:
            - Inject a user-friendly error hint into the conversation.
            - Agent can then decide: retry with different params, or inform user.
            """
            last = state["messages"][-1]
            if not isinstance(last, ToolMessage):
                return {"messages": []}
            try:
                data = json.loads(str(last.content))
                if data.get("status") == "error":
                    err_msg = data.get("message", "Неизвестная ошибка инструмента")
                    return {"messages": [AIMessage(content=(
                        f"⚠️ Инструмент вернул ошибку: {err_msg}. "
                        "Проинформируй пользователя понятным языком."
                    ))]}
            except (json.JSONDecodeError, TypeError):
                pass
            return {"messages": []}

        # ── Wire the graph ─────────────────────────────────────────────────────
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools=self._all_tools))
        workflow.add_node("validator", validator)
        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "validator")
        workflow.add_edge("validator", "agent")

        return workflow.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["tools"],
        )

    # ── Forced tool call (deterministic bypass for CREATE_DOCUMENT) ──────────

    async def _try_forced_tool_call(
            self,
            context: ContextParams,
            inputs: dict,
            original_message: str,
    ) -> bool:
        """Bypass LLM for CREATE_DOCUMENT + local file — inject tool call directly."""
        import uuid as _uuid
        from edms_ai_assistant.tools.create_document_from_file import _extract_category_from_message

        if context.intent != UserIntent.CREATE_DOCUMENT:
            return False
        clean_path = str(context.file_path or "").strip()
        if not clean_path or _is_valid_uuid(clean_path):
            return False

        doc_category = _extract_category_from_message(original_message) or "APPEAL"
        tool_call_id = f"forced_{_uuid.uuid4().hex[:12]}"
        forced_ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "create_document_from_file",
                "args": {
                    "token": context.user_token,
                    "file_path": clean_path,
                    "doc_category": doc_category,
                    "file_name": context.uploaded_file_name or "",
                    "autofill": True,
                },
                "id": tool_call_id,
            }],
        )
        sys_msg = inputs["messages"][0]
        human_msg = inputs["messages"][1] if len(inputs["messages"]) > 1 else HumanMessage(content=original_message)
        await self.state_manager.update_state(
            context.thread_id, [sys_msg, human_msg, forced_ai_msg], as_node="agent"
        )
        logger.info("Forced tool call injected: create_document_from_file id=%s", tool_call_id)
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_memory_manager(self, thread_id: str) -> ConversationMemoryManager:
        """Get or create a memory manager for this thread."""
        if thread_id not in self._memory_managers:
            self._memory_managers[thread_id] = ConversationMemoryManager(llm=self._llm)
        return self._memory_managers[thread_id]

    async def _build_context(self, request: AgentRequest) -> ContextParams:
        ctx = request.user_context
        first_name = (ctx.get("firstName") or ctx.get("first_name") or "").strip()
        last_name = (ctx.get("lastName") or ctx.get("last_name") or "").strip()
        full_name = (ctx.get("fullName") or ctx.get("full_name") or ctx.get("name") or "").strip()
        user_id = ctx.get("id") or ctx.get("userId") or ctx.get("user_id")
        display_name = first_name or last_name or full_name or "пользователь"
        return ContextParams(
            user_token=request.user_token,
            document_id=request.context_ui_id,
            file_path=request.file_path,
            uploaded_file_name=request.file_name or None,
            thread_id=request.thread_id or "default",
            user_name=display_name,
            user_first_name=first_name or None,
            user_last_name=last_name or None,
            user_full_name=full_name or None,
            user_id=user_id,
            user_context=ctx,
        )

    async def _fetch_document(self, context: ContextParams) -> DocumentDto | None:
        try:
            async with DocumentClient() as client:
                raw = await client.get_document_metadata(context.user_token, context.document_id)
                return DocumentDto.model_validate(raw)
        except Exception as exc:
            logger.error("Failed to fetch document %s: %s", context.document_id, exc)
            return None

    @staticmethod
    def _build_semantic_xml(semantic_context: Any) -> str:
        return (
            "\n<semantic_context>\n"
            f"  <intent>{semantic_context.query.intent.value}</intent>\n"
            f"  <complexity>{semantic_context.query.complexity.value}</complexity>\n"
            f"  <original>{semantic_context.query.original}</original>\n"
            f"  <refined>{semantic_context.query.refined}</refined>\n"
            "</semantic_context>"
        )
