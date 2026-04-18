# edms_ai_assistant/agent.py
"""
EDMS AI Assistant — Production-Ready Agent v3.1

Fixes vs v3.0:
  BUG-1: QueryComplexity.HIGH → use VERY_COMPLEX/COMPLEX  (router_planner.py)
  BUG-2: Search results lost → ResponseAssembler now correctly picks up
          AIMessage content over ToolMessage JSON
  BUG-3: file_path validator rejected plain filenames → model.py fix
  BUG-4: Summarize returned "Анализ завершён" → validator no longer stops
          graph on compliance-only pattern; assembler skips empty AIMessage

Architecture:
  Part 1 — Memory:   ConversationMemoryManager (L0/L1/L2)
  Part 2 — Router:   AgentRouter + AgentPlanner
  Part 3 — Supervisor: DocumentSubAgent / WorkflowSubAgent / SearchSubAgent
  Part 4 — Tools:    ToolCallPatchPipeline
  Part 5 — Safety:   GuardrailPipeline, SemanticCache, HITL, CircuitBreaker
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.model import (
    AgentRequest,
    AgentResponse,
    AgentState,
    AgentStatus,
    ActionType,
    ContextParams,
    _is_mutation_response,
)
from edms_ai_assistant.response_assembler import ResponseAssembler, build_response_assembler
from edms_ai_assistant.router_planner import AgentPlanner, AgentRouter, RouteDecision
from edms_ai_assistant.services.nlp_service import SemanticDispatcher, UserIntent
from edms_ai_assistant.tools import all_tools
from edms_ai_assistant.tools.router import estimate_tools_tokens, get_tools_for_intent
from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

USE_LEAN_PROMPT: bool = False

# Optional components — graceful degradation if not present
try:
    from edms_ai_assistant.memory import ConversationMemoryManager, MAX_WORKING_MESSAGES
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False
    MAX_WORKING_MESSAGES = 40
    ConversationMemoryManager = None  # type: ignore

try:
    from edms_ai_assistant.guardrails import GuardrailPipeline
    _GUARDRAILS_AVAILABLE = True
except ImportError:
    _GUARDRAILS_AVAILABLE = False
    GuardrailPipeline = None  # type: ignore

try:
    from edms_ai_assistant.semantic_cache import SemanticCache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    SemanticCache = None  # type: ignore

try:
    from edms_ai_assistant.resilience import CircuitBreaker
    _RESILIENCE_AVAILABLE = True
except ImportError:
    _RESILIENCE_AVAILABLE = False
    CircuitBreaker = None  # type: ignore

try:
    from edms_ai_assistant.tool_call_patch_pipeline import ToolCallPatchPipeline
    from edms_ai_assistant.tool_args_patcher import ToolArgsPatcher
    from edms_ai_assistant.tool_call_router import ToolCallRouter
    from edms_ai_assistant.tool_call_guard import ToolCallGuard
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    ToolCallPatchPipeline = None  # type: ignore
    ToolArgsPatcher = None  # type: ignore
    ToolCallRouter = None  # type: ignore
    ToolCallGuard = None  # type: ignore

try:
    from edms_ai_assistant.agent_config import AgentConfig, DEFAULT_CONFIG
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    AgentConfig = None  # type: ignore
    DEFAULT_CONFIG = None  # type: ignore

try:
    from edms_ai_assistant.agent_components import build_tool_call_guard
    _COMPONENTS_AVAILABLE = True
except ImportError:
    _COMPONENTS_AVAILABLE = False
    build_tool_call_guard = None  # type: ignore


def _is_valid_uuid(value: str) -> bool:
    return bool(UUID_RE.match(value.strip()))


__all__ = [
    "EdmsDocumentAgent",
    "AgentStatus",
    "ActionType",
    "AgentRequest",
    "AgentResponse",
    "ContextParams",
    "ContentExtractor",
    "AgentStateManager",
]

# ── Mutation phrases for requires_reload signal ───────────────────────────────
_MUTATION_SUCCESS_PHRASES: tuple[str, ...] = (
    "успешно добавлен", "успешно создан", "список ознакомления",
    "поручение создано", "поручение успешно", "обращение заполнено",
    "карточка заполнена", "добавлено в список", "ознакомление создано",
    "задача создана", "заголовок обновлен", "заголовок изменен",
    "адрес заявителя обновлен", "телефон в карточке обновлен",
    "изменение выполнено успешно", "операция выполнена успешно",
    "автозаполнен", "обращение автоматически заполнен",
    "карточка обращения заполнен",
)

# ── Tools that need document_id injected ──────────────────────────────────────
_TOOLS_REQUIRING_DOCUMENT_ID: frozenset[str] = frozenset({
    "doc_get_details", "doc_get_versions", "doc_compare_documents",
    "doc_get_file_content", "doc_compare_attachment_with_local",
    "doc_summarize_text", "doc_search_tool",
    "introduction_create_tool", "task_create_tool",
})

# ── Placeholder values for local_file_path ────────────────────────────────────
_COMPARE_LOCAL_PLACEHOLDERS: frozenset[str] = frozenset({
    "", "local_file", "local_file_path", "/path/to/file",
    "path/to/file", "none", "null", "<local_file_path>", "<path>",
})

# ── Tools involved in disambiguation flow ────────────────────────────────────
_DISAMBIGUATION_TOOLS: frozenset[str] = frozenset({
    "introduction_create_tool", "task_create_tool",
})


# ─────────────────────────────────────────────────────────────────────────────
# DocumentRepository
# ─────────────────────────────────────────────────────────────────────────────


class DocumentRepository:
    async def get_document(self, token: str, doc_id: str) -> DocumentDto | None:
        try:
            async with DocumentClient() as client:
                raw_data = await client.get_document_metadata(token, doc_id)
                return DocumentDto.model_validate(raw_data)
        except Exception as exc:
            logger.error("Failed to fetch document %s: %s", doc_id, exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder
# ─────────────────────────────────────────────────────────────────────────────


class PromptBuilder:
    """Builds system prompts with dynamic context injection."""

    CORE_TEMPLATE = """<role>
Ты — экспертный ИИ-помощник системы электронного документооборота (EDMS/СЭД).
Специализация: анализ документов, управление персоналом, автоматизация рутинных задач.
</role>

<context>
- Пользователь (имя): {user_name}
- Пользователь (фамилия): {user_last_name}
- Пользователь (полное имя): {user_full_name}
- Текущая дата: {current_date} (год: {current_year})
- Активный документ в EDMS: {context_ui_id}
- Загруженный файл/вложение: {local_file}
- Имя загруженного файла (показывай пользователю): {uploaded_file_name}
<local_file_path>{local_file}</local_file_path>
</context>

<current_user_rules>
Когда пользователь говорит "добавь меня", "я", "моя фамилия":
- Его фамилия: {user_last_name}
- Его полное имя: {user_full_name}
- Используй эти данные напрямую — НЕ спрашивай фамилию у пользователя.
</current_user_rules>

<critical_rules>
1. token и document_id добавляются АВТОМАТИЧЕСКИ. Не указывай их при вызове инструментов.
2. Если есть локальный файл — ПРИОРИТЕТ: используй его через read_local_file_content.
3. Один инструмент за раз. Дождись результата.
4. При requires_disambiguation — покажи список, жди выбора пользователя.
5. Финальный ответ ТОЛЬКО на РУССКОМ, без UUID, без JSON, без технических деталей.
6. UUID в ответах пользователю — ЗАПРЕЩЕНЫ. Вместо UUID → имя/название.
7. doc_compliance_check — вызывать ОДИН РАЗ с check_all=True, сразу формулировать ответ.
8. Результаты doc_search_tool — ТОЛЬКО таблица: | № | id | Рег. номер | Дата | Краткое содержание | Автор | Статус |
   Колонку id НИКОГДА не убирать.
9. После получения результата любого инструмента — НЕМЕДЛЕННО формулируй ответ пользователю.
   Не вызывай инструменты повторно если данные уже есть.
</critical_rules>

<available_tools_guide>
| Сценарий                                 | Инструменты                                                  |
|------------------------------------------|--------------------------------------------------------------|
| Анализ документа целиком                 | doc_get_details → doc_get_file_content → doc_summarize_text  |
| Анализ конкретного вложения (UUID)       | doc_get_file_content → doc_summarize_text                    |
| Анализ загруженного файла                | read_local_file_content → doc_summarize_text                 |
| Сравнение файла с вложением [ЕСТЬ файл]  | doc_compare_attachment_with_local                            |
| Вопрос о документе                       | doc_get_details                                              |
| Сравнение версий документа [НЕТ файла]   | doc_get_versions                                             |
| Поиск документов в базе EDMS             | doc_search_tool                                              |
| Поиск сотрудника                         | employee_search_tool                                         |
| Добавление в лист ознакомления           | introduction_create_tool                                     |
| Создание поручения                       | task_create_tool                                             |
| Автозаполнение обращения                 | autofill_appeal_document                                     |
| Создать документ из файла                | create_document_from_file                                    |
| Проверка соответствия карточки           | doc_compliance_check (один раз, check_all=True)              |
</available_tools_guide>"""

    LEAN_TEMPLATE = """<role>Ты — AI-помощник EDMS/СЭД.</role>
<context>
Пользователь: {user_name} ({user_last_name})
Дата: {current_date} (год: {current_year})
Документ: {context_ui_id}
Файл: {local_file}
</context>
<rules>
1. token/document_id инжектируются системой.
2. Локальный файл ({local_file}) — приоритет.
3. Один инструмент за раз.
4. При requires_disambiguation — покажи список, жди выбора.
5. Ответ только на русском, без UUID.
</rules>"""

    _SNIPPETS: dict = {}

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        semantic_xml: str,
        plan_hint: str = "",
        *,
        lean: bool = False,
    ) -> str:
        if lean:
            base = cls.LEAN_TEMPLATE.format(
                user_name=context.user_first_name or context.user_name,
                user_last_name=context.user_last_name or "Не указана",
                user_full_name=context.user_full_name or context.user_name,
                current_date=context.current_date,
                current_year=context.current_year,
                context_ui_id=context.document_id or "Не указан",
                local_file=context.file_path or "Не загружен",
            )
        else:
            base = cls.CORE_TEMPLATE.format(
                user_name=context.user_first_name or context.user_name,
                user_last_name=context.user_last_name or "Не указана",
                user_full_name=context.user_full_name or context.user_name,
                current_date=context.current_date,
                current_year=context.current_year,
                context_ui_id=context.document_id or "Не указан",
                local_file=context.file_path or "Не загружен",
                uploaded_file_name=context.uploaded_file_name or "Не определено",
            )
        snippet = cls._SNIPPETS.get(intent, "")
        return base + snippet + plan_hint + semantic_xml


PromptBuilder._SNIPPETS = {
    UserIntent.CREATE_INTRODUCTION: """
<introduction_workflow>
1. introduction_create_tool(last_names=[...])
2. При requires_disambiguation → покажи список → selected_employee_ids
3. Сообщи об успехе с именами добавленных сотрудников
</introduction_workflow>""",
    UserIntent.CREATE_TASK: """
<task_guide>
task_create_tool(task_text=..., executor_last_names=[...])
Дата если упомянута → ISO 8601 ("T23:59:59Z"), год={current_year}.
При disambiguation → покажи список → selected_employee_ids.
</task_guide>""",
    UserIntent.SUMMARIZE: """
<summarize_guide>
ШАГ 1: Получи текст:
  - Файл: read_local_file_content(file_path=<путь>)
  - UUID: doc_get_file_content(attachment_id=<UUID>)
ШАГ 2: doc_summarize_text(text=<полученный текст>, summary_type=<тип или None>)
  Если тип НЕ указан → передай None, инструмент спросит пользователя.
ЗАПРЕЩЕНО: вызывать doc_summarize_text без текста.
</summarize_guide>""",
    UserIntent.COMPARE: """
<compare_guide>
ЕСТЬ файл → doc_compare_attachment_with_local (только этот инструмент).
НЕТ файла → doc_get_versions (сравнивает все версии автоматически).
</compare_guide>""",
    UserIntent.SEARCH: """
<search_guide>
doc_search_tool → ОБЯЗАТЕЛЬНО выводи таблицу:
| № | id | Рег. номер | Дата | Краткое содержание | Автор | Статус |
Колонку id НИКОГДА не убирать — она нужна для навигации.
</search_guide>""",
    UserIntent.ANALYZE: """
<analyze_guide>
doc_get_details → doc_get_file_content → doc_summarize_text(summary_type='thesis')
</analyze_guide>""",
    UserIntent.FILE_ANALYSIS: """
<file_guide>
read_local_file_content(file_path=<из контекста>) → doc_summarize_text
</file_guide>""",
    UserIntent.CREATE_DOCUMENT: """
<create_doc_guide>
create_document_from_file(file_path=<автоматически>, doc_category=<из запроса>, autofill=True)
Один вызов. Не читай файл заранее.
</create_doc_guide>""",
    UserIntent.COMPLIANCE_CHECK: """
<compliance_guide>
ШАГ 1: doc_compliance_check(check_all=True) — ОДИН РАЗ.
ШАГ 2: СРАЗУ формируй ответ пользователю. НЕ вызывай повторно.
ok → всё совпадает.
has_mismatches → перечисли расхождения (поле: было → стало).
cannot_verify → поля не найдены в файле.
</compliance_guide>""",
    UserIntent.QUESTION: """
<question_guide>
Метаданные: doc_get_details
Содержимое: doc_get_file_content → ответь на основе текста
Сотрудник: employee_search_tool
Без документа: отвечай напрямую
</question_guide>""",
}


# ─────────────────────────────────────────────────────────────────────────────
# ContentExtractor (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────


class ContentExtractor:
    """Extracts final human-readable content from a LangGraph message chain."""

    _SKIP_PATTERNS: tuple[str, ...] = (
        "вызвал инструмент", "tool call", '"name"', '"id"', '"tool_calls"',
    )
    MIN_CONTENT_LENGTH = 30
    _JSON_PRIORITY_FIELDS: tuple[str, ...] = (
        "content", "message", "text", "text_preview", "result",
    )

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
                    data: dict[str, Any] = json.loads(raw)
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
# AgentStateManager
# ─────────────────────────────────────────────────────────────────────────────


class AgentStateManager:
    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
        self.graph = graph
        self.checkpointer = checkpointer

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def get_state(self, thread_id: str) -> Any:
        return await self.graph.aget_state(self._config(thread_id))

    async def update_state(
        self, thread_id: str, messages: list[BaseMessage], as_node: str = "agent"
    ) -> None:
        await self.graph.aupdate_state(
            self._config(thread_id), {"messages": messages}, as_node=as_node
        )

    async def invoke(
        self, inputs: dict[str, Any] | None, thread_id: str, timeout: float = 120.0
    ) -> None:
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
            logger.warning("Thread %s repaired: %d synthetic ToolMessages", thread_id, len(synthetic))
            return True
        except Exception as exc:
            logger.error("Thread repair failed: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Sub-Agents (Supervisor pattern)
# ─────────────────────────────────────────────────────────────────────────────


class SubAgentBase:
    domain: str = "base"

    def __init__(self, llm: Any, tool_names: list[str], all_tools_list: list[Any]) -> None:
        self._tools = [t for t in all_tools_list if getattr(t, "name", None) in set(tool_names)]
        self._model = llm.bind_tools(self._tools) if self._tools else llm

    @property
    def tools(self) -> list[Any]:
        return self._tools

    @property
    def model_with_tools(self) -> Any:
        return self._model


class DocumentSubAgent(SubAgentBase):
    domain = "document"

    def __init__(self, llm: Any, all_tools_list: list[Any]) -> None:
        super().__init__(llm, [
            "doc_get_details", "doc_get_file_content", "doc_get_versions",
            "doc_compare_documents", "doc_compare_attachment_with_local",
            "doc_summarize_text", "read_local_file_content",
            "doc_compliance_check", "doc_update_field",
        ], all_tools_list)


class WorkflowSubAgent(SubAgentBase):
    domain = "workflow"

    def __init__(self, llm: Any, all_tools_list: list[Any]) -> None:
        super().__init__(llm, [
            "task_create_tool", "introduction_create_tool",
            "autofill_appeal_document", "create_document_from_file",
            "doc_send_notification",
        ], all_tools_list)


class SearchSubAgent(SubAgentBase):
    domain = "search"

    def __init__(self, llm: Any, all_tools_list: list[Any]) -> None:
        super().__init__(llm, [
            "doc_search_tool", "employee_search_tool", "doc_get_details",
        ], all_tools_list)


# ─────────────────────────────────────────────────────────────────────────────
# EdmsDocumentAgent — Main Supervisor
# ─────────────────────────────────────────────────────────────────────────────


class EdmsDocumentAgent:
    MAX_ITERATIONS: int = 10
    EXECUTION_TIMEOUT: float = 120.0

    def __init__(
        self,
        config: Any = None,
        document_repo: Any = None,
        semantic_dispatcher: Any = None,
    ):
        self._config = config
        self._enable_guardrails = getattr(config, "enable_guardrails", False)
        self._max_iterations = getattr(config, "max_iterations", self.MAX_ITERATIONS)
        self._execution_timeout = getattr(config, "execution_timeout", self.EXECUTION_TIMEOUT)

        try:
            self._llm = get_chat_model()
            self._all_tools = all_tools
            self._checkpointer = MemorySaver()

            self.document_repo = document_repo or DocumentRepository()
            self.dispatcher = semantic_dispatcher or SemanticDispatcher()

            # Part 3: Sub-agents
            self._document_agent = DocumentSubAgent(self._llm, self._all_tools)
            self._workflow_agent = WorkflowSubAgent(self._llm, self._all_tools)
            self._search_agent = SearchSubAgent(self._llm, self._all_tools)

            # Active model/tools — swapped by _dispatch_to_sub_agent()
            self._active_model = self._llm.bind_tools(self._all_tools)
            self._active_tools = self._all_tools

            # Part 2: Router + Planner
            self._router = AgentRouter(llm=self._llm)
            self._planner = AgentPlanner(llm=self._llm)

            # Part 1: Memory managers
            self._memory_managers: dict[str, Any] = {}

            # Part 5: Optional cross-cutting concerns
            self._guardrail_pipeline = GuardrailPipeline() if _GUARDRAILS_AVAILABLE and GuardrailPipeline else None
            self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0) if _RESILIENCE_AVAILABLE and CircuitBreaker else None
            self._semantic_cache = SemanticCache() if _CACHE_AVAILABLE and SemanticCache else None

            # Response assembler (Part 4 output)
            self._response_assembler = ResponseAssembler(
                guardrail_pipeline=self._guardrail_pipeline,
                enable_guardrails=self._enable_guardrails,
            )

            # Tool binding cache
            self._tool_bindings: dict[str, Any] = {}
            self._tool_call_guard: Any = None

            # LangGraph
            self._compiled_graph = self._build_graph()
            self.state_manager = AgentStateManager(self._compiled_graph, self._checkpointer)

            logger.info(
                "EdmsDocumentAgent initialized (v3.1)",
                extra={"tools": len(self._all_tools)},
            )
        except Exception as exc:
            logger.error("Agent initialization failed", exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {exc}") from exc

    # ── Public API ─────────────────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        return {
            "model": self._llm is not None,
            "tools": len(self._all_tools),
            "graph": self._compiled_graph is not None,
            "sub_agents": {
                "document": len(self._document_agent.tools),
                "workflow": len(self._workflow_agent.tools),
                "search": len(self._search_agent.tools),
            },
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
        """Main entry point for agent interaction."""
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

            # Thread repair
            if await self.state_manager.is_thread_broken(context.thread_id):
                repaired = await self.state_manager.repair_thread(context.thread_id)
                logger.warning("Broken thread %s repaired=%s", context.thread_id, repaired)

            state = await self.state_manager.get_state(context.thread_id)

            # HITL resume from interrupt
            if human_choice and state.next:
                return await self._handle_human_choice(context, human_choice)

            # Document fetch
            document: DocumentDto | None = None
            if context.document_id:
                document = await self.document_repo.get_document(
                    context.user_token, context.document_id
                )

            # Semantic analysis
            semantic_ctx = self.dispatcher.build_context(
                request.message, document, context.file_path
            )
            context.intent = semantic_ctx.query.intent
            logger.info(
                "Intent: %s complexity: %s thread: %s",
                semantic_ctx.query.intent.value,
                semantic_ctx.query.complexity.value,
                context.thread_id,
            )

            # Part 2: Router
            route_result = self._router.route(
                intent=semantic_ctx.query.intent,
                complexity=semantic_ctx.query.complexity,
                has_document_context=bool(context.document_id),
                message=request.message,
            )

            if route_result.decision == RouteDecision.DIRECT_ANSWER and not human_choice:
                return await self._direct_answer(context, request.message)

            # Part 2: Planner
            plan = self._planner.plan(
                intent=semantic_ctx.query.intent,
                has_file=bool(context.file_path),
                has_document=bool(context.document_id),
                message=request.message,
            )
            plan_hint = plan.to_prompt_hint() if plan else ""

            # Part 3: Dispatch sub-agent
            self._dispatch_to_sub_agent(semantic_ctx.query.intent, context)

            # Build system prompt
            system_prompt = PromptBuilder.build(
                context,
                semantic_ctx.query.intent,
                self._build_semantic_xml(semantic_ctx),
                plan_hint=plan_hint,
                lean=USE_LEAN_PROMPT,
            )

            # Part 1: Memory
            memory_mgr = self._get_memory_manager(context.thread_id)
            raw_messages: list[BaseMessage] = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=semantic_ctx.query.refined),
            ]
            if _MEMORY_AVAILABLE and memory_mgr:
                try:
                    prepared = await memory_mgr.prepare(raw_messages, system_prompt)
                    inputs = {"messages": prepared}
                except Exception:
                    inputs = {"messages": raw_messages}
            else:
                inputs = {"messages": raw_messages}

            # Forced tool call bypass
            forced = await self._try_forced_tool_call(context, inputs, request.message)

            return await self._orchestrate(
                context=context,
                inputs=None if forced else inputs,
                is_choice_active=bool(human_choice),
                iteration=0,
            )

        except Exception as exc:
            logger.error("Chat error: %s", exc, exc_info=True)
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {exc}",
            ).model_dump()

    # ── HITL ──────────────────────────────────────────────────────────────────

    async def _handle_human_choice(
        self, context: ContextParams, human_choice: str
    ) -> dict[str, Any]:
        # Compliance field fix
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
            logger.warning("human_choice after graph END — treating as new message")
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
        patched = []
        for tc in raw_calls:
            t_args = dict(tc["args"])
            t_name = tc["name"]

            if t_name == "doc_summarize_text":
                t_args["summary_type"] = human_choice.strip()

            elif t_name in _DISAMBIGUATION_TOOLS:
                raw_ids = [x.strip() for x in human_choice.split(",") if x.strip()]
                valid_ids = []
                for raw_id in raw_ids:
                    try:
                        UUID(raw_id)
                        valid_ids.append(raw_id)
                    except ValueError:
                        logger.warning("Invalid UUID in human_choice: %s", raw_id)
                if valid_ids:
                    t_args["selected_employee_ids"] = valid_ids
                    t_args.pop("last_names", None)
                    t_args.pop("executor_last_names", None)
                    t_args.pop("recipient_last_names", None)

            elif t_name == "doc_compare_attachment_with_local":
                choice = human_choice.strip()
                t_args["attachment_id"] = choice
                if context.document_id:
                    t_args["document_id"] = context.document_id
                if context.uploaded_file_name:
                    t_args["original_filename"] = context.uploaded_file_name

            patched.append({"name": t_name, "args": t_args, "id": tc["id"]})
        return patched

    # ── Direct answer ──────────────────────────────────────────────────────────

    async def _direct_answer(self, context: ContextParams, message: str) -> dict[str, Any]:
        try:
            messages = [
                SystemMessage(content=(
                    f"Ты — ИИ-помощник EDMS. "
                    f"Пользователь: {context.user_name}. "
                    f"Дата: {context.current_date}. "
                    f"Отвечай кратко и по-русски."
                )),
                HumanMessage(content=message),
            ]
            response = await self._llm.ainvoke(messages)
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                content=str(response.content).strip(),
            ).model_dump()
        except Exception as exc:
            return AgentResponse(status=AgentStatus.ERROR, message=str(exc)).model_dump()

    # ── Core orchestration ─────────────────────────────────────────────────────

    async def _orchestrate(
        self,
        context: ContextParams,
        inputs: dict[str, Any] | None,
        is_choice_active: bool,
        iteration: int,
    ) -> dict[str, Any]:
        if iteration > self._max_iterations:
            return AgentResponse(status=AgentStatus.ERROR, message="Превышен лимит итераций.").model_dump()

        # Minimal toolset on first iteration
        if iteration == 0:
            selected_tools = get_tools_for_intent(
                context.intent or UserIntent.UNKNOWN,
                self._active_tools,
            )
            cache_key = ",".join(sorted(getattr(t, "name", "") for t in selected_tools))
            if cache_key not in self._tool_bindings:
                self._tool_bindings[cache_key] = self._llm.bind_tools(selected_tools)
                logger.info(
                    "Tool binding: intent=%s tools=%d (~%d tokens)",
                    getattr(context.intent, "value", "?"),
                    len(selected_tools),
                    estimate_tools_tokens(selected_tools),
                )
            self._active_model = self._tool_bindings[cache_key]
            self._active_tools = selected_tools

        try:
            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self._execution_timeout,
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
                result = self._response_assembler.assemble(messages, context)
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                return result

            # Process tool calls
            if not last_has_calls:
                # No tool calls but graph not finished — assemble what we have
                return self._response_assembler.assemble(messages, context)

            raw_calls = list(getattr(last_msg, "tool_calls", []))

            # Block parallel tool calls — keep only first
            if len(raw_calls) > 1:
                logger.warning(
                    "Parallel calls blocked: keeping %s, dropping %s",
                    raw_calls[0]["name"],
                    [tc["name"] for tc in raw_calls[1:]],
                )
                raw_calls = raw_calls[:1]

            last_tool_text = ContentExtractor.extract_last_tool_text(messages)

            # Detect if we're after a compare disambiguation
            _after_compare_disambiguation = self._detect_after_compare_disambiguation(messages)

            patched_calls = []
            for tc in raw_calls:
                t_name = tc["name"]
                t_args = dict(tc["args"])
                t_id = tc["id"]

                patched = self._patch_tool_call(
                    t_name=t_name,
                    t_args=t_args,
                    t_id=t_id,
                    context=context,
                    messages=messages,
                    last_tool_text=last_tool_text,
                    is_choice_active=is_choice_active,
                    after_compare_disambiguation=_after_compare_disambiguation,
                )
                patched_calls.append(patched)

            await self.state_manager.update_state(
                context.thread_id,
                [AIMessage(
                    content=last_msg.content or "",
                    tool_calls=patched_calls,
                    id=last_msg.id,
                )],
                as_node="agent",
            )

            # Propagate is_choice_active correctly
            next_choice_active = is_choice_active
            if is_choice_active and patched_calls:
                last_tool_name = patched_calls[-1]["name"]
                next_choice_active = last_tool_name in (
                    "doc_compare_attachment_with_local",
                    "doc_summarize_text",
                )

            return await self._orchestrate(
                context=context,
                inputs=None,
                is_choice_active=next_choice_active,
                iteration=iteration + 1,
            )

        except TimeoutError:
            return AgentResponse(status=AgentStatus.ERROR, message="Превышено время ожидания.").model_dump()

        except Exception as exc:
            err_str = str(exc)
            logger.error("Orchestration error iter=%d: %s", iteration, exc, exc_info=True)

            _broken_signals = (
                "tool_calls must be followed by tool messages",
                "tool_call_ids did not have response messages",
                "invalid_request_error",
            )
            if any(s in err_str for s in _broken_signals) and iteration == 0:
                repaired = await self.state_manager.repair_thread(context.thread_id)
                if repaired:
                    return AgentResponse(
                        status=AgentStatus.ERROR,
                        message="Предыдущий запрос завершился некорректно. Повторите вопрос.",
                    ).model_dump()

            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            return AgentResponse(status=AgentStatus.ERROR, message=f"Ошибка: {exc}").model_dump()

    # ── Tool call patching (inlined from ToolCallPatchPipeline) ───────────────

    def _patch_tool_call(
        self,
        t_name: str,
        t_args: dict,
        t_id: str,
        context: ContextParams,
        messages: list[BaseMessage],
        last_tool_text: str | None,
        is_choice_active: bool,
        after_compare_disambiguation: bool,
    ) -> dict:
        """Apply all patching rules to a single tool call."""
        clean_path = str(context.file_path or "").strip()
        path_is_uuid = _is_valid_uuid(clean_path) if clean_path else False
        path_is_local = bool(clean_path) and not path_is_uuid

        # 1. Inject auth token
        t_args["token"] = context.user_token

        # 2. create_document_from_file special injection
        if t_name == "create_document_from_file":
            if not t_args.get("doc_category"):
                from edms_ai_assistant.tools.create_document_from_file import _extract_category_from_message
                for _m in reversed(messages):
                    if isinstance(_m, HumanMessage):
                        detected = _extract_category_from_message(str(_m.content))
                        if detected:
                            t_args["doc_category"] = detected
                        break
            if not t_args.get("file_path") and path_is_local:
                t_args["file_path"] = clean_path
            if not t_args.get("file_name") and context.uploaded_file_name:
                t_args["file_name"] = context.uploaded_file_name

        # 3. Inject document_id
        if context.document_id and t_name in _TOOLS_REQUIRING_DOCUMENT_ID:
            cur = str(t_args.get("document_id", "")).strip()
            if not cur or not _is_valid_uuid(cur):
                t_args["document_id"] = context.document_id

        # 4. Route file references
        if path_is_local:
            if t_name == "doc_get_versions":
                t_name = "doc_compare_attachment_with_local"
                t_args = {"local_file_path": clean_path}
                if context.document_id:
                    t_args["document_id"] = context.document_id
                logger.warning("GUARD: doc_get_versions → doc_compare_attachment_with_local (local file present)")

            elif t_name == "doc_compare_documents":
                t_name = "doc_compare_attachment_with_local"
                t_args["local_file_path"] = clean_path
                t_args.pop("document_id_1", None)
                t_args.pop("document_id_2", None)

            elif t_name == "doc_get_file_content" and not t_args.get("attachment_id"):
                t_name = "read_local_file_content"
                t_args["file_path"] = clean_path
                t_args.pop("attachment_id", None)

        elif path_is_uuid:
            if t_name == "read_local_file_content":
                t_name = "doc_get_file_content"
                t_args["attachment_id"] = clean_path
                t_args.pop("file_path", None)
            elif t_name == "doc_get_file_content":
                cur = str(t_args.get("attachment_id", "")).strip()
                if not cur or not _is_valid_uuid(cur):
                    t_args["attachment_id"] = clean_path

        # 5. Fallback: compare_with_local without file → version compare
        if (
            t_name == "doc_compare_attachment_with_local"
            and not clean_path
            and not after_compare_disambiguation
            and not (is_choice_active and t_args.get("attachment_id"))
        ):
            t_name = "doc_compare_documents"
            t_args.pop("local_file_path", None)
            t_args.pop("attachment_id", None)

        # 6. Force-inject local_file_path for compare_with_local
        if t_name == "doc_compare_attachment_with_local" and path_is_local:
            cur_local = str(t_args.get("local_file_path", "")).strip()
            if not cur_local or cur_local.lower() in _COMPARE_LOCAL_PLACEHOLDERS or not Path(cur_local).exists():
                t_args["local_file_path"] = clean_path
            if context.uploaded_file_name and not t_args.get("original_filename"):
                t_args["original_filename"] = context.uploaded_file_name

        # 7. Block doc_compare_documents after disambiguation
        if t_name == "doc_compare_documents" and (after_compare_disambiguation or (is_choice_active and path_is_local)):
            t_name = "doc_compare_attachment_with_local"
            t_args = {
                "token": context.user_token,
                "document_id": context.document_id,
                "local_file_path": clean_path if path_is_local else t_args.get("local_file_path"),
                "attachment_id": t_args.get("document_id_2") or t_args.get("attachment_id"),
                "original_filename": context.uploaded_file_name,
            }

        # 8. Fix placeholder file_path for read_local_file_content
        if path_is_local and t_name == "read_local_file_content":
            cur_fp = str(t_args.get("file_path", "")).strip()
            if not cur_fp or cur_fp.lower() in ("local_file", "file_path", "none", "null", ""):
                t_args["file_path"] = clean_path

        # 9. Inject text for doc_summarize_text + summary_type fallback
        if t_name == "doc_summarize_text":
            if last_tool_text:
                t_args["text"] = last_tool_text
            if not t_args.get("summary_type"):
                pref = context.user_context.get("preferred_summary_format", "")
                if is_choice_active:
                    t_args["summary_type"] = "extractive"
                    logger.warning("safety-net: summary_type=extractive (is_choice_active)")
                elif pref and pref != "ask":
                    t_args["summary_type"] = pref

        # Re-inject document_id after possible rename
        if context.document_id and t_name in _TOOLS_REQUIRING_DOCUMENT_ID:
            cur = str(t_args.get("document_id", "")).strip()
            if not cur or not _is_valid_uuid(cur):
                t_args["document_id"] = context.document_id

        return {"name": t_name, "args": t_args, "id": t_id}

    @staticmethod
    def _detect_after_compare_disambiguation(messages: list[BaseMessage]) -> bool:
        for prev_msg in reversed(messages[-15:]):
            if isinstance(prev_msg, ToolMessage):
                try:
                    data = json.loads(str(prev_msg.content))
                    if (
                        data.get("status") == "requires_disambiguation"
                        and prev_msg.name == "doc_compare_attachment_with_local"
                    ):
                        return True
                except (json.JSONDecodeError, AttributeError):
                    pass
            if isinstance(prev_msg, HumanMessage):
                break
        return False

    # ── Graph compilation ──────────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)

        async def call_model(state: AgentState) -> dict[str, Any]:
            sys_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
            non_sys = [m for m in state["messages"] if not isinstance(m, SystemMessage)]

            if len(non_sys) > MAX_WORKING_MESSAGES:
                non_sys = non_sys[-MAX_WORKING_MESSAGES:]

            # Inject compliance-fix hint when relevant
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    raw = str(msg.content)
                    if raw.startswith("{"):
                        try:
                            data = json.loads(raw)
                            if data.get("status") == "success" and "fields" in data:
                                sys_msgs.append(SystemMessage(content=(
                                    "ВНИМАНИЕ: в истории есть результат compliance check. "
                                    "При запросе на исправление — используй correct_value "
                                    "из результатов и вызывай doc_update_field."
                                )))
                        except json.JSONDecodeError:
                            pass
                    break

            # Sanitize dangling AIMessages
            candidate_msgs = sys_msgs + non_sys
            final_msgs: list[BaseMessage] = []
            for i, msg in enumerate(candidate_msgs):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    nxt = candidate_msgs[i + 1] if i + 1 < len(candidate_msgs) else None
                    if not isinstance(nxt, ToolMessage):
                        final_msgs.append(AIMessage(content=msg.content or "", id=msg.id))
                        continue
                final_msgs.append(msg)

            response = await self._active_model.ainvoke(final_msgs)
            return {"messages": [response]}

        async def validator(state: AgentState) -> dict[str, Any]:
            """
            Post-tool validator.

            BUG-4 FIX: Do NOT stop graph on plain success statuses.
            Only stop for:
            - compliance results (status=success + fields list)
            - interactive statuses (requires_choice / requires_disambiguation)
            - empty tool results

            The graph must continue to call_model so the LLM can formulate
            a proper response for the user (e.g. after doc_summarize_text).
            """
            last = state["messages"][-1]
            if not isinstance(last, ToolMessage):
                return {"messages": []}

            raw = str(last.content).strip()

            # Empty result
            if not raw or raw in ("None", "{}", "null"):
                return {"messages": [AIMessage(content=(
                    "⚠️ Инструмент вернул пустой результат. Попробуй другой подход."
                ))]}

            if raw.startswith("{"):
                try:
                    tool_data = json.loads(raw)
                    interactive_status = tool_data.get("status", "")

                    # Compliance result — stop and let ResponseAssembler handle it
                    if (
                        interactive_status == "success"
                        and isinstance(tool_data.get("fields"), list)
                        and "overall" in tool_data
                    ):
                        logger.info("Validator: compliance result → stopping graph")
                        return {"messages": [AIMessage(
                            content="Проверка соответствия завершена. Результаты готовы."
                        )]}

                    # Interactive statuses — stop graph so assembler detects them
                    if interactive_status in (
                        "requires_choice", "requires_disambiguation", "requires_action"
                    ):
                        logger.info("Validator: interactive status '%s' → stopping graph", interactive_status)
                        return {"messages": [AIMessage(content="")]}

                    # NOTE: Do NOT stop on plain success — let LLM formulate response

                except json.JSONDecodeError:
                    pass

            return {"messages": []}

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

    # ── Forced tool call ───────────────────────────────────────────────────────

    async def _try_forced_tool_call(
        self, context: ContextParams, inputs: dict, original_message: str
    ) -> bool:
        import uuid as _uuid

        try:
            from edms_ai_assistant.tools.create_document_from_file import _extract_category_from_message
        except ImportError:
            return False

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

    # ── Sub-agent dispatch ────────────────────────────────────────────────────

    def _dispatch_to_sub_agent(self, intent: UserIntent, context: ContextParams) -> None:
        """Manager decides which sub-agent handles this intent."""
        workflow_intents = {
            UserIntent.CREATE_TASK, UserIntent.CREATE_INTRODUCTION,
            UserIntent.CREATE_DOCUMENT,
        }
        search_intents = {UserIntent.SEARCH}

        if intent in workflow_intents:
            sub = self._workflow_agent
        elif intent in search_intents:
            sub = self._search_agent
        else:
            sub = self._document_agent

        self._active_model = sub.model_with_tools
        self._active_tools = sub.tools

    # ── Memory ────────────────────────────────────────────────────────────────

    def _get_memory_manager(self, thread_id: str) -> Any | None:
        if not _MEMORY_AVAILABLE or not ConversationMemoryManager:
            return None
        if thread_id not in self._memory_managers:
            self._memory_managers[thread_id] = ConversationMemoryManager(llm=self._llm)
        return self._memory_managers[thread_id]

    # ── Context builder ───────────────────────────────────────────────────────

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