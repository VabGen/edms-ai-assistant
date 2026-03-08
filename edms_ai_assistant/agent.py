# edms_ai_assistant/agent.py
"""
EDMS AI Assistant — Core Agent Module.

Production-ready LangGraph agent for Electronic Document Management System.
Implements multi-node graph: agent → tools → validator → agent.

Architecture:
    - EdmsDocumentAgent: Top-level orchestrator (entry point)
    - AgentStateManager: LangGraph state/checkpoint wrapper
    - PromptBuilder: Strategy pattern for dynamic system-prompt construction
    - ContentExtractor: Utilities for parsing final answer from message chain
    - ToolCallPatcher: Centralized tool-argument injection (token, document_id, etc.)
    - DisambiguationContext: Pending disambiguation state between turns

NOTE: ``from __future__ import annotations`` (PEP-563) is intentionally absent.
LangGraph calls typing.get_type_hints() on node functions at graph compile time.
With PEP-563 lazy string annotations, AgentState TypedDict cannot be resolved
in get_type_hints() global namespace, causing NameError at startup.
"""

import asyncio
import json
import logging
import re
import uuid as _uuid_module
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.services.nlp_service import SemanticDispatcher, UserIntent
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AgentState — ОБЯЗАТЕЛЬНО на уровне модуля.
#
# Причина: LangGraph вызывает typing.get_type_hints(should_continue) при
# add_conditional_edges(). Этот вызов разрешает аннотации в ГЛОБАЛЬНОМ
# namespace модуля. Если AgentState определён локально внутри _build_graph(),
# он там виден, но НЕ виден в get_type_hints() → NameError при старте.
#
# ВАЖНО: только поле messages с add_messages. Добавление extra-полей в
# TypedDict ломает LangGraph checkpointer если они не умеют сериализоваться.
# pending_disambiguation храним в отдельном in-memory словаре (см. ниже).
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """LangGraph state schema: накопленная цепочка сообщений."""

    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# UUID-паттерн (скомпилирован один раз, используется в нескольких местах)
# ---------------------------------------------------------------------------
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Перечисления
# ---------------------------------------------------------------------------


class AgentStatus(str, Enum):
    """Статусы обработки запроса агентом."""

    SUCCESS = "success"
    ERROR = "error"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"


class ActionType(str, Enum):
    """Типы интерактивных действий, требующих участия пользователя."""

    SUMMARIZE_SELECTION = "summarize_selection"
    DISAMBIGUATION = "requires_disambiguation"
    CONFIRMATION = "requires_confirmation"


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextParams:
    """Immutable контекст для выполнения агента.

    Attributes:
        user_token: Bearer-токен текущего пользователя.
        document_id: UUID активного документа в EDMS (опционально).
        file_path: Путь к файлу или UUID вложения (опционально).
        thread_id: Идентификатор диалоговой сессии LangGraph.
        user_name: Отображаемое имя пользователя.
        user_first_name: Имя пользователя (опционально).
        current_date: Текущая дата в формате ДД.ММ.ГГГГ.
    """

    user_token: str
    document_id: Optional[str] = None
    file_path: Optional[str] = None
    thread_id: str = "default"
    user_name: str = "пользователь"
    user_first_name: Optional[str] = None
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%d.%m.%Y")
    )

    def __post_init__(self) -> None:
        """Validate required fields."""
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")


class AgentRequest(BaseModel):
    """Валидированный входящий запрос к агенту."""

    message: str = Field(..., min_length=1, max_length=8000)
    user_token: str = Field(..., min_length=20)
    context_ui_id: Optional[str] = Field(None, pattern=r"^[0-9a-f-]{36}$|^$")
    thread_id: Optional[str] = Field(None, max_length=255)
    user_context: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = Field(None, max_length=500)
    human_choice: Optional[str] = Field(None, max_length=100)

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Strip leading/trailing whitespace from message."""
        return v.strip()

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate file_path as UUID or filesystem path.

        Args:
            v: Raw file_path value.

        Returns:
            Validated path string or None.

        Raises:
            ValueError: If format is unrecognized.
        """
        if not v:
            return None

        if re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            v,
            re.I,
        ):
            return v

        if len(v) < 500:
            if v.startswith("/"):
                return v
            if re.match(r"^[A-Za-z]:\\", v):
                return v
            if re.match(r"^[^/\\]+[\\/]", v):
                return v

        raise ValueError(f"Invalid file_path format: {v!r}")


class AgentResponse(BaseModel):
    """Стандартизированный ответ агента."""

    status: AgentStatus
    content: Optional[str] = None
    message: Optional[str] = None
    action_type: Optional[ActionType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Репозиторий документов
# ---------------------------------------------------------------------------


class IDocumentRepository(Protocol):
    """Интерфейс для работы с документами (Dependency Inversion)."""

    async def get_document(self, token: str, doc_id: str) -> Optional[DocumentDto]:
        """Получить метаданные документа."""
        ...


class DocumentRepository:
    """Реализация репозитория документов."""

    async def get_document(self, token: str, doc_id: str) -> Optional[DocumentDto]:
        """Fetch document metadata from EDMS API.

        Args:
            token: Bearer auth token.
            doc_id: Document UUID.

        Returns:
            DocumentDto if found, None on error.
        """
        try:
            async with DocumentClient() as client:
                raw_data = await client.get_document_metadata(token, doc_id)
                doc = DocumentDto.model_validate(raw_data)
                logger.info("Document fetched", extra={"doc_id": doc_id})
                return doc
        except Exception as e:
            logger.error(
                "Failed to fetch document",
                exc_info=True,
                extra={"doc_id": doc_id, "error": str(e)},
            )
            return None


# ---------------------------------------------------------------------------
# DisambiguationContext — in-memory хранилище pending-выборов
#
# Намеренно НЕ помещается в AgentState/LangGraph checkpointer:
#  - LangGraph MemorySaver сериализует state через Pydantic/json.
#    Произвольные dict-поля могут сломать snapshot.
#  - Disambiguation — короткоживущий контекст (один turn).
#  - Храним в dict[thread_id → pending_ctx] на уровне агента.
# ---------------------------------------------------------------------------


class DisambiguationContext:
    """In-memory registry of pending disambiguation states per thread.

    Thread-safe for single-process async usage (asyncio event loop).
    """

    def __init__(self) -> None:
        """Initialize empty context store."""
        self._store: Dict[str, Dict[str, Any]] = {}

    def save(self, thread_id: str, ctx: Dict[str, Any]) -> None:
        """Persist disambiguation context for a thread.

        Args:
            thread_id: LangGraph thread identifier.
            ctx: Context dict with tool_name, tool_args, ambiguous_matches.
        """
        self._store[thread_id] = ctx
        logger.info(
            "Saved disambiguation context",
            extra={
                "thread_id": thread_id,
                "tool": ctx.get("tool_name"),
                "groups": len(ctx.get("ambiguous_matches", [])),
            },
        )

    def get(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve pending disambiguation context.

        Args:
            thread_id: LangGraph thread identifier.

        Returns:
            Context dict or None if no pending disambiguation.
        """
        return self._store.get(thread_id)

    def clear(self, thread_id: str) -> None:
        """Remove disambiguation context after resolution.

        Args:
            thread_id: LangGraph thread identifier.
        """
        self._store.pop(thread_id, None)


# ---------------------------------------------------------------------------
# DisambiguationResolver — резолвит выбор пользователя → UUID
# ---------------------------------------------------------------------------


class DisambiguationResolver:
    """Resolves user's free-text choice to employee UUID(s).

    Supports three strategies (in priority order):
    1. Direct UUID → accept immediately.
    2. Integer N → pick N-th candidate from flattened list (1-based).
    3. Name substring match (case-insensitive).
    """

    @staticmethod
    def resolve(
        user_message: str,
        pending: Dict[str, Any],
    ) -> Optional[List[str]]:
        """Resolve user choice to employee UUID list.

        Supports two ``ambiguous_matches`` formats:

        - **Grouped** (task_create_tool):
          ``[{"search_query": "...", "matches": [{id, full_name, ...}]}, ...]``
        - **Flat** (introduction_create_tool):
          ``[{"id": "...", "full_name": "...", "search_term": "..."}, ...]``

        Args:
            user_message: Raw text the user sent after seeing the list.
            pending: Pending disambiguation context dict.

        Returns:
            List of UUID strings, or None if cannot resolve.
        """
        text = user_message.strip()
        groups: List[Dict[str, Any]] = pending.get("ambiguous_matches", [])

        if not groups:
            return None

        # Нормализуем в плоский список кандидатов независимо от формата
        all_candidates: List[Dict[str, Any]] = []
        first = groups[0]
        if "matches" in first:
            # Grouped format
            for g in groups:
                all_candidates.extend(g.get("matches", []))
        else:
            # Flat format — каждый элемент уже является кандидатом
            all_candidates = groups

        if not all_candidates:
            return None

        # 1. Прямой UUID
        if _UUID_RE.match(text):
            for c in all_candidates:
                if c.get("id", "").lower() == text.lower():
                    return [c["id"]]
            return [text]

        # 2. Номер из списка (1-based)
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(all_candidates):
                chosen = all_candidates[idx]
                logger.info(
                    "Disambiguation resolved by index %d → %s (%s)",
                    idx + 1,
                    chosen.get("full_name"),
                    chosen.get("id"),
                )
                return [chosen["id"]]
            return None

        # 3. Частичное совпадение по ФИО
        text_lower = text.lower()
        matched = [
            c for c in all_candidates
            if text_lower in c.get("full_name", "").lower()
        ]
        if matched:
            if len(matched) > 1:
                logger.warning(
                    "Disambiguation name match ambiguous ('%s' → %d hits), picking first",
                    text,
                    len(matched),
                )
            logger.info(
                "Disambiguation resolved by name '%s' → %s (%s)",
                text,
                matched[0].get("full_name"),
                matched[0].get("id"),
            )
            return [matched[0]["id"]]

        return None

    @staticmethod
    def is_likely_selection(user_message: str) -> bool:
        """Heuristically check if message looks like a list selection.

        Args:
            user_message: Raw user input.

        Returns:
            True if input looks like a number, UUID, or short name-answer.
        """
        text = user_message.strip()
        if text.isdigit():
            return True
        if _UUID_RE.match(text):
            return True
        # Короткое сообщение без вопросительного знака — вероятно ФИО
        if len(text) <= 80 and "?" not in text:
            return True
        return False


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Strategy для построения системных промптов с динамическим контекстом."""

    CORE_TEMPLATE = """<role>
Ты — экспертный помощник системы электронного документооборота (EDMS/СЭД).
Помогаешь с анализом документов, управлением персоналом и делегированием задач.
</role>

<context>
- Пользователь: {user_name}
- Текущая дата: {current_date}
- Активный документ: {context_ui_id}
- Загруженный файл: {local_file}
</context>

<critical_rules>
1. **Автоинъекция**: Параметры `token` и `document_id` добавляются АВТОМАТИЧЕСКИ системой. Не указывай их явно.

2. **Обработка LOCAL_FILE**:
   - UUID формат (например: 0c2216e1-...) → Вызови `doc_get_file_content(attachment_id=LOCAL_FILE)`
   - Путь к файлу (/tmp/...) → Вызови `read_local_file_content(file_path=LOCAL_FILE)`
   - Пустое значение ("Не загружен") → Вызови `doc_get_details()` для поиска вложений

3. **Обработка requires_action**:
   - Статус "requires_choice" → Предложи пользователю выбрать формат анализа (факты/пересказ/тезисы)
   - Статус "requires_disambiguation" → Покажи список кандидатов пользователю, ОЖИДАЙ его выбора. НЕ вызывай инструмент повторно сам.

4. **ВАЖНО**: После вызова инструментов ВСЕГДА формулируй финальный ответ на русском языке.
   Если инструмент вернул status=success — опиши результат пользователю.
   Если инструмент вернул requires_disambiguation — ТОЛЬКО покажи список, не создавай запись.

5. **Язык**: Только русский. Обращайся к пользователю по имени: {user_name}

6. **Сравнение версий**: Сначала вызови `doc_get_versions`, затем передай РАЗНЫЕ UUID версий в `doc_compare`.
   Поля document_id_1 и document_id_2 должны быть UUID из поля "id" каждой версии.
</critical_rules>

<tool_selection>
**Типичные сценарии**:
- Анализ документа: doc_get_details → doc_get_file_content → doc_summarize_text
- Анализ файла (UUID): doc_get_file_content → doc_summarize_text
- Поиск сотрудника: employee_search_tool
- Список ознакомления: introduction_create_tool
- Создание поручения: task_create_tool
- Создание обращения: autofill_appeal_document
</tool_selection>

<response_format>
✅ Структурировано, кратко, по делу
❌ Многословие, технические детали API, сырые JSON-данные
</response_format>"""

    CONTEXT_SNIPPETS: Dict[Any, str] = {
        UserIntent.CREATE_INTRODUCTION: """
<introduction_guide>
При создании списка ознакомления:
- Вызови introduction_create_tool с last_names
- Если статус "requires_disambiguation" → ТОЛЬКО покажи список найденных сотрудников пользователю
- После выбора пользователя система автоматически вызовет инструмент с selected_employee_ids
- НЕ пытайся угадать выбор или вызвать инструмент повторно самостоятельно
</introduction_guide>""",
        UserIntent.CREATE_TASK: """
<task_guide>
При создании поручения:
- executor_last_names: обязательно (минимум 1)
- responsible_last_name: опционально
- planed_date_end: опционально, формат ISO 8601 с timezone: "2026-02-15T23:59:59Z"
- Если статус "requires_disambiguation" → ТОЛЬКО покажи список, жди выбора пользователя
- После выбора пользователя система сама вызовет инструмент с selected_employee_ids
</task_guide>""",
        UserIntent.SUMMARIZE: """
<summarize_guide>
При суммаризации документа:
- Сначала получи текст через doc_get_file_content или doc_get_details
- Затем вызови doc_summarize_text(text=..., summary_type=None) — система спросит формат
- Если пользователь уже указал формат (факты/пересказ/тезисы) — передай его напрямую
</summarize_guide>""",
    }

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        semantic_xml: str,
    ) -> str:
        """Build full system prompt for the given context and intent.

        Args:
            context: Immutable execution context.
            intent: Detected user intent for dynamic snippet injection.
            semantic_xml: XML block from semantic analysis.

        Returns:
            Full system prompt string.
        """
        base_prompt = cls.CORE_TEMPLATE.format(
            user_name=context.user_name,
            current_date=context.current_date,
            context_ui_id=context.document_id or "Не указан",
            local_file=context.file_path or "Не загружен",
        )

        dynamic_context = cls.CONTEXT_SNIPPETS.get(intent, "")
        return base_prompt + dynamic_context + semantic_xml


# ---------------------------------------------------------------------------
# ContentExtractor
# ---------------------------------------------------------------------------


class ContentExtractor:
    """Извлечение финального контента из цепочки сообщений."""

    # Паттерны контента, который НЕ нужно показывать пользователю
    SKIP_PATTERNS = [
        "вызвал инструмент",
        "tool call",
        '"name"',
        '"id"',
    ]

    # Технические JSON-поля — признак сырого ответа инструмента
    TECHNICAL_JSON_KEYS = {
        "ambiguous_matches",
        "document_id_1",
        "document_id_2",
        "differences",
        "tool_name",
        "tool_args",
    }

    MIN_CONTENT_LENGTH = 10
    JSON_FIELDS = ["content", "text", "text_preview", "message", "summary"]

    @classmethod
    def extract_final_content(cls, messages: List[BaseMessage]) -> Optional[str]:
        """Extract best human-readable content from message chain.

        Priority: AIMessage text → ToolMessage JSON fields → fallbacks.
        Skips raw technical JSON that should not be shown to users.

        Args:
            messages: Full message chain from LangGraph state.

        Returns:
            Best human-readable string, or None if nothing found.
        """
        # 1. Первый приоритет — осмысленный AIMessage
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if cls._is_skip_content(content):
                    continue
                if len(content) > cls.MIN_CONTENT_LENGTH:
                    logger.debug("Extracted AIMessage: %d chars", len(content))
                    return content

        # 2. Второй приоритет — осмысленные поля ToolMessage JSON
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._extract_from_tool_message(m)
                if extracted:
                    return extracted

        # 3. Fallback — любой AIMessage с контентом
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if content:
                    return content

        return None

    @classmethod
    def extract_last_text(cls, messages: List[BaseMessage]) -> Optional[str]:
        """Extract last substantial text from ToolMessage results.

        Used to provide text content for doc_summarize_text auto-injection.

        Args:
            messages: Full message chain.

        Returns:
            Text string suitable for summarisation, or None.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                raw = str(m.content or "")
                if raw.startswith("{"):
                    data = json.loads(raw)
                    text = (
                        data.get("content")
                        or data.get("text_preview")
                        or data.get("text")
                    )
                    if text and len(str(text)) > 100:
                        return str(text)
                if len(raw) > 100:
                    return raw
            except (json.JSONDecodeError, TypeError):
                raw = str(m.content or "")
                if len(raw) > 100:
                    return raw
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """Remove raw JSON wrapper artifacts from content string.

        Args:
            content: Potentially JSON-wrapped content string.

        Returns:
            Clean human-readable string.
        """
        if content.startswith('{"status"'):
            try:
                data = json.loads(content)
                if "content" in data:
                    content = data["content"]
                elif "message" in data:
                    content = data["message"]
            except (json.JSONDecodeError, TypeError):
                pass

        content = content.replace('{"status": "success", "content": "', "")
        content = content.replace('"}', "")
        content = content.replace('\\"', '"')
        content = content.replace("\\n", "\n")
        return content.strip()

    @classmethod
    def _is_skip_content(cls, content: str) -> bool:
        """Check if content should be skipped (technical noise).

        Args:
            content: Content string to check.

        Returns:
            True if content should be skipped.
        """
        lower = content.lower()
        return any(skip in lower for skip in cls.SKIP_PATTERNS)

    @classmethod
    def _is_raw_technical_json(cls, data: Dict[str, Any]) -> bool:
        """Check if parsed JSON dict looks like raw tool output.

        Args:
            data: Parsed JSON dict.

        Returns:
            True if it contains technical keys that should not be shown.
        """
        return bool(cls.TECHNICAL_JSON_KEYS & set(data.keys()))

    @classmethod
    def _extract_from_tool_message(cls, message: ToolMessage) -> Optional[str]:
        """Extract human-readable value from ToolMessage JSON content.

        Args:
            message: ToolMessage from LangGraph state.

        Returns:
            Extracted string or None.
        """
        try:
            raw = str(message.content or "").strip()
            if not raw.startswith("{"):
                return None
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None

            # Пропускаем сырой технический JSON
            if cls._is_raw_technical_json(data):
                logger.debug("Skipping raw technical JSON from ToolMessage")
                return None

            for field_name in cls.JSON_FIELDS:
                val = data.get(field_name)
                if val and len(str(val)) >= cls.MIN_CONTENT_LENGTH:
                    logger.debug(
                        "ToolMessage JSON[%s]: %d chars", field_name, len(str(val))
                    )
                    return str(val).strip()
        except (json.JSONDecodeError, TypeError):
            pass
        return None


# ---------------------------------------------------------------------------
# AgentStateManager
# ---------------------------------------------------------------------------


class AgentStateManager:
    """Управление состоянием LangGraph агента.

    Wrapper вокруг CompiledStateGraph для изоляции LangGraph API.
    """

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
        """Initialize state manager.

        Args:
            graph: Compiled LangGraph state graph.
            checkpointer: MemorySaver instance for state persistence.

        Raises:
            ValueError: If graph or checkpointer is None.
        """
        if graph is None:
            raise ValueError("Graph cannot be None")
        if checkpointer is None:
            raise ValueError("Checkpointer cannot be None")

        self.graph = graph
        self.checkpointer = checkpointer

    async def get_state(self, thread_id: str) -> Any:
        """Get current graph state snapshot.

        Args:
            thread_id: LangGraph thread identifier.

        Returns:
            StateSnapshot object.
        """
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aget_state(config)

    async def update_state(
        self,
        thread_id: str,
        messages: List[BaseMessage],
        as_node: str = "agent",
    ) -> None:
        """Update graph state with new messages.

        Args:
            thread_id: LangGraph thread identifier.
            messages: Messages to inject into state.
            as_node: Node name for state update attribution.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await self.graph.aupdate_state(
            config, {"messages": messages}, as_node=as_node
        )

    async def invoke(
        self,
        inputs: Optional[Dict[str, Any]],
        thread_id: str,
        timeout: float = 120.0,
    ) -> None:
        """Invoke graph with timeout protection.

        Args:
            inputs: Input dict or None (resume from checkpoint).
            thread_id: LangGraph thread identifier.
            timeout: Max seconds to wait.

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await asyncio.wait_for(
            self.graph.ainvoke(inputs, config=config), timeout=timeout
        )


# ---------------------------------------------------------------------------
# EdmsDocumentAgent — главный оркестратор
# ---------------------------------------------------------------------------


class EdmsDocumentAgent:
    """Production-ready мультиагентная система для EDMS.

    Граф выполнения:
        START → agent → [interrupt_before tools] → tools → validator → agent → END

    Основные возможности:
        - Автоинъекция токена и document_id в tool_calls.
        - Семантический анализ запроса перед LLM-вызовом.
        - Защита от бесконечных циклов (MAX_ITERATIONS).
        - Обработка human_choice (выбор типа суммаризации).
        - Обработка disambiguation (выбор сотрудника из списка).
        - Структурированные логи для каждого шага.
    """

    MAX_ITERATIONS: int = 10
    EXECUTION_TIMEOUT: float = 120.0

    def __init__(
        self,
        document_repo: Optional[IDocumentRepository] = None,
        semantic_dispatcher: Optional[SemanticDispatcher] = None,
    ) -> None:
        """Initialize agent with all components.

        Args:
            document_repo: Document repository implementation (DI).
            semantic_dispatcher: NLP dispatcher for intent detection (DI).

        Raises:
            RuntimeError: If initialization of any component fails.
        """
        try:
            self.model = get_chat_model()
            self.tools = all_tools
            self.document_repo = document_repo or DocumentRepository()
            self.dispatcher = semantic_dispatcher or SemanticDispatcher()

            self._checkpointer = MemorySaver()

            # In-memory disambiguation store (не в LangGraph state)
            self._dis_ctx = DisambiguationContext()

            self._compiled_graph = self._build_graph()
            if self._compiled_graph is None:
                raise RuntimeError("Graph compilation returned None")

            self.state_manager = AgentStateManager(
                graph=self._compiled_graph,
                checkpointer=self._checkpointer,
            )

            logger.info(
                "EdmsDocumentAgent initialized successfully",
                extra={"tools_count": len(self.tools)},
            )

        except Exception as e:
            logger.error("Failed to initialize EdmsDocumentAgent", exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """Return component health status dict.

        Returns:
            Dict with boolean health flags per component.
        """
        return {
            "model": self.model is not None,
            "tools": len(self.tools) > 0,
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": hasattr(self, "_compiled_graph")
            and self._compiled_graph is not None,
            "state_manager": hasattr(self, "state_manager")
            and self.state_manager is not None,
            "checkpointer": hasattr(self, "_checkpointer")
            and self._checkpointer is not None,
        }

    # ------------------------------------------------------------------
    # Граф LangGraph
    # ------------------------------------------------------------------

    def _build_graph(self) -> CompiledStateGraph:
        """Compile LangGraph execution graph.

        Graph topology:
            START → agent → (should_continue) → tools → validator → agent
                                              ↓
                                             END

        The graph is compiled with ``interrupt_before=["tools"]`` so that
        tool_calls can be patched (token injection, etc.) before execution.

        Returns:
            Compiled LangGraph state graph.

        Raises:
            RuntimeError: On compilation failure.
        """
        # AgentState определён на уровне модуля — используем напрямую
        workflow = StateGraph(AgentState)

        async def call_model(state: AgentState) -> Dict[str, Any]:
            """Invoke LLM with bound tools.

            Keeps only the latest SystemMessage to prevent prompt bloat.
            """
            model_with_tools = self.model.bind_tools(self.tools)
            msgs = state["messages"]
            sys_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
            non_sys = [m for m in msgs if not isinstance(m, SystemMessage)]
            final_messages = ([sys_msgs[-1]] if sys_msgs else []) + non_sys
            response = await model_with_tools.ainvoke(final_messages)
            return {"messages": [response]}

        async def validator(state: AgentState) -> Dict[str, Any]:
            """Validate tool results; inject system notification on errors."""
            messages = state["messages"]
            last_message = messages[-1] if messages else None

            if not isinstance(last_message, ToolMessage):
                return {"messages": []}

            content_raw = str(last_message.content or "").strip()

            if not content_raw or content_raw in ("None", "{}", "null"):
                return {
                    "messages": [
                        HumanMessage(
                            content="[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Инструмент вернул пустой результат."
                        )
                    ]
                }

            # Проверяем явные ошибки Python (не JSON-статус "error")
            if (
                "traceback" in content_raw.lower()
                or "exception" in content_raw.lower()
            ):
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Техническая ошибка: {content_raw[:300]}"
                        )
                    ]
                }

            return {"messages": []}

        def should_continue(state: AgentState) -> str:
            """Route: continue to tools or finish."""
            last = state["messages"][-1] if state["messages"] else None
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validator", validator)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "validator")
        workflow.add_edge("validator", "agent")

        try:
            compiled = workflow.compile(
                checkpointer=self._checkpointer,
                interrupt_before=["tools"],
            )
            logger.debug("Graph compiled successfully")
            return compiled
        except Exception as e:
            logger.error("Graph compilation failed", exc_info=True)
            raise RuntimeError(f"Failed to compile graph: {e}") from e

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    async def chat(
        self,
        message: str,
        user_token: str,
        context_ui_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
        human_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry point for user messages.

        Handles three cases in priority order:
        1. human_choice (summarization type selection) → resume interrupted graph.
        2. Pending disambiguation (employee selection) → replay tool with UUID.
        3. Normal new message → full pipeline (semantic → prompt → graph).

        Args:
            message: User message text.
            user_token: Bearer authentication token.
            context_ui_id: Active document UUID in EDMS (optional).
            thread_id: LangGraph conversation thread (optional).
            user_context: User metadata dict (firstName, lastName, etc.).
            file_path: Local file path or attachment UUID (optional).
            human_choice: User choice for pending action (optional).

        Returns:
            AgentResponse serialized as dict.
        """
        try:
            request = AgentRequest(
                message=message,
                user_token=user_token,
                context_ui_id=context_ui_id,
                thread_id=thread_id,
                user_context=user_context or {},
                file_path=file_path,
                human_choice=human_choice,
            )

            context = await self._build_context(request)

            # ── 1. human_choice (суммаризация) ──────────────────────
            if human_choice:
                state = await self.state_manager.get_state(context.thread_id)
                if state.next:
                    return await self._handle_human_choice(context, human_choice)

            # ── 2. Pending disambiguation (выбор исполнителя/ознакомления) ──
            pending_dis = self._dis_ctx.get(context.thread_id)
            if pending_dis and DisambiguationResolver.is_likely_selection(message):
                resolved_ids = DisambiguationResolver.resolve(message, pending_dis)
                if resolved_ids:
                    return await self._replay_with_selection(
                        context=context,
                        pending=pending_dis,
                        selected_ids=resolved_ids,
                    )
                else:
                    # Не смогли распознать выбор — сообщаем пользователю
                    human_text = self._build_disambiguation_message(
                        pending_dis.get("ambiguous_matches", [])
                    )
                    return AgentResponse(
                        status=AgentStatus.REQUIRES_ACTION,
                        action_type=ActionType.DISAMBIGUATION,
                        content=(
                            "Не удалось определить ваш выбор. "
                            "Пожалуйста, введите номер из списка.\n\n"
                            + human_text
                        ),
                        metadata={"ambiguous_matches": pending_dis.get("ambiguous_matches", [])},
                    ).model_dump()

            # ── 3. Обычный новый запрос ──────────────────────────────
            document: Optional[DocumentDto] = None
            if context.document_id:
                document = await self.document_repo.get_document(
                    context.user_token, context.document_id
                )

            semantic_context = self.dispatcher.build_context(request.message, document)
            logger.info(
                "Semantic analysis complete",
                extra={
                    "intent": semantic_context.query.intent.value,
                    "complexity": semantic_context.query.complexity.value,
                    "thread_id": context.thread_id,
                },
            )

            refined_message = semantic_context.query.refined
            user_intent = semantic_context.query.intent
            semantic_xml = self._build_semantic_xml(semantic_context)
            full_prompt = PromptBuilder.build(context, user_intent, semantic_xml)

            inputs: Dict[str, Any] = {
                "messages": [
                    SystemMessage(content=full_prompt),
                    HumanMessage(content=refined_message),
                ]
            }

            return await self._orchestrate(
                context=context,
                inputs=inputs,
                is_choice_active=bool(human_choice),
                iteration=0,
            )

        except Exception as exc:
            logger.error(
                "Chat error",
                exc_info=True,
                extra={"user_message_preview": message[:100]},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {exc}",
            ).model_dump()

    # ------------------------------------------------------------------
    # Human-in-the-loop: обработка выбора типа суммаризации
    # ------------------------------------------------------------------

    async def _handle_human_choice(
        self, context: ContextParams, human_choice: str
    ) -> Dict[str, Any]:
        """Resume interrupted graph with user's summarization type choice.

        Args:
            context: Execution context.
            human_choice: Canonical summary type string ('extractive', etc.).

        Returns:
            AgentResponse serialized as dict.
        """
        try:
            state = await self.state_manager.get_state(context.thread_id)
            messages = state.values.get("messages", [])
            if not messages:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Не найдено активное состояние для применения выбора.",
                ).model_dump()

            last_msg = messages[-1]
            original_calls = getattr(last_msg, "tool_calls", [])

            if not original_calls:
                logger.warning(
                    "No pending tool_calls for human choice",
                    extra={"thread_id": context.thread_id},
                )
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Нет ожидающих операций для применения выбора.",
                ).model_dump()

            fixed_calls = []
            for tc in original_calls:
                t_args = dict(tc["args"])
                t_name = tc["name"]

                # Явно устанавливаем выбор пользователя ПЕРВЫМ —
                # до ToolCallPatcher, чтобы патчер его не перезаписал
                if t_name == "doc_summarize_text":
                    t_args["summary_type"] = human_choice
                    logger.info(
                        "Applied human_choice to summary_type",
                        extra={"human_choice": human_choice},
                    )

                fixed_calls.append({"name": t_name, "args": t_args, "id": tc["id"]})

            await self.state_manager.update_state(
                context.thread_id,
                [
                    AIMessage(
                        content=last_msg.content or "",
                        tool_calls=fixed_calls,
                        id=last_msg.id,
                    )
                ],
                as_node="agent",
            )

            return await self._orchestrate(
                context=context,
                inputs=None,
                is_choice_active=True,
                iteration=0,
            )

        except Exception as exc:
            logger.error("_handle_human_choice error", exc_info=True)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка применения выбора: {exc}",
            ).model_dump()

    # ------------------------------------------------------------------
    # Disambiguation: повторный вызов инструмента с UUID
    # ------------------------------------------------------------------

    async def _replay_with_selection(
        self,
        context: ContextParams,
        pending: Dict[str, Any],
        selected_ids: List[str],
    ) -> Dict[str, Any]:
        """Replay a tool call with user-selected employee UUIDs.

        Builds a synthetic AIMessage(tool_calls=[...]) and resumes
        orchestration so LangGraph executes the tool correctly.

        Args:
            context: Immutable execution context.
            pending: Saved disambiguation context dict.
            selected_ids: Resolved employee UUID strings.

        Returns:
            AgentResponse serialized as dict.
        """
        tool_name: str = pending.get("tool_name", "task_create_tool")
        tool_args: Dict[str, Any] = dict(pending.get("tool_args", {}))

        # Внедряем selected_employee_ids, убираем двусмысленные фамилии
        tool_args["selected_employee_ids"] = selected_ids
        tool_args.pop("executor_last_names", None)
        tool_args.pop("last_names", None)
        tool_args["token"] = context.user_token
        if context.document_id and "document_id" not in tool_args:
            tool_args["document_id"] = context.document_id

        synthetic_tc_id = f"tc_{_uuid_module.uuid4().hex[:12]}"

        logger.info(
            "Replaying %s with selected_employee_ids=%s",
            tool_name,
            selected_ids,
        )

        # Очищаем disambiguation контекст
        self._dis_ctx.clear(context.thread_id)

        # Синтетический AIMessage с tool_call
        synthetic_ai = AIMessage(
            content="",
            tool_calls=[
                {"name": tool_name, "args": tool_args, "id": synthetic_tc_id}
            ],
        )
        await self.state_manager.update_state(
            context.thread_id,
            [synthetic_ai],
            as_node="agent",
        )

        return await self._orchestrate(
            context=context,
            inputs=None,
            is_choice_active=True,
            iteration=0,
        )

    # ------------------------------------------------------------------
    # Основной цикл оркестрации
    # ------------------------------------------------------------------

    async def _orchestrate(
        self,
        context: ContextParams,
        inputs: Optional[Dict[str, Any]],
        is_choice_active: bool = False,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """Core execution loop: invoke graph, patch tool_calls, recurse.

        Args:
            context: Immutable execution context.
            inputs: Initial inputs or None (resume from checkpoint).
            is_choice_active: Whether user has made a pending choice.
            iteration: Current recursion depth.

        Returns:
            AgentResponse serialized as dict.
        """
        if iteration > self.MAX_ITERATIONS:
            logger.error(
                "Max iterations exceeded",
                extra={"thread_id": context.thread_id, "max": self.MAX_ITERATIONS},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышен лимит итераций обработки. Пожалуйста, переформулируйте запрос.",
            ).model_dump()

        try:
            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self.EXECUTION_TIMEOUT,
            )

            state = await self.state_manager.get_state(context.thread_id)
            messages: List[BaseMessage] = state.values.get("messages", [])

            if not messages:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Пустое состояние агента после выполнения.",
                ).model_dump()

            last_msg = messages[-1]

            logger.debug(
                "Orchestrate state snapshot",
                extra={
                    "thread_id": context.thread_id,
                    "iteration": iteration,
                    "messages_count": len(messages),
                    "last_msg_type": type(last_msg).__name__,
                    "state_next": state.next,
                },
            )

            # ── Граф завершён (нет pending tool_calls) ──────────────
            has_pending_tools = (
                bool(state.next)
                and isinstance(last_msg, AIMessage)
                and bool(getattr(last_msg, "tool_calls", None))
            )

            if not has_pending_tools:
                # Проверяем ToolMessages на requires_disambiguation / requires_choice
                action_resp = self._check_tool_action_required(messages, context.thread_id)
                if action_resp:
                    return action_resp

                final_content = ContentExtractor.extract_final_content(messages)
                if final_content:
                    final_content = ContentExtractor.clean_json_artifacts(final_content)
                    logger.info(
                        "Execution completed successfully",
                        extra={
                            "thread_id": context.thread_id,
                            "content_length": len(final_content),
                            "iterations": iteration + 1,
                        },
                    )
                    return AgentResponse(
                        status=AgentStatus.SUCCESS,
                        content=final_content,
                    ).model_dump()

                logger.warning(
                    "No final content extracted",
                    extra={"thread_id": context.thread_id},
                )
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content="Операция выполнена.",
                ).model_dump()

            # ── Патчим tool_calls и продолжаем ──────────────────────
            last_extracted_text = ContentExtractor.extract_last_text(messages)

            patched_calls = self._patch_tool_calls(
                tool_calls=list(last_msg.tool_calls),
                context=context,
                last_extracted_text=last_extracted_text,
                is_choice_active=is_choice_active,
            )

            await self.state_manager.update_state(
                context.thread_id,
                [
                    AIMessage(
                        content=last_msg.content or "",
                        tool_calls=patched_calls,
                        id=last_msg.id,
                    )
                ],
                as_node="agent",
            )

            return await self._orchestrate(
                context=context,
                inputs=None,
                is_choice_active=True,
                iteration=iteration + 1,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Execution timeout",
                extra={"thread_id": context.thread_id},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения. Попробуйте позже.",
            ).model_dump()

        except Exception as exc:
            logger.error(
                "Orchestration error",
                exc_info=True,
                extra={"thread_id": context.thread_id, "iteration": iteration},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка выполнения: {exc}",
            ).model_dump()

    # ------------------------------------------------------------------
    # Обнаружение requires_action в результатах инструментов
    # ------------------------------------------------------------------

    def _check_tool_action_required(
        self,
        messages: List[BaseMessage],
        thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Scan ToolMessages for statuses requiring user action.

        Checks for:
        - ``requires_disambiguation``: user must pick from a candidate list.
        - ``requires_choice``: user must choose summarisation type.

        For disambiguation, saves context to DisambiguationContext so the
        next user message can replay the tool with correct UUIDs.

        Must run BEFORE ContentExtractor so we don't let the LLM narrate
        a fake "success" over a result that needs user input.

        Args:
            messages: Full message chain from LangGraph state.
            thread_id: Active thread identifier.

        Returns:
            AgentResponse dict if action required, else None.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            raw = str(m.content or "").strip()
            if not raw.startswith("{"):
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            status = data.get("status", "")

            # ── requires_disambiguation ──────────────────────────────
            if status == "requires_disambiguation":
                ambiguous = data.get("ambiguous_matches", [])

                # Ищем родительский AIMessage с tool_call для сохранения контекста
                tool_call_id = getattr(m, "tool_call_id", None)
                pending_tool_name = ""
                pending_tool_args: Dict[str, Any] = {}
                for msg in reversed(messages):
                    if not isinstance(msg, AIMessage):
                        continue
                    for tc in getattr(msg, "tool_calls", []):
                        if tc.get("id") == tool_call_id:
                            pending_tool_name = tc.get("name", "")
                            pending_tool_args = dict(tc.get("args", {}))
                            break
                    if pending_tool_name:
                        break

                # Сохраняем в in-memory хранилище
                self._dis_ctx.save(
                    thread_id,
                    {
                        "tool_name": pending_tool_name,
                        "tool_args": pending_tool_args,
                        "ambiguous_matches": ambiguous,
                    },
                )

                human_text = self._build_disambiguation_message(ambiguous)
                return AgentResponse(
                    status=AgentStatus.REQUIRES_ACTION,
                    action_type=ActionType.DISAMBIGUATION,
                    content=human_text,
                    metadata={
                        "ambiguous_matches": ambiguous,
                        "raw_tool_status": status,
                    },
                ).model_dump()

            # ── requires_choice (суммаризация) ───────────────────────
            if status == "requires_choice":
                suggestion = data.get("suggestion", {})
                recommended = suggestion.get("recommended", "extractive")
                logger.info(
                    "Returning requires_choice to UI",
                    extra={"recommended": recommended},
                )
                return AgentResponse(
                    status=AgentStatus.REQUIRES_ACTION,
                    action_type=ActionType.SUMMARIZE_SELECTION,
                    content=data.get("message", "Выберите формат анализа документа:"),
                    metadata={"suggestion": suggestion, "recommended": recommended},
                ).model_dump()

        return None

    # ------------------------------------------------------------------
    # Патчинг tool_calls
    # ------------------------------------------------------------------

    def _patch_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: ContextParams,
        last_extracted_text: Optional[str],
        is_choice_active: bool,
    ) -> List[Dict[str, Any]]:
        """Apply all patches to tool_calls before graph execution.

        Patches applied:
        1. Auto-inject ``token`` into every tool call.
        2. Convert UUID file_path → attachment_id for read_local_file_content.
        3. Auto-inject ``document_id`` into doc_* and creation tools.
        4. Patch ``doc_summarize_text``: inject text, preserve human_choice type.

        Args:
            tool_calls: Original tool_calls list from AIMessage.
            context: Immutable execution context.
            last_extracted_text: Last tool text result (for summarize injection).
            is_choice_active: Kept for API compatibility; not used in summarize patch.

        Returns:
            Patched list of tool_call dicts.
        """
        clean_path = str(context.file_path).strip() if context.file_path else ""
        is_uuid_path = bool(_UUID_RE.match(clean_path))

        patched: List[Dict[str, Any]] = []
        for tc in tool_calls:
            t_name: str = tc["name"]
            t_args: Dict[str, Any] = dict(tc["args"])
            t_id: str = tc["id"]

            # 1. Автоинъекция token
            t_args["token"] = context.user_token

            # 2. UUID file_path → attachment_id
            if is_uuid_path and t_name == "read_local_file_content":
                t_name = "doc_get_file_content"
                t_args["attachment_id"] = clean_path
                t_args.pop("file_path", None)
                logger.info(
                    "Converted read_local_file_content → doc_get_file_content",
                    extra={"attachment_id_prefix": clean_path[:8]},
                )

            # 3. Автоинъекция document_id
            if context.document_id and (
                t_name.startswith("doc_")
                or "document_id" in t_args
                or t_name in ("introduction_create_tool", "task_create_tool")
            ):
                t_args["document_id"] = context.document_id

            # 4. Патч doc_summarize_text
            if t_name == "doc_summarize_text":
                t_args = self._patch_summarize(
                    t_args,
                    last_extracted_text=last_extracted_text,
                )

            patched.append({"name": t_name, "args": t_args, "id": t_id})

        return patched

    def _patch_summarize(
        self,
        t_args: Dict[str, Any],
        last_extracted_text: Optional[str],
        is_choice_active: bool,
    ) -> Dict[str, Any]:
        """Apply summarize-specific patches to tool args.

        Design decision: we NEVER auto-select summary_type here.
        The doc_summarize_text tool itself returns ``requires_choice``
        when summary_type is None, which is the correct UX flow —
        the user should always explicitly choose the analysis format.

        Patch rules:
        - Inject ``text`` from last_extracted_text if not already set.
        - If ``summary_type`` already set (injected by _handle_human_choice) → keep it.
        - Otherwise → leave as None so the tool prompts the user.

        Args:
            t_args: Tool arguments dict to patch.
            last_extracted_text: Extracted text from previous tool results.
            is_choice_active: True if user already chose summary type this turn.

        Returns:
            Patched arguments dict.
        """
        # Инъекция текста (если ещё не задан LLM-ом)
        if last_extracted_text and not t_args.get("text"):
            t_args["text"] = str(last_extracted_text)

        current_type = t_args.get("summary_type")

        # Если summary_type уже задан (через _handle_human_choice) — не трогаем
        if current_type:
            logger.info(
                "summary_type already set by human_choice, preserving: %s", current_type
            )
            return t_args

        # Во всех остальных случаях оставляем None —
        # инструмент вернёт requires_choice и UI покажет кнопки выбора
        logger.info(
            "summary_type=None → tool will return requires_choice to ask user"
        )
        t_args["summary_type"] = None
        return t_args

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    async def _build_context(self, request: AgentRequest) -> ContextParams:
        """Build immutable ContextParams from validated request.

        Args:
            request: Validated AgentRequest.

        Returns:
            ContextParams instance.
        """
        user_name = (
            request.user_context.get("firstName")
            or request.user_context.get("name")
            or "пользователь"
        ).strip()

        return ContextParams(
            user_token=request.user_token,
            document_id=request.context_ui_id or None,
            file_path=request.file_path,
            thread_id=request.thread_id or "default",
            user_name=user_name,
            user_first_name=request.user_context.get("firstName"),
        )

    @staticmethod
    def _build_semantic_xml(semantic_context: Any) -> str:
        """Build semantic context XML snippet for prompt injection.

        Args:
            semantic_context: SemanticContext object from NLP dispatcher.

        Returns:
            XML string block.
        """
        try:
            return (
                "\n<semantic_context>\n"
                "  <user_query>\n"
                f"    <original>{semantic_context.query.original}</original>\n"
                f"    <refined>{semantic_context.query.refined}</refined>\n"
                f"    <intent>{semantic_context.query.intent.value}</intent>\n"
                f"    <complexity>{semantic_context.query.complexity.value}</complexity>\n"
                "  </user_query>\n"
                "</semantic_context>"
            )
        except AttributeError:
            return ""

    @staticmethod
    def _build_disambiguation_message(
        ambiguous_matches: List[Dict[str, Any]],
    ) -> str:
        """Build human-readable Russian-language disambiguation list.

        Handles two formats from different tools:

        **Grouped** (task_create_tool):
            ``[{"search_query": "Петров", "matches": [{id, full_name, ...}, ...]}, ...]``

        **Flat** (introduction_create_tool via _build_disambiguation_response):
            ``[{"id": "...", "full_name": "...", "post": "...", "department": "...",
               "search_term": "Иванов"}, ...]``

        Args:
            ambiguous_matches: List of match entries from tool result.

        Returns:
            Formatted Russian-language selection prompt string.
        """
        if not ambiguous_matches:
            return "Найдено несколько совпадений. Уточните выбор."

        lines: List[str] = ["Найдено несколько сотрудников. Выберите нужного:"]

        # Определяем формат: если первый элемент содержит "matches" → grouped
        # Если содержит "full_name" напрямую → flat (introduction tool)
        first = ambiguous_matches[0]
        is_grouped = "matches" in first

        if is_grouped:
            # Grouped format: task_create_tool
            global_idx = 1
            for group in ambiguous_matches:
                query = group.get("search_query", "")
                matches = group.get("matches", [])
                if len(ambiguous_matches) > 1 and query:
                    lines.append(f'\nПо запросу «{query}»:')
                for match in matches:
                    full_name = match.get("full_name", "—")
                    post = match.get("post", "должность не указана")
                    dept = match.get("department", "отдел не указан")
                    lines.append(f"{global_idx}. {full_name} — {post}, {dept}")
                    global_idx += 1
        else:
            # Flat format: introduction_create_tool
            # Группируем по search_term чтобы показать подзаголовки если запросов несколько
            from collections import defaultdict as _dd
            groups: Dict[str, List[Dict[str, Any]]] = _dd(list)
            for match in ambiguous_matches:
                key = match.get("search_term", "")
                groups[key].append(match)

            global_idx = 1
            for query, matches in groups.items():
                if len(groups) > 1 and query:
                    lines.append(f'\nПо запросу «{query}»:')
                for match in matches:
                    full_name = match.get("full_name", "—")
                    post = match.get("post", "должность не указана")
                    dept = match.get("department", "отдел не указан")
                    lines.append(f"{global_idx}. {full_name} — {post}, {dept}")
                    global_idx += 1

        lines.append("\nНапишите номер сотрудника из списка или его ФИО.")
        return "\n".join(lines)