# edms_ai_assistant/agent.py
"""
EDMS AI Assistant — Core Agent Module.

Production-ready LangGraph agent for Electronic Document Management System.
Implements multi-node graph: agent → tools → validator → agent.

NOTE: ``from __future__ import annotations`` (PEP-563) is intentionally absent.
LangGraph calls typing.get_type_hints() on node functions at compile time.
With lazy string annotations, AgentState TypedDict cannot be resolved
in get_type_hints() global namespace, causing NameError. Eager evaluation fixes this.
"""

import asyncio
import json
import logging
import re
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
# AgentState — определён на уровне модуля, чтобы get_type_hints() в LangGraph
# мог разрешить тип в глобальном namespace (иначе NameError внутри _build_graph)
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """LangGraph state schema: накопленная цепочка сообщений."""

    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Скомпилированный UUID-паттерн (используется в нескольких местах)
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
    """Immutable контекст для одного вызова агента.

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

        # UUID format
        if _UUID_RE.match(v):
            return v

        # Unix absolute path
        if v.startswith("/"):
            return v

        # Windows absolute path
        if re.match(r"^[A-Za-z]:\\", v):
            return v

        # Relative path with separator
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
        """Fetch document metadata.

        Args:
            token: User bearer token.
            doc_id: Document UUID.

        Returns:
            DocumentDto or None on failure.
        """
        ...


class DocumentRepository:
    """Реализация репозитория документов через DocumentClient."""

    async def get_document(self, token: str, doc_id: str) -> Optional[DocumentDto]:
        """Fetch document metadata from EDMS API.

        Args:
            token: User bearer token.
            doc_id: Document UUID.

        Returns:
            Validated DocumentDto or None if fetch fails.
        """
        try:
            async with DocumentClient() as client:
                raw_data = await client.get_document_metadata(token, doc_id)
                doc = DocumentDto.model_validate(raw_data)
                logger.info("Document fetched", extra={"doc_id": doc_id})
                return doc
        except Exception as exc:
            logger.error(
                "Failed to fetch document",
                exc_info=True,
                extra={"doc_id": doc_id, "error": str(exc)},
            )
            return None


# ---------------------------------------------------------------------------
# PromptBuilder — Strategy для построения системного промпта
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Строит системный промпт с динамическим контекстом и семантическим XML."""

    # ------------------------------------------------------------------
    # Базовый шаблон системного промпта
    # ------------------------------------------------------------------
    CORE_TEMPLATE: str = """<role>
Ты — экспертный помощник системы электронного документооборота (EDMS/СЭД).
Помогаешь с анализом документов, управлением персоналом и делегированием задач.
Отвечаешь на ВСЕ вопросы пользователя: как по EDMS, так и общие вопросы.
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
   - Статус "summarize_selection" → Предложи формат анализа (факты/пересказ/тезисы)
   - Статус "requires_disambiguation" → Покажи список, дождись выбора пользователя

4. **Ответ пользователю**: После вызова инструментов ВСЕГДА формулируй финальный ответ на русском языке.
   Если инструмент вернул данные — проанализируй и объясни их пользователю.
   Если инструменты не нужны — отвечай напрямую без вызовов инструментов.

5. **Язык**: Только русский. Обращайся к пользователю по имени: {user_name}

6. **Общие вопросы**: На приветствия, вопросы "что ты умеешь?", общие темы — отвечай дружелюбно и без вызова инструментов.

7. **Никогда не оставляй ответ пустым**: Если не можешь выполнить запрос — объясни почему.
</critical_rules>

<tool_selection>
**Типичные сценарии**:
- Анализ документа: doc_get_details → doc_get_file_content → doc_summarize_text
- Анализ файла (UUID): doc_get_file_content → doc_summarize_text
- Поиск сотрудника: employee_search_tool
- Список ознакомления: introduction_create_tool
- Создание поручения: task_create_tool
- Создание обращения: autofill_appeal_document
- Общий вопрос / приветствие: ответь напрямую, НЕ вызывай инструменты
</tool_selection>

<response_format>
✅ Структурировано, кратко, по делу
✅ На все вопросы — содержательный ответ
❌ Многословие, технические детали API
❌ Пустые ответы или молчание
</response_format>"""

    # ------------------------------------------------------------------
    # Динамические снипеты по intent
    # ------------------------------------------------------------------
    CONTEXT_SNIPPETS: Dict[UserIntent, str] = {
        UserIntent.CREATE_INTRODUCTION: """
<introduction_guide>
При создании списка ознакомления:
- Если статус "requires_disambiguation" → Покажи список найденных сотрудников
- Дождись выбора пользователя
- Повторный вызов: introduction_create_tool(selected_employee_ids=[uuid1, uuid3])
</introduction_guide>""",
        UserIntent.CREATE_TASK: """
<task_guide>
При создании поручения:
- executor_last_names: обязательно (минимум 1)
- responsible_last_name: опционально (если НЕ указан → первый исполнитель)
- planed_date_end: опционально (если НЕ указан → +7 дней)
- Даты должны быть в формате ISO 8601 с timezone (например: "2026-02-15T23:59:59Z")
</task_guide>""",
        UserIntent.SUMMARIZE: """
<summarize_guide>
При суммаризации:
- extractive → ключевые факты и цитаты
- abstractive → краткий пересказ своими словами
- thesis → тезисный план
Преобразование дат в ISO 8601:
- "до 15 февраля" → "2026-02-15T23:59:59Z"
- "через неделю" → +7 дней от текущей даты
</summarize_guide>""",
    }

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        semantic_xml: str,
    ) -> str:
        """Build complete system prompt.

        Args:
            context: Immutable execution context.
            intent: Detected user intent for optional snippet injection.
            semantic_xml: Pre-built semantic context XML block.

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
        return base_prompt + dynamic_context + "\n" + semantic_xml


# ---------------------------------------------------------------------------
# ContentExtractor — Утилиты для извлечения финального текста
# ---------------------------------------------------------------------------


class ContentExtractor:
    """Извлечение финального контента из цепочки сообщений LangGraph."""

    # Паттерны, по которым AIMessage считается «техническим» (не ответ пользователю)
    SKIP_PATTERNS: List[str] = ["вызвал инструмент", "tool call", '"name"', '"id"']
    MIN_CONTENT_LENGTH: int = 10  # снижен порог для коротких ответов
    JSON_FIELDS: List[str] = ["content", "text", "text_preview", "message"]

    @classmethod
    def extract_final_content(cls, messages: List[BaseMessage]) -> Optional[str]:
        """Extract the most recent meaningful AI response.

        Strategy (priority order):
            1. Last AIMessage with sufficient non-technical content.
            2. Structured JSON field from last ToolMessage.
            3. Any AIMessage content (fallback).
            4. Raw ToolMessage content (last resort).

        Args:
            messages: Full message chain from LangGraph state.

        Returns:
            Extracted content string or None.
        """
        # 1. Ищем последний AIMessage с содержательным ответом
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if not cls._is_skip_content(content) and len(content) >= cls.MIN_CONTENT_LENGTH:
                    logger.debug("Extracted AIMessage content: %d chars", len(content))
                    return content

        # 2. ToolMessage → JSON поля
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._extract_from_tool_message(m)
                if extracted:
                    return extracted

        # 3. Любой AIMessage (fallback)
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if content:
                    logger.debug("Fallback AIMessage: %d chars", len(content))
                    return content

        # 4. Сырой ToolMessage (последний резерв)
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                content = str(m.content).strip()
                if len(content) >= cls.MIN_CONTENT_LENGTH:
                    logger.debug("Fallback ToolMessage: %d chars", len(content))
                    return content

        return None

    @classmethod
    def extract_last_text(cls, messages: List[BaseMessage]) -> Optional[str]:
        """Extract document/tool text from the last ToolMessage for summarization.

        Args:
            messages: Full message chain.

        Returns:
            Extracted text content or None.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                raw = str(m.content)
                if raw.strip().startswith("{"):
                    data = json.loads(raw)
                    for key in ("content", "text_preview", "text"):
                        val = data.get(key)
                        if val and len(str(val)) > 50:
                            return str(val)
                if len(raw) > 50:
                    return raw
            except (json.JSONDecodeError, TypeError):
                if len(str(m.content)) > 50:
                    return str(m.content)
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """Remove JSON wrapper artifacts from content string.

        Args:
            content: Raw content that may contain JSON wrapping.

        Returns:
            Cleaned content string.
        """
        if not content:
            return content

        # Попытка распарсить как JSON-объект с полем "content"
        stripped = content.strip()
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
                if isinstance(data, dict) and "content" in data:
                    return str(data["content"]).strip()
            except (json.JSONDecodeError, TypeError):
                pass

        # Простая очистка от JSON-оберток
        content = content.replace('{"status": "success", "content": "', "")
        content = content.replace('"}', "")
        content = content.replace('\\"', '"')
        content = content.replace("\\n", "\n")
        return content.strip()

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    @classmethod
    def _is_skip_content(cls, content: str) -> bool:
        """Check if content should be skipped as technical noise."""
        lower = content.lower()
        return any(pattern in lower for pattern in cls.SKIP_PATTERNS)

    @classmethod
    def _extract_from_tool_message(cls, message: ToolMessage) -> Optional[str]:
        """Extract meaningful content from a ToolMessage JSON payload."""
        try:
            raw = str(message.content).strip()
            if raw.startswith("{"):
                data = json.loads(raw)
                if isinstance(data, dict):
                    for field_name in cls.JSON_FIELDS:
                        val = data.get(field_name)
                        if val and len(str(val)) >= cls.MIN_CONTENT_LENGTH:
                            logger.debug(
                                "ToolMessage JSON[%s]: %d chars",
                                field_name,
                                len(str(val)),
                            )
                            return str(val).strip()
        except (json.JSONDecodeError, TypeError):
            pass
        return None


# ---------------------------------------------------------------------------
# ToolCallPatcher — централизованная инъекция аргументов в tool_calls
# ---------------------------------------------------------------------------


class ToolCallPatcher:
    """Патчит аргументы tool_calls перед их выполнением.

    Обязанности:
        - Автоинъекция ``token`` во все вызовы.
        - Автоинъекция ``document_id`` в doc_* инструменты.
        - Конвертация UUID file_path → attachment_id для read_local_file_content.
        - Автовыбор summary_type для doc_summarize_text.
    """

    @classmethod
    def patch(
        cls,
        tool_calls: List[Dict[str, Any]],
        context: ContextParams,
        last_extracted_text: Optional[str],
        is_choice_active: bool,
    ) -> List[Dict[str, Any]]:
        """Apply all patches to a list of tool calls.

        Args:
            tool_calls: Original tool_calls from AIMessage.
            context: Immutable execution context.
            last_extracted_text: Last meaningful text from ToolMessage (for summarize).
            is_choice_active: Whether user already made a choice.

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

            # --- 1. Автоинъекция token ---
            t_args["token"] = context.user_token

            # --- 2. UUID file_path → attachment_id ---
            if is_uuid_path and t_name == "read_local_file_content":
                t_name = "doc_get_file_content"
                t_args["attachment_id"] = clean_path
                t_args.pop("file_path", None)
                logger.info(
                    "Converted read_local_file_content → doc_get_file_content",
                    extra={"attachment_id_prefix": clean_path[:8]},
                )

            # --- 3. Автоинъекция document_id ---
            if context.document_id and cls._needs_document_id(t_name, t_args):
                t_args["document_id"] = context.document_id

            # --- 4. Патч doc_summarize_text ---
            if t_name == "doc_summarize_text":
                t_args = cls._patch_summarize(
                    t_args,
                    last_extracted_text=last_extracted_text,
                    is_choice_active=is_choice_active,
                )

            patched.append({"name": t_name, "args": t_args, "id": t_id})

        return patched

    # ------------------------------------------------------------------

    @staticmethod
    def _needs_document_id(t_name: str, t_args: Dict[str, Any]) -> bool:
        """Check if tool requires document_id injection."""
        return (
            t_name.startswith("doc_")
            or "document_id" in t_args
            or t_name in ("introduction_create_tool", "task_create_tool")
        )

    @staticmethod
    def _patch_summarize(
        t_args: Dict[str, Any],
        last_extracted_text: Optional[str],
        is_choice_active: bool,
    ) -> Dict[str, Any]:
        """Patch doc_summarize_text arguments.

        Args:
            t_args: Current tool arguments.
            last_extracted_text: Previously extracted document text.
            is_choice_active: Whether user made a summarization-type choice.

        Returns:
            Updated arguments dict.
        """
        # Подставляем текст из предыдущего шага, если не задан явно
        if last_extracted_text and not t_args.get("text"):
            t_args["text"] = str(last_extracted_text)

        # Авто-выбор типа суммаризации
        if not t_args.get("summary_type"):
            if is_choice_active:
                t_args["summary_type"] = "extractive"
                logger.info("Summarize fallback: using 'extractive' (choice was active)")
            else:
                try:
                    from edms_ai_assistant.services.nlp_service import (
                        EDMSNaturalLanguageService,
                    )

                    suggestion = EDMSNaturalLanguageService().suggest_summarize_format(
                        str(last_extracted_text) if last_extracted_text else ""
                    )
                    recommended = suggestion.get("recommended", "extractive")
                    t_args["summary_type"] = recommended
                    logger.info(
                        "Auto-selected summary_type: %s (reason: %s)",
                        recommended,
                        suggestion.get("reason", "n/a"),
                    )
                except Exception as exc:
                    logger.warning(
                        "suggest_summarize_format failed, using extractive: %s", exc
                    )
                    t_args["summary_type"] = "extractive"

        return t_args


# ---------------------------------------------------------------------------
# AgentStateManager — обёртка над LangGraph checkpointer/state API
# ---------------------------------------------------------------------------


class AgentStateManager:
    """Управляет состоянием и жизненным циклом LangGraph-графа."""

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
        """Initialize state manager.

        Args:
            graph: Compiled LangGraph state graph.
            checkpointer: Memory checkpointer instance.

        Raises:
            ValueError: If graph or checkpointer is None.
        """
        if graph is None:
            raise ValueError("Graph cannot be None")
        if checkpointer is None:
            raise ValueError("Checkpointer cannot be None")

        self.graph = graph
        self.checkpointer = checkpointer
        logger.debug(
            "AgentStateManager initialized",
            extra={
                "graph_type": type(graph).__name__,
                "checkpointer_type": type(checkpointer).__name__,
            },
        )

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
            messages: Messages to inject.
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
        START → agent → tools (interrupt_before) → validator → agent → … → END

    Основные возможности:
        - Автоинъекция токена и document_id в tool_calls.
        - Семантический анализ запроса перед LLM-вызовом.
        - Защита от бесконечных циклов (MAX_ITERATIONS).
        - Обработка пользовательских выборов (human_choice).
        - Структурированные JSON-логи для каждого шага.
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
            semantic_dispatcher: NLP semantic dispatcher (DI).

        Raises:
            RuntimeError: If any component fails to initialize.
        """
        try:
            self.model = get_chat_model()
            self.tools = all_tools
            self.document_repo: IDocumentRepository = (
                document_repo or DocumentRepository()
            )
            self.dispatcher: SemanticDispatcher = (
                semantic_dispatcher or SemanticDispatcher()
            )

            logger.debug("Base components initialized")

            self._checkpointer = MemorySaver()
            self._compiled_graph = self._build_graph()

            if self._compiled_graph is None:
                raise RuntimeError("Graph compilation returned None")

            self.state_manager = AgentStateManager(
                graph=self._compiled_graph,
                checkpointer=self._checkpointer,
            )

            logger.info(
                "EdmsDocumentAgent initialized successfully",
                extra={
                    "tools_count": len(self.tools),
                    "model": str(self.model),
                },
            )

        except Exception as exc:
            logger.error(
                "Failed to initialize EdmsDocumentAgent",
                exc_info=True,
            )
            raise RuntimeError(f"Agent initialization failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return component-level health status dict.

        Returns:
            Dict with boolean health flags per component.
        """
        return {
            "model": self.model is not None,
            "tools": len(self.tools) > 0,
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": getattr(self, "_compiled_graph", None) is not None,
            "state_manager": getattr(self, "state_manager", None) is not None,
            "checkpointer": getattr(self, "_checkpointer", None) is not None,
        }

    # ------------------------------------------------------------------
    # Построение графа
    # ------------------------------------------------------------------

    def _build_graph(self) -> CompiledStateGraph:
        """Compile LangGraph execution graph.

        Graph topology:
            START → agent → [tools | END]
                         tools → validator → agent

        Returns:
            Compiled state graph with MemorySaver checkpointer.

        Raises:
            RuntimeError: On compilation failure.
        """

        # AgentState определён на уровне модуля — используем напрямую
        workflow = StateGraph(AgentState)

        # ── Узел: agent ──────────────────────────────────────────────
        async def call_model(state: AgentState) -> Dict[str, Any]:
            """Invoke LLM with bound tools.

            Filters system messages to keep only the latest one,
            preventing prompt bloat across turns.
            """
            model_with_tools = self.model.bind_tools(self.tools)
            msgs = state["messages"]

            # Держим только последний SystemMessage + все не-системные
            sys_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
            non_sys = [m for m in msgs if not isinstance(m, SystemMessage)]
            final_messages = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            try:
                response = await model_with_tools.ainvoke(final_messages)
                logger.debug(
                    "LLM response received",
                    extra={
                        "has_tool_calls": bool(
                            getattr(response, "tool_calls", None)
                        ),
                        "content_length": len(str(response.content or "")),
                    },
                )
                return {"messages": [response]}
            except Exception as exc:
                logger.error("LLM invocation error", exc_info=True)
                # Возвращаем сообщение об ошибке вместо падения
                error_msg = AIMessage(
                    content=f"Произошла ошибка при обработке запроса: {exc}. "
                            "Пожалуйста, попробуйте переформулировать вопрос."
                )
                return {"messages": [error_msg]}

        # ── Узел: validator ──────────────────────────────────────────
        async def validator(state: AgentState) -> Dict[str, Any]:
            """Validate tool results and inject system notifications on errors.

            Returns empty messages list if tool result is healthy.
            Injects HumanMessage notification if tool returned empty/error.
            """
            messages = state["messages"]
            if not messages:
                return {"messages": []}

            last_message = messages[-1]

            if not isinstance(last_message, ToolMessage):
                return {"messages": []}

            content_raw = str(last_message.content or "").strip()

            if not content_raw or content_raw in ("None", "{}", "null"):
                logger.warning(
                    "Tool returned empty result",
                    extra={"tool_call_id": getattr(last_message, "tool_call_id", None)},
                )
                return {
                    "messages": [
                        HumanMessage(
                            content="[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Инструмент вернул пустой результат. "
                                    "Сообщи пользователю об этом и предложи альтернативу."
                        )
                    ]
                }

            lower = content_raw.lower()
            if "error" in lower or "exception" in lower or "traceback" in lower:
                logger.warning(
                    "Tool returned error content",
                    extra={"content_preview": content_raw[:200]},
                )
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Техническая ошибка при выполнении операции. "
                                    f"Детали: {content_raw[:500]}. "
                                    "Сообщи пользователю и предложи альтернативный вариант."
                        )
                    ]
                }

            return {"messages": []}

        # ── Рёбра графа ──────────────────────────────────────────────
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
        except Exception as exc:
            logger.error("Graph compilation failed", exc_info=True)
            raise RuntimeError(f"Failed to compile graph: {exc}") from exc

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

            # Проверяем наличие pending human_choice
            if human_choice:
                state = await self.state_manager.get_state(context.thread_id)
                if state.next:
                    return await self._handle_human_choice(context, human_choice)

            # Опционально загружаем метаданные документа
            document: Optional[DocumentDto] = None
            if context.document_id:
                document = await self.document_repo.get_document(
                    context.user_token, context.document_id
                )

            # Семантический анализ
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

            # Строим промпт и входные сообщения
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
    # Обработка пользовательского выбора (human-in-the-loop)
    # ------------------------------------------------------------------

    async def _handle_human_choice(
        self, context: ContextParams, human_choice: str
    ) -> Dict[str, Any]:
        """Resume interrupted graph execution with user's choice.

        Applies human_choice as summary_type for doc_summarize_text,
        then resumes orchestration from the interrupted state.

        Args:
            context: Execution context.
            human_choice: User's choice string (e.g. 'extractive').

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
                # Нет незавершённых вызовов — просто продолжаем с новым сообщением
                inputs = {
                    "messages": [HumanMessage(content=human_choice)]
                }
                return await self._orchestrate(
                    context=context,
                    inputs=inputs,
                    is_choice_active=True,
                    iteration=0,
                )

            # Патчим tool_calls: вставляем human_choice в summary_type
            fixed_calls = []
            for tc in original_calls:
                t_args = dict(tc["args"])
                if tc["name"] == "doc_summarize_text":
                    t_args["summary_type"] = human_choice
                fixed_calls.append({"name": tc["name"], "args": t_args, "id": tc["id"]})

            # Также патчим token + document_id
            last_text = ContentExtractor.extract_last_text(messages)
            fixed_calls = ToolCallPatcher.patch(
                tool_calls=fixed_calls,
                context=context,
                last_extracted_text=last_text,
                is_choice_active=True,
            )

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
            logger.error(
                "Error handling human choice",
                exc_info=True,
                extra={"thread_id": context.thread_id},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки выбора: {exc}",
            ).model_dump()

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
        """Recursive orchestration loop with tool injection.

        Invokes graph, inspects state, patches tool_calls, and recurses
        until graph reaches END or MAX_ITERATIONS is exceeded.

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
                    "has_tool_calls": bool(
                        isinstance(last_msg, AIMessage)
                        and getattr(last_msg, "tool_calls", None)
                    ),
                },
            )

            # ── Завершение: нет pending tool_calls ──────────────────
            has_pending_tools = (
                state.next
                and isinstance(last_msg, AIMessage)
                and bool(getattr(last_msg, "tool_calls", None))
            )

            if not has_pending_tools:
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

                # Финальный fallback — не должны сюда попасть
                logger.warning(
                    "No final content extracted, returning stub",
                    extra={"thread_id": context.thread_id},
                )
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content="Операция выполнена. Уточните запрос, если нужна дополнительная информация.",
                ).model_dump()

            # ── Pending tool_calls → патчим и продолжаем ────────────
            last_extracted_text = ContentExtractor.extract_last_text(messages)

            patched_calls = ToolCallPatcher.patch(
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
                extra={"thread_id": context.thread_id, "timeout": self.EXECUTION_TIMEOUT},
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
        except AttributeError as exc:
            logger.warning("Failed to build semantic XML: %s", exc)
            return ""