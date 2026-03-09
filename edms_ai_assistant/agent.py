# EDMS AI Assistant - Core Agent Module
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol
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
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.services.nlp_service import (
    EDMSNaturalLanguageService,
    SemanticDispatcher,
    UserIntent,
)
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class ContextParams:
    """Immutable контекст для выполнения агента."""

    user_token: str
    document_id: Optional[str] = None
    file_path: Optional[str] = None
    thread_id: str = "default"
    user_name: str = "пользователь"
    user_first_name: Optional[str] = None
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%d.%m.%Y")
    )

    def __post_init__(self):
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
        """Removes leading/trailing whitespace from the message."""
        return v.strip()

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Validates file_path as UUID or filesystem path.

        Supported formats:
        - UUID: 550e8400-e29b-41d4-a716-446655440000
        - Unix path: /tmp/file.docx
        - Windows path: C:\\Users\\...\\file.docx
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

        raise ValueError(f"Invalid file_path format: {v}")


class AgentResponse(BaseModel):
    """Стандартизированный ответ агента."""

    status: AgentStatus
    content: Optional[str] = None
    message: Optional[str] = None
    action_type: Optional[ActionType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IDocumentRepository(Protocol):
    """Интерфейс для работы с документами (Dependency Inversion)."""

    async def get_document(self, token: str, doc_id: str) -> Optional[DocumentDto]:
        """Fetches document metadata by ID."""
        ...


class DocumentRepository:
    """Реализация репозитория документов."""

    async def get_document(self, token: str, doc_id: str) -> Optional[DocumentDto]:
        """
        Fetches and validates document metadata from the EDMS API.

        Args:
            token: JWT authorization token.
            doc_id: UUID of the document.

        Returns:
            Validated DocumentDto or None on failure.
        """
        try:
            async with DocumentClient() as client:
                raw_data = await client.get_document_metadata(token, doc_id)
                doc = DocumentDto.model_validate(raw_data)
                logger.info(f"Document fetched: {doc_id}", extra={"doc_id": doc_id})
                return doc
        except Exception as e:
            logger.error(
                f"Failed to fetch document {doc_id}: {e}",
                exc_info=True,
                extra={"doc_id": doc_id, "error": str(e)},
            )
            return None


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
   - Статус "summarize_selection" → Предложи формат анализа (факты/пересказ/тезисы)
   - Статус "requires_disambiguation" → Покажи список, дождись выбора пользователя

4. **ВАЖНО**: После вызова инструментов ВСЕГДА формулируй финальный ответ на русском языке.

5. **Язык**: Только русский. Обращайся к пользователю по имени: {user_name}

6. **Строгая последовательность инструментов**:
   - Вызывай РОВНО ОДИН инструмент за раз
   - Дождись результата, затем решай, нужен ли следующий вызов
   - НИКОГДА не вызывай doc_summarize_text одновременно с doc_get_file_content
   - Правильный порядок: сначала получи текст (doc_get_file_content), потом анализируй (doc_summarize_text)
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
❌ Многословие, технические детали API
</response_format>"""

    CONTEXT_SNIPPETS = {
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
<date_parsing>
Преобразование дат в ISO 8601:
- "до 15 февраля" → "2026-02-15T23:59:59Z"
- "через неделю" → +7 дней от текущей даты
Всегда добавляй суффикс 'Z' (UTC timezone).
</date_parsing>""",
    }

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        semantic_xml: str,
    ) -> str:
        """
        Builds a full system prompt from context, intent, and semantic XML.

        Args:
            context: Immutable execution context.
            intent: Detected user intent for dynamic snippet injection.
            semantic_xml: Pre-built semantic XML block.

        Returns:
            Complete system prompt string.
        """
        base_prompt = cls.CORE_TEMPLATE.format(
            user_name=context.user_name,
            current_date=context.current_date,
            context_ui_id=context.document_id or "Не указан",
            local_file=context.file_path or "Не загружен",
        )

        dynamic_context = cls.CONTEXT_SNIPPETS.get(intent, "")
        return base_prompt + dynamic_context + semantic_xml


class ContentExtractor:
    """Извлечение финального контента из цепочки сообщений."""

    SKIP_PATTERNS = ["вызвал инструмент", "tool call", '"name"', '"id"']
    MIN_CONTENT_LENGTH = 50
    JSON_FIELDS = ["content", "text", "text_preview", "message"]

    @classmethod
    def extract_final_content(cls, messages: List[BaseMessage]) -> Optional[str]:
        """
        Extracts the final human-readable content from the message chain.

        Priority: last AIMessage with content → cleaned ToolMessage → fallback AIMessage.

        Args:
            messages: Full LangGraph message chain.

        Returns:
            Cleaned content string or None.
        """
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if cls._is_skip_content(content):
                    continue
                if len(content) > cls.MIN_CONTENT_LENGTH:
                    logger.debug(f"Extracted AIMessage: {len(content)} chars")
                    return content

        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._extract_from_tool_message(m)
                if extracted:
                    return extracted

        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if content:
                    logger.debug(f"Fallback AIMessage: {len(content)} chars")
                    return content

        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._extract_from_tool_message(m)
                if extracted:
                    logger.debug(
                        f"Fallback ToolMessage (cleaned): {len(extracted)} chars"
                    )
                    return extracted

        return None

    @classmethod
    def extract_last_text(cls, messages: List[BaseMessage]) -> Optional[str]:
        """
        Extracts text content from the most recent ToolMessage.

        Used to pass file content into doc_summarize_text.

        Args:
            messages: Full LangGraph message chain.

        Returns:
            Text content string or None.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                if isinstance(m.content, str) and m.content.startswith("{"):
                    data = json.loads(m.content)
                    text = (
                        data.get("content")
                        or data.get("text_preview")
                        or data.get("text")
                    )
                    if text and len(str(text)) > 100:
                        return str(text)
                if len(str(m.content)) > 100:
                    return str(m.content)
            except json.JSONDecodeError:
                if len(str(m.content)) > 100:
                    return str(m.content)
        return None

    _TECHNICAL_FIELDS = {
        "status", "meta", "format_used", "was_truncated",
        "text_length", "suggestion", "action_type",
        "added_count", "not_found", "partial_success",
    }

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """
        Strips technical JSON wrappers from the final content.

        Args:
            content: Raw content that may contain JSON envelopes.

        Returns:
            Clean text without technical metadata.
        """
        stripped = content.strip()
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
                for field_name in ("content", "message", "text"):
                    if data.get(field_name) and isinstance(data[field_name], str):
                        extracted = data[field_name].strip()
                        if len(extracted) > cls.MIN_CONTENT_LENGTH:
                            return extracted.replace("\\n", "\n").replace('\\"', '"')
            except json.JSONDecodeError:
                pass

        content = re.sub(r'\{"status":\s*"success",\s*"content":\s*"', "", content)
        content = re.sub(r'",\s*"meta":\s*\{[^}]*\}\s*\}', "", content)
        content = re.sub(r'"\s*\}$', "", content)
        content = content.replace('\\"', '"').replace("\\n", "\n")
        return content.strip()

    @classmethod
    def _is_skip_content(cls, content: str) -> bool:
        """Returns True if content is a technical marker that should be skipped."""
        return any(skip in content.lower() for skip in cls.SKIP_PATTERNS)

    @classmethod
    def _extract_from_tool_message(cls, message: ToolMessage) -> Optional[str]:
        """
        Safely extracts human-readable content from a ToolMessage.

        Args:
            message: LangGraph ToolMessage to parse.

        Returns:
            Extracted text or None.
        """
        try:
            if isinstance(message.content, str) and message.content.strip().startswith(
                "{"
            ):
                data = json.loads(message.content)
                for field_name in cls.JSON_FIELDS:
                    if field_name in data and data[field_name]:
                        content = str(data[field_name]).strip()
                        if len(content) > cls.MIN_CONTENT_LENGTH:
                            logger.debug(
                                f"ToolMessage JSON[{field_name}]: {len(content)} chars"
                            )
                            return content
        except json.JSONDecodeError:
            pass
        return None


class AgentStateManager:
    """Управление состоянием LangGraph агента."""

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver):
        """
        Initializes the state manager.

        Args:
            graph: Compiled LangGraph state graph.
            checkpointer: In-memory checkpoint storage.
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
        """Retrieves current graph state for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aget_state(config)

    async def update_state(
        self,
        thread_id: str,
        messages: List[BaseMessage],
        as_node: str = "agent",
    ) -> None:
        """Updates graph state with new messages."""
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
        """Invokes the graph with optional inputs and a timeout."""
        config = {"configurable": {"thread_id": thread_id}}
        await asyncio.wait_for(
            self.graph.ainvoke(inputs, config=config), timeout=timeout
        )


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


def _is_valid_uuid(value: str) -> bool:
    """Returns True if value matches the UUID4 pattern."""
    return bool(_UUID_RE.match(value))


class EdmsDocumentAgent:
    """Production-ready мультиагентная система для EDMS."""

    MAX_ITERATIONS = 10
    EXECUTION_TIMEOUT = 120.0

    def __init__(
        self,
        document_repo: Optional[IDocumentRepository] = None,
        semantic_dispatcher: Optional[SemanticDispatcher] = None,
    ):
        """
        Initializes the EDMS document agent and compiles the LangGraph workflow.

        Args:
            document_repo: Document repository implementation (DI).
            semantic_dispatcher: NLP semantic dispatcher (DI).
        """
        try:
            self.model = get_chat_model()
            self.tools = all_tools
            self.document_repo = document_repo or DocumentRepository()
            self.dispatcher = semantic_dispatcher or SemanticDispatcher()

            logger.debug("Base components initialized")

            self._checkpointer = MemorySaver()
            logger.debug("Checkpointer created")

            self._compiled_graph = self._build_graph()

            if self._compiled_graph is None:
                raise RuntimeError("Graph compilation returned None")

            logger.debug(
                "Graph compiled successfully",
                extra={"graph_type": type(self._compiled_graph).__name__},
            )

            self.state_manager = AgentStateManager(
                graph=self._compiled_graph,
                checkpointer=self._checkpointer,
            )

            logger.info(
                "EdmsDocumentAgent initialized successfully",
                extra={
                    "tools_count": len(self.tools),
                    "model": str(self.model),
                    "has_state_manager": hasattr(self, "state_manager"),
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize EdmsDocumentAgent: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """Returns agent component health status."""
        return {
            "model": self.model is not None,
            "tools": len(self.tools) > 0,
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": (
                hasattr(self, "_compiled_graph")
                and self._compiled_graph is not None
            ),
            "state_manager": (
                hasattr(self, "state_manager") and self.state_manager is not None
            ),
            "checkpointer": (
                hasattr(self, "_checkpointer") and self._checkpointer is not None
            ),
        }

    def _build_graph(self) -> CompiledStateGraph:
        """
        Compiles the LangGraph workflow with interrupt_before=['tools'].

        Graph topology: agent → (tools → validator → agent)* → END

        Returns:
            Compiled and interrupt-enabled state graph.
        """
        from typing import Annotated, List, TypedDict

        class AgentState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]

        workflow = StateGraph(AgentState)

        async def call_model(state: AgentState) -> Dict:
            """
            Invokes the LLM with bound tools.

            ВАЖНО: parallel_tool_calls НЕ передаётся — кастомная модель
            не поддерживает этот параметр и игнорирует все tool_calls при
            его наличии. Параллельные вызовы блокируются в _orchestrate.
            """
            # bind_tools без parallel_tool_calls — модель поддерживает только стандартный API
            model_with_tools = self.model.bind_tools(self.tools)

            non_sys = [
                m for m in state["messages"] if not isinstance(m, SystemMessage)
            ]
            sys_msgs = [
                m for m in state["messages"] if isinstance(m, SystemMessage)
            ]
            final_messages = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            response = await model_with_tools.ainvoke(final_messages)
            return {"messages": [response]}

        async def validator(state: AgentState) -> Dict:
            """
            Post-tool validator: injects error notifications for failed tool calls.
            """
            messages = state["messages"]
            last_message = messages[-1]

            if not isinstance(last_message, ToolMessage):
                return {"messages": []}

            content_raw = str(last_message.content).strip()

            if not content_raw or content_raw in ("None", "{}"):
                return {
                    "messages": [
                        HumanMessage(
                            content="[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Инструмент вернул пустой результат."
                        )
                    ]
                }

            if "error" in content_raw.lower() or "exception" in content_raw.lower():
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Техническая ошибка: {content_raw}"
                        )
                    ]
                }

            return {"messages": []}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validator", validator)
        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and getattr(
                last_message, "tool_calls", None
            ):
                return "tools"
            return END

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
            logger.error(f"Graph compilation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to compile graph: {e}") from e

    async def chat(
        self,
        message: str,
        user_token: str,
        context_ui_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_context: Optional[Dict] = None,
        file_path: Optional[str] = None,
        human_choice: Optional[str] = None,
    ) -> Dict:
        """
        Main entry point for agent interaction.

        Args:
            message: User message text.
            user_token: JWT authorization token.
            context_ui_id: Active document UUID in the EDMS UI.
            thread_id: Conversation thread identifier.
            user_context: Optional user profile dict.
            file_path: UUID of an EDMS attachment or local filesystem path.
            human_choice: Disambiguation or summarization type choice from UI.

        Returns:
            Serialized AgentResponse dict.
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
            state = await self.state_manager.get_state(context.thread_id)

            if human_choice and state.next:
                return await self._handle_human_choice(context, human_choice)

            document = None
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

            sys_msg = SystemMessage(content=full_prompt)
            hum_msg = HumanMessage(content=refined_message)
            inputs = {"messages": [sys_msg, hum_msg]}

            return await self._orchestrate(
                context=context,
                inputs=inputs,
                is_choice_active=bool(human_choice),
                iteration=0,
            )

        except Exception as e:
            logger.error(
                f"Chat error: {e}", exc_info=True, extra={"user_message": message}
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {str(e)}",
            ).model_dump()

    async def _handle_human_choice(
        self, context: ContextParams, human_choice: str
    ) -> Dict:
        """
        Resumes a paused graph after the user makes a disambiguation or
        summarization type choice.

        Args:
            context: Immutable execution context.
            human_choice: Raw user choice string (UUID list or summary type).

        Returns:
            Serialized AgentResponse dict.
        """
        state = await self.state_manager.get_state(context.thread_id)
        last_msg = state.values["messages"][-1]

        fixed_calls = []
        for tc in getattr(last_msg, "tool_calls", []):
            t_args = dict(tc["args"])
            t_name = tc["name"]

            if t_name == "doc_summarize_text":
                t_args["summary_type"] = human_choice

            elif t_name == "introduction_create_tool":
                raw_ids = [x.strip() for x in human_choice.split(",") if x.strip()]
                valid_ids = []
                for raw_id in raw_ids:
                    try:
                        UUID(raw_id)
                        valid_ids.append(raw_id)
                    except ValueError:
                        logger.warning(f"Invalid UUID in human_choice: {raw_id}")
                if valid_ids:
                    t_args["selected_employee_ids"] = valid_ids
                    t_args.pop("last_names", None)
                    logger.info(
                        "Resolved disambiguation for introduction",
                        extra={"employee_ids": valid_ids},
                    )

            elif t_name == "task_create_tool":
                raw_ids = [x.strip() for x in human_choice.split(",") if x.strip()]
                valid_ids = [rid for rid in raw_ids if _is_valid_uuid(rid)]
                if valid_ids:
                    t_args["selected_employee_ids"] = valid_ids
                    t_args.pop("executor_last_names", None)
                    logger.info(
                        "Resolved disambiguation for task",
                        extra={"employee_ids": valid_ids},
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

    async def _orchestrate(
        self,
        context: ContextParams,
        inputs: Optional[Dict],
        is_choice_active: bool = False,
        iteration: int = 0,
    ) -> Dict:
        """
        Core orchestration loop: invokes the graph, injects token/doc_id,
        handles parallel tool_calls by keeping only the first call per turn,
        and resumes until END or max iterations.

        Args:
            context: Immutable execution context.
            inputs: Initial graph inputs (None when resuming from interrupt).
            is_choice_active: True when resuming after a human choice.
            iteration: Current recursion depth (guards against infinite loops).

        Returns:
            Serialized AgentResponse dict.
        """
        if iteration > self.MAX_ITERATIONS:
            logger.error(
                "Max iterations exceeded",
                extra={"thread_id": context.thread_id},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышен лимит итераций обработки.",
            ).model_dump()

        try:
            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self.EXECUTION_TIMEOUT,
            )

            state = await self.state_manager.get_state(context.thread_id)
            messages = state.values.get("messages", [])

            logger.debug(
                "State snapshot",
                extra={
                    "thread_id": context.thread_id,
                    "iteration": iteration,
                    "messages_count": len(messages),
                    "last_message_type": (
                        type(messages[-1]).__name__ if messages else None
                    ),
                    "has_tool_calls": (
                        bool(
                            isinstance(messages[-1], AIMessage)
                            and getattr(messages[-1], "tool_calls", None)
                        )
                        if messages
                        else False
                    ),
                    "state_next": state.next,
                },
            )

            if not messages:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Пустое состояние агента.",
                ).model_dump()

            last_msg = messages[-1]

            if (
                not state.next
                or not isinstance(last_msg, AIMessage)
                or not getattr(last_msg, "tool_calls", None)
            ):
                final_content = ContentExtractor.extract_final_content(messages)
                if final_content:
                    final_content = ContentExtractor.clean_json_artifacts(
                        final_content
                    )
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
                    "No final content found",
                    extra={"thread_id": context.thread_id},
                )
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content="Анализ завершен.",
                ).model_dump()

            last_extracted_text = ContentExtractor.extract_last_text(messages)

            raw_tool_calls = last_msg.tool_calls

            if len(raw_tool_calls) > 1:
                logger.warning(
                    "Model returned parallel tool_calls — keeping only the first",
                    extra={
                        "total": len(raw_tool_calls),
                        "kept": raw_tool_calls[0]["name"],
                        "dropped": [tc["name"] for tc in raw_tool_calls[1:]],
                        "thread_id": context.thread_id,
                    },
                )
                raw_tool_calls = raw_tool_calls[:1]

            fixed_calls = []

            for tc in raw_tool_calls:
                t_name = tc["name"]
                t_args = dict(tc["args"])
                t_id = tc["id"]

                t_args["token"] = context.user_token

                clean_path = (
                    str(context.file_path).strip() if context.file_path else ""
                )
                is_uuid = _is_valid_uuid(clean_path)

                if is_uuid and t_name == "read_local_file_content":
                    t_name = "doc_get_file_content"
                    t_args["attachment_id"] = clean_path
                    t_args.pop("file_path", None)
                    logger.info(
                        "Converted UUID path to doc_get_file_content",
                        extra={"attachment_id": clean_path[:8] + "..."},
                    )

                if t_name == "doc_get_file_content" and is_uuid:
                    current_att = str(t_args.get("attachment_id", "")).strip()
                    if not current_att or not _is_valid_uuid(current_att):
                        t_args["attachment_id"] = clean_path
                        logger.info(
                            "Injected attachment_id from context",
                            extra={"attachment_id": clean_path[:8] + "..."},
                        )

                if context.document_id and (
                    t_name.startswith("doc_")
                    or "document_id" in t_args
                    or t_name
                    in ("introduction_create_tool", "task_create_tool")
                ):
                    t_args["document_id"] = context.document_id

                if t_name == "doc_summarize_text":
                    if last_extracted_text:
                        t_args["text"] = str(last_extracted_text)

                    if not t_args.get("summary_type"):
                        if is_choice_active:
                            t_args["summary_type"] = "extractive"
                            logger.info(
                                "Fallback summary_type=extractive "
                                "(choice_active, no type in args)"
                            )
                        else:
                            nlp = EDMSNaturalLanguageService()
                            suggestion = nlp.suggest_summarize_format(
                                str(last_extracted_text)
                                if last_extracted_text
                                else ""
                            )
                            recommended_type = suggestion.get(
                                "recommended", "extractive"
                            )
                            t_args["summary_type"] = recommended_type
                            logger.info(
                                "Auto-selected summary_type",
                                extra={
                                    "type": recommended_type,
                                    "reason": suggestion.get("reason"),
                                    "text_length": suggestion.get(
                                        "stats", {}
                                    ).get("chars", 0),
                                },
                            )

                fixed_calls.append({"name": t_name, "args": t_args, "id": t_id})

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
                is_choice_active=is_choice_active,
                iteration=iteration + 1,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Execution timeout", extra={"thread_id": context.thread_id}
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения.",
            ).model_dump()
        except Exception as e:
            logger.error(
                f"Orchestration error: {e}",
                exc_info=True,
                extra={"thread_id": context.thread_id},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка оркестрации: {str(e)}",
            ).model_dump()

    async def _build_context(self, request: AgentRequest) -> ContextParams:
        """
        Builds an immutable ContextParams from a validated AgentRequest.

        Args:
            request: Validated agent request.

        Returns:
            Immutable ContextParams instance.
        """
        user_name = (
            request.user_context.get("firstName")
            or request.user_context.get("name")
            or "пользователь"
        ).strip()

        return ContextParams(
            user_token=request.user_token,
            document_id=request.context_ui_id,
            file_path=request.file_path,
            thread_id=request.thread_id or "default",
            user_name=user_name,
            user_first_name=request.user_context.get("firstName"),
        )

    @staticmethod
    def _build_semantic_xml(semantic_context: Any) -> str:
        """
        Serializes semantic context into an XML block for the system prompt.

        Args:
            semantic_context: Output of SemanticDispatcher.build_context().

        Returns:
            XML string with user query metadata.
        """
        return f"""
<semantic_context>
  <user_query>
    <original>{semantic_context.query.original}</original>
    <refined>{semantic_context.query.refined}</refined>
    <intent>{semantic_context.query.intent.value}</intent>
    <complexity>{semantic_context.query.complexity.value}</complexity>
  </user_query>
</semantic_context>"""