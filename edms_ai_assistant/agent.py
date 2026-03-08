import ast
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Annotated, Dict, List, Optional, Protocol, TypedDict

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

_CHOICE_NORM: dict[str, str] = {
    "1": "extractive",
    "факты": "extractive",
    "ключевые факты": "extractive",
    "extractive": "extractive",
    "2": "abstractive",
    "пересказ": "abstractive",
    "краткий пересказ": "abstractive",
    "abstractive": "abstractive",
    "3": "thesis",
    "тезисы": "thesis",
    "тезисный план": "thesis",
    "thesis": "thesis",
}

_MAX_HISTORY_MESSAGES = 20


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
    current_date: str = field(default_factory=lambda: datetime.now().strftime("%d.%m.%Y"))

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
        return v.strip()

    @field_validator("human_choice")
    @classmethod
    def normalize_human_choice(cls, v: Optional[str]) -> Optional[str]:
        """Normalize human_choice aliases to canonical summary type values.

        Args:
            v: Raw human_choice value from request.

        Returns:
            Canonical summary type string or None.
        """
        if not v:
            return None
        normalized = _CHOICE_NORM.get(v.strip().lower())
        return normalized if normalized else v.strip().lower()

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate file_path as UUID or filesystem path.

        Args:
            v: Raw file path value.

        Returns:
            Validated path string or None.

        Raises:
            ValueError: If format is invalid.
        """
        if not v:
            return None
        if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", v, re.I):
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
        """Fetch document metadata by ID.

        Args:
            token: User bearer token.
            doc_id: Document UUID.

        Returns:
            DocumentDto or None.
        """
        ...


class DocumentRepository:
    """Реализация репозитория документов."""

    async def get_document(self, token: str, doc_id: str) -> Optional[DocumentDto]:
        """Fetch document metadata from EDMS API.

        Args:
            token: User bearer token.
            doc_id: Document UUID.

        Returns:
            Validated DocumentDto or None on error.
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
- Если инструмент вернул "requires_disambiguation" → НЕМЕДЛЕННО покажи пользователю список сотрудников для выбора
- НЕ выбирай первого сотрудника автоматически — всегда жди явного выбора пользователя
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
        human_choice: Optional[str] = None,
    ) -> str:
        """Build a full system prompt for the given context and intent.

        Args:
            context: Immutable context parameters.
            intent: Detected user intent for dynamic snippet injection.
            semantic_xml: XML block with semantic analysis results.
            human_choice: Pre-selected summarisation format (if provided by UI).

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

        choice_block = ""
        if human_choice:
            choice_block = f"""
<summarize_instruction>
ФОРМАТ АНАЛИЗА УЖЕ ВЫБРАН ПОЛЬЗОВАТЕЛЕМ: "{human_choice}"
Обязательный порядок действий:
1. Вызови doc_get_file_content чтобы получить текст вложения
2. Сразу после получения текста вызови doc_summarize_text(summary_type="{human_choice}")
НЕ спрашивай у пользователя формат — он уже выбран. НЕ показывай текст вложения напрямую.
</summarize_instruction>"""

        return base_prompt + dynamic_context + choice_block + semantic_xml


class ContentExtractor:
    """Извлечение финального контента из цепочки сообщений."""

    SKIP_PATTERNS = ["вызвал инструмент", "tool call", '"name"', '"id"']
    MIN_CONTENT_LENGTH = 50
    JSON_FIELDS = ["content", "text", "text_preview", "message"]

    @classmethod
    def extract_final_content(cls, messages: List[BaseMessage]) -> Optional[str]:
        """Extract the best final content from message chain.

        Args:
            messages: Full message list from agent state.

        Returns:
            Best content string or None.
        """
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                content = str(m.content).strip()
                if cls._is_skip_content(content):
                    continue
                if len(content) > cls.MIN_CONTENT_LENGTH:
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
                    return content

        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                content = str(m.content).strip()
                if len(content) > cls.MIN_CONTENT_LENGTH:
                    return content

        return None

    @classmethod
    def extract_last_text(cls, messages: List[BaseMessage]) -> Optional[str]:
        """Extract the last substantial text from tool messages.

        Args:
            messages: Full message list from agent state.

        Returns:
            Text string or None.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                raw = str(m.content)
                if raw.startswith("{"):
                    # Поддержка JSON и Python repr
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        try:
                            data = ast.literal_eval(raw)
                        except (ValueError, SyntaxError):
                            data = {}
                    if isinstance(data, dict):
                        # Не извлекаем text из disambiguation — там нет текста файла
                        if data.get("status") in ("requires_disambiguation", "requires_action"):
                            continue
                        text = data.get("content") or data.get("text_preview") or data.get("text")
                        if text and len(str(text)) > 100:
                            return str(text)
                if len(raw) > 100:
                    return raw
            except Exception:
                pass
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """Remove JSON envelope artifacts from content strings.

        Args:
            content: Raw content string potentially containing JSON wrappers.

        Returns:
            Cleaned content string.
        """
        if content.startswith('{"status"'):
            try:
                data = json.loads(content)
                if "content" in data:
                    content = data["content"]
            except json.JSONDecodeError:
                pass

        content = content.replace('{"status": "success", "content": "', "")
        content = content.replace('"}', "")
        content = content.replace('\\"', '"')
        content = content.replace("\\n", "\n")
        return content.strip()

    @classmethod
    def _is_skip_content(cls, content: str) -> bool:
        return any(skip in content.lower() for skip in cls.SKIP_PATTERNS)

    @classmethod
    def _extract_from_tool_message(cls, message: ToolMessage) -> Optional[str]:
        try:
            raw = message.content
            if isinstance(raw, str) and raw.strip().startswith("{"):
                # Поддержка и JSON и Python repr (одинарные кавычки от ToolNode)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        return None
                if not isinstance(data, dict):
                    return None
                # Не возвращаем requires_disambiguation как content — это служебный статус
                if data.get("status") in ("requires_disambiguation", "requires_action"):
                    return None
                for f in cls.JSON_FIELDS:
                    if f in data and data[f]:
                        content_val = str(data[f]).strip()
                        if len(content_val) > cls.MIN_CONTENT_LENGTH:
                            return content_val
        except Exception:
            pass
        return None


def _trim_history(messages: List[BaseMessage], max_messages: int) -> List[BaseMessage]:
    """Trim message history while always preserving the last SystemMessage.

    Keeps the most recent SystemMessage and the last ``max_messages`` non-system
    messages to prevent unbounded context growth and hallucinations from stale history.

    Args:
        messages: Full message list from agent state.
        max_messages: Maximum number of non-system messages to retain.

    Returns:
        Trimmed message list with one SystemMessage prepended.
    """
    sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_sys = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_sys) > max_messages:
        logger.debug(
            "Trimming history: %d → %d non-system messages",
            len(non_sys),
            max_messages,
        )
        non_sys = non_sys[-max_messages:]

    return ([sys_msgs[-1]] if sys_msgs else []) + non_sys


class AgentStateManager:
    """Управление состоянием LangGraph агента."""

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
        if graph is None:
            raise ValueError("Graph cannot be None")
        if checkpointer is None:
            raise ValueError("Checkpointer cannot be None")
        self.graph = graph
        self.checkpointer = checkpointer

    async def get_state(self, thread_id: str) -> Any:
        """Retrieve current state snapshot for thread.

        Args:
            thread_id: LangGraph thread identifier.

        Returns:
            State snapshot object.
        """
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aget_state(config)

    async def update_state(
        self,
        thread_id: str,
        messages: List[BaseMessage],
        as_node: str = "agent",
    ) -> None:
        """Update agent state with new messages.

        Args:
            thread_id: LangGraph thread identifier.
            messages: Messages to inject into state.
            as_node: Node name to attribute the update to.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await self.graph.aupdate_state(config, {"messages": messages}, as_node=as_node)

    async def invoke(
        self,
        inputs: Dict[str, Any],
        thread_id: str,
        timeout: float = 120.0,
    ) -> None:
        """Invoke the graph with timeout protection.

        Args:
            inputs: Graph input dict (None resumes from interrupt).
            thread_id: LangGraph thread identifier.
            timeout: Maximum execution seconds before TimeoutError.
        """
        config = {"configurable": {"thread_id": thread_id}}
        await asyncio.wait_for(self.graph.ainvoke(inputs, config=config), timeout=timeout)


class EdmsDocumentAgent:
    """Production-ready мультиагентная система для EDMS."""

    MAX_ITERATIONS = 10
    EXECUTION_TIMEOUT = 120.0

    def __init__(
        self,
        document_repo: Optional[IDocumentRepository] = None,
        semantic_dispatcher: Optional[SemanticDispatcher] = None,
    ) -> None:
        try:
            self.model = get_chat_model()
            self.tools = all_tools
            self.document_repo = document_repo or DocumentRepository()
            self.dispatcher = semantic_dispatcher or SemanticDispatcher()

            self._checkpointer = MemorySaver()
            self._compiled_graph = self._build_graph()

            if self._compiled_graph is None:
                raise RuntimeError("Graph compilation returned None")

            self.state_manager = AgentStateManager(
                graph=self._compiled_graph,
                checkpointer=self._checkpointer,
            )

            self._pending_disambiguations: Dict[str, Dict[str, Any]] = {}

            logger.info(
                "EdmsDocumentAgent initialized successfully",
                extra={
                    "tools_count": len(self.tools),
                    "model": str(self.model),
                },
            )

        except Exception as e:
            logger.error("Failed to initialize EdmsDocumentAgent: %s", e, exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """Return health indicators for readiness probe.

        Returns:
            Dict of component name to boolean status.
        """
        return {
            "model": self.model is not None,
            "tools": len(self.tools) > 0,
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": hasattr(self, "_compiled_graph") and self._compiled_graph is not None,
            "state_manager": hasattr(self, "state_manager") and self.state_manager is not None,
            "checkpointer": hasattr(self, "_checkpointer") and self._checkpointer is not None,
        }

    def _build_graph(self) -> CompiledStateGraph:
        """Compile the LangGraph state machine.

        Returns:
            Compiled graph ready for invocation.

        Raises:
            RuntimeError: If compilation fails.
        """

        class AgentState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]

        workflow = StateGraph(AgentState)

        async def call_model(state: AgentState) -> Dict:
            model_with_tools = self.model.bind_tools(self.tools)
            trimmed = _trim_history(state["messages"], _MAX_HISTORY_MESSAGES)
            response = await model_with_tools.ainvoke(trimmed)
            return {"messages": [response]}

        async def validator(state: AgentState) -> Dict:
            """Validate tool results and inject system notifications on errors."""
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
            if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
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
            logger.error("Graph compilation failed: %s", e, exc_info=True)
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
        """Main entry point: process a user message through the agent.

        Args:
            message: User text input.
            user_token: JWT bearer token.
            context_ui_id: Active document UUID.
            thread_id: Conversation thread identifier.
            user_context: Dict with user profile data.
            file_path: Local file path or attachment UUID.
            human_choice: User's selection for summarization format or disambiguation.

        Returns:
            AgentResponse dict.
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

            normalized_choice = request.human_choice

            if normalized_choice and state.next:
                return await self._handle_human_choice(context, normalized_choice)

            # Проверяем pending disambiguation ДО запуска графа.
            # Если пользователь написал "3", "Иванов Игорь" и т.д. после показа списка —
            # сразу резолвим выбор и вызываем инструмент напрямую, без LLM.
            if not normalized_choice:
                disambiguation_result = await self._handle_disambiguation_choice(
                    context=context,
                    user_message=message,
                )
                if disambiguation_result is not None:
                    return disambiguation_result

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
                    "human_choice": normalized_choice,
                },
            )

            refined_message = semantic_context.query.refined
            user_intent = semantic_context.query.intent

            semantic_xml = self._build_semantic_xml(semantic_context)
            full_prompt = PromptBuilder.build(context, user_intent, semantic_xml, human_choice=normalized_choice)

            sys_msg = SystemMessage(content=full_prompt)
            hum_msg = HumanMessage(content=refined_message)
            inputs = {"messages": [sys_msg, hum_msg]}

            return await self._orchestrate(
                context=context,
                inputs=inputs,
                human_choice=normalized_choice,
                iteration=0,
            )

        except Exception as e:
            logger.error("Chat error: %s", e, exc_info=True, extra={"user_message": message})
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {str(e)}",
            ).model_dump()

    async def _handle_human_choice(
        self, context: ContextParams, human_choice: Optional[str]
    ) -> Dict:
        """Resume execution after user selected a summarization format or employee.

        The normalized human_choice (already canonical via AgentRequest validator)
        is injected into the pending tool call arguments so the correct format
        is used — regardless of what the LLM originally placed there.

        Args:
            context: Current execution context.
            human_choice: Canonical choice string (e.g. 'extractive', 'abstractive', 'thesis').

        Returns:
            AgentResponse dict.
        """
        state = await self.state_manager.get_state(context.thread_id)
        last_msg = state.values["messages"][-1]

        if not isinstance(last_msg, AIMessage) or not getattr(last_msg, "tool_calls", None):
            logger.warning(
                "handle_human_choice: no pending tool calls in state",
                extra={"thread_id": context.thread_id},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Нет ожидающих действий для обработки выбора.",
            ).model_dump()

        fixed_calls = []
        for tc in last_msg.tool_calls:
            t_args = dict(tc["args"])
            t_name = tc["name"]

            if t_name == "doc_summarize_text" and human_choice:
                t_args["summary_type"] = human_choice
                logger.info(
                    "Injected user summary_type into tool call",
                    extra={"summary_type": human_choice, "thread_id": context.thread_id},
                )

            fixed_calls.append({"name": t_name, "args": t_args, "id": tc["id"]})

        await self.state_manager.update_state(
            context.thread_id,
            [AIMessage(content=last_msg.content or "", tool_calls=fixed_calls, id=last_msg.id)],
            as_node="agent",
        )

        return await self._orchestrate(
            context=context,
            inputs=None,
            human_choice=human_choice,
            iteration=0,
        )

    async def _orchestrate(
        self,
        context: ContextParams,
        inputs: Optional[Dict],
        human_choice: Optional[str] = None,
        iteration: int = 0,
    ) -> Dict:
        """Core orchestration loop: invoke graph, inspect state, handle interrupts.

        Args:
            context: Execution context parameters.
            inputs: Graph inputs dict (None when resuming from interrupt).
            human_choice: Canonical user choice for summarization (already normalized).
            iteration: Current recursion depth counter.

        Returns:
            AgentResponse dict.
        """
        if iteration > self.MAX_ITERATIONS:
            logger.error("Max iterations exceeded", extra={"thread_id": context.thread_id})
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
                # Проверяем disambiguation ПЕРВЫМ — tools уже выполнились (iteration >= 1)
                # и ToolMessage с requires_disambiguation мог появиться после нашего
                # предыдущего invoke. Извлекаем tool_name из предшествующего AIMessage.
                if iteration >= 1:
                    prior_fixed_calls = self._extract_prior_tool_calls(messages)
                    requires_disambiguation = self._check_tool_disambiguation(
                        messages=messages,
                        thread_id=context.thread_id,
                        fixed_calls=prior_fixed_calls,
                    )
                    if requires_disambiguation:
                        return requires_disambiguation

                if human_choice and iteration == 0:
                    pending_text = ContentExtractor.extract_last_text(messages)
                    if pending_text and len(pending_text) > 100:
                        logger.info(
                            "LLM skipped doc_summarize_text but human_choice present — forcing summarize call",
                            extra={
                                "human_choice": human_choice,
                                "text_length": len(pending_text),
                                "thread_id": context.thread_id,
                            },
                        )
                        import uuid as _uuid
                        forced_tool_id = f"forced_summarize_{_uuid.uuid4().hex[:8]}"
                        forced_call = AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "doc_summarize_text",
                                    "args": {
                                        "text": pending_text,
                                        "summary_type": human_choice,
                                        "token": context.user_token,
                                    },
                                    "id": forced_tool_id,
                                }
                            ],
                        )
                        await self.state_manager.update_state(
                            context.thread_id,
                            [forced_call],
                            as_node="agent",
                        )
                        return await self._orchestrate(
                            context=context,
                            inputs=None,
                            human_choice=human_choice,
                            iteration=iteration + 1,
                        )

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

                logger.warning("No final content found", extra={"thread_id": context.thread_id})
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    content="Анализ завершен.",
                ).model_dump()

            last_extracted_text = ContentExtractor.extract_last_text(messages)
            fixed_calls = []

            for tc in last_msg.tool_calls:
                t_name, t_args, t_id = tc["name"], dict(tc["args"]), tc["id"]

                t_args["token"] = context.user_token

                clean_path = str(context.file_path).strip() if context.file_path else ""
                is_uuid = bool(
                    re.match(
                        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                        clean_path,
                        re.I,
                    )
                )

                if is_uuid and t_name == "read_local_file_content":
                    t_name = "doc_get_file_content"
                    t_args["attachment_id"] = clean_path
                    t_args.pop("file_path", None)
                    logger.info("Converted UUID to attachment_id: %s...", clean_path[:8])

                if is_uuid and t_name == "doc_get_file_content" and not t_args.get("attachment_id"):
                    t_args["attachment_id"] = clean_path
                    logger.info(
                        "Injected file_path UUID as attachment_id: %s...",
                        clean_path[:8],
                    )

                if context.document_id and (
                    t_name.startswith("doc_")
                    or "document_id" in t_args
                    or t_name in ["introduction_create_tool", "task_create_tool"]
                ):
                    t_args["document_id"] = context.document_id

                if t_name == "doc_summarize_text":
                    if last_extracted_text:
                        t_args["text"] = str(last_extracted_text)

                    # Защита: если текст пустой или слишком мал — пропустить этот tool call.
                    # Это происходит когда LLM ставит doc_summarize_text и doc_get_file_content
                    # в один пакет. doc_summarize_text будет вызван на следующей итерации
                    # после того как doc_get_file_content вернёт текст.
                    current_text = t_args.get("text", "")
                    if len(str(current_text).strip()) < 200:
                        logger.info(
                            "doc_summarize_text skipped: text too short (%d chars), "
                            "will be called after doc_get_file_content",
                            len(str(current_text).strip()),
                            extra={"thread_id": context.thread_id},
                        )
                        continue  # Пропускаем этот tool call в fixed_calls

                    if human_choice:
                        t_args["summary_type"] = human_choice
                        logger.info(
                            "Applied user choice to summary_type",
                            extra={
                                "summary_type": human_choice,
                                "thread_id": context.thread_id,
                            },
                        )
                    elif t_args.get("summary_type"):
                        logger.info(
                            "LLM provided summary_type: %s",
                            t_args["summary_type"],
                            extra={"thread_id": context.thread_id},
                        )
                    else:
                        logger.info(
                            "No summary_type and no human_choice — asking user",
                            extra={"thread_id": context.thread_id},
                        )
                        return AgentResponse(
                            status=AgentStatus.REQUIRES_ACTION,
                            action_type=ActionType.SUMMARIZE_SELECTION,
                            message=(
                                "Выберите формат анализа:\n"
                                "1. Ключевые факты\n"
                                "2. Краткий пересказ\n"
                                "3. Тезисный план"
                            ),
                        ).model_dump()

                fixed_calls.append({"name": t_name, "args": t_args, "id": t_id})

            await self.state_manager.update_state(
                context.thread_id,
                [AIMessage(content=last_msg.content or "", tool_calls=fixed_calls, id=last_msg.id)],
                as_node="agent",
            )

            return await self._orchestrate(
                context=context,
                inputs=None,
                human_choice=human_choice,
                iteration=iteration + 1,
            )

        except asyncio.TimeoutError:
            logger.error("Execution timeout", extra={"thread_id": context.thread_id})
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения.",
            ).model_dump()
        except Exception as e:
            logger.error(
                "Orchestration error: %s", e,
                exc_info=True,
                extra={"thread_id": context.thread_id},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка оркестрации: {str(e)}",
            ).model_dump()

    def _check_tool_disambiguation(
        self,
        messages: List[BaseMessage],
        thread_id: str = "",
        fixed_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict]:
        """Inspect recent ToolMessages for requires_disambiguation status.

        When a tool (task_create_tool, introduction_create_tool, employee_search_tool)
        returns ``requires_disambiguation``, the agent surfaces the employee list to the
        user and stores pending state so the next user message can resume directly.

        Args:
            messages: Current message list from agent state.
            thread_id: LangGraph thread identifier for pending state storage.
            fixed_calls: Resolved tool call list to capture original args for resume.

        Returns:
            AgentResponse dict if disambiguation is needed, else None.
        """
        # Ищем ToolMessage с disambiguation в ПОСЛЕДНЕМ блоке ToolMessage-ов.
        # Блок определяется как все ToolMessage между двумя AIMessage-ами.
        # Нельзя просто брать с конца — после tools идут validator HumanMessage и финальный AIMessage.
        tool_block: List[ToolMessage] = []
        collecting = False
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                collecting = True
                tool_block.append(m)
            elif collecting:
                # Первый не-ToolMessage после блока ToolMessage — остановка
                break

        for m in tool_block:
            try:
                content_str = str(m.content).strip()
                if not content_str.startswith("{"):
                    continue
                # ToolNode может сериализовать dict через str() → одинарные кавычки.
                # json.loads не справится — используем ast.literal_eval как fallback.
                try:
                    data = json.loads(content_str)
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(content_str)
                    except (ValueError, SyntaxError):
                        continue
                if not isinstance(data, dict):
                    continue
                status = data.get("status", "")
                if status in ("requires_disambiguation", "requires_action"):
                    action = data.get("action_type", "select_employee")
                    choices = data.get("ambiguous_matches") or data.get("choices", [])
                    tool_message = data.get("message", "Найдено несколько совпадений.")

                    formatted_list = self._format_employee_choices(choices)
                    # Используем только форматированный список — без "укажите ID" от LLM.
                    # tool_message содержит только заголовок типа "Найдено несколько совпадений."
                    # Конкретный призыв "укажите номер" убираем — пользователь понимает из контекста.
                    if formatted_list:
                        full_message = f"{tool_message}\n\n{formatted_list}"
                    else:
                        full_message = tool_message

                    # Сохраняем pending state: tool_name + original_args для resume
                    if thread_id and fixed_calls:
                        originating_call = next(
                            (
                                c for c in fixed_calls
                                if c["name"] in (
                                    "introduction_create_tool",
                                    "task_create_tool",
                                    "employee_search_tool",
                                )
                            ),
                            None,
                        )
                        if originating_call:
                            self._store_pending_disambiguation(
                                thread_id=thread_id,
                                tool_name=originating_call["name"],
                                choices=choices,
                                original_tool_args=dict(originating_call["args"]),
                            )

                    logger.info(
                        "Disambiguation required from tool",
                        extra={"action_type": action, "choices_count": len(choices)},
                    )
                    return AgentResponse(
                        status=AgentStatus.REQUIRES_ACTION,
                        action_type=ActionType.DISAMBIGUATION,
                        message=full_message,
                        metadata={"choices": choices, "action_type": action},
                    ).model_dump()
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    @staticmethod
    def _format_employee_choices(choices: List[Dict[str, Any]]) -> str:
        """Format employee choice list into a human-readable Russian string.

        Args:
            choices: List of employee dicts with id, full_name, post, department keys.

        Returns:
            Formatted multiline string or empty string if list is empty.
        """
        if not choices:
            return ""
        lines = []
        for i, emp in enumerate(choices, 1):
            name = emp.get("full_name", "Не указано")
            post = emp.get("post", "Должность не указана")
            dept = emp.get("department") or emp.get("dept", "Отдел не указан")
            lines.append(f"{i}. {name} — {post}, {dept}")
        return "\n".join(lines)

    @staticmethod
    def _extract_prior_tool_calls(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Extract tool call list from the last AIMessage that preceded ToolMessages.

        Traverses message history in reverse to find the AIMessage with tool_calls
        that was executed just before the most recent ToolMessage block.
        Correctly skips the final AIMessage (which contains the text reply, not tool calls).

        Args:
            messages: Full message list from agent state.

        Returns:
            List of tool call dicts compatible with fixed_calls format.
        """
        # Алгоритм: идём с конца
        # Фаза 1: пропускаем финальный AIMessage (без tool_calls) и любые HumanMessage/SystemMessage
        # Фаза 2: собираем все ToolMessage в блок
        # Фаза 3: берём первый AIMessage с tool_calls перед блоком ToolMessage-ов
        phase = 0  # 0=skip_final_ai, 1=collect_tool_msgs, 2=find_ai_with_calls
        found_tool_msgs = False
        for m in reversed(messages):
            if phase == 0:
                # Пропускаем финальный AIMessage (text ответ) и вспомогательные сообщения
                if isinstance(m, ToolMessage):
                    found_tool_msgs = True
                    phase = 1
                continue
            if phase == 1:
                if isinstance(m, ToolMessage):
                    continue  # продолжаем собирать блок ToolMessage
                if isinstance(m, AIMessage):
                    phase = 2
                    # Это AIMessage с tool_calls — именно он нам нужен
                    if getattr(m, "tool_calls", None):
                        return [
                            {"name": tc["name"], "args": dict(tc["args"]), "id": tc["id"]}
                            for tc in m.tool_calls
                        ]
                continue
            if phase == 2:
                break

        # Fallback: если фазовый алгоритм не сработал — старый подход
        found_tool_msgs = False
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                found_tool_msgs = True
                continue
            if isinstance(m, AIMessage) and found_tool_msgs:
                if getattr(m, "tool_calls", None):
                    return [
                        {"name": tc["name"], "args": dict(tc["args"]), "id": tc["id"]}
                        for tc in m.tool_calls
                    ]
                break
        return []

    def _store_pending_disambiguation(
        self,
        thread_id: str,
        tool_name: str,
        choices: List[Dict[str, Any]],
        original_tool_args: Dict[str, Any],
    ) -> None:
        """Store disambiguation state for the given thread.

        Args:
            thread_id: LangGraph thread identifier.
            tool_name: Name of the tool that produced the disambiguation.
            choices: List of employee choice dicts.
            original_tool_args: Original tool arguments to reuse on retry.
        """
        self._pending_disambiguations[thread_id] = {
            "tool_name": tool_name,
            "choices": choices,
            "original_tool_args": original_tool_args,
        }
        logger.info(
            "Stored pending disambiguation",
            extra={"thread_id": thread_id, "tool_name": tool_name, "choices": len(choices)},
        )

    def _pop_pending_disambiguation(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Pop and return pending disambiguation state for thread.

        Args:
            thread_id: LangGraph thread identifier.

        Returns:
            Disambiguation dict or None.
        """
        return self._pending_disambiguations.pop(thread_id, None)

    def _resolve_employee_choice(
        self,
        user_message: str,
        choices: List[Dict[str, Any]],
    ) -> Optional[List[str]]:
        """Resolve a user's textual or numeric selection to employee UUID list.

        Supports:
        - Single number: "3" → third employee
        - Comma-separated numbers: "1, 3" → first and third employees
        - Partial name match: "Иванов Игорь" → fuzzy match against full_name

        Args:
            user_message: Raw user message.
            choices: Ordered list of employee choice dicts from disambiguation.

        Returns:
            List of selected employee UUID strings or None if unresolvable.
        """
        if not choices:
            return None

        msg = user_message.strip()

        numbers = re.findall(r"\b(\d+)\b", msg)
        if numbers:
            selected_ids = []
            for n_str in numbers:
                idx = int(n_str) - 1
                if 0 <= idx < len(choices):
                    emp_id = choices[idx].get("id")
                    if emp_id:
                        selected_ids.append(emp_id)
            if selected_ids:
                return selected_ids

        msg_lower = msg.lower()
        matched_ids = []
        for emp in choices:
            full_name = emp.get("full_name", "").lower()
            if full_name and (msg_lower in full_name or full_name in msg_lower):
                emp_id = emp.get("id")
                if emp_id:
                    matched_ids.append(emp_id)
        if matched_ids:
            return matched_ids

        return None

    async def _handle_disambiguation_choice(
        self,
        context: ContextParams,
        user_message: str,
    ) -> Optional[Dict]:
        """Handle user reply to a pending disambiguation request.

        Looks up stored disambiguation state for the thread, resolves the
        user's choice to employee UUIDs, then directly invokes the original
        tool with selected_employee_ids injected — no LLM round-trip needed.

        Args:
            context: Current execution context.
            user_message: Raw user message (e.g. "3", "1,2", "Иванов Игорь").

        Returns:
            AgentResponse dict if handled, None if no pending disambiguation.
        """
        pending = self._pending_disambiguations.get(context.thread_id)
        if not pending:
            # Диагностика: показываем все known thread_ids для отладки
            known = list(self._pending_disambiguations.keys())
            logger.debug(
                "No pending disambiguation for thread. Known threads: %s",
                known,
                extra={"thread_id": context.thread_id},
            )
            return None

        choices = pending["choices"]
        tool_name = pending["tool_name"]
        original_args = dict(pending["original_tool_args"])

        selected_ids = self._resolve_employee_choice(user_message, choices)
        if not selected_ids:
            formatted = self._format_employee_choices(choices)
            logger.info(
                "Could not resolve disambiguation choice: '%s'",
                user_message,
                extra={"thread_id": context.thread_id},
            )
            return AgentResponse(
                status=AgentStatus.REQUIRES_ACTION,
                action_type=ActionType.DISAMBIGUATION,
                message=(
                    f"Не удалось определить выбор. Пожалуйста, укажите номер сотрудника из списка:\n\n{formatted}"
                ),
                metadata={"choices": choices, "action_type": "select_employee"},
            ).model_dump()

        self._pop_pending_disambiguation(context.thread_id)

        original_args["token"] = context.user_token
        if context.document_id:
            original_args["document_id"] = context.document_id
        original_args["selected_employee_ids"] = selected_ids
        original_args.pop("last_names", None)
        original_args.pop("executor_last_names", None)

        selected_names = [
            choices[int(n) - 1].get("full_name", "")
            for n in re.findall(r"\b(\d+)\b", user_message)
            if 0 < int(n) <= len(choices)
        ] or [
            emp.get("full_name", "") for emp in choices
            if emp.get("id") in selected_ids
        ]

        import uuid as _uuid
        forced_id = f"disamb_resume_{_uuid.uuid4().hex[:8]}"
        forced_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": original_args,
                    "id": forced_id,
                }
            ],
        )
        await self.state_manager.update_state(
            context.thread_id,
            [forced_call],
            as_node="agent",
        )

        logger.info(
            "Resuming %s with selected_employee_ids=%s",
            tool_name,
            selected_ids,
            extra={"thread_id": context.thread_id},
        )

        return await self._orchestrate(
            context=context,
            inputs=None,
            human_choice=None,
            iteration=0,
        )

    async def _build_context(self, request: AgentRequest) -> ContextParams:
        """Build immutable ContextParams from validated AgentRequest.

        Args:
            request: Validated agent request.

        Returns:
            Frozen ContextParams instance.
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
        """Serialize semantic context to XML block for prompt injection.

        Args:
            semantic_context: Result from SemanticDispatcher.build_context.

        Returns:
            XML string fragment.
        """
        metadata = getattr(semantic_context, "metadata", {}) or {}
        doc_category = metadata.get("document_category", "")
        doc_status = metadata.get("document_status", "")
        attachments_count = metadata.get("attachments_count", 0)

        category_hint = ""
        if doc_category == "APPEAL":
            category_hint = (
                "\n  <document_hint>Это документ категории APPEAL (Обращение). "
                "Инструмент для ознакомления: introduction_create_tool. "
                "Инструмент для автозаполнения карточки: autofill_appeal_document. "
                "При выборе сотрудника из списка — СРАЗУ вызывай introduction_create_tool(selected_employee_ids=[...]) без уточнений.</document_hint>"
            )

        return f"""
<semantic_context>
  <user_query>
    <original>{semantic_context.query.original}</original>
    <refined>{semantic_context.query.refined}</refined>
    <intent>{semantic_context.query.intent.value}</intent>
    <complexity>{semantic_context.query.complexity.value}</complexity>
  </user_query>
  <document_meta>
    <category>{doc_category}</category>
    <status>{doc_status}</status>
    <attachments_count>{attachments_count}</attachments_count>
  </document_meta>{category_hint}
</semantic_context>"""