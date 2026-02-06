# edms_ai_assistant/agent.py
import logging
import asyncio
import re
import json
from typing import Dict, List, Annotated, TypedDict, Optional
from datetime import datetime

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)

CORE_SYSTEM_PROMPT = """<role>
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
   - UUID формат (550e8400-...) → Вызови `doc_get_file_content(attachment_id=LOCAL_FILE)`
   - Путь к файлу (/tmp/...) → Вызови `read_local_file_content(file_path=LOCAL_FILE)`
   - Пустое значение → Вызови `doc_get_details()` для поиска вложений

3. **Обработка requires_action**:
   - Статус "summarize_selection" → Предложи формат анализа (факты/пересказ/тезисы)
   - Статус "requires_disambiguation" → Покажи список, дождись выбора пользователя
   - Статус "select_employee" → Покажи сотрудников для выбора

4. **Язык**: Только русский. Обращайся к пользователю по имени: {user_name}
</critical_rules>

<tool_selection>
**Анализ документа**: doc_get_details → doc_get_file_content → doc_summarize_text
**Поиск сотрудника**: employee_search_tool
**Список ознакомления**: introduction_create_tool
**Создание поручения**: task_create_tool
</tool_selection>

<response_format>
✅ Структурировано, кратко, по делу
❌ Многословие, технические детали API
</response_format>"""

# ========================================
# DYNAMIC CONTEXT SNIPPETS
# ========================================

CONTEXT_SNIPPETS = {
    "introduction_disambiguation": """
<introduction_guide>
При создании списка ознакомления:
- Если статус "requires_disambiguation" → Покажи список найденных сотрудников
- Дождись выбора пользователя (например: "первый и третий")
- Повторный вызов: introduction_create_tool(selected_employee_ids=[uuid1, uuid3])
</introduction_guide>""",

    "task_creation": """
<task_guide>
При создании поручения:
- executor_last_names: ["Иванов", "Петров"] — обязательно (минимум 1)
- responsible_last_name: "Петров" — опционально (если НЕ указан → первый исполнитель ответственный)
- planed_date_end: "2026-02-15T23:59:59Z" — опционально (если НЕ указан → +7 дней)
</task_guide>""",

    "date_parsing": """
<date_parsing>
Преобразование дат в ISO 8601:
- "до 15 февраля" → "2026-02-15T23:59:59Z"
- "через неделю" → +7 дней от текущей даты
- "к концу месяца" → последний день текущего месяца 23:59:59
Всегда добавляй суффикс 'Z' (UTC timezone).
</date_parsing>""",
}


def get_dynamic_context(message: str) -> str:
    """
    Динамически загружает релевантный контекст в зависимости от запроса.
    """
    context_parts = []
    msg_lower = message.lower()

    if any(trigger in msg_lower for trigger in [
        "ознакомление", "ознакомь", "добавь в список", "список ознакомления"
    ]):
        context_parts.append(CONTEXT_SNIPPETS["introduction_disambiguation"])

    # Триггеры для создания поручений
    if any(trigger in msg_lower for trigger in [
        "поручение", "создай поручение", "дай задание", "назначь исполнителей",
        "исполнитель", "ответственный"
    ]):
        context_parts.append(CONTEXT_SNIPPETS["task_creation"])

    # Триггеры для парсинга дат
    if any(trigger in msg_lower for trigger in [
        "до ", "через ", "к ", "срок", "дедлайн", "до конца"
    ]):
        context_parts.append(CONTEXT_SNIPPETS["date_parsing"])

    return "\n".join(context_parts)

class AgentState(TypedDict):
    """State for LangGraph agent."""
    messages: Annotated[List[BaseMessage], add_messages]

class EdmsDocumentAgent:
    """
    Основной агент для работы с EDMS.

    Capabilities:
    - Анализ документов (чтение, суммаризация)
    - Управление персоналом (поиск сотрудников, списки ознакомления)
    - Делегирование задач (создание поручений)
    """

    def __init__(self):
        self.model = get_chat_model()
        self.tools = all_tools
        self.checkpointer = MemorySaver()
        self.base_prompt_template = CORE_SYSTEM_PROMPT
        self.agent = self._build_graph()

        logger.info(f"EdmsDocumentAgent initialized with {len(self.tools)} tools")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Node 1: Agent (LLM with tools)
        async def call_model(state: AgentState) -> Dict:
            model_with_tools = self.model.bind_tools(self.tools)

            non_sys = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
            sys_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
            final_messages = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            response = await model_with_tools.ainvoke(final_messages)
            return {"messages": [response]}

        async def validator(state: AgentState) -> Dict:
            messages = state["messages"]
            last_message = messages[-1]

            if not isinstance(last_message, ToolMessage):
                return {"messages": []}

            content_raw = str(last_message.content).strip()

            if not content_raw or content_raw in ("None", "{}"):
                return {
                    "messages": [
                        HumanMessage(
                            content="[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Инструмент вернул пустой результат. Проверь параметры или попробуй другой метод."
                        )
                    ]
                }

            if "error" in content_raw.lower() or "exception" in content_raw.lower():
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Техническая ошибка: {content_raw}. Попробуй исправить параметры."
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
            "agent",
            should_continue,
            {"tools": "tools", END: END}
        )

        workflow.add_edge("tools", "validator")
        workflow.add_edge("validator", "agent")

        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["tools"]
        )

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
        config = {"configurable": {"thread_id": thread_id or "default"}}

        user_context = user_context or {}
        user_name = (
                user_context.get("firstName")
                or user_context.get("name")
                or "пользователь"
        ).strip()

        state = await self.agent.aget_state(config)

        if human_choice and state.next:
            return await self._handle_human_choice(
                config, user_token, context_ui_id, file_path, human_choice
            )

        current_date = datetime.now().strftime("%d.%m.%Y")

        # 1. Base prompt (always)
        base_prompt = self.base_prompt_template.format(
            user_name=user_name,
            current_date=current_date,
            context_ui_id=context_ui_id or "Не указан",
            local_file=file_path or "Не загружен"
        )

        # 2. Dynamic context (conditional)
        dynamic_context = get_dynamic_context(message)

        # 3. Final prompt
        full_prompt = base_prompt + dynamic_context

        # Log token estimate (for monitoring)
        if logger.isEnabledFor(logging.DEBUG):
            import tiktoken
            try:
                encoder = tiktoken.encoding_for_model("gpt-4")
                token_count = len(encoder.encode(full_prompt))
                logger.debug(f"System prompt tokens: {token_count}")
            except Exception:
                pass  # Skip if tiktoken not available

        # Build messages
        sys_msg = SystemMessage(content=full_prompt)
        hum_msg = HumanMessage(content=message if message else "Продолжи анализ.")
        inputs = {"messages": [sys_msg, hum_msg]}

        # Execute agent
        return await self._orchestrate(
            inputs=inputs,
            config=config,
            token=user_token,
            doc_id=context_ui_id,
            file_path=file_path,
            is_choice_active=bool(human_choice),
            human_choice=human_choice,
        )

    async def _handle_human_choice(
            self,
            config: Dict,
            token: str,
            doc_id: Optional[str],
            file_path: Optional[str],
            human_choice: str,
    ) -> Dict:
        state = await self.agent.aget_state(config)
        last_msg = state.values["messages"][-1]

        fixed_calls = []
        for tc in getattr(last_msg, "tool_calls", []):
            t_args = dict(tc["args"])
            t_name = tc["name"]

            if t_name == "doc_summarize_text":
                t_args["summary_type"] = human_choice

            fixed_calls.append({
                "name": t_name,
                "args": t_args,
                "id": tc["id"]
            })

        await self.agent.aupdate_state(
            config,
            {
                "messages": [
                    AIMessage(
                        content=last_msg.content or "",
                        tool_calls=fixed_calls,
                        id=last_msg.id,
                    )
                ]
            },
            as_node="agent",
        )

        return await self._orchestrate(
            inputs=None,
            config=config,
            token=token,
            doc_id=doc_id,
            file_path=file_path,
            is_choice_active=True,
            human_choice=human_choice,
        )

    async def _orchestrate(
            self,
            inputs: Optional[Dict],
            config: Dict,
            token: str,
            doc_id: Optional[str],
            file_path: Optional[str],
            is_choice_active: bool = False,
            iteration: int = 0,
            human_choice: Optional[str] = None,
    ) -> Dict:
        if iteration > 10:
            logger.error(f"Max iterations reached for thread {config['configurable']['thread_id']}")
            return {
                "status": "error",
                "message": "Цикл обработки слишком длинный. Уточните запрос.",
            }

        try:
            await asyncio.wait_for(
                self.agent.ainvoke(inputs, config=config),
                timeout=120.0
            )

            state = await self.agent.aget_state(config)
            messages = state.values.get("messages", [])

            if not messages:
                return {"status": "error", "message": "Пустое состояние агента."}

            last_msg = messages[-1]

            if (
                    not state.next
                    or not isinstance(last_msg, AIMessage)
                    or not getattr(last_msg, "tool_calls", None)
            ):
                for m in reversed(messages):
                    if isinstance(m, AIMessage) and m.content:
                        return {"status": "success", "content": m.content}

                return {"status": "success", "content": "Готово."}

            last_extracted_text = self._extract_last_text(messages)

            fixed_calls = []

            for tc in last_msg.tool_calls:
                t_name = tc["name"]
                t_args = dict(tc["args"])
                t_id = tc["id"]
                t_args["token"] = token

                clean_path = str(file_path).strip() if file_path else ""
                is_uuid = bool(re.match(
                    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                    clean_path,
                    re.I
                ))

                if is_uuid and t_name == "read_local_file_content":
                    t_name = "doc_get_file_content"
                    t_args["attachment_id"] = clean_path
                    t_args.pop("file_path", None)

                if doc_id and (
                        t_name.startswith("doc_")
                        or "document_id" in t_args
                        or t_name in ["introduction_create_tool", "task_create_tool"]
                ):
                    t_args["document_id"] = doc_id

                if t_name == "doc_summarize_text":
                    if last_extracted_text:
                        t_args["text"] = str(last_extracted_text)

                    if human_choice:
                        t_args["summary_type"] = human_choice

                    if not t_args.get("summary_type") and not is_choice_active:
                        return {
                            "status": "requires_action",
                            "action_type": "summarize_selection",
                            "message": "В каком формате подготовить анализ документа?",
                        }

                fixed_calls.append({
                    "name": t_name,
                    "args": t_args,
                    "id": t_id
                })

            await self.agent.aupdate_state(
                config,
                {
                    "messages": [
                        AIMessage(
                            content=last_msg.content or "",
                            tool_calls=fixed_calls,
                            id=last_msg.id,
                        )
                    ]
                },
                as_node="agent",
            )

            return await self._orchestrate(
                inputs=None,
                config=config,
                token=token,
                doc_id=doc_id,
                file_path=file_path,
                is_choice_active=is_choice_active,
                iteration=iteration + 1,
                human_choice=human_choice,
            )

        except asyncio.TimeoutError:
            logger.error("Agent execution timeout")
            return {
                "status": "error",
                "message": "Превышено время ожидания. Попробуйте упростить запрос.",
            }

        except Exception as e:
            logger.error(f"Agent orchestration error: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Ошибка обработки запроса: {str(e)}",
            }

    def _extract_last_text(self, messages: List[BaseMessage]) -> Optional[str]:
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
                        return text

                if len(str(m.content)) > 100:
                    return m.content

            except json.JSONDecodeError:
                if len(str(m.content)) > 100:
                    return m.content

        return None