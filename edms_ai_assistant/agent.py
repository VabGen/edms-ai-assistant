import logging
import asyncio
import re
import json
from typing import Dict, List, Annotated, TypedDict
from datetime import datetime

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage
)
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


class EdmsDocumentAgent:
    def __init__(self):
        self.model = get_chat_model()
        self.tools = all_tools
        self.checkpointer = MemorySaver()

        self.tool_manifesto_template = (
            "### ROLE: EXPERT EDMS ANALYST (СЭД)\n"
            "Ты — ведущий аналитик СЭД. Твоя цель: безупречный анализ данных.\n\n"
            "### CONTEXT DATA:\n"
            "- Пользователь: {user_name}\n"
            "- Активный ID документа в СЭД: {context_ui_id}\n\n"
            "### SOURCE PRIORITIZATION:\n"
            "1. **LOCAL_FILE / ATTACHMENT**:\n"
            "   - Если LOCAL_FILE — это UUID (например, 550e8400...), это системное вложение. ТЫ ОБЯЗАН вызвать `doc_get_file_content(attachment_id=LOCAL_FILE)`.\n"
            "   - Если LOCAL_FILE — это путь к файлу, используй `read_local_file_content`.\n"
            "2. **EDMS DOCUMENT**: Если LOCAL_FILE пуст, используй `doc_get_details` для поиска вложений в документе {context_ui_id}.\n\n"
            "### ALGORITHMIC STEPS:\n"
            "1. Получи текст документа одним из инструментов.\n"
            "2. Передай текст в `doc_summarize_text(text=..., summary_type=...)`.\n\n"
            "### GUARDRAILS:\n"
            "- Если LOCAL_FILE содержит ID, ЗАПРЕЩЕНО отвечать, что файл не загружен. Сначала вызови инструмент.\n"
            "- ЯЗЫК: Строго русский. Обращайся по имени: {user_name}."
        )

        self.agent = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 1. Узел вызова модели
        async def call_model(state: AgentState):
            model_with_tools = self.model.bind_tools(self.tools)

            non_sys = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
            sys_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]

            final_messages = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            response = await model_with_tools.ainvoke(final_messages)
            return {"messages": [response]}

        # 2. Узел валидации ответов инструментов
        async def validator(state: AgentState):
            """Проверяет результат работы инструментов перед тем, как вернуть его модели."""
            messages = state["messages"]
            last_message = messages[-1]

            if isinstance(last_message, ToolMessage):
                content_raw = str(last_message.content).strip()

                error_detected = False
                error_reason = ""

                if not content_raw or content_raw == "None" or content_raw == "{}":
                    error_detected = True
                    error_reason = "Инструмент вернул пустой результат. Проверь входные данные или попробуй другой метод."

                elif "error" in content_raw.lower() or "exception" in content_raw.lower():
                    error_detected = True
                    error_reason = f"Техническая ошибка при выполнении: {content_raw}. Попробуй исправить параметры вызова."

                if error_detected:
                    return {
                        "messages": [HumanMessage(content=f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: {error_reason}")]
                    }

            return {"messages": []}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validator", validator)

        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
                return "tools"
            return END

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )

        workflow.add_edge("tools", "validator")
        workflow.add_edge("validator", "agent")

        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["tools"]
        )

    async def chat(self, message: str, user_token: str, context_ui_id: str = None,
                   thread_id: str = None, user_context: Dict = None,
                   file_path: str = None, human_choice: str = None) -> Dict:

        config = {"configurable": {"thread_id": thread_id or "default"}}
        user_context = user_context or {}
        user_name = (user_context.get('firstName') or user_context.get('name') or "пользователь").strip()

        state = await self.agent.aget_state(config)

        if human_choice and state.next:
            last_msg = state.values["messages"][-1]
            fixed_calls = []
            for tc in getattr(last_msg, "tool_calls", []):
                t_args = dict(tc["args"])
                t_name = tc["name"]

                if t_name == "doc_summarize_text":
                    t_args["summary_type"] = human_choice

                fixed_calls.append({"name": t_name, "args": t_args, "id": tc["id"]})

            await self.agent.aupdate_state(config, {
                "messages": [AIMessage(content=last_msg.content or "", tool_calls=fixed_calls, id=last_msg.id)]
            }, as_node="agent")

            return await self._orchestrate(None, config, user_token, context_ui_id, file_path,
                                           is_choice_active=True, human_choice=human_choice)

        current_date = datetime.now().strftime("%d.%m.%Y")
        manifesto = self.tool_manifesto_template.format(
            context_ui_id=context_ui_id or "Не указан",
            user_name=user_name
        )
        env = (
            f"\n### ENVIRONMENT\n"
            f"CURRENT_DATE: {current_date}\n"
            f"ACTIVE_DOC_ID: {context_ui_id}\n"
            f"LOCAL_FILE: {file_path or 'Не загружен'}\n"
        )

        sys_msg = SystemMessage(content=manifesto + env)
        hum_msg = HumanMessage(content=message if message else "Продолжи анализ.")
        inputs = {"messages": [sys_msg, hum_msg]}

        return await self._orchestrate(inputs, config, user_token, context_ui_id, file_path,
                                       is_choice_active=bool(human_choice), human_choice=human_choice)

    async def _orchestrate(self, inputs, config, token, doc_id, file_path,
                           is_choice_active=False, iteration=0, human_choice=None):
        if iteration > 12:
            return {"status": "error", "message": "Слишком сложная цепочка действий. Попробуйте уточнить запрос."}

        try:
            await asyncio.wait_for(self.agent.ainvoke(inputs, config=config), timeout=180.0)

            state = await self.agent.aget_state(config)
            messages = state.values.get("messages", [])
            if not messages:
                return {"status": "error", "message": "Ошибка инициализации диалога."}

            last_msg = messages[-1]

            if not state.next or not isinstance(last_msg, AIMessage) or not getattr(last_msg, "tool_calls", None):
                for m in reversed(messages):
                    if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                        return {"status": "success", "content": m.content}
                return {"status": "success", "content": "Выполнено."}

            fixed_calls = []
            newly_read_content = None

            for m in reversed(messages):
                if isinstance(m, ToolMessage):
                    content_str = str(m.content)
                    if content_str.strip().startswith("{"):
                        try:
                            data = json.loads(content_str)
                            newly_read_content = data.get("content") or data.get("text")
                        except:
                            pass
                    elif len(content_str) > 20:
                        newly_read_content = content_str
                    if newly_read_content: break

            for tc in last_msg.tool_calls:
                t_name = tc["name"]
                t_args = dict(tc["args"])
                t_id = tc["id"]
                t_args["token"] = token

                clean_path = str(file_path).strip() if file_path else ""
                is_uuid = bool(
                    re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', clean_path, re.I))

                if is_uuid:
                    if t_name == "read_local_file_content":
                        logger.info(f"!!! REDIRECT TRIGGERED: Swapping local tool to EDMS for UUID {clean_path}")
                        t_name = "doc_get_file_content"
                        t_args["attachment_id"] = clean_path
                        t_args.pop("file_path", None)

                    if t_name == "doc_get_file_content" and "file_path" in t_args:
                        t_args["attachment_id"] = t_args.pop("file_path")

                    if doc_id:
                        t_args["document_id"] = doc_id

                elif clean_path and t_name == "read_local_file_content":
                    t_args["file_path"] = clean_path

                if doc_id and (t_name.startswith("doc_") or "document_id" in t_args):
                    t_args["document_id"] = doc_id

                if t_name == "doc_summarize_text":
                    if newly_read_content:
                        t_args["text"] = newly_read_content

                    if human_choice:
                        t_args["summary_type"] = human_choice
                        logger.info(f"FORCE SUMMARY TYPE: {human_choice}")

                    if not t_args.get("summary_type"):
                        usr_msg = next((m.content.lower() for m in reversed(messages) if isinstance(m, HumanMessage)),
                                       "")
                        keywords = {"факты": "extractive", "тезисы": "thesis", "пересказ": "abstractive"}
                        for kw, fmt in keywords.items():
                            if kw in usr_msg:
                                t_args["summary_type"] = fmt
                                break

                    if not t_args.get("summary_type") and not is_choice_active:
                        await self.agent.aupdate_state(config, {
                            "messages": [AIMessage(content=last_msg.content or "Выполняю анализ...",
                                                   tool_calls=[{"name": t_name, "args": t_args, "id": t_id}],
                                                   id=last_msg.id)]
                        }, as_node="agent")
                        return {
                            "status": "requires_action",
                            "action_type": "summarize_selection",
                            "message": "В каком формате подготовить анализ?"
                        }

                fixed_calls.append({"name": t_name, "args": t_args, "id": t_id})

            await self.agent.aupdate_state(config, {
                "messages": [AIMessage(content=last_msg.content or "", tool_calls=fixed_calls, id=last_msg.id)]
            }, as_node="agent")

            return await self._orchestrate(None, config, token, doc_id, file_path,
                                           is_choice_active, iteration + 1, human_choice)

        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            return {"status": "error", "message": "Техническая ошибка при обработке файла."}
