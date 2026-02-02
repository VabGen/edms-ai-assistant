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
    BaseMessage,
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

        # self.tool_manifesto_template = (
        #     "### ROLE: EXPERT EDMS ANALYST (СЭД)\n"
        #     "Ты — ведущий аналитик СЭД. Твоя цель: безупречный анализ данных.\n\n"
        #     "### CONTEXT DATA:\n"
        #     "- Пользователь: {user_name}\n"
        #     "- Активный ID документа в СЭД: {context_ui_id}\n\n"
        #     "### SOURCE PRIORITIZATION:\n"
        #     "1. **LOCAL_FILE / ATTACHMENT**:\n"
        #     "   - Если LOCAL_FILE — это UUID (например, 550e8400...), это системное вложение. ТЫ ОБЯЗАН вызвать `doc_get_file_content(attachment_id=LOCAL_FILE)`.\n"
        #     "   - Если LOCAL_FILE — это путь к файлу, используй `read_local_file_content`.\n"
        #     "2. **EDMS DOCUMENT**: Если LOCAL_FILE пуст, используй `doc_get_details` для поиска вложений в документе {context_ui_id}.\n\n"
        #     "### ALGORITHMIC STEPS:\n"
        #     "1. Получи текст документа одним из инструментов.\n"
        #     "2. Передай текст в `doc_summarize_text(text=..., summary_type=...)`.\n\n"
        #     "### GUARDRAILS:\n"
        #     "- Если LOCAL_FILE содержит ID, ЗАПРЕЩЕНО отвечать, что файл не загружен. Сначала вызови инструмент.\n"
        #     "- ЯЗЫК: Строго русский. Обращайся по имени: {user_name}."
        # )

        self.tool_manifesto_template = (
            "### ROLE: EXPERT EDMS ANALYST (СЭД)\n"
            "Ты — ведущий аналитик СЭД. Твоя цель: безупречный анализ данных.\n\n"

            "### CONTEXT DATA:\n"
            "- Пользователь: {user_name}\n"
            "- Активный ID документа в СЭД: {context_ui_id}\n\n"

            "### АВТОЗАПОЛНЕНИЕ ОБРАЩЕНИЙ:\n"
            "Если пользователь просит 'заполнить обращение', 'автозаполнить обращение' или подобное:\n"
            "1. Проверь, что ACTIVE_DOC_ID указан (это ID документа)\n"
            "2. Вызови инструмент `autofill_appeal_document(document_id=ACTIVE_DOC_ID, token=..., attachment_id=LOCAL_FILE если есть)`\n"
            "3. Инструмент автоматически:\n"
            "   - Найдет вложение (если attachment_id не указан)\n"
            "   - Извлечет текст\n"
            "   - Заполнит все поля обращения\n"
            "4. Верни пользователю красиво отформатированный результат\n\n"

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
            "- ЯЗЫК: Строго русский. Обращайся по имени: {user_name}.\n"
            "- При автозаполнении ВСЕГДА используй `autofill_appeal_document`, а не пытайся заполнять поля вручную."
        )

        self.agent = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        async def call_model(state: AgentState):
            model_with_tools = self.model.bind_tools(self.tools)

            non_sys = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
            sys_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
            final_messages = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            response = await model_with_tools.ainvoke(final_messages)
            return {"messages": [response]}

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

                elif (
                    "error" in content_raw.lower() or "exception" in content_raw.lower()
                ):
                    error_detected = True
                    error_reason = f"Техническая ошибка при выполнении: {content_raw}. Попробуй исправить параметры вызова."

                if error_detected:
                    return {
                        "messages": [
                            HumanMessage(
                                content=f"[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: {error_reason}"
                            )
                        ]
                    }

            return {"messages": []}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validator", validator)

        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState):
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

        return workflow.compile(
            checkpointer=self.checkpointer, interrupt_before=["tools"]
        )

    async def chat(
        self,
        message: str,
        user_token: str,
        context_ui_id: str = None,
        thread_id: str = None,
        user_context: Dict = None,
        file_path: str = None,
        human_choice: str = None,
    ) -> Dict:

        config = {"configurable": {"thread_id": thread_id or "default"}}
        user_context = user_context or {}
        user_name = (
            user_context.get("firstName") or user_context.get("name") or "пользователь"
        ).strip()

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
                None,
                config,
                user_token,
                context_ui_id,
                file_path,
                is_choice_active=True,
                human_choice=human_choice,
            )

        current_date = datetime.now().strftime("%d.%m.%Y")
        manifesto = self.tool_manifesto_template.format(
            context_ui_id=context_ui_id or "Не указан", user_name=user_name
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

        return await self._orchestrate(
            inputs,
            config,
            user_token,
            context_ui_id,
            file_path,
            is_choice_active=bool(human_choice),
            human_choice=human_choice,
        )

    async def _orchestrate(
        self,
        inputs,
        config,
        token,
        doc_id,
        file_path,
        is_choice_active=False,
        iteration=0,
        human_choice=None,
    ):
        if iteration > 10:
            return {
                "status": "error",
                "message": "Цикл обработки слишком длинный. Уточните запрос.",
            }

        try:
            await asyncio.wait_for(
                self.agent.ainvoke(inputs, config=config), timeout=120.0
            )

            state = await self.agent.aget_state(config)
            messages = state.values.get("messages", [])
            if not messages:
                return {"status": "error", "message": "Пустое состояние агента."}

            last_msg = messages[-1]

            if (
                not state.next
                or not isinstance(last_msg, AIMessage)
                or not last_msg.tool_calls
            ):
                for m in reversed(messages):
                    if isinstance(m, AIMessage) and m.content:
                        return {"status": "success", "content": m.content}
                return {"status": "success", "content": "Готово."}

            fixed_calls = []
            last_extracted_text = None
            for m in reversed(messages):
                if isinstance(m, ToolMessage):
                    try:
                        data = (
                            json.loads(m.content)
                            if isinstance(m.content, str) and m.content.startswith("{")
                            else {}
                        )
                        last_extracted_text = (
                            data.get("content")
                            or data.get("text_preview")
                            or (m.content if len(str(m.content)) > 100 else None)
                        )
                    except:
                        last_extracted_text = (
                            m.content if len(str(m.content)) > 100 else None
                        )
                if last_extracted_text:
                    break

            for tc in last_msg.tool_calls:
                t_name = tc["name"]
                t_args = dict(tc["args"])
                t_id = tc["id"]
                t_args["token"] = token
                clean_path = str(file_path).strip() if file_path else ""
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

                if doc_id and (t_name.startswith("doc_") or "document_id" in t_args):
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

                fixed_calls.append({"name": t_name, "args": t_args, "id": t_id})

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
                None,
                config,
                token,
                doc_id,
                file_path,
                is_choice_active,
                iteration + 1,
                human_choice,
            )

        except Exception as e:
            logger.error(f"Agent Orchestration Error: {e}", exc_info=True)
            return {"status": "error", "message": f"Ошибка анализа: {str(e)}"}
