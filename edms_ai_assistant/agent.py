import logging
import operator
from typing import List, Optional, Annotated, Dict, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)


class EdmsAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context_ui_id: Optional[str]
    user_token: str
    user_context: Optional[Dict[str, Any]]
    file_path: Optional[str]


class EdmsDocumentAgent:
    def __init__(self):
        self.model = get_chat_model()
        self.tools = all_tools
        self.checkpointer = MemorySaver()

        self.system_template = (
            "Ты — экспертный ИИ-ассистент системы электронного документооборота (EDMS).\n"
            "Текущий UUID документа: {context_ui_id}\n"
            "ПРАВИЛА:\n"
            "1. Если нужно узнать список файлов или метаданные — используй doc_get_details.\n"
            "2. Если пользователь спрашивает о содержании вложений — ОБЯЗАТЕЛЬНО сначала получи текст через doc_get_file_content,\n"
            "   а затем ВСЕГДА передавай этот текст в doc_summarize_text для формирования ответа.\n"
            "3. НЕ ПЕРЕСКАЗЫВАЙ содержимое файлов самостоятельно без вызова doc_summarize_text."
        )

        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            interrupt_before=["tools"]
        )

    async def chat(
            self,
            message: str,
            user_token: str,
            context_ui_id: Optional[str] = None,
            thread_id: Optional[str] = None,
            user_context: Optional[Dict[str, Any]] = None,
            file_path: Optional[str] = None,
            human_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id or "default_session"}}
            current_sys = self.system_template.format(context_ui_id=context_ui_id or "не задан")

            if human_choice:
                snapshot = await self.agent.aget_state(config)
                if snapshot.values and "messages" in snapshot.values:
                    last_message = snapshot.values["messages"][-1]
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        tool_call = last_message.tool_calls[0]
                        new_args = {**tool_call["args"], "summary_type": human_choice}

                        await self.agent.aupdate_state(
                            config,
                            {"messages": [last_message.copy(update={"tool_calls": [{**tool_call, "args": new_args}]})]},
                            as_node="agent"
                        )
                await self.agent.ainvoke(None, config=config)

            else:
                route_check = await self.model.ainvoke([
                    SystemMessage(content="Ты — классификатор. Отвечай только 'YES' или 'NO'."),
                    HumanMessage(content=f"Нужны инструменты для запроса '{message}'?")
                ])

                if "NO" in route_check.content.upper():
                    response = await self.model.ainvoke(
                        [SystemMessage(content=current_sys), HumanMessage(content=message)])
                    return {"status": "success", "content": response.content}

                context_info = f"TOKEN: {user_token}\nID: {context_ui_id}\n"

                if file_path:
                    context_info += f"[ДОСТУПЕН ЛОКАЛЬНЫЙ ФАЙЛ]: {file_path}\n"

                if user_context:
                    context_info += f"USER_CONTEXT: {user_context}\n"

                inputs = {
                    "messages": [
                        SystemMessage(content=current_sys),
                        HumanMessage(content=f"{context_info}ЗАПРОС: {message}")
                    ],
                    "user_token": user_token,
                    "context_ui_id": context_ui_id,
                    "user_context": user_context,
                    "file_path": file_path
                }
                await self.agent.ainvoke(inputs, config=config)

            while True:
                state = await self.agent.aget_state(config)
                if not state.next:
                    break

                last_msg = state.values["messages"][-1]
                tool_calls = getattr(last_msg, 'tool_calls', [])

                if tool_calls:
                    t_name = tool_calls[0]["name"]

                    if t_name == "doc_summarize_text" and not human_choice:
                        return {
                            "status": "requires_action",
                            "action_type": "summarize_selection",
                            "message": "Выберите режим суммаризации для анализа файла:"
                        }

                    human_choice = None
                    await self.agent.ainvoke(None, config=config)
                else:
                    break

            final_state = await self.agent.aget_state(config)
            final_msg = final_state.values["messages"][-1]

            if not final_msg.content and hasattr(final_msg, 'tool_calls') and final_msg.tool_calls:
                if final_msg.tool_calls[0]["name"] == "doc_summarize_text":
                    return {
                        "status": "requires_action",
                        "action_type": "summarize_selection",
                        "message": "Для ответа выберите тип резюме:"
                    }

            return {"status": "success", "content": final_msg.content or "Не удалось получить ответ."}

        except Exception as e:
            logger.error(f"Ошибка в EdmsDocumentAgent: {e}", exc_info=True)
            return {"status": "error", "message": "Ошибка обработки запроса."}
