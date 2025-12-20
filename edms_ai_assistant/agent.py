# edms_ai_assistant\agent.py
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
        # Получаем объект модели из llm.py
        self.model = get_chat_model()
        self.tools = all_tools
        self.checkpointer = MemorySaver()

        self.system_template = (
            "<identity>\n"
            "Ты — экспертный ИИ-ассистент системы электронного документооборота (EDMS).\n"
            "</identity>\n\n"
            "<context>\n"
            "- Текущий UUID документа в системе: {context_ui_id}\n"
            "</context>\n\n"
            "<tool_policy>\n"
            "   <local_files>\n"
            "   Если предоставлен `file_path`, используй `read_local_file_content`.\n"
            "   </local_files>\n"
            "   <edms_files>\n"
            "   Протокол: 1. doc_get_details -> 2. Выбор ID -> 3. doc_get_file_content.\n"
            "   </edms_files>\n"
            "</tool_policy>\n\n"
            "ПРАВИЛО HITL:\n"
            "Для суммаризации используй `doc_summarize_text`. Система остановится для выбора типа (Extractive/Abstractive/Thesis).\n"
        )

        # Передаем объект модели напрямую.
        # interrupt_before=["tools"] создает точку прерывания для HITL.
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
            config: RunnableConfig = {
                "configurable": {"thread_id": thread_id or "default_session"}
            }

            if human_choice:
                # ЛОГИКА ПРОДОЛЖЕНИЯ ПОСЛЕ ВЫБОРА ПОЛЬЗОВАТЕЛЯ
                snapshot = await self.agent.aget_state(config)
                if snapshot.values and "messages" in snapshot.values:
                    last_message = snapshot.values["messages"][-1]

                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        tool_call = last_message.tool_calls[0]
                        new_args = {**tool_call["args"]}

                        # Подставляем выбор пользователя в аргумент 'focus' (для суммаризации)
                        if tool_call["name"] == "doc_summarize_text":
                            new_args["focus"] = human_choice

                        updated_tool_calls = [{
                            "name": tool_call["name"],
                            "args": new_args,
                            "id": tool_call["id"],
                            "type": "tool_call"
                        }]

                        # Обновляем состояние агента перед продолжением
                        await self.agent.aupdate_state(
                            config,
                            {"messages": [last_message.copy(update={"tool_calls": updated_tool_calls})]},
                            as_node="agent"
                        )
                # Продолжаем выполнение с None (агент возьмет данные из обновленного состояния)
                result = await self.agent.ainvoke(None, config=config)
            else:
                # ПЕРВИЧНЫЙ ЗАПУСК
                current_sys = self.system_template.format(context_ui_id=context_ui_id or "не задан")
                context_header = f"КОНТЕКСТ:\n- Токен: {user_token}\n- ID документа: {context_ui_id or 'не задан'}\n"

                if user_context:
                    context_header += f"- Данные профиля: {user_context}\n"

                if file_path:
                    context_header += f"- Локальный файл: {file_path}\n"

                inputs: EdmsAgentState = {
                    "messages": [
                        SystemMessage(content=current_sys),
                        HumanMessage(content=f"{context_header}\nЗАПРОС: {message}")
                    ],
                    "user_token": user_token,
                    "context_ui_id": context_ui_id,
                    "user_context": user_context or {},
                    "file_path": file_path
                }

                # ПЕРЕДАЕМ inputs как словарь (без распаковки **)
                result = await self.agent.ainvoke(inputs, config=config)

            # ПРОВЕРКА СОСТОЯНИЯ: НУЖНО ЛИ ПРЕРЫВАНИЕ (HITL)?
            final_snapshot = await self.agent.aget_state(config)

            if final_snapshot.next:
                last_msg = final_snapshot.values["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    tool_name = last_msg.tool_calls[0]["name"]

                    # Если вызван инструмент суммаризации — возвращаем сигнал для фронтенда
                    if tool_name == "doc_summarize_text":
                        return {
                            "status": "requires_action",
                            "action_type": "summarize_selection",
                            "message": "Файл загружен. Выберите режим анализа для создания выжимки:"
                        }
                    # Для остальных инструментов (чтение файла и т.д.) — авто-продолжение
                    else:
                        logger.info(f"Авто-продолжение для инструмента: {tool_name}")
                        result = await self.agent.ainvoke(None, config=config)

            # ВОЗВРАТ УСПЕШНОГО РЕЗУЛЬТАТА
            if result and "messages" in result:
                return {
                    "status": "success",
                    "content": result["messages"][-1].content
                }

            return {"status": "error", "message": "Агент не вернул сообщений."}

        except Exception as e:
            logger.error(f"Ошибка в EdmsDocumentAgent.chat: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}