import logging
import operator
from typing import List, Optional, Annotated, Dict, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

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

        self.system_message = (
            "<identity>\n"
            "Ты — экспертный ИИ-ассистент системы электронного документооборота (EDMS). "
            "Твоя задача: профессиональный анализ метаданных и содержимого документов.\n"
            "</identity>\n\n"

            "<context>\n"
            "- Текущий UUID документа в системе: {context_ui_id}\n"
            "- Все действия выполняются от лица авторизованного пользователя.\n"
            "</context>\n\n"

            "<tool_policy>\n"
            "   <local_files>\n"
            "   Если в контексте предоставлен `file_path`, для любых вопросов по этому файлу используй "
            "исключительно инструмент `read_local_file_content`. Игнорируй инструменты EDMS для локальных файлов.\n"
            "   </local_files>\n\n"

            "   <edms_files>\n"
            "   При работе с вложениями внутри EDMS (context_ui_id) строго соблюдай протокол:\n"
            "   1. **Discovery**: Если пользователь упоминает файл по типу (РКК, Договор) или имени, "
            "ТЫ ОБЯЗАН сначала вызвать `doc_get_details`.\n"
            "   2. **Selection**: Проанализируй массив `attachmentDocument`. Найди объект, где `name` или "
            "`attachmentDocumentType` наиболее релевантны запросу.\n"
            "   3. **Extraction**: Используй полученный `id` вложения как аргумент `attachment_id` в инструменте `doc_get_file_content`.\n"
            "   4. **Fallback**: Вызывай `doc_get_file_content` без `attachment_id` ТОЛЬКО если в метаданных присутствует ровно одно вложение.\n"
            "   </edms_files>\n"
            "</tool_policy>\n\n"

            "<constraints>\n"
            "- Запрещено галлюцинировать именами файлов. Используй только данные из `doc_get_details`.\n"
            "- Если подходящий файл не найден, выведи список доступных имен файлов и попроси уточнения.\n"
            "- Всегда сообщай название файла, с которым работаешь.\n"
            "</constraints>\n\n"

            "<thought_process>\n"
            "Перед вызовом инструмента проговори про себя (внутренний монолог):\n"
            "1. Какой тип файла ищет пользователь?\n"
            "2. Знаю ли я UUID этого вложения? (Если нет — вызываю doc_get_details).\n"
            "3. Какой инструмент соответствует источнику (локальный диск или облако EDMS)?\n"
            "</thought_process>"
        )

        try:
            self.agent = create_react_agent(
                model=self.model,
                tools=self.tools,
                state_modifier=self.system_message
            )
        except (TypeError, ValueError):
            self.agent = create_react_agent(
                model=self.model,
                tools=self.tools
            )

    async def chat(
            self,
            message: str,
            user_token: str,
            context_ui_id: Optional[str] = None,
            thread_id: Optional[str] = None,
            user_context: Optional[Dict[str, Any]] = None,
            file_path: Optional[str] = None
    ) -> str:
        try:
            current_system_message = self.system_message.replace("{context_ui_id}", context_ui_id or "не задан")

            if file_path:
                file_info = f"\n[ДОСТУПЕН ЛОКАЛЬНЫЙ ФАЙЛ]:\nПУТЬ: {file_path}"
            else:
                file_info = ""

            context_header = (
                f"КОНТЕКСТ СИСТЕМЫ:\n"
                f"- Токен: {user_token}\n"
                f"- ID текущего документа: {context_ui_id or 'не задан'}"
                f"{file_info}\n"
            )

            new_messages = [
                SystemMessage(content=current_system_message),
                HumanMessage(content=f"{context_header}\nЗАПРОС ПОЛЬЗОВАТЕЛЯ: {message}")
            ]

            inputs = {
                "messages": new_messages,
                "user_token": user_token,
                "context_ui_id": context_ui_id,
                "user_context": user_context or {},
                "file_path": file_path
            }

            config: RunnableConfig = {
                "configurable": {"thread_id": thread_id or "default_session"}
            }

            result = await self.agent.ainvoke(inputs, config=config)

            if result and "messages" in result:
                return result["messages"][-1].content

            return "Извините, я не смог сформировать ответ."
        except Exception as e:
            logger.error(f"Ошибка в chat: {e}", exc_info=True)
            return f"Техническая ошибка: {str(e)}"
