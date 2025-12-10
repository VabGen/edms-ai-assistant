# edms_ai_assistant.sub_agents.documents_agent

import logging
import json
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from edms_ai_assistant.core.sub_agents import register_agent
from edms_ai_assistant.core.orchestrator import OrchestratorState
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.tools.document import get_document_tool, search_documents_tool
from edms_ai_assistant.tools.attachment import summarize_attachment_tool
from edms_ai_assistant.utils.format_utils import format_document_response

logger = logging.getLogger(__name__)


class DocumentIdSchema(BaseModel):
    document_id: str = Field(description="Уникальный идентификатор документа (UUID).")


class SearchFiltersSchema(BaseModel):
    filters: Dict[str, Any] = Field(description="Словарь фильтров для поиска документов.")


class SummarizeAttachmentSchema(BaseModel):
    document_id: str = Field(description="Уникальный идентификатор документа (UUID).")
    attachment_id: str = Field(description="Уникальный идентификатор вложения (UUID).")
    attachment_name: str = Field(description="Имя файла вложения, которое нужно суммировать.")


# ----------------------------------------------------------------------------------------------------

class GetDocumentToolWrapped(BaseTool):
    """Обёрнутый инструмент для получения документа, передающий токен."""
    name: str = "get_document_tool_wrapped"
    description: str = "Используется для получения данных о конкретном документе по его идентификатору (ID)."
    args_schema: type[BaseModel] = DocumentIdSchema
    user_token: str = Field(exclude=True)

    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool is designed for async-only invocation. Use ainvoke.")

    async def _arun(self, document_id: str) -> str:
        logger.info(f"Вызов get_document_tool с document_id: {document_id}")
        return await get_document_tool(document_id, self.user_token)

    async def ainvoke(self, input: Dict[str, Any], **kwargs) -> str:
        document_id = input.get("document_id")
        if not document_id:
            raise ValueError("document_id is required")
        return await self._arun(document_id=document_id)


class SearchDocumentsToolWrapped(BaseTool):
    """Обёрнутый инструмент для поиска документов, передающий токен."""
    name: str = "search_documents_tool_wrapped"
    description: str = "Используется для поиска документов по фильтрам (например, по имени, дате, типу)."
    args_schema: type[BaseModel] = SearchFiltersSchema
    user_token: str = Field(exclude=True)

    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool is designed for async-only invocation. Use ainvoke.")

    async def _arun(self, filters: Dict[str, Any]) -> str:
        return await search_documents_tool(filters, self.user_token)

    async def ainvoke(self, input: Dict[str, Any], **kwargs) -> str:
        filters = input.get("filters")
        if filters is None:
            raise ValueError("filters dictionary is required")
        return await self._arun(filters=filters)


class SummarizeAttachmentToolWrapped(BaseTool):
    """Обёрнутый инструмент для суммаризации вложения, передающий токен."""
    name: str = "summarize_attachment_tool_wrapped"
    description: str = (
        "Используется для суммаризации содержимого конкретного вложения документа. "
        "Требует document_id, attachment_id и attachment_name (имя файла). "
        "Используй это, если пользователь спрашивает о содержании вложений, и у тебя уже есть список вложений."
    )
    args_schema: type[BaseModel] = SummarizeAttachmentSchema
    user_token: str = Field(exclude=True)

    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool is designed for async-only invocation. Use ainvoke.")

    async def _arun(self, document_id: str, attachment_id: str, attachment_name: str) -> str:
        """Асинхронное выполнение, использующее внутренний user_token."""
        tool_input = {
            "document_id": document_id,
            "attachment_id": attachment_id,
            "attachment_name": attachment_name,
            "service_token": self.user_token
        }

        return await summarize_attachment_tool.ainvoke(tool_input)

    async def ainvoke(self, input: Dict[str, Any], **kwargs) -> str:
        return await self._arun(
            document_id=input.get("document_id"),
            attachment_id=input.get("attachment_id"),
            attachment_name=input.get("attachment_name")
        )


# ----------------------------------------------------------------------------------------------

async def documents_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел под-агента для работы с документами.
    Реализует внутренний цикл ReAct с инструментами.
    """
    messages = state['messages']
    user_token = state['user_token']
    file_path = state.get('file_path')
    context = state.get('context', {})
    document_id = context.get('document_id')

    logger.debug(f"documents_agent_node: {messages}, {context}, {file_path}, {user_token}")

    llm = get_chat_model()

    tools_to_bind = [
        GetDocumentToolWrapped(user_token=user_token),
        SearchDocumentsToolWrapped(user_token=user_token),
        SummarizeAttachmentToolWrapped(user_token=user_token),
    ]

    current_messages = list(messages)
    max_iterations = 5

    context_prompt = ""
    if file_path:
        context_prompt += f"Внимание: Пользователь загрузил файл. Путь: {file_path}. Учитывай его при ответе."

    if context:
        if document_id:
            context_prompt += f" ТЕКУЩИЙ КОНТЕКСТ: Пользователь находится на странице документа. ID документа: {document_id}."
        else:
            context_prompt += f" ТЕКУЩИЙ КОНТЕКСТ ПОЛЬЗОВАТЕЛЯ: {json.dumps(context, ensure_ascii=False)}"

    if context_prompt:
        current_messages.append(HumanMessage(content=context_prompt))

    format_instruction = (
        "Сформулируй финальный ответ о документе в структурированном формате Markdown, "
        "используя заголовки, списки и **жирный шрифт** для ключевых деталей. "
        "**Если запрос пользователя конкретен (например, 'Кто автор?'), отвечай максимально лаконично, "
        "предоставляя только запрошенную информацию и минимальный связанный контекст (должность/отдел), "
        "и СТРОГО ИСКЛЮЧАЯ все остальные разделы (краткое содержание, вложения, суммы контрактов и т.д.), о которых не просили. "
        "Строго исключи из финального ответа всю служебную информацию, такую как **ID** (документа или вложений), "
        "**UUID**, **размеры файлов**, **даты загрузки** и другие технические поля, не имеющие прямого смысла для пользователя. "
        "Не используй управляющие символы и в тексте ответа."
    )
    current_messages.append(HumanMessage(content=format_instruction))

    react_instruction = (
        "ПРИМЕЧАНИЕ ДЛЯ РАБОТЫ С ДОКУМЕНТАМИ: Прежде чем отвечать на вопросы о содержимом "
        "(особенно о вложениях), СТРОГО используй инструмент 'get_document_tool_wrapped' "
        "для получения списка доступных вложений (их ID и имен). "
        "Только после этого используй 'summarize_attachment_tool_wrapped' с реальными ID. "
        "**Главная цель — ответить на текущий вопрос пользователя.**"
    )
    current_messages.append(HumanMessage(content=react_instruction))

    # --- ReAct Loop ---
    tool_choice_mode = "required"
    final_ai_message: Optional[AIMessage] = None
    iteration_count = 0
    attachments_to_process: List[Dict[str, Any]] = []

    document_data_fetched = False

    while iteration_count < max_iterations:
        iteration_count += 1

        if attachments_to_process and not final_ai_message:
            tool_choice_mode = "required"
        elif document_data_fetched and not final_ai_message:
            tool_choice_mode = "none"

        logger.debug(f"Документальный агент: Итерация {iteration_count}, tool_choice: {tool_choice_mode}")

        llm_with_tools = llm.bind_tools(tools_to_bind, tool_choice=tool_choice_mode)

        response = await llm_with_tools.ainvoke(current_messages)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            current_messages.append(response)
            tool_messages_to_add = []

            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']
                tool_instance = next((t for t in tools_to_bind if t.name == tool_name), None)

                tool_result = None

                if tool_instance:
                    try:
                        tool_result = await tool_instance.ainvoke(tool_args)

                        tool_message = ToolMessage(content=tool_result, tool_call_id=tool_id)
                        tool_messages_to_add.append(tool_message)

                        if tool_name == 'get_document_tool_wrapped':
                            document_data_fetched = True
                            try:
                                doc_data = json.loads(tool_result)
                                attachments = doc_data.get('attachmentDocument', [])
                                if attachments:
                                    attachments_to_process.clear()
                                    attachments_to_process.extend(attachments)

                            except json.JSONDecodeError:
                                logger.error("Не удалось декодировать JSON из tool_result.")

                        if tool_name == 'summarize_attachment_tool_wrapped':
                            if attachments_to_process:
                                attachments_to_process.pop(0)

                    except Exception as e:
                        logger.error(f"Ошибка выполнения инструмента {tool_name} (ID: {tool_id}): {e}", exc_info=True)

                        error_msg = (
                            f"Ошибка выполнения инструмента: {tool_name} не удалось выполнить. "
                            f"Детали ошибки: {type(e).__name__}. Переход к генерации ответа на основе имеющихся данных."
                        )
                        tool_message = ToolMessage(content=error_msg, tool_call_id=tool_id)
                        tool_messages_to_add.append(tool_message)

                        if tool_name == 'summarize_attachment_tool_wrapped':
                            attachments_to_process.clear()

                        tool_choice_mode = "none"


                else:
                    logger.warning(f"Инструмент '{tool_name}' не найден в списке tools_to_bind.")

            if tool_messages_to_add:
                current_messages.extend(tool_messages_to_add)

                if document_id and attachments_to_process:
                    attachment_to_call = attachments_to_process[0]
                    enforcement_message = f"""
                        СТРОГОЕ ПРЕДПИСАНИЕ: Ты получила данные о документе, и в очереди на обработку {len(attachments_to_process)} вложений. 
                        ТЕПЕРЬ СТРОГО ВЫЗОВИ ИНСТРУМЕНТ 'summarize_attachment_tool_wrapped' 
                        со следующими параметрами для обработки вложения ('{attachment_to_call['name']}'):
                        - document_id: {document_id}
                        - attachment_id: {attachment_to_call['id']}
                        - attachment_name: {attachment_to_call['name']}
                        """
                    current_messages.append(HumanMessage(content=enforcement_message))

                    tool_choice_mode = "required"

                elif document_data_fetched:
                    tool_choice_mode = "none"

                else:
                    tool_choice_mode = "required"

            else:
                break

        else:
            final_ai_message = response
            break

    if final_ai_message is None:
        final_ai_message = AIMessage(
            content="Процесс превысил допустимое количество шагов или произошла неисправимая ошибка. Попробуйте перефразировать запрос.")

    formatted_content = format_document_response(final_ai_message.content)

    return {
        "final_response": formatted_content,
        "messages": [AIMessage(content=formatted_content)],
        "subagent_result": formatted_content,
        "called_subagent": "documents_agent",
    }


@register_agent("documents_agent")
def create_documents_agent_graph():
    """Создает и компилирует граф агента по документам."""
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("executor", documents_agent_node)
    workflow.set_entry_point("executor")
    workflow.add_edge("executor", END)

    return workflow.compile()
