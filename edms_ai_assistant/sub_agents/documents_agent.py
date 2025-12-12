# edms_ai_assistant.sub_agents.documents_agent

import logging
import json
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from typing import Dict, Any
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from edms_ai_assistant.core.sub_agents import register_agent
from edms_ai_assistant.core.orchestrator import OrchestratorState, _extract_summary_intent
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.tools.document import get_document_tool, search_documents_tool
from edms_ai_assistant.tools.attachment import summarize_attachment_tool, extract_and_summarize_file_async_tool
from edms_ai_assistant.utils.format_utils import format_document_response

try:
    from edms_ai_assistant.constants import SUMMARY_TYPES
except ImportError:
    SUMMARY_TYPES = {}

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------
# --- Schemas & Wrapped Tools (Оставлены без изменений, кроме ExtractAndSummarizeFileToolWrapped - добавлен summary_type) ---
# ----------------------------------------------------------------------------------------------


class DocumentIdSchema(BaseModel):
    document_id: str = Field(description="Уникальный идентификатор документа (UUID).")


class SearchFiltersSchema(BaseModel):
    filters: Dict[str, Any] = Field(
        description="Словарь фильтров для поиска документов."
    )


class SummarizeAttachmentSchema(BaseModel):
    document_id: str = Field(description="Уникальный идентификатор документа (UUID).")
    attachment_id: str = Field(description="Уникальный идентификатор вложения (UUID).")
    attachment_name: str = Field(
        description="Имя файла вложения, которое нужно суммировать."
    )
    # Добавляем summary_type, чтобы LLM мог передать его
    summary_type: Optional[str] = Field(description="Тип суммаризации (e.g., 'SHORT', 'DETAILED', 'LEGAL').")


class SummarizeFileSchema(BaseModel):
    file_path: str = Field(description="Локальный путь к файлу, который нужно суммировать.")
    # Добавляем summary_type
    summary_type: Optional[str] = Field(description="Тип суммаризации (e.g., 'SHORT', 'DETAILED', 'LEGAL').")


class ExtractAndSummarizeFileToolWrapped(BaseTool):
    """Обёрнутый инструмент для суммаризации локально загруженного файла."""

    name: str = "extract_and_summarize_file_async_tool_wrapped"
    description: str = (
        "Используется для суммаризации содержимого локально загруженного файла. "
        "Требует file_path и service_token. Используй СТРОГО, если file_path присутствует в state."
    )
    args_schema: type[BaseModel] = SummarizeFileSchema
    user_token: str = Field(exclude=True)

    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool is designed for async-only invocation. Use ainvoke.")

    async def _arun(self, file_path: str, summary_type: Optional[str]) -> str:
        """Асинхронное выполнение, использующее внутренний user_token."""
        # 📌 summary_type по умолчанию передается как None, если не задан.
        return await extract_and_summarize_file_async_tool(
            file_path=file_path,
            service_token=self.user_token,
            summary_type=summary_type
        )

    async def ainvoke(self, input: Dict[str, Any], **kwargs) -> str:
        return await self._arun(
            file_path=input.get("file_path"),
            summary_type=input.get("summary_type"),
        )


class GetDocumentToolWrapped(BaseTool):
    # Оставлен без изменений
    """Обёрнутый инструмент для получения документа, передающий токен."""

    name: str = "get_document_tool_wrapped"
    description: str = (
        "Используется для получения данных о конкретном документе по его идентификатору (ID)."
    )
    args_schema: type[BaseModel] = DocumentIdSchema
    user_token: str = Field(exclude=True)

    def _run(self, *args, **kwargs):
        raise NotImplementedError(
            "This tool is designed for async-only invocation. Use ainvoke."
        )

    async def _arun(self, document_id: str) -> str:
        logger.info(f"Вызов get_document_tool с document_id: {document_id}")
        return await get_document_tool(document_id, self.user_token)

    async def ainvoke(self, input: Dict[str, Any], **kwargs) -> str:
        document_id = input.get("document_id")
        if not document_id:
            raise ValueError("document_id is required")
        return await self._arun(document_id=document_id)


class SearchDocumentsToolWrapped(BaseTool):
    # Оставлен без изменений
    """Обёрнутый инструмент для поиска документов, передающий токен."""

    name: str = "search_documents_tool_wrapped"
    description: str = (
        "Используется для поиска документов по фильтрам (например, по имени, дате, типу)."
    )
    args_schema: type[BaseModel] = SearchFiltersSchema
    user_token: str = Field(exclude=True)

    def _run(self, *args, **kwargs):
        raise NotImplementedError(
            "This tool is designed for async-only invocation. Use ainvoke."
        )

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
        "Требует document_id, attachment_id, attachment_name и summary_type. "
        "Используй это, если пользователь спрашивает о содержании вложений."
    )
    args_schema: type[BaseModel] = SummarizeAttachmentSchema
    user_token: str = Field(exclude=True)

    def _run(self, *args, **kwargs):
        raise NotImplementedError(
            "This tool is designed for async-only invocation. Use ainvoke."
        )

    async def _arun(
            self, document_id: str, attachment_id: str, attachment_name: str, summary_type: Optional[str]
    ) -> str:
        """Асинхронное выполнение, использующее внутренний user_token."""
        tool_input = {
            "document_id": document_id,
            "attachment_id": attachment_id,
            "attachment_name": attachment_name,
            "service_token": self.user_token,
            "summary_type": summary_type  # Передача summary_type
        }

        return await summarize_attachment_tool.ainvoke(tool_input)

    async def ainvoke(self, input: Dict[str, Any], **kwargs) -> str:
        return await self._arun(
            document_id=input.get("document_id"),
            attachment_id=input.get("attachment_id"),
            attachment_name=input.get("attachment_name"),
            summary_type=input.get("summary_type"),
        )


# ----------------------------------------------------------------------------------------------
# --- НОВЫЕ УЗЛЫ LANGGRAPH (CODE & LLM) ---
# ----------------------------------------------------------------------------------------------


async def fetch_document_data_node(state: OrchestratorState) -> Dict[str, Any]:
    """Узел, управляемый кодом: получает данные о документе, определяет намерение суммирования и сохраняет данные."""

    document_id = state.get("context", {}).get("document_id")
    user_token = state["user_token"]
    messages = state["messages"]

    # 1. ОПРЕДЕЛЯЕМ И СОХРАНЯЕМ НАМЕРЕНИЕ
    is_summary_request = state.get("is_summary_request_initial", False)

    if not is_summary_request:
        initial_query = ""
        # ❗ ПРОВЕРКА: Извлекаем контент из первого HumanMessage
        if messages and isinstance(messages[0], HumanMessage):
            initial_query = messages[0].content

        # 1.1. Определяем намерение по начальному запросу
        is_summary_request = _extract_summary_intent(initial_query)

    # 1.2. ЛОГИКА ДЛЯ HITL-ОТВЕТА ('3') - КЛЮЧЕВОЙ МОМЕНТ
    last_message_content = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
    if state.get("is_hitl_active") and last_message_content.isdigit():
        # Если последний ответ - цифра, это явный запрос на суммирование после HITL
        is_summary_request = True
        state["summary_type"] = SUMMARY_TYPES.get(last_message_content, "SHORT")
        logger.debug(f"HITL-ответ распознан как явный запрос на суммирование. Тип: {state['summary_type']}.")

    # 2. Если есть локальный файл, пропускаем получение данных EDMS, но сохраняем флаг
    if state.get("file_path"):
        return {"document_data_fetched": False, "is_summary_request_initial": is_summary_request}

    if not document_id:
        # Если нет ID, завершаем, но сохраняем флаг
        return {"document_data_fetched": False, "is_summary_request_initial": is_summary_request}

    # Инициализируем инструмент
    get_doc_tool = GetDocumentToolWrapped(user_token=user_token)

    try:
        raw_result = await get_doc_tool.ainvoke({"document_id": document_id})
        doc_data = json.loads(raw_result)
        tool_result = json.dumps(doc_data, ensure_ascii=False)

        # 3. СОХРАНЯЕМ ВЛОЖЕНИЯ
        attachments = doc_data.get("attachmentDocument", [])

        if attachments and is_summary_request:
            # Обрабатываем только первое вложение (если хотим ограничиться одним)
            attachments_to_process = [attachments[0]]
            logger.info(f"Найдено {len(attachments)} вложений. Выбрано 1 для обработки: {attachments[0].get('name')}.")
        else:
            attachments_to_process = []
            if is_summary_request:
                logger.warning("Запрос на суммирование, но в документе не найдено вложений.")

        # Создаем ToolMessage с ПОЛНЫМИ данными для истории.
        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=f"doc_fetch_{document_id}",
            name="get_document_tool_wrapped"
        )

        # Возвращаем обновленное состояние
        return {
            "document_data_fetched": True,
            "document_data_json": tool_result,
            "attachments_to_process": attachments_to_process,
            "messages": messages + [tool_message],
            "is_summary_request_initial": is_summary_request  # Пересохраняем флаг!
        }

    except Exception as e:
        logger.error(f"Ошибка получения документа {document_id}: {e}")
        error_msg = f"Ошибка получения данных документа: {type(e).__name__}."

        return {
            "document_data_fetched": False,
            "messages": messages + [AIMessage(content=error_msg)],
        }


async def summarize_attachment_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел, управляемый LLM: вызывает инструмент для суммаризации вложения.
    """
    messages = state["messages"]
    user_token = state["user_token"]
    attachments_to_process = state.get("attachments_to_process", [])
    document_id = state.get("context", {}).get("document_id")
    summary_type = state.get("summary_type")

    # Если нет вложений для обработки
    if not attachments_to_process:
        return {"messages": messages}  # Переходим к финалу без ошибок

    attachment = attachments_to_process[0]

    # 🛑 ИСПРАВЛЕНИЕ КЛЮЧЕЙ: Используем 'id' и 'name' из DTO/JSON
    attachment_id = attachment.get("id")
    attachment_name = attachment.get("name")

    if not attachment_id or not attachment_name:
        logger.warning("Некорректные данные вложения (отсутствует id или name). Пропускаем суммирование.")
        # Удаляем обработанное вложение и пытаемся перейти к следующему (если бы их было несколько)
        return {"attachments_to_process": attachments_to_process[1:]}

    llm = get_chat_model()

    # Инструкция для LLM, чтобы он вызвал нужный инструмент (включая summary_type)
    system_instruction = (
            f"Ты должен вызвать инструмент 'summarize_attachment_tool_wrapped' с аргументами: "
            f"document_id='{document_id}', attachment_id='{attachment_id}', attachment_name='{attachment_name}'"
            + (f", summary_type='{summary_type}'" if summary_type else "") + "."
    )

    # 📌 УБРАНО: HumanMessage(content=system_instruction) - LLM не нуждается в этом для вызова инструмента

    tools_to_bind = [SummarizeAttachmentToolWrapped(user_token=user_token)]
    llm_with_tool = llm.bind_tools(
        tools_to_bind,
        tool_choice={'type': 'function', 'function': {'name': 'summarize_attachment_tool_wrapped'}}
    )

    try:
        # LLM получает текущую историю и неявный вызов инструмента через bind_tools
        # Добавляем лишь короткий промпт, чтобы инициировать tool_call
        tool_call_prompt = [AIMessage(content=f"Суммируй вложение {attachment_name} по запрошенному типу.")]
        response = await llm_with_tool.ainvoke(messages + tool_call_prompt)

        # Обработка вызова инструмента
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]

            tool_args = dict(tool_call['args'])
            # 📌 ПРИНУДИТЕЛЬНОЕ ДОБАВЛЕНИЕ summary_type, если LLM его не добавил
            if summary_type and 'summary_type' not in tool_args:
                tool_args['summary_type'] = summary_type

            tool_instance = next(t for t in tools_to_bind if t.name == tool_call["name"])
            tool_result = await tool_instance.ainvoke(tool_args)

            # Обновляем состояние: удаляем обработанное вложение и добавляем ToolMessage
            new_attachments_list = attachments_to_process[1:]

            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            )

            # ❗ ДОБАВЛЕН ЛОГ УСПЕШНОГО ВЫЗОВА
            logger.info(f"Успешное суммирование вложения {attachment_name}.")

            return {
                "attachments_to_process": new_attachments_list,
                "messages": messages + [tool_message],
            }
        else:
            logger.error(
                "LLM не вызвал инструмент в summarize_attachment_node. Возможно, проблема в промпте/bind_tools.")
            return {"messages": messages,
                    "attachments_to_process": attachments_to_process[1:]}  # Идем дальше без суммирования

    except Exception as e:
        logger.error(f"Ошибка в узле summarize_attachment_node: {e}")
        error_msg = f"Ошибка суммирования вложения: {type(e).__name__}."

        # Добавляем ошибку в историю и переходим к финализации
        return {
            "attachments_to_process": attachments_to_process[1:],
            "messages": messages + [AIMessage(content=error_msg)],
        }


async def summarize_file_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел, управляемый LLM: вызывает инструмент для суммаризации локального файла.
    """
    messages = state["messages"]
    user_token = state["user_token"]
    file_path = state["file_path"]
    summary_type = state.get("summary_type")

    llm = get_chat_model()

    tools_to_bind = [ExtractAndSummarizeFileToolWrapped(user_token=user_token)]
    llm_with_tool = llm.bind_tools(
        tools_to_bind,
        tool_choice={'type': 'function', 'function': {'name': 'extract_and_summarize_file_async_tool_wrapped'}}
    )

    try:
        # LLM получает текущую историю и неявный вызов инструмента через bind_tools
        tool_call_prompt = [AIMessage(content=f"Суммируй загруженный файл по запрошенному типу.")]
        response = await llm_with_tool.ainvoke(messages + tool_call_prompt)

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]

            tool_args = dict(tool_call['args'])
            # 📌 ПРИНУДИТЕЛЬНОЕ ДОБАВЛЕНИЕ summary_type
            if summary_type and 'summary_type' not in tool_args:
                tool_args['summary_type'] = summary_type

            tool_instance = next(t for t in tools_to_bind if t.name == tool_call["name"])
            tool_result = await tool_instance.ainvoke(tool_args)

            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            )

            return {
                "messages": messages + [tool_message],
                "file_path": None,  # Сбрасываем, чтобы не вызывать повторно
            }
        else:
            logger.error("LLM не вызвал инструмент в summarize_file_node.")
            return {"messages": messages, "file_path": None}

    except Exception as e:
        logger.error(f"Ошибка в узле summarize_file_node: {e}")
        error_msg = f"Ошибка суммирования файла: {type(e).__name__}."

        return {
            "messages": messages + [AIMessage(content=error_msg)],
            "file_path": None,
        }


async def generate_final_response_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел, управляемый LLM: генерирует финальный, отформатированный ответ.
    """
    messages = state["messages"]
    is_summary_request = state.get("is_summary_request_initial", False)

    # ❗ ЛОГИКА ФИЛЬТРАЦИИ ДАННЫХ ПЕРЕД ОТПРАВКОЙ LLM (Оставлена без изменений - она корректна)
    if not is_summary_request and state.get("document_data_json"):

        full_data = json.loads(state["document_data_json"])

        if "attachmentDocument" in full_data:
            full_data.pop("attachmentDocument")
            logger.debug("Вопрос не о содержании (фильтр перед финальным LLM). Удаляем 'attachmentDocument'.")

        cleaned_tool_result = json.dumps(full_data, ensure_ascii=False)
        cleaned_tool_message = ToolMessage(
            content=cleaned_tool_result,
            tool_call_id=f"doc_fetch_cleaned_{full_data.get('id')}",
            name="get_document_tool_wrapped"
        )

        new_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "get_document_tool_wrapped" and msg.content == state[
                "document_data_json"]:
                new_messages.append(cleaned_tool_message)
            else:
                new_messages.append(msg)

        final_messages_for_llm = new_messages
    else:
        final_messages_for_llm = messages

    llm = get_chat_model()

    # 📌 ФИНАЛЬНЫЙ СТРОГИЙ ПРОМПТ
    final_instruction = (
        "Ты — специализированный под-агент, отвечающий за работу с документами СЭД. "
        "Твоя задача — дать **финальный, отформатированный, структурированный и строго релевантный** ответ на исходный запрос пользователя, основываясь на всей истории сообщений, включая полученные метаданные и результаты суммирования. "

        "### ПРАВИЛА ГЕНЕРАЦИИ ОТВЕТА (Markdown)\n"
        "1. **РЕЛЕВАНТНОСТЬ (ПРИОРИТЕТ):** Ответ должен **СТРОГО** соответствовать исходному вопросу. Если спрашивали только об авторе, отвечаешь только об авторе.\n"
        "2. **СТРУКТУРА:** Используй заголовки, списки и **жирный шрифт** для ключевых деталей.\n"
        "3. **СОДЕРЖАНИЕ ВЛОЖЕНИЙ:** Раздел о вложениях включай **ТОЛЬКО** если: а) ты получил результат суммирования вложения ИЛИ б) вопрос явно касался вложений.\n"
        "4. **ЗАПРЕТЫ:** Категорически запрещено упоминать названия инструментов, технические шаги (например, 'необходимо извлечь', 'проанализировать') или любую служебную информацию (UUID, ID, размеры, даты)."
    )

    final_prompt = [HumanMessage(content=final_instruction)] + final_messages_for_llm

    try:
        response = await llm.ainvoke(final_prompt)
        raw_content = response.content or ""

        # 📌 ИСПОЛЬЗУЕМ ФОРМАТТЕР (post-processing)
        formatted_content = format_document_response(raw_content)
        final_result = formatted_content if formatted_content else raw_content

    except Exception as e:
        logger.error(f"Ошибка генерации финального ответа: {e}")
        final_result = "Не удалось получить осмысленный ответ от модели."

    return {
        "final_response": final_result,
        "messages": [AIMessage(content=final_result)],
        "subagent_result": final_result,
        "called_subagent": "documents_agent",
        "attachments_to_process": [],  # Очистка
    }


def route_next_step(state: Dict[str, Any]) -> str:
    """
    Определяет следующий шаг: суммирование или финальный ответ,
    опираясь исключительно на флаги, установленные в предыдущих узлах.
    """

    is_summary_request = state.get("is_summary_request_initial", False)
    file_path = state.get("file_path")
    attachments = state.get("attachments_to_process", [])

    # ❗ ДОБАВЛЕН ЛОГ ДЛЯ ПОНИМАНИЯ МАРШРУТИЗАЦИИ
    logger.debug(
        f"Route check: summary={is_summary_request}, file_path={file_path}, attachments_count={len(attachments)}")

    # --- ЛОГИКА МАРШРУТИЗАЦИИ ---

    # 1. Если есть ЛОКАЛЬНЫЙ файл И запрошено суммирование (PRIORITY)
    if file_path and is_summary_request:
        return "summarize_file"

    # 2. Если запрошено СУММИРОВАНИЕ, И есть вложения для обработки
    if is_summary_request and attachments:
        return "summarize_attachment"

    # 3. Во всех остальных случаях:
    # - Нет запроса на суммирование.
    # - Есть запрос, но нет подходящих вложений/файла.
    # - Нужен финальный ответ по метаданным.
    return "generate_final_response"


# ----------------------------------------------------------------------------------------------
# --- LANGGRAPH ASSEMBLY (Окончательная сборка) ---
# ----------------------------------------------------------------------------------------------

@register_agent("documents_agent")
def create_documents_agent_graph():
    """Создает и компилирует граф агента по документам."""

    nodes_map = {
        "fetch_data": fetch_document_data_node,
        "summarize_attachment": summarize_attachment_node,
        "summarize_file": summarize_file_node,
        "generate_final_response": generate_final_response_node,
    }

    workflow = StateGraph(OrchestratorState)

    # Добавление узлов
    for name, node in nodes_map.items():
        workflow.add_node(name, node)

    # 1. Определение начальной точки
    workflow.set_entry_point("fetch_data")

    # 2. Маршрутизация после получения данных (КОД)
    workflow.add_conditional_edges(
        "fetch_data",
        route_next_step,
        {
            "summarize_file": "summarize_file",
            "summarize_attachment": "summarize_attachment",
            "generate_final_response": "generate_final_response",
        }
    )

    # 3. Маршрутизация после суммаризации файла (всегда к финалу)
    workflow.add_edge("summarize_file", "generate_final_response")

    # 4. Маршрутизация после суммаризации вложения (может быть несколько вложений, но пока идем к финалу)
    # Если вы хотите обрабатывать ВСЕ вложения, здесь должна быть условная петля обратно к summarize_attachment.
    # Но так как вы ограничили список одним элементом, идем к финалу.
    workflow.add_edge("summarize_attachment", "generate_final_response")

    # 5. Завершение
    workflow.add_edge("generate_final_response", END)

    return workflow.compile()