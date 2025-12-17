import json
import logging
from typing import Dict, Any, List, Optional, Callable
from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
    BaseMessage,
    AIMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain.tools import BaseTool
from jsonpath_ng.ext import parse as jsonpath_parse
from pydantic import ValidationError

from .models import OrchestratorState, Plan, ToolCallRequest

logger = logging.getLogger(__name__)


def get_llm_with_tools_binding(llm: BaseChatModel, tools: Optional[List[BaseTool]] = None) -> BaseChatModel:
    """
    Привязывает инструменты к LLM, если это возможно.
    Используется для LLM, которые поддерживают Tool Calling (например, OpenAI).
    """
    if tools and hasattr(llm, 'bind_tools') and callable(llm.bind_tools):
        logger.debug(f"Привязка {len(tools)} инструментов к LLM.")
        return llm.bind_tools(tools)
    return llm


def _execute_jsonpath(path_or_value: Any, tool_results_history: List[Dict[str, Any]]) -> Any:
    """
    Разрешает JSONPath выражения, используя историю исполнения инструментов.
    Возвращает исходное значение, если это не JSONPath (или None при ошибке).
    """
    if isinstance(path_or_value, str):
        path_or_value = path_or_value.strip()
        # Убираем возможные скобки, которые иногда добавляет LLM
        if path_or_value.startswith('(') and path_or_value.endswith(')'):
            path_or_value = path_or_value[1:-1].strip()

    if not isinstance(path_or_value, str) or not path_or_value.startswith('$.STEPS'):
        return path_or_value

    try:
        root_data = {"STEPS": tool_results_history}
        jsonpath_expression = jsonpath_parse(path_or_value)
        match = jsonpath_expression.find(root_data)

        if match:
            result = match[0].value

            if isinstance(result, list) and len(result) == 1:
                return result[0]

            return result

        logger.warning(f"JSONPath не нашел совпадений: {path_or_value}")
        return None

    except Exception as e:
        logger.error(f"Ошибка при выполнении JSONPath {path_or_value}: {e}", exc_info=True)
        return None


# ----------------------------------------------------------------
# 1. УЗЕЛ ПЛАНИРОВЩИКА (PLANNER NODE)
# ----------------------------------------------------------------

def planner_node_factory(llm_factory: Callable[[], BaseChatModel], tools: List[BaseTool]) -> Callable:
    """
    Фабрика для создания узла Планировщика с внедренными зависимостями.
    """
    planner_llm = get_llm_with_tools_binding(
        llm_factory(),
        tools=tools
    ).with_structured_output(Plan)

    tool_names = [t.name for t in tools]

    async def planner_node(state: OrchestratorState) -> Dict[str, Any]:
        """
        LLM (Планировщик) анализирует запрос и составляет план действий.
        """
        messages: List[BaseMessage] = state["messages"]
        user_context = state.get("user_context")
        user_context_str = json.dumps(user_context,
                                      ensure_ascii=False) if user_context else "Контекст пользователя не предоставлен."
        context_ui_id = state.get("context_ui_id")
        context_ui_id_str = f"ID контекстного документа: {context_ui_id}" if context_ui_id else "ID контекстного документа: Отсутствует."
        required_file_name = state.get("required_file_name")
        required_file_name_str = f"ТРЕБУЕМЫЙ ФАЙЛ: {required_file_name}" if required_file_name else "ТРЕБУЕМЫЙ ФАЙЛ: Не указан."
        # required_attachment_id_str = f"ТРЕБУЕМЫЙ ID ВЛОЖЕНИЯ: {required_file_id}"

        # Генерация правильного JSONPath для Planner
        attachment_id_jsonpath = (
            f"$.STEPS[0].result.metadata.attachmentDocument[?(@.name=='{required_file_name}')].id"
            if required_file_name else
            "$.STEPS[0].result.metadata.attachmentDocument[0].id"
        )
        # attachment_id_jsonpath = attachment_id_jsonpath.replace("][0].id", "].id")

        attachment_id_jsonpath = (
            "$.STEPS[0].result.metadata.attachmentDocument[0].id"
        )

        # summarize_content_jsonpath = "$.STEPS[1].result.content"
        summarize_content_jsonpath = "$.STEPS[1].result.file_name"

        system_prompt_content = f"""
                        <ROLE>
                        Ты - центральный Оркестратор СЭД. Твоя задача - составить четкий, структурированный план действий в виде списка вызовов инструментов (<steps>) для ответа на запрос пользователя.
                        Ты должен использовать ТОЛЬКО ИНСТРУМЕНТЫ ИЗ СПИСКА <AVAILABLE_TOOLS_SUMMARY>.
                        </ROLE>

                        <AVAILABLE_TOOLS_SUMMARY>
                        Доступные инструменты: {', '.join(tool_names)}
                        </AVAILABLE_TOOLS_SUMMARY>

                        <GLOBAL_CONTEXT>
                        Дополнительный контекст пользователя (роль, права): {user_context_str}
                        {context_ui_id_str}
                        {required_file_name_str}
                        </GLOBAL_CONTEXT>

                        <PLANNING_INSTRUCTIONS>
                        1. Многошаговые задачи: Используй JSONPath ($.STEPS[Индекс_Шага].result.Ключ) для передачи результатов предыдущих шагов в аргументы последующих. Индекс начинается с 0.
                        2. Обязательный аргумент: Всегда передавай 'token' (ключ 'user_token' из состояния) в каждый инструмент.
                        3. **КОНТЕКСТ ДОКУМЕНТА**: Используй `context_ui_id` (значение: {context_ui_id}) как аргумент `document_id` при вызове любых инструментов документации.
                        4. **ОБЯЗАТЕЛЬНАЯ ЦЕПОЧКА ДЛЯ СВОДКИ**:
                           - Шаг 0: Всегда начинай с `doc_metadata_get_by_id_tool` для получения метаданных.
                           - Шаг 1: Используй `doc_attachment_get_content_tool`. **Обязательно включи аргументы `document_id` (значение из `<GLOBAL_CONTEXT>`) и `attachment_id` (через JSONPath).**
                             **КРАЙНЕ ВАЖНО**: 
                             a) Аргумент `document_id` должен быть равен `context_ui_id` (значение: {context_ui_id}).
                             b) Аргумент `attachment_id` должен быть получен с помощью JSONPath из результатов Шага 0. Используй точный путь, включая фильтр:
                                (Пример для attachment_id: `$.STEPS[0].result.metadata.attachmentDocument[0].id`)
                           - Шаг 2: Используй `doc_content_summarize_tool`.
                              a) Аргумент `content`: Получи с помощью JSONPath из результатов Шага 1 (ключ `content`). Пример: `$.STEPS[1].result.content`
                              b) Аргумент **`file_name`**: Получи с помощью JSONPath из результатов Шага 1 (ключ `file_name`). Пример: `{summarize_content_jsonpath}`
                              c) Аргумент **`document_id`**: Используй `context_ui_id` (значение: e56b2ce9-adb4-11f0-980d-1831bf272b96).
                        </PLANNING_INSTRUCTIONS>

                        <FINAL_INSTRUCTION>
                        Если для ответа не требуется вызов инструментов EDMS (например, это приветствие, благодарность или общий вопрос), верни ПУСТОЙ список шагов ([]).
                        Твой вывод должен строго соответствовать схеме {Plan.__name__}.
                        </FINAL_INSTRUCTION>
                        """

        current_messages = list(messages)
        if state.get("tool_results_history"):
            tool_history_message = ToolMessage(
                content=json.dumps(state["tool_results_history"], ensure_ascii=False, indent=2),
                tool_call_id="history_context"
            )
            current_messages.append(tool_history_message)

        messages_with_system = [SystemMessage(content=system_prompt_content)] + current_messages

        try:
            llm_output = await planner_llm.ainvoke(messages_with_system)

            if isinstance(llm_output, Plan):
                response = llm_output
            else:
                response_data = json.loads(llm_output.content)
                response = Plan(**response_data)
            logger.debug(f"PLANNER: Сгенерированный план:\n{response.model_dump_json(indent=2)}")

        except (Exception, ValidationError) as e:
            logger.error(f"Критическая ошибка парсинга/вызова LLM Планировщика: {type(e).__name__}: {e}", exc_info=True)
            return {"tools_to_call": []}

        valid_steps = []
        for step in response.steps:
            try:
                if isinstance(step, dict):
                    step = ToolCallRequest(**step)

                if step.tool_name in tool_names:
                    valid_steps.append(step.model_dump())
                else:
                    logger.warning(f"Планировщик предложил невалидный инструмент: {step.tool_name}")

            except (ValidationError, AttributeError):
                logger.error(f"Некорректный формат шага: {step}")

        logger.info(f"PLANNER: Запланировано {len(valid_steps)} валидных шагов. Причина: {response.reasoning}")

        return {
            "tools_to_call": valid_steps,
        }

    return planner_node


# ----------------------------------------------------------------
# 2. УЗЕЛ ИСПОЛНИТЕЛЯ (TOOL EXECUTOR)
# ----------------------------------------------------------------
def tool_executor_node_factory(tools: List[BaseTool]) -> Callable:
    tool_map = {t.name: t for t in tools}

    async def tool_executor_node(state: OrchestratorState) -> Dict[str, Any]:

        tools_to_call = state.get("tools_to_call", [])
        executed_history = state.get("tool_results_history", [])
        user_token = state.get("user_token")
        context_doc_id = state.get("document_id")

        if not tools_to_call:
            return {"tool_results_history": executed_history, "tools_to_call": []}

        tool_to_call = tools_to_call[0]
        tool_name = tool_to_call.get("tool_name")
        args = tool_to_call.get("arguments", {})

        tool = tool_map.get(tool_name)
        tool_result = None
        resolved_args = args

        if tool is None:
            tool_result = {"error": f"Инструмент '{tool_name}' не найден."}
        else:
            try:
                resolved_args = {k: _execute_jsonpath(v, executed_history) for k, v in args.items()}

                if tool_name == "doc_attachment_get_content_tool" and resolved_args.get("attachment_id") is None:
                    raw_path = args.get("attachment_id", "")
                    required_file_name = state.get('required_file_name')

                    if required_file_name and 'name==' in raw_path:
                        logger.warning(
                            f"Фильтр JSONPath по имени файла ('{required_file_name}') вернул NULL. "
                            f"Попытка использовать ID первого вложения: $.STEPS[0].result.metadata.attachmentDocument[0].id"
                        )
                        first_attachment_path = "$.STEPS[0].result.metadata.attachmentDocument[0].id"

                        resolved_args["attachment_id"] = _execute_jsonpath(first_attachment_path, executed_history)

                        if resolved_args["attachment_id"] is None:
                            raise ValueError(
                                f"JSONPath вернул NULL. Не удалось найти ID вложения по имени '{required_file_name}' или получить ID первого файла в документе."
                            )
                        logger.warning(
                            f"Обход успешен. Используется ID первого вложения: {resolved_args['attachment_id']}")
                    else:
                        file_name_hint = required_file_name or (
                            raw_path if "name==" in raw_path else "None")
                        raise ValueError(
                            f"JSONPath вернул NULL. Не удалось найти ID вложения по имени '{file_name_hint}' в метаданных документа. Проверьте имя файла или его наличие."
                        )

                if tool_name == "doc_content_summarize_tool" and resolved_args.get("content") is None:
                    raise ValueError(
                        "JSONPath вернул NULL. Не удалось получить контент файла для сводки. Шаг 1 (получение контента) завершился неудачей.")

                tool_kwargs = {}
                for k, v in resolved_args.items():
                    if k in tool.args_schema.model_fields:
                        tool_kwargs[k] = v

                tool_kwargs['token'] = user_token

                if "document_id" in tool.args_schema.model_fields:
                    if "document_id" not in tool_kwargs and context_doc_id:
                        tool_kwargs["document_id"] = context_doc_id

                logger.info(f"EXECUTOR: Исполнение: {tool_name}")
                tool_result = await tool.ainvoke(input=tool_kwargs)

                if isinstance(tool_result, dict):
                    import json
                    for key in ["metadata", "results"]:
                        if key in tool_result and isinstance(tool_result[key], str):
                            try:
                                tool_result[key] = json.loads(tool_result[key])
                                logger.debug(f"EXECUTOR: Поле '{key}' успешно десериализовано из строки в JSON.")
                            except json.JSONDecodeError:
                                pass

            except Exception as e:
                error_msg = str(e)
                logger.error(f"EXECUTOR: Ошибка исполнения: {error_msg}")
                tool_result = {"error": error_msg}

        executed_history.append({
            "tool_name": tool_name,
            "arguments": args,
            "resolved_arguments": resolved_args,
            "result": tool_result
        })

        return {
            "tool_results_history": executed_history,
            "tools_to_call": tools_to_call[1:]
        }

    return tool_executor_node


# ----------------------------------------------------------------
# 3. УЗЕЛ ПРОВЕРКИ РЕЗУЛЬТАТОВ (CHECKER NODE)
# ----------------------------------------------------------------

TOOL_METADATA = 'doc_metadata_get_by_id_tool'
TOOL_SEARCH_EMP = 'employee_tools_search_tool'


def tool_result_checker_node(state: OrchestratorState) -> Dict[str, Any]:
    executed_history = state.get("tool_results_history", [])
    if not executed_history:
        return {"next_route": "responder" if not state.get("tools_to_call") else "executor"}

    last_execution = executed_history[-1]
    last_result = last_execution.get("result", {})
    tool_name = last_execution.get("tool_name")

    if isinstance(last_result, dict) and "error" in last_result:
        logger.error(f"CHECKER: Обнаружена ошибка исполнения: {last_result['error']}. Переход к Responder.")
        state["tools_to_call"] = []
        return {"next_route": "responder"}

    if state.get("tools_to_call"):
        logger.info("CHECKER: В очереди есть инструменты. Переход к исполнителю.")
        return {"next_route": "executor"}

    if tool_name == TOOL_METADATA:
        metadata = last_result.get("metadata")

        attachments = metadata.get("attachmentDocument") if isinstance(metadata, dict) else []
        if not isinstance(attachments, list):
            attachments = []

        valid_attachments = [a for a in attachments if isinstance(a, dict) and a.get("id")]

        required_file_name = state.get("required_file_name")

        if len(valid_attachments) > 1 and not required_file_name:
            logger.warning(
                f"CHECKER: Обнаружено {len(valid_attachments)} валидных вложений. Требуется HiTL для выбора.")
            return {
                "next_route": "responder",
                "selection_context": {
                    "reason": "multiple_attachments",
                    "attachments": [
                        {
                            "id": att["id"],
                            "name": att["name"],
                            "type": att.get("attachmentDocumentType", "UNKNOWN")
                        }
                        for att in valid_attachments
                    ]
                }
            }

    if tool_name == TOOL_SEARCH_EMP:
        results = last_result.get("results")
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except:
                results = []

        if isinstance(results, list) and len(results) > 1:
            logger.warning("CHECKER: Обнаружено несколько совпадений сотрудников. Требуется HiTL.")

            return {
                "next_route": "responder",
                "selection_context": {
                    "reason": "multiple_employees",
                    "employees": results
                }
            }

    logger.info("CHECKER: Все запланированные шаги выполнены или HiTL не требуется. Переход к ответу.")
    return {"next_route": "responder"}


# ----------------------------------------------------------------
# 4. УЗЕЛ ФИНАЛЬНОГО ОТВЕТА (RESPONDER NODE)
# ----------------------------------------------------------------

def responder_node_factory(llm_factory: Callable[[], BaseChatModel]) -> Callable:
    """
    Фабрика для создания узла Генератора Ответа с внедренным LLM.
    """
    responder_llm = llm_factory()

    async def responder_node(state: OrchestratorState) -> Dict[str, Any]:
        """
        Генерирует финальный ответ пользователю.
        """
        messages: List[BaseMessage] = state["messages"]
        history: List[Dict[str, Any]] = state.get("tool_results_history", [])

        # ----------------------------------------------------------------
        # 4.1. ОБРАБОТКА HUMAN-IN-THE-LOOP (HiTL)
        # ----------------------------------------------------------------

        selection_context = state.get("selection_context")

        if selection_context:
            logger.info("RESPONDER: Генерация запроса на уточнение данных (Human-in-the-Loop).")

            response_content = "Требуется уточнение, но я не смог определить, что именно. Пожалуйста, повторите запрос."

            reason = selection_context.get("reason")

            if reason == "multiple_attachments":
                attachment_list = selection_context.get("attachments", [])

                numbered_attachments = []
                for i, att in enumerate(attachment_list):
                    name = att.get('name', 'Неизвестный файл')
                    file_type = att.get('type', 'UNKNOWN')

                    display_name = name
                    if file_type == "PRINT_DOCUMENT":
                        display_name += " (Печатная форма)"

                    numbered_attachments.append(
                        f"{len(numbered_attachments) + 1}. **{display_name}**")

                list_items = "\n".join(numbered_attachments)

                summary_options = (
                    "Доступные форматы сводки:\n"
                    "  - `TL;DR`: Кратчайшая выжимка (1-2 предложения).\n"
                    "  - `Развернутая`: Развернутый обзор (3-5 предложений, по умолчанию).\n"
                    "  - `Структурированная`: Структурированная сводка (ключевые пункты)."
                )

                if not numbered_attachments:
                    response_content = "Метаданные содержат несколько вложений, но не удалось определить основные файлы для обработки. Пожалуйста, уточните имя файла вручную."
                else:
                    response_content = (
                        "Обнаружено несколько вложений. Пожалуйста, укажите, какой файл нужно обработать (по номеру или названию), "
                        "и, при необходимости, выберите формат сводки:\n\n"
                        "**Доступные вложения:**\n"
                        f"{list_items}\n\n"
                        f"{summary_options}\n\n"
                        "Например: `Сделай структурированную сводку для файла 2` или `Сводка для файла 'Договор.pdf'`"
                    )

            elif reason == "multiple_employees":
                employee_list = selection_context.get("employees", [])
                list_items = "\n".join([
                    f"{i + 1}. **{emp.get('full_name', 'N/A')}** ({emp.get('position', 'N/A')})"
                    for i, emp in enumerate(employee_list)
                ])
                response_content = (
                    "Найдено несколько сотрудников. Пожалуйста, уточните, данные какого сотрудника вы хотите получить:\n\n"
                    "**Доступные сотрудники:**\n"
                    f"{list_items}\n\n"
                    "Например: `Выбери сотрудника 1` или `Данные для Иванова Ивана`"
                )

            response = AIMessage(content=response_content)

            return {"messages": state["messages"] + [response], "tools_to_call": [], "selection_context": None,
                    "next_route": None}

        # ----------------------------------------------------------------
        # 4.2. СТАНДАРТНЫЙ ОТВЕТ (Используем ToolMessage)
        # ----------------------------------------------------------------
        attachment_summary = ""
        for item in history:
            if item.get("tool_name") == 'doc_metadata_get_by_id_tool':
                metadata = item["result"].get("metadata")

                if isinstance(metadata, dict):
                    attachments = metadata.get("attachmentDocument")

                    if attachments and isinstance(attachments, list):
                        attachment_list = []
                        for att in attachments:
                            if isinstance(att, dict) and att.get("name") and att.get("id"):
                                name = att["name"]
                                file_type = att.get("attachmentDocumentType", "UNKNOWN")

                                display_name = name
                                if file_type == "PRINT_DOCUMENT":
                                    display_name += " (Печатная форма)"

                                attachment_list.append(display_name)

                        if attachment_list:
                            attachment_summary = "Обнаруженные вложения документа (список для включения в ответ):\n* " + "\n* ".join(
                                attachment_list)
                            break
                elif "error" in item["result"]:
                    attachment_summary = f"**Ошибка получения метаданных:** {item['result']['error']}"
                    break
        # ====================================================================================

        tool_messages: List[ToolMessage] = []
        for idx, item in enumerate(history):
            if "error" in item["result"]:
                result_content = f"ToolCall ({item['tool_name']}) ОШИБКА: {item['result']['error']}"
            else:
                try:
                    result_str = json.dumps(item['result'], ensure_ascii=False)
                    result_content = "ToolCall OK. Результат:\n" + result_str[:4000] + (
                        "..." if len(result_str) > 4000 else "")
                except:
                    result_content = "ToolCall OK. Результат: (невозможно сериализовать)"

            fake_tool_call_id = f"call_{item['tool_name']}_{idx}"

            tool_messages.append(ToolMessage(
                content=result_content,
                tool_call_id=fake_tool_call_id
            ))

        user_context = state.get("user_context")
        user_context_str = json.dumps(user_context, ensure_ascii=False) if user_context else "Не предоставлен."

        system_prompt_content = f"""
        <ROLE>
        Ты - AI-ассистент СЭД. Твоя задача: проанализировать историю диалога, результаты вызовов инструментов (<TOOL_RESULTS>) и контекст пользователя (<USER_CONTEXT>).
        Сформулируй краткий, точный и полезный ФИНАЛЬНЫЙ ответ пользователю.
        </ROLE>

        <USER_CONTEXT>
        Контекст пользователя: {user_context_str}
        </USER_CONTEXT>

        <GUARANTEED_ATTACHMENTS>
        {attachment_summary if attachment_summary else "Документация о вложениях отсутствует в результатах или не требуется для ответа."}
        </GUARANTEED_ATTACHMENTS>

        <INSTRUCTIONS>
        1. Используй <TOOL_RESULTS> и информацию из <GUARANTEED_ATTACHMENTS> для ответа. Если <GUARANTEED_ATTACHMENTS> содержит список вложений, **обязательно** перечисли его в финальном ответе.
        2. Если в <TOOL_RESULTS> есть ошибки, извинись и объясни причину ошибки (например, "Не удалось найти документ").
        3. Если <TOOL_RESULTS> пуст, а запрос пользователя является общим (приветствие, вопрос о погоде), ответь вежливо и спроси, чем ты можешь помочь в рамках СЭД.
        4. Ответ должен быть только текстовым, без вызова функций.
        </INSTRUCTIONS>
        """

        final_messages = [SystemMessage(content=system_prompt_content)] + messages + tool_messages

        logger.info("RESPONDER: Генерация финального ответа.")
        response = await responder_llm.ainvoke(final_messages)

        new_messages: List[BaseMessage] = state["messages"] + [response]

        return {"messages": new_messages, "next_route": None}

    return responder_node


# ----------------------------------------------------------------
# 5. УЗЕЛ УДАЛЕНИЯ ИСТОРИИ (DELETE HISTORY NODE)
# ----------------------------------------------------------------

def delete_history_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел, который сохраняет только последние 1-2 сообщения для контекста
    следующего запроса (для Checkpointer).
    """
    messages: List[BaseMessage] = state["messages"]

    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    human_messages = [m for m in messages if m.type == 'human']

    retained_messages: List[BaseMessage] = []

    if human_messages:
        retained_messages.append(human_messages[-1])

    if ai_messages:
        retained_messages.append(ai_messages[-1])

    logger.info(f"DELETER: Сохранено {len(retained_messages)} сообщений для следующего цикла.")

    state_update = {
        "messages": retained_messages,
        "tools_to_call": None,
        "tool_results_history": None,
        "selection_context": None,
        "next_route": None,
    }

    return state_update