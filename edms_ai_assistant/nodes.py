# Файл: edms_ai_assistant/nodes.py

import json
import logging
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, RemoveMessage, BaseMessage, AIMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.language_models import BaseChatModel
from langchain.tools import BaseTool
from jsonpath_ng.ext import parse as jsonpath_parse

from .models import OrchestratorState, Plan, ToolCallRequest
from .tools import ALL_TOOLS
from .llm import get_chat_model as get_base_chat_model

logger = logging.getLogger(__name__)


# ... (Вспомогательные функции get_chat_model и _execute_jsonpath остаются без изменений)
def get_chat_model(tools: List[BaseTool] = None) -> BaseChatModel:
    """Обертка для get_chat_model, добавляющая инструменты."""
    llm = get_base_chat_model()
    if tools:
        return llm.bind_tools(tools)
    return llm


def _execute_jsonpath(path_or_value: Any, tool_results_history: List[Dict[str, Any]]) -> Any:
    """
    Разрешает JSONPath выражения, используя историю исполнения инструментов.
    Возвращает исходное значение, если это не JSONPath.
    """
    # ИЗМЕНЕНИЕ: Теперь ожидаем, что путь начинается с '$.STEPS'
    if not isinstance(path_or_value, str) or not path_or_value.startswith('$.STEPS'):
        return path_or_value

    try:
        # Создаем корневой объект, который включает историю
        root_data = {"STEPS": tool_results_history}

        # Парсим и выполняем JSONPath
        # Убедимся, что path_or_value корректно начинается с $
        jsonpath_expression = jsonpath_parse(path_or_value)
        match = jsonpath_expression.find(root_data)

        if match:
            # Возвращаем значение первого совпадения
            return match[0].value

        logger.warning(f"JSONPath не нашел совпадений: {path_or_value}")
        return None  # Если путь не найден, возвращаем None

    except Exception as e:
        logger.error(f"Ошибка при выполнении JSONPath {path_or_value}: {e}")
        return None


# ----------------------------------------------------------------
# 1. УЗЕЛ ПЛАНИРОВЩИКА (PLANNER NODE) - ИСПРАВЛЕНИЕ JSONPATH
# ----------------------------------------------------------------

async def planner_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LLM (Планировщик) анализирует запрос и составляет план действий.
    Добавлена логика Plan Override для запроса сводки документа.
    """

    messages: List[Any] = state["messages"]

    # ----------------------------------------------------------------
    # 1. PLAN OVERRIDE (Для запроса сводки с ID - Исправление Теста 1)
    # ----------------------------------------------------------------
    # if (state.get("context_ui_id")
    #         and messages[-1].content.lower().strip().startswith("сделай сводку")):
    #     logger.info("PLANNER: Обнаружен запрос сводки с ID. Принудительное выполнение 3-шагового плана.")
    #
    #     doc_id = state["context_ui_id"]
    #
    #     # Шаг 1: Получить метаданные
    #     step1 = ToolCallRequest(
    #         tool_name="doc_metadata_get_by_id",
    #         arguments={"document_id": doc_id}
    #     )
    #     # Шаг 2: Получить контент вложения
    #     step2 = ToolCallRequest(
    #         tool_name="doc_attachment_get_content",
    #         arguments={
    #             "document_id": doc_id,
    #             # ИСПРАВЛЕНО: Добавлена точка для корректного JSONPath
    #             "attachment_id": "$.STEPS[0].result.attachmentDocument[0].id"
    #         }
    #     )
    #     # Шаг 3: Сделать сводку
    #     step3 = ToolCallRequest(
    #         tool_name="doc_content_summarize",
    #         arguments={
    #             "document_id": doc_id,
    #             # ИСПРАВЛЕНО: Добавлена точка для корректного JSONPath
    #             "content_key": "$.STEPS[1].result.content_key",
    #             "file_name": "$.STEPS[0].result.attachmentDocument[0].fileName",
    #             "metadata_context": "$.STEPS[0].result"
    #         }
    #     )
    #
    #     plan_steps = [step1, step2, step3]
    #     logger.info(f"PLANNER: Принудительно запланировано {len(plan_steps)} шагов.")
    #     return {"tools_to_call": [step.dict() for step in plan_steps]}

    # ----------------------------------------------------------------
    # 2. STANDARD LLM PLANNING (Для всех остальных запросов)
    # ----------------------------------------------------------------
    user_context = state.get("user_context")
    user_context_str = json.dumps(user_context,
                                  ensure_ascii=False) if user_context else "Контекст пользователя не предоставлен."

    context_ui_id = state.get("context_ui_id")
    context_ui_id_str = f"ID контекстного документа: {context_ui_id}" if context_ui_id else ""

    system_prompt_content = f"""
            <ROLE>
            Ты - центральный Оркестратор СЭД. Твоя задача - составить четкий, структурированный план действий в виде списка вызовов инструментов (<steps>).
            </ROLE>

            <AVAILABLE_TOOLS_SUMMARY>
            Используй ТОЛЬКО следующие имена инструментов:
            1. doc_metadata_get_by_id: Получить метаданные документа.
            2. doc_attachment_get_content: Скачать вложение документа.
            3. doc_content_summarize: Сделать сводку по скачанному контенту.
            4. employee_tools_search: Найти сотрудника по частичному имени (ИСПОЛЬЗУЙ ИМЯ АРГУМЕНТА 'search_query').
            5. employee_tools_get_by_id: Получить полные данные сотрудника по ID.
            </AVAILABLE_TOOLS_SUMMARY>

            <GLOBAL_CONTEXT>
            Дополнительный контекст пользователя: {user_context_str}
            {context_ui_id_str}
            </GLOBAL_CONTEXT>
    
            <PLANNING_EXAMPLE>
            Для выполнения многошаговых задач используй JSONPath для передачи данных между шагами.
    
            ПРИМЕР ПОЛНОГО И КОРРЕКТНОГО ПЛАНА (Сводка документа по ID):
            Шаг 0: doc_metadata_get_by_id (Использует document_id из GLOBAL_CONTEXT).
            Шаг 1: doc_attachment_get_content (Использует document_id из GLOBAL_CONTEXT и attachment_id из Шага 0).
            Шаг 2: doc_content_summarize (Требует document_id, content_key, file_name, И КОНТЕКСТ МЕТАДАННЫХ). // <-- Улучшение текста
    
            Корректный план из 3-х шагов для сводки (используй ID из GLOBAL_CONTEXT):
            [
                {{
                  "tool_name": "doc_metadata_get_by_id",
                  "arguments": {{
                    "document_id": "{context_ui_id or 'ID_ИЗ_КОНТЕКСТА'}"
                  }}
                }},
                {{
                  "tool_name": "doc_attachment_get_content",
                  "arguments": {{
                    "document_id": "{context_ui_id or 'ID_ИЗ_КОНТЕКСТА'}",
                    "attachment_id": "$.STEPS[0].result.attachmentDocument[0].id"
                  }}
                }},
                {{
                  "tool_name": "doc_content_summarize",
                  "arguments": {{
                    "document_id": "{context_ui_id or 'ID_ИЗ_КОНТЕКСТА'}",
                    "content_key": "$.STEPS[1].result.content_key",
                    "file_name": "$.STEPS[0].result.attachmentDocument[0].name", 
                    "metadata_context": "$.STEPS[0].result" // <-- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ В ПРИМЕРЕ
                  }}
                }}
            ]
            </PLANNING_EXAMPLE>
    
            <FINAL_INSTRUCTION>
            Проанализируй запрос и доступные инструменты. Если информации достаточно или запрос не требует инструментов, верни пустой список шагов. 
            ОБЯЗАТЕЛЬНО используй JSONPath для передачи ID вложения в doc_attachment_get_content.
            ОБЯЗАТЕЛЬНО передай **все четыре аргумента** (document_id, content_key, file_name, metadata_context) в doc_content_summarize, используя правильные JSONPath. // <-- Улучшение текста
            </FINAL_INSTRUCTION>
            """

    messages_with_system = [SystemMessage(content=system_prompt_content)] + messages
    llm = get_chat_model(tools=ALL_TOOLS).with_structured_output(Plan)

    try:
        llm_output = await llm.ainvoke(messages_with_system)

        if isinstance(llm_output, Plan):
            response = llm_output
        elif hasattr(llm_output, 'content') and llm_output.content:
            response_data = json.loads(llm_output.content)
            response = Plan(**response_data)
        else:
            response = Plan(reasoning="Failed to parse LLM output.", steps=[])


    except Exception as e:
        logger.error(f"Ошибка парсинга плана LLM: {type(e).__name__}: {e}")
        return {"tools_to_call": []}

    logger.info(f"PLANNER: Запланировано {len(response.steps)} шагов. Причина: {response.reasoning}")

    return {
        "tools_to_call": [step.dict() for step in response.steps],
    }


# ----------------------------------------------------------------
# 2. УЗЕЛ ИСПОЛНИТЕЛЯ (TOOL EXECUTOR)
# ----------------------------------------------------------------
async def tool_executor_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Исполняет запланированные вызовы инструментов.
    Устойчив к ошибкам отсутствия инструмента и неверным аргументам.
    """
    tools_to_call = state.get("tools_to_call", [])
    executed_history = state.get("tool_results_history", [])

    user_token = state.get("user_token")
    logger.debug(
        f"EXECUTOR: Токен в состоянии: {'ПРИСУТСТВУЕТ' if user_token and len(user_token) > 5 else 'ОТСУТСТВУЕТ'}")
    if not user_token:
        error_msg = "User token отсутствует в состоянии. Инструменты не могут быть выполнены."
        logger.error(f"EXECUTOR: CRITICAL ERROR: {error_msg}")
        return {"tool_results_history": executed_history, "tools_to_call": [],
                "messages": state["messages"] + [AIMessage(content=error_msg)]}

    # Обрабатываем только первый инструмент в очереди
    if not tools_to_call:
        return {"tool_results_history": executed_history, "tools_to_call": []}

    tool_to_call = tools_to_call[0]
    tool_name = tool_to_call["tool_name"]
    args = tool_to_call["arguments"]

    tool = next((t for t in ALL_TOOLS if t.name == tool_name), None)

    tool_result = None
    error_message = None
    resolved_args = args  # Будет обновлено ниже

    if tool is None:
        error_message = f"Инструмент '{tool_name}' не найден в списке доступных инструментов."
        logger.error(f"EXECUTOR: Ошибка при вызове {tool_name}: {error_message}")
        tool_result = {"error": error_message}

    else:
        try:
            # 1. Разрешение JSONPath для всех аргументов
            resolved_args = {k: _execute_jsonpath(v, executed_history) for k, v in args.items()}

            # 2. Определение разрешенных аргументов на основе Pydantic-схемы
            tool_schema = tool.args_schema.schema()
            allowed_args = set(tool_schema.get('properties', {}).keys())

            # Разрешаем metadata_context для SummarizeContentArgs
            if tool_name == "doc_content_summarize":
                allowed_args.add("metadata_context")

            # 3. Формируем аргументы, передаваемые в tool_func, отфильтровывая лишние
            tool_kwargs = {
                k: v
                for k, v in resolved_args.items()
                if k in allowed_args
            }

            # 4. Добавляем токен
            tool_kwargs['token'] = user_token

            # 5. !!! КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: ВОЗВРАЩАЕМ 'config' !!!
            # Требуется для StructuredTool._arun в этой конкретной версии/конфигурации.
            tool_kwargs['config'] = {}

            logger.info(f"EXECUTOR: Исполнение: {tool_name} с аргументами: {tool_kwargs} (токен передан)")

            tool_func = None

            # 6. Используем _arun
            if hasattr(tool, '_arun') and callable(tool._arun):
                tool_func = tool._arun
            else:
                raise AttributeError(
                    f"Инструмент '{tool_name}' не имеет доступной асинхронной исполняемой функции (_arun)."
                )

            # 7. Асинхронный вызов
            tool_result = await tool_func(**tool_kwargs)

        except Exception as e:
            error_message = f"Ошибка выполнения: {type(e).__name__}: {e}"
            logger.error(f"EXECUTOR: Ошибка при вызове {tool_name}: {error_message}")
            tool_result = {"error": error_message}

    # Сохраняем результат
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


# ----------------------------------------------------------------
# 3. УЗЕЛ ПРОВЕРКИ РЕЗУЛЬТАТОВ (CHECKER NODE)
# ----------------------------------------------------------------
def tool_result_checker_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Определяет, что делать дальше: продолжить планирование, перейти к ответу или
    привлечь человека (Human in the Loop).

    *** Возвращает DICT с ключом 'next_step' ***
    """
    executed_history = state.get("tool_results_history", [])

    if not executed_history:
        # Если история пуста, это ошибка.
        return {"next_step": "responder"}

    last_result = executed_history[-1]["result"]
    tool_name = executed_history[-1]["tool_name"]
    logger.info(f"CHECKER: Проверка результата Tool: {tool_name}")

    # Проверка на наличие ошибки
    if isinstance(last_result, dict) and "error" in last_result:
        logger.error(f"CHECKER: Обнаружена ошибка исполнения: {last_result['error']}")
        return {"next_step": "responder"}

    # Проверка на сценарий "Human in the Loop"
    if tool_name == "employee_tools_search" and len(last_result) > 1:
        logger.warning("CHECKER: Обнаружено несколько совпадений. Требуется Human-in-the-Loop.")
        return {"next_step": "human_in_the_loop"}

    # Проверка, остались ли еще шаги в плане
    if state["tools_to_call"]:
        logger.info("CHECKER: В очереди есть инструменты. Возврат к планировщику.")
        # ВОЗВРАЩАЕМ 'planner' (который должен перейти к executor)
        return {"next_step": "planner"}

    # Если инструментов в очереди нет и ошибок не было, переходим к финальному ответу
    logger.info("CHECKER: Все запланированные шаги выполнены. Переход к ответу.")
    return {"next_step": "responder"}


# ----------------------------------------------------------------
# 4. УЗЕЛ ФИНАЛЬНОГО ОТВЕТА (RESPONDER NODE) - без изменений
# ... (код responder_node без изменений)
async def responder_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Генерирует финальный ответ пользователю на основе истории сообщений и результатов инструментов.
    """
    messages: List[Any] = state["messages"]
    history: List[Dict[str, Any]] = state.get("tool_results_history", [])

    # Собираем историю ToolCall и ToolResult в сообщения
    tool_messages: List[ToolMessage] = []

    for idx, item in enumerate(history):
        # ... (Логика формирования tool_messages остается без изменений)
        if "error" in item["result"]:
            result_content = f"ToolCall ({item['tool_name']}) ОШИБКА: {item['result']['error']}"
        else:
            try:
                result_str = json.dumps(item['result'], ensure_ascii=False)
                result_content = "ToolCall OK. Результат:\n" + result_str[:1000] + (
                    "..." if len(result_str) > 1000 else "")
            except:
                result_content = "ToolCall OK. Результат: (невозможно сериализовать)"

        fake_tool_call_id = f"call_{item['tool_name']}_{idx}"

        tool_messages.append(ToolMessage(
            content=result_content,
            tool_call_id=fake_tool_call_id
        ))

    # Система видит весь контекст и результаты инструментов.
    user_context = state.get("user_context")
    user_context_str = json.dumps(user_context, ensure_ascii=False) if user_context else "Не предоставлен."

    system_prompt_content = f"""
    <ROLE>
    Ты - AI-ассистент СЭД. 
    Твоя задача: проанализировать историю диалога, результаты вызовов инструментов (<TOOL_RESULTS>) и контекст пользователя (<USER_CONTEXT>).
    Сформулируй краткий, точный и полезный финальный ответ пользователю.
    </ROLE>

    <USER_CONTEXT>
    Дополнительный контекст пользователя (роль, права): {user_context_str}
    </USER_CONTEXT>

    <TOOL_RESULTS>
    Это результаты всех выполненных тобой шагов. Используй их для формулирования ответа:
    {json.dumps([h for h in history if "result" in h], ensure_ascii=False, indent=2)}
    </TOOL_RESULTS>

    <INSTRUCTIONS>
    1. Если в результатах есть ошибки, извинись и объясни причину ошибки (например, "Инструмент не найден" или "Не удалось найти сотрудника").
    2. Если результатов несколько (например, найдено 2 сотрудника), попроси пользователя уточнить выбор.
    3. Если все успешно, дай прямой ответ на вопрос, используя информацию из <TOOL_RESULTS>.
    4. Если <TOOL_RESULTS> пуст, а запрос пользователя является общим (приветствие, вопрос о погоде), ответь вежливо и спроси, чем ты можешь помочь в рамках СЭД.
    </INSTRUCTIONS>
    """

    final_messages = [SystemMessage(content=system_prompt_content)] + messages + tool_messages

    llm = get_chat_model()

    logger.info("RESPONDER: Генерация финального ответа.")
    response = await llm.ainvoke(final_messages)

    # Обновляем состояние, добавляя финальный ответ LLM
    new_messages: List[BaseMessage] = state["messages"] + [response]  # <-- Response - это AIMessage

    return {"messages": new_messages}


# ----------------------------------------------------------------
# 5. УЗЕЛ УДАЛЕНИЯ ИСТОРИИ (DELETE HISTORY NODE) - ИСПРАВЛЕНИЕ ДЛЯ SAVER
# ----------------------------------------------------------------
def delete_history_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел, который удаляет все сообщения, кроме последнего ответа AI,
    используя явные команды RemoveMessage (самый надежный способ).
    """
    messages: List[BaseMessage] = state["messages"]

    if messages:
        last_message = messages[-1]
        logger.info("DELETER: Сохраняется только последнее сообщение.")

        # 1. Сообщения для удаления: все, кроме последнего
        messages_to_remove = messages[:-1]

        new_messages = []

        # 2. Создаем команду RemoveMessage для каждого старого сообщения.
        # Это заставляет Checkpointer удалить их по ID.
        for msg in messages_to_remove:
            # BaseMessage гарантирует наличие ID
            new_messages.append(RemoveMessage(id=msg.id))

        # 3. Добавляем последнее сообщение, которое должно быть сохранено.
        new_messages.append(last_message)

        # Checkpointer выполнит удаление и добавит одно сообщение.
        return {"messages": new_messages}

    logger.warning("DELETER: Не удалось найти финальное сообщение для сохранения.")
    return {}
