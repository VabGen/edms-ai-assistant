# core/orchestrator.py

import logging
import json
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.tools import get_available_tools  # <-- Фабрика инструментов

logger = logging.getLogger(__name__)


# --- 1. State ---

class OrchestratorState(TypedDict):
    """Единое состояние для LangGraph."""
    messages: List[BaseMessage]
    user_token: str
    tools_to_call: List[Dict[str, Any]]
    tool_results_history: List[Dict[str, Any]]


# --- 2. Schemas for Planning (Обязательный JSON-формат для Planner'а) ---

class ToolCallRequest(BaseModel):
    tool_name: str = Field(...,
                           description="Имя инструмента (например, 'doc_metadata.get_by_id', 'employee_tools.search').")
    arguments: Dict[str, Any] = Field(...,
                                      description="Аргументы для инструмента. Для передачи данных используй синтаксис JSONPath: '$STEPS[0].result.attachmentDocument[0].id'.")


class Plan(BaseModel):
    """Список шагов для выполнения."""
    steps: List[ToolCallRequest] = Field(..., description="Список инструментов, которые нужно вызвать.")
    reasoning: str = Field(..., description="Объяснение плана.")


# --- 3. Вспомогательный класс для обработки JSONPath ---

class ArgumentMapper:
    """Простой класс для маппинга аргументов, используя 'STEPS[index].result.path'."""

    @staticmethod
    def map_arguments(args: Dict[str, Any], tool_results_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Заменяет заполнители JSONPath на фактические значения."""
        mapped_args = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$STEPS["):
                try:
                    # 1. Извлекаем индекс и путь
                    start_index = value.find('[') + 1
                    end_index = value.find(']')
                    step_index_str = value[start_index:end_index]
                    step_index = int(step_index_str)

                    path_start = value.find('.result.') + len('.result.')
                    path = value[path_start:]

                    current_val = None
                    if step_index < len(tool_results_history):
                        result_str = tool_results_history[step_index].get('result')
                        if result_str:
                            result_data = json.loads(result_str)
                            current_val = result_data
                            path_parts = path.split('.')

                            for part in path_parts:
                                if current_val is None:
                                    break
                                # Обработка массива (attachmentDocument[0])
                                if '[' in part and ']' in part:
                                    arr_name, index_str = part.split('[')
                                    try:
                                        index = int(index_str[:-1])
                                        current_val = current_val.get(arr_name, [])[index]
                                    except (ValueError, IndexError):
                                        raise KeyError(f"Неверный формат или индекс массива: {part}")
                                else:
                                    # Обработка обычного ключа
                                    current_val = current_val.get(part)

                    mapped_args[key] = current_val
                    logger.debug(f"Маппинг: {value} -> {current_val}")

                except Exception as e:
                    logger.error(f"Сбой парсинга JSONPath в {value}: {type(e).__name__}: {e}")
                    mapped_args[key] = None
            else:
                mapped_args[key] = value
        return mapped_args


# --- 4. Nodes ---

async def planner_node(state: OrchestratorState):
    """LLM анализирует запрос и составляет план вызовов API."""
    messages = state["messages"]
    tool_results_history = state.get("tool_results_history", [])

    system_prompt = """
        Ты - центральный Оркестратор системы Chancellor NEXT. Твоя задача - составить план действий.

        **ОСОБОЕ ВНИМАНИЕ К КОНТЕКСТУ UI:** Если ты видишь в истории 'КОНТЕКСТ UI' с ID сущности (чистый UUID), ИСПОЛЬЗУЙ ЧИСТЫЙ UUID ИЗ ЭТОГО КОНТЕКСТА для заполнения аргумента 'document_id' или 'employee_id'.

        **ЦЕПОЧКА ШАГОВ (JSONPath):** Если результат предыдущего шага (ToolMessage) содержит данные, необходимые для текущего шага, ИСПОЛЬЗУЙ СЛЕДУЮЩИЙ СИНТАКСИС ДЛЯ МЭППИНГА АРГУМЕНТОВ:
        Например, чтобы получить ID вложения: 'attachment_id': '$STEPS[0].result.attachmentDocument[0].id'.

        **☝️ ПРАВИЛО ПОСЛЕДОВАТЕЛЬНОСТИ:** При запросе сводки вложения ('о чем документ?') ВСЕГДА планируй два шага:
        1. doc_metadata.get_by_id (чтобы получить ID вложения).
        2. doc_attachment.summarize (используя данные из Шага 1 через JSONPath $STEPS[0].result...).

        **⚠️ ПРАВИЛО СВОДКИ (Summarize):** Для вызова **doc_attachment.summarize** требуются **'document_id'** (из КОНТЕКСТА UI) и **'attachment_id'** с **'file_name'** (из результата Шага 1). Используй синтаксис JSONPath для получения 'attachment_id' и 'file_name'.

        Также, если это контракт, передай туда ключевые поля: **'contract_sum', 'reg_date' и 'duration_end'**, используя JSONPath из Шага 1.

        Доступные инструменты:
        1. **doc_metadata.get_by_id(...):** Получить все детальные метаданные (автор, статус, суммы, даты) о документе по его ID (UUID).
        2. **doc_attachment.summarize(...):** Скачать и прочитать содержимое вложения, а затем создать его сводку.
        3. **employee_tools.search(...):** Используется для поиска списка сотрудников по частичному совпадению (фамилия, имя, должность).
        4. **employee_tools.get_by_id(...):** Используется для получения полных метаданных сотрудника по его UUID.

        Проанализируй историю. Если нужно получить информацию, создай список вызовов. Если информации достаточно для ответа пользователю, верни пустой список шагов.
        """

    llm = get_chat_model().with_structured_output(Plan)

    # 1. Вызов LLM
    llm_output = await llm.ainvoke([HumanMessage(content=system_prompt)] + messages)

    # 2. Обработка результата
    try:
        # 🚨 ИСПРАВЛЕНИЕ: Проверяем, является ли результат уже объектом Plan
        if isinstance(llm_output, Plan):
            # Успех: LangChain вернул Pydantic объект напрямую
            response = llm_output
        elif hasattr(llm_output, 'content') and llm_output.content:
            # Резервный сценарий: если LLM вернул BaseMessage с JSON строкой внутри
            response_data = json.loads(llm_output.content)
            response = Plan(**response_data)
        else:
            # Неизвестный случай, возможно, LLM вернул сырую JSON-строку
            response_data = json.loads(llm_output)
            response = Plan(**response_data)

    except Exception as e:
        # Отлавливаем любые ошибки, включая JSONDecodeError и unexpected type errors
        logger.error(f"Ошибка парсинга плана LLM: {type(e).__name__}: {e}")
        return {"tools_to_call": [], "tool_results_history": tool_results_history}

    logger.info(f"PLANNER: Запланировано {len(response.steps)} шагов: {response.reasoning}")

    return {"tools_to_call": [step.dict() for step in response.steps], "tool_results_history": tool_results_history}


async def executor_node(state: OrchestratorState):
    """Выполняет запланированные инструменты."""
    tools_to_call = state.get("tools_to_call", [])
    tool_results_history = state.get("tool_results_history", [])

    llm_summarizer = get_chat_model()

    # 🚨 Инициализация и маппинг методов
    tools_by_class = get_available_tools(state["user_token"], llm_summarizer)
    tools_map = {}
    for class_name, tool_instance in tools_by_class.items():
        for attr_name in dir(tool_instance):
            attr = getattr(tool_instance, attr_name)
            if callable(attr) and not attr_name.startswith('_') and attr_name not in ['__init__']:
                tools_map[f"{class_name}.{attr_name}"] = attr

    new_messages = []

    for tool_index, tool_request in enumerate(tools_to_call):
        full_tool_name = tool_request["tool_name"]
        raw_args = tool_request["arguments"]

        args = ArgumentMapper.map_arguments(raw_args, tool_results_history)
        tool_function = tools_map.get(full_tool_name)

        if tool_function:
            try:
                # 🚨 Вызов метода инструмента
                result = await tool_function(**args)

                new_tool_message = ToolMessage(
                    content=result,
                    tool_call_id=f"call_{full_tool_name}_{tool_index}",
                    name=full_tool_name
                )
                new_messages.append(new_tool_message)
                tool_results_history.append({"name": full_tool_name, "result": result})

            except Exception as e:
                error_msg = f"Error during {full_tool_name} call: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                new_messages.append(ToolMessage(
                    content=json.dumps({"error": error_msg}),
                    tool_call_id=f"err_{full_tool_name}",
                    name=full_tool_name
                ))
        else:
            logger.warning(f"EXECUTOR: Tool '{full_tool_name}' not found in map.")

    all_messages = state["messages"] + new_messages
    return {"messages": all_messages, "tools_to_call": [], "tool_results_history": tool_results_history}


async def synthesizer_node(state: OrchestratorState):
    """Генерирует финальный ответ."""
    llm = get_chat_model()

    system_msg_content = (
        "Ты - ассистент Chancellor NEXT. Используй информацию из ToolMessages (результаты API), "
        "чтобы ответить на вопрос пользователя. "
        "**ОТВЕЧАЙ СТРОГО И ТОЛЬКО ПО СУТИ ЗАПРОСА**. Не добавляй лишнюю информацию, если она не была явно запрошена. "
        "Не показывай JSON. Отвечай вежливо и по сути. "
        "**Если в сводке документа есть незаполненные поля или плейсхолдеры, явно укажи, что это, вероятно, шаблон, и приведи всю доступную информацию.**"
        "Если данных нет или они нерелевантны, скажи об этом."
    )

    messages_for_llm = [HumanMessage(content=system_msg_content)] + state["messages"]

    response_message = await llm.ainvoke(messages_for_llm)

    return {"messages": [response_message]}


# --- 5. Graph Construction ---

def router_logic(state: OrchestratorState):
    """Определяет следующий шаг."""
    if state.get("tools_to_call"):
        return "executor_node"
    return "synthesizer_node"


def create_edms_graph():
    """Создает и компилирует LangGraph."""
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("executor_node", executor_node)
    workflow.add_node("synthesizer_node", synthesizer_node)

    workflow.set_entry_point("planner_node")

    # После планировщика: есть шаги -> executor, нет шагов -> synthesis
    workflow.add_conditional_edges(
        "planner_node",
        router_logic,
        {"executor_node": "executor_node", "synthesizer_node": "synthesizer_node"}
    )

    # После экзекьютора -> Синтезатор (мы не планируем многошаговых вызовов инструментов)
    workflow.add_edge("executor_node", "synthesizer_node")
    workflow.add_edge("synthesizer_node", END)

    return workflow.compile()
