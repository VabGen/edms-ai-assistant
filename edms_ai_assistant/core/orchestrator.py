import logging
from typing import TypedDict, List, Dict, Any, Optional, Literal
from pydantic import Field, create_model
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.core.sub_agents import (
    get_available_agent_names,
    get_sub_agent_executor,
    run_discovery_if_needed,
)

logger = logging.getLogger(__name__)

try:
    from edms_ai_assistant.constants import SUMMARY_TYPES
except ImportError:
    SUMMARY_TYPES = {}


def _extract_summary_intent(query: str) -> bool:
    """Определяет, запрашивает ли пользователь суммирование или содержание документа/вложения."""

    query = query.lower()

    # 📌 ДОБАВИТЬ: Условия для вложений
    summary_keywords = [
        "о чем",
        "содержание",
        "резюме",
        "кратко",
        "суммируй",
        "вложение",
        "файл"
        "что внутри",
    ]

    # 📌 Проверяем наличие ключевых слов
    if any(keyword in query for keyword in summary_keywords):
        return True

    return False


def _extract_summary_type(text: str) -> Optional[str]:
    """Извлекает тип резюме по номеру или названию."""
    text_lower = text.lower().replace("-", "").replace(" ", "")
    for key, details in SUMMARY_TYPES.items():
        if text_lower == key or text_lower == details["name"].lower().replace(
                "-", ""
        ).replace(" ", ""):
            return key
    return None


def _generate_hitl_prompt() -> str:
    """Генерирует запрос на уточнение типа резюме."""
    prompt = " Чтобы я мог правильно суммаризировать документ, уточните, пожалуйста, тип резюме, который вам нужен:\n\n"

    for num, details in SUMMARY_TYPES.items():
        prompt += f"{num}. **{details['name']}** — {details['description']}\n"

    prompt += "\nПросто введите номер (1-7) или название типа."
    return prompt


class OrchestratorState(TypedDict):
    """
    Определение состояния графа LangGraph (Orchestrator).
    """
    messages: List[BaseMessage]
    user_token: str
    file_path: Optional[str]
    context: Optional[Dict[str, Any]]
    subagent_result: Optional[str]
    called_subagent: Optional[str]
    final_response: Optional[str]
    agent_history: List[str]
    summary_type: Optional[str]
    is_hitl_query: bool


run_discovery_if_needed()

AVAILABLE_AGENTS = get_available_agent_names()
logger.info(f"Доступные агенты для маршрутизации: {AVAILABLE_AGENTS}")

if not AVAILABLE_AGENTS:
    logger.warning(
        "НЕ НАЙДЕНО ЗАРЕГИСТРИРОВАННЫХ АГЕНТОВ! Оркестратор будет использовать резервный вариант."
    )

AgentLiteral = (
    Literal[tuple(AVAILABLE_AGENTS)] if AVAILABLE_AGENTS else Literal["general_agent"]
)

DynamicRouteDecision = create_model(
    "DynamicRouteDecision",
    next_agent=(
        AgentLiteral,
        Field(..., description="Имя агента, которому нужно передать задачу. Должно быть одним из: " + ', '.join(
            AVAILABLE_AGENTS) + "."),
    ),
    reasoning=(str, Field(..., description="Почему выбран именно этот агент.")),
)


async def orchestrate_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел оркестратора.
    Включает логику определения намерения, HITL-механизм и LLM-маршрутизацию.
    """
    messages = state["messages"]
    last_message_content = messages[-1].content if messages else ""

    if state.get("is_hitl_query", False):
        summary_type = _extract_summary_type(last_message_content)
        if summary_type:
            logger.info(f"Получен HITL-ответ. Тип резюме: {summary_type}")
            return {
                "called_subagent": "documents_agent",
                "summary_type": summary_type,
                "is_hitl_query": False,
                "agent_history": state.get("agent_history", []) + ["hitl_response"],
            }
        else:
            hitl_prompt = f"Не удалось распознать тип резюме. Пожалуйста, введите номер или название типа из списка.{_generate_hitl_prompt()}"
            return {
                "final_response": hitl_prompt,
                "messages": [AIMessage(content=hitl_prompt)],
                "is_hitl_query": True,
                "called_subagent": "end_node",
            }

    if _extract_summary_intent(last_message_content):
        summary_type = _extract_summary_type(last_message_content)

        if not summary_type and "documents_agent" in AVAILABLE_AGENTS:
            hitl_prompt = _generate_hitl_prompt()
            logger.info("Активирован HITL-запрос для уточнения типа резюме.")
            return {
                "final_response": hitl_prompt,
                "messages": [AIMessage(content=hitl_prompt)],
                "is_hitl_query": True,
                "called_subagent": "end_node",
                "agent_history": state.get("agent_history", []) + ["hitl_query"]
            }

        if summary_type:
            state["summary_type"] = summary_type

    context = state.get("context", {})

    last_message = (
        messages[-1] if messages else HumanMessage(content="Пустое сообщение")
    )
    enhanced_message_content = f"Пользователь сказал: '{last_message.content}'"

    document_id_from_context = context.get("document_id")
    current_page = context.get("current_page", "unknown")
    file_path = state.get("file_path")

    if document_id_from_context:
        enhanced_message_content += f"\nКонтекст: Пользователь находится на странице документа с ID: {document_id_from_context}."
    elif current_page != "unknown":
        enhanced_message_content += (
            f"\nКонтекст: Пользователь находится на странице: {current_page}."
        )

    if file_path:
        enhanced_message_content += f"\nВложение: Пользователь загрузил файл. Приоритет отдается агентам, способным работать с файлами."

    if state.get('summary_type'):
        enhanced_message_content += f"\nИНСТРУКЦИЯ: Пользователь запросил резюме типа: {SUMMARY_TYPES.get(state['summary_type'], {}).get('name', 'Multi-sentence')}."

    llm = get_chat_model()
    orchestrator_llm = llm.with_structured_output(DynamicRouteDecision)

    system_prompt = f"""Ты - маршрутизатор AI-ассистента для СЭД (edms).
    Твоя задача - строго определить, какой из специализированных под-агентов должен обработать запрос пользователя.
    Доступные под-агенты: {', '.join(AVAILABLE_AGENTS)}.
    Проанализируй следующий запрос пользователя и контекст. Ответь строго в формате Pydantic-модели DynamicRouteDecision."""

    llm_input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": enhanced_message_content},
    ]

    try:
        decision: DynamicRouteDecision = await orchestrator_llm.ainvoke(
            llm_input_messages
        )
        logger.info(
            f"Оркестратор выбрал под-агента: {decision.next_agent}. Причина: {decision.reasoning}"
        )

        return {
            "called_subagent": decision.next_agent,
            "agent_history": state.get("agent_history", []) + [decision.next_agent],
        }

    except Exception as e:
        logger.error(f"Ошибка в orchestrate_node (LLM/Structured Output): {e}")
        return {
            "called_subagent": "general_agent",  # Безопасный fallback
            "subagent_result": f"Ошибка оркестратора: {e}",
            "final_response": "Извините, произошла внутренняя ошибка при определении действия.",
            "agent_history": state.get("agent_history", [])
                             + ["orchestrator_error"],
        }


def route_logic(state: OrchestratorState) -> str:
    """
    Определяет, в какой узел графа перейти после orchestrate_node.
    """
    agent_to_call = state.get("called_subagent", "general_agent")

    if agent_to_call == "end_node":
        return "end_node"

    if agent_to_call in AVAILABLE_AGENTS:
        return agent_to_call

    logger.warning(f"Неизвестный агент '{agent_to_call}', используем general_agent")
    return "general_agent"


# ... (Весь код до функции make_agent_node остается без изменений) ...

def make_agent_node(agent_name: str):
    """
    Создает функцию-узел для конкретного под-агента.
    """

    async def agent_node(state: OrchestratorState) -> Dict[str, Any]:
        logger.info(f"Запуск под-агента: {agent_name}")

        executor = get_sub_agent_executor(agent_name)
        if not executor:
            error_msg = f"Ошибка конфигурации: Агент {agent_name} не найден или не скомпилирован."
            logger.error(error_msg)
            return {
                "final_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "subagent_result": error_msg,
            }

        # 💡 Передаем текущее состояние целиком
        sub_agent_inputs = state

        try:
            agent_output = await executor.ainvoke(sub_agent_inputs)

            final_response = agent_output.get(
                "final_response", "Под-агент завершил работу."
            )

            # 📌 ИЗМЕНЕНИЕ: Сброс summary_type после выполнения документального агента
            # Это предотвратит его использование в следующих, не связанных запросах
            if agent_name == 'documents_agent' and state.get('summary_type'):
                logger.debug(f"Сброс summary_type: {state['summary_type']} после documents_agent.")
                state['summary_type'] = None # Сброс в текущем state

            return {
                # 📌 ИЗМЕНЕНИЕ: Возвращаем обновленный state
                "messages": agent_output.get(
                    "messages", [AIMessage(content=final_response)]
                ),
                "final_response": final_response,
                "subagent_result": final_response,
                "called_subagent": agent_name,
                # 💡 Явно возвращаем None для summary_type, если он был сброшен
                "summary_type": state.get('summary_type'),
            }

        except Exception as e:
            logger.error(f"Ошибка при выполнении {agent_name}: {e}", exc_info=True)
            error_msg = f"Извините, возникла ошибка при работе с {agent_name.replace('_', ' ')}."
            return {
                "final_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "subagent_result": f"Ошибка под-агента {agent_name}: {e}",
            }

    return agent_node


def create_orchestrator_graph():
    """
    Создаёт и компилирует граф оркестратора с динамическими узлами и условными переходами.
    """
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("orchestrate", orchestrate_node)

    conditional_map = {}
    for agent_name in AVAILABLE_AGENTS:
        workflow.add_node(agent_name, make_agent_node(agent_name))
        conditional_map[agent_name] = agent_name
        workflow.add_edge(agent_name, END)

    if not AVAILABLE_AGENTS:
        workflow.add_node("general_agent", make_agent_node("general_agent"))
        conditional_map["general_agent"] = "general_agent"
        workflow.add_edge("general_agent", END)

    workflow.add_node("end_node", lambda state: state)
    conditional_map["end_node"] = "end_node"
    workflow.add_edge("end_node", END)

    workflow.set_entry_point("orchestrate")

    workflow.add_conditional_edges(
        "orchestrate",
        route_logic,
        conditional_map
    )

    app = workflow.compile()
    return app
