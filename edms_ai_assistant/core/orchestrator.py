# edms_ai_assistant\core\orchestrator.py
import logging
from typing import TypedDict, List, Dict, Any, Optional, Literal
from pydantic import Field, create_model
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.core.sub_agents import get_available_agent_names, get_sub_agent_executor, run_discovery_if_needed

logger = logging.getLogger(__name__)


class OrchestratorState(TypedDict):
    """
    Определение состояния графа LangGraph (Orchestrator).
    Все поля 'initial_state' в app.py должны соответствовать этим типам.
    """
    messages: List[BaseMessage]
    user_token: str
    file_path: Optional[str]
    context: Optional[Dict[str, Any]]
    subagent_result: Optional[str]
    called_subagent: Optional[str]
    final_response: Optional[str]
    agent_history: List[str]


run_discovery_if_needed()

AVAILABLE_AGENTS = get_available_agent_names()
logger.info(f"Доступные агенты для маршрутизации: {AVAILABLE_AGENTS}")

if not AVAILABLE_AGENTS:
    logger.warning("НЕ НАЙДЕНО ЗАРЕГИСТРИРОВАННЫХ АГЕНТОВ! Оркестратор будет использовать резервный вариант.")

AgentLiteral = Literal[tuple(AVAILABLE_AGENTS)] if AVAILABLE_AGENTS else Literal["general_agent"]

DynamicRouteDecision = create_model(
    'DynamicRouteDecision',
    next_agent=(AgentLiteral, Field(..., description="Имя агента, которому нужно передать задачу.")),
    reasoning=(str, Field(..., description="Почему выбран именно этот агент.")),
)


async def orchestrate_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел оркестратора.
    Использует LLM с Pydantic-схемой для надежного выбора под-агента.
    """
    messages = state['messages']
    context = state.get('context', {})

    last_message = messages[-1] if messages else HumanMessage(content="Пустое сообщение")
    enhanced_message_content = f"Пользователь сказал: '{last_message.content}'"

    document_id_from_context = context.get('document_id')
    current_page = context.get('current_page', 'unknown')
    file_path = state.get('file_path')

    if document_id_from_context:
        enhanced_message_content += f"\nКонтекст: Пользователь находится на странице документа с ID: {document_id_from_context}."
    elif current_page != 'unknown':
        enhanced_message_content += f"\nКонтекст: Пользователь находится на странице: {current_page}."

    if file_path:
        enhanced_message_content += f"\nВложение: Пользователь загрузил файл. Приоритет отдается агентам, способным работать с файлами."

    llm = get_chat_model()
    orchestrator_llm = llm.with_structured_output(DynamicRouteDecision)

    system_prompt = f"""Ты - маршрутизатор AI-ассистента для СЭД (Chancellor NEXT).
    Твоя задача - строго определить, какой из специализированных под-агентов должен обработать запрос пользователя.
    Доступные под-агенты: {', '.join(AVAILABLE_AGENTS)}.
    Проанализируй следующий запрос пользователя и контекст. Ответь строго в формате Pydantic-модели RouteDecision."""

    llm_input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": enhanced_message_content}
    ]

    try:
        decision: DynamicRouteDecision = await orchestrator_llm.ainvoke(llm_input_messages)
        logger.info(f"Оркестратор выбрал под-агента: {decision.next_agent}. Причина: {decision.reasoning}")
        return {
            "called_subagent": decision.next_agent,
            "agent_history": state.get("agent_history", []) + [decision.next_agent]
        }

    except Exception as e:
        logger.error(f"Ошибка в orchestrate_node (LLM/Structured Output): {e}")
        return {
            "called_subagent": "general_agent",
            "subagent_result": f"Ошибка оркестратора: {e}",
            "final_response": "Извините, произошла внутренняя ошибка при определении действия.",
            "agent_history": state.get("agent_history", []) + ["orchestrator_error"]
        }


def route_logic(state: OrchestratorState) -> str:
    """
    Определяет, в какой узел графа перейти после orchestrate_node.
    """
    agent_to_call = state.get("called_subagent", "general_agent")
    if agent_to_call in AVAILABLE_AGENTS:
        return agent_to_call
    else:
        logger.warning(f"Неизвестный агент '{agent_to_call}', используем general_agent")
        return "general_agent"


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
                "subagent_result": error_msg
            }

        sub_agent_inputs = state

        try:
            agent_output = await executor.ainvoke(sub_agent_inputs)

            final_response = agent_output.get("final_response", "Под-агент завершил работу.")

            return {
                "messages": agent_output.get("messages", [AIMessage(content=final_response)]),
                "final_response": final_response,
                "subagent_result": final_response,
                "called_subagent": agent_name,
            }

        except Exception as e:
            logger.error(f"Ошибка при выполнении {agent_name}: {e}", exc_info=True)
            error_msg = f"Извините, возникла ошибка при работе с {agent_name.replace('_', ' ')}."
            return {
                "final_response": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "subagent_result": f"Ошибка под-агента {agent_name}: {e}"
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

    workflow.set_entry_point("orchestrate")

    workflow.add_conditional_edges(
        "orchestrate",
        route_logic,
        conditional_map
    )

    app = workflow.compile()
    return app
