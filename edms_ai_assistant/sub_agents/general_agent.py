# edms_ai_assistant/sub_agents/general_agent.py
import logging
import json
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.core.sub_agents import register_agent
from edms_ai_assistant.core.orchestrator import OrchestratorState

logger = logging.getLogger(__name__)


async def general_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел под-агента для общих вопросов.
    """
    messages = state['messages']

    llm = get_chat_model()

    context = state.get('context', {})
    context_prompt = ""
    if context:
        context_prompt = f"Учитывай, что пользователь находится в контексте: {json.dumps(context, ensure_ascii=False)}"

    system_prompt = (
        "Ты - универсальный AI-ассистент для СЭД (Chancellor NEXT). Отвечай на общие вопросы, "
        "предоставляй справку и веди диалог. Не используй инструменты, так как ты - 'general_agent'. "
        f"{context_prompt}"
    )

    augmented_messages = [HumanMessage(content=system_prompt)] + messages

    try:
        response = await llm.ainvoke(augmented_messages)
        ai_message_content = response.content if hasattr(response, 'content') else str(response)

        new_messages = messages + [response]

        logger.info(f"general_agent_node: {ai_message_content}")

        return {
            "subagent_result": ai_message_content,
            "final_response": ai_message_content,
            "messages": new_messages
        }
    except Exception as e:
        logger.error(f"Ошибка в general_agent_node: {e}")
        error_msg = "Извините, не удалось обработать запрос общего характера."

        return {
            "subagent_result": f"general_agent_error: {e}",
            "final_response": error_msg,
            "messages": messages + [AIMessage(content=error_msg)]
        }


@register_agent("general_agent")
def create_general_agent_graph():
    """Создает и компилирует граф агента для общих вопросов."""
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("executor", general_agent_node)
    workflow.set_entry_point("executor")
    workflow.add_edge("executor", END)

    logger.info("Граф 'general_agent' скомпилирован.")
    return workflow.compile()
