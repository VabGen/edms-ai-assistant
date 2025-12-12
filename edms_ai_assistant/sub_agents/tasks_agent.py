# edms_ai_assistant/sub_agents/tasks_agent.py

import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from edms_ai_assistant.core.sub_agents import register_agent
from edms_ai_assistant.core.orchestrator import OrchestratorState
from edms_ai_assistant.llm import get_chat_model

# from edms_ai_assistant.tools.task import create_task_tool

logger = logging.getLogger(__name__)


# --- Узел под-агента ---
async def tasks_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел под-агента для работы с задачами.
    Использует инструменты через привязку к LLM.
    """
    messages = state["messages"]
    user_token = state["user_token"]
    context = state.get("context", {})

    # --- ИНИЦИАЛИЗАЦИЯ ИНСТРУМЕНТОВ ---
    # from langchain_core.tools import tool
    # @tool
    # async def create_task_tool_wrapped(...) -> str:
    #     # Обёртка, передающая user_token
    #     ...
    # tools_to_bind = [create_task_tool_wrapped, ...]

    # --- ПРИВЯЗКА ИНСТРУМЕНТОВ К LLM ---
    # llm = get_chat_model()
    # llm_with_tools = llm.bind_tools(tools_to_bind)

    # --- ВРЕМЕННАЯ РЕАЛИЗАЦИЯ БЕЗ ИНСТРУМЕНТА ---
    # Если инструмент для задачи не готов, просто возвращаем сообщение.
    last_message_content = messages[-1].content if messages else "Пустое сообщение"
    document_id_from_context = context.get("document_id")
    if document_id_from_context:
        result_text = f"Tasks Agent: Обработка запроса '{last_message_content}' для задачи, связанной с документом ID {document_id_from_context}."
    else:
        result_text = f"Tasks Agent: Обработка запроса '{last_message_content}' (ID документа не предоставлен)."
    logger.info(result_text)
    ai_message_content = result_text

    # --- УДАЛИТЬ или ЗАМЕНИТЬ на ReAct логику, когда инструмент будет готов ---
    # current_messages = messages
    # max_iterations = 5
    # iteration_count = 0
    # while iteration_count < max_iterations:
    #     # ... (логика ReAct, как в других агентах) ...
    #     break # Пока выходим сразу
    # final_ai_message = AIMessage(content=ai_message_content)

    final_ai_message = AIMessage(content=ai_message_content)

    return {
        "result": "tasks_agent_success",
        "final_response": ai_message_content,
        "messages": [final_ai_message],
    }


@register_agent("tasks_agent")
def create_tasks_agent_graph():
    """Создает и компилирует граф агента по задачам."""
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("executor", tasks_agent_node)
    workflow.set_entry_point("executor")
    workflow.add_edge("executor", END)

    logger.info("Граф 'tasks_agent' скомпилирован.")
    return workflow.compile()
