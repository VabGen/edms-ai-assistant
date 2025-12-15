# edms_ai_assistant/sub_agents/employees_agent.py

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from e_a_a.core import register_agent
from e_a_a.core import OrchestratorState
from edms_ai_assistant.llm import get_chat_model
from e_a_a.tools.employee import (
    find_responsible_tool,
    get_employee_by_id_tool,
)

logger = logging.getLogger(__name__)


# --- Узел под-агента ---
async def employees_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Узел под-агента для работы с сотрудниками.
    Использует инструменты через привязку к LLM.
    """
    messages = state["messages"]
    user_token = state["user_token"]

    # --- ИНИЦИАЛИЗАЦИЯ ИНСТРУМЕНТОВ ---
    from langchain_core.tools import tool

    @tool
    async def find_responsible_tool_wrapped(query: str) -> str:
        """Обёрнутый инструмент для поиска сотрудника, передающий токен."""
        return await find_responsible_tool(query, user_token)

    @tool
    async def get_employee_by_id_tool_wrapped(employee_id: str) -> str:
        """Обёрнутый инструмент для получения сотрудника по ID, передающий токен."""
        return await get_employee_by_id_tool(employee_id, user_token)

    llm = get_chat_model()
    tools_to_bind = [
        find_responsible_tool_wrapped,
        get_employee_by_id_tool_wrapped,
        # ... другие инструменты ...
    ]
    llm_with_tools = llm.bind_tools(tools_to_bind)

    # --- ВЫЗОВ LLM С ИНСТРУМЕНТАМИ (упрощённый ReAct) ---
    current_messages = messages
    max_iterations = 5
    iteration_count = 0

    while iteration_count < max_iterations:
        iteration_count += 1
        logger.debug(f"Кадровый агент: Итерация {iteration_count}")
        response = await llm_with_tools.ainvoke(current_messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_name = response.tool_calls[0]["name"]
            tool_args = response.tool_calls[0]["args"]
            tool_to_execute = next(
                (t for t in tools_to_bind if t.name == tool_name), None
            )
            if tool_to_execute:
                try:
                    tool_result = await tool_to_execute.ainvoke(tool_args)
                    from langchain_core.messages import ToolMessage

                    tool_message = ToolMessage(
                        content=tool_result, tool_call_id=response.tool_calls[0]["id"]
                    )
                    current_messages += [response, tool_message]
                except Exception as e:
                    logger.error(f"Ошибка выполнения инструмента {tool_name}: {e}")
                    error_msg = f"Ошибка выполнения инструмента: {e}"
                    from langchain_core.messages import ToolMessage

                    tool_message = ToolMessage(
                        content=error_msg, tool_call_id=response.tool_calls[0]["id"]
                    )
                    current_messages += [response, tool_message]
            else:
                logger.error(f"Инструмент {tool_name} не найден!")
                break
        else:
            final_ai_message = response
            break
    else:
        logger.warning("Кадровый агент: Достигнут лимит итераций")
        final_ai_message = AIMessage(
            content="Процесс превысил допустимое количество шагов."
        )

    return {
        "result": "employees_agent_success",
        "final_response": final_ai_message.content,
        "messages": [final_ai_message],
    }


@register_agent("employees_agent")
def create_employees_agent_graph():
    """Создает и компилирует граф агента по сотрудникам."""
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("executor", employees_agent_node)
    workflow.set_entry_point("executor")
    workflow.add_edge("executor", END)

    logger.info("Граф 'employees_agent' скомпилирован (с внутренней логикой ReAct).")
    return workflow.compile()
