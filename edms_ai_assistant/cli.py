# edms_ai_assistant/cli.py


def main():
    """Заглушка для CLI."""
    print("CLI интерфейс EDMS Assistant.")


if __name__ == "__main__":
    main()



# # edms_ai_assistant/sub_agents/documents_agent.py
#
# import logging
# import json
# from langchain_core.tools import BaseTool
# from pydantic import Field
# from typing import Dict, Any, Optional
# from langgraph.graph import StateGraph, END
# from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
# from edms_ai_assistant.core.sub_agents import register_agent
# from edms_ai_assistant.core.orchestrator import OrchestratorState
# from edms_ai_assistant.llm import get_chat_model
# from edms_ai_assistant.tools.document import get_document_tool, search_documents_tool
# from edms_ai_assistant.tools.employee import find_responsible_tool
# # from langchain_core.tools import tool
#
# logger = logging.getLogger(__name__)
#
#
# @tool
# async def get_document_tool_wrapped(document_id: str, user_token: str) -> str:
#     """Обёрнутый инструмент для получения документа, передающий токен."""
#     logger.info(f"Вызов get_document_tool с document_id: {document_id}, user_token: {user_token}")
#     return await get_document_tool(document_id, user_token)
#
#
# @tool
# async def search_documents_tool_wrapped(filters: Dict[str, Any], user_token: str) -> str:
#     """Обёрнутый инструмент для поиска документов, передающий токен."""
#     return await search_documents_tool(filters, user_token)
#
#
# @tool
# async def find_responsible_tool_wrapped(query: str, user_token: str) -> str:
#     """Обёрнутый инструмент для поиска сотрудника, передающий токен."""
#     # return await find_responsible_tool(query, user_token)
#     return json.dumps({"result": "Employee tool is not available in this context."}, ensure_ascii=False)
#
#
# # ---------------------------------------------
#
# async def documents_agent_node(state: OrchestratorState) -> Dict[str, Any]:
#     """
#     Узел под-агента для работы с документами.
#     Реализует внутренний цикл ReAct с инструментами.
#     """
#     messages = state['messages']
#     user_token = state['user_token']
#     file_path = state.get('file_path')
#     context = state.get('context', {})
#
#     logger.debug(f"documents_agent_node: {messages}, {context}, {file_path}, {user_token}")
#
#     # @tool
#     # async def get_document_tool_wrapped(document_id: str) -> str:
#     #     """Обёрнутый инструмент для получения документа, передающий токен."""
#     #     return await get_document_tool(document_id, user_token)
#     #
#     # @tool
#     # async def search_documents_tool_wrapped(filters: Dict[str, Any]) -> str:
#     #     """Обёрнутый инструмент для поиска документов, передающий токен."""
#     #     return await search_documents_tool(filters, user_token)
#     #
#     # @tool
#     # async def find_responsible_tool_wrapped(query: str) -> str:
#     #     """Обёрнутый инструмент для поиска сотрудника, передающий токен."""
#     #     # return await find_responsible_tool(query, user_token)
#     #     return json.dumps({"result": "Employee tool is not available in this context."}, ensure_ascii=False)
#
#     llm = get_chat_model()
#     tools_to_bind = [
#         get_document_tool_wrapped,
#         search_documents_tool_wrapped,
#         find_responsible_tool_wrapped,
#     ]
#
#     llm_with_tools = llm.bind_tools(tools_to_bind, tool_choice="required")
#
#     current_messages = list(messages)
#     max_iterations = 5
#     iteration_count = 0
#
#     # context_prompt = ""
#     # if file_path:
#     #     context_prompt += f"Внимание: Пользователь загрузил файл. Путь: {file_path}. Учитывай его при ответе."
#     # if context:
#     #     context_prompt += f" Текущий контекст пользователя: {json.dumps(context, ensure_ascii=False)}"
#     #
#     # if context_prompt:
#     #     current_messages.append(HumanMessage(content=context_prompt))
#
#     # --- ReAct Loop ---
#     final_ai_message: Optional[AIMessage] = None
#     while iteration_count < max_iterations:
#         iteration_count += 1
#         logger.debug(f"Документальный агент: Итерация {iteration_count}")
#         response = await llm_with_tools.ainvoke(current_messages)
#
#         if hasattr(response, 'tool_calls') and response.tool_calls:
#             tool_call = response.tool_calls[0]
#             tool_name = tool_call['name']
#             tool_args = tool_call['args']
#             tool_id = tool_call['id']
#             tool_to_execute = next((t for t in tools_to_bind if t.name == tool_name), None)
#
#             if tool_to_execute:
#                 try:
#                     tool_args_with_token = {**tool_args, "user_token": user_token}
#                     tool_result = await tool_to_execute.ainvoke(tool_args_with_token)
#                     tool_message = ToolMessage(content=tool_result, tool_call_id=tool_id)
#                     current_messages += [response, tool_message]
#                 except Exception as e:
#                     logger.error(f"Ошибка выполнения инструмента {tool_name}: {e}")
#                     error_msg = f"Ошибка выполнения инструмента: {tool_name} не удалось выполнить. Попробуй другой подход или ответь без инструмента."
#                     tool_message = ToolMessage(content=error_msg, tool_call_id=tool_id)
#                     current_messages += [response, tool_message]
#             else:
#                 break
#         else:
#             final_ai_message = response
#             break
#     else:
#         final_ai_message = AIMessage(
#             content="Процесс превысил допустимое количество шагов. Попробуйте перефразировать запрос.")
#
#     if final_ai_message is None:
#         final_ai_message = AIMessage(content="Извините, не удалось получить ответ от LLM.")
#
#     return {
#         "final_response": final_ai_message.content,
#         "messages": [final_ai_message],
#         "subagent_result": final_ai_message.content,
#         "called_subagent": "documents_agent",
#     }
#
#
# @register_agent("documents_agent")
# def create_documents_agent_graph():
#     """Создает и компилирует граф агента по документам."""
#     workflow = StateGraph(OrchestratorState)
#     workflow.add_node("executor", documents_agent_node)
#     workflow.set_entry_point("executor")
#     workflow.add_edge("executor", END)
#
#     return workflow.compile()
