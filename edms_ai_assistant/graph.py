from typing import Literal
from langgraph.graph import StateGraph, END
from .models import OrchestratorState
from .nodes import (
    tool_result_checker_node,
    planner_node_factory,
    tool_executor_node_factory,
    responder_node_factory
)
from .tools import get_all_tools
from .llm import get_chat_model


def route_tool_result_checker(state: OrchestratorState) -> Literal["executor", "responder"]:
    """
    Маршрутизатор после выполнения инструмента (узла "checker").
    Определяет, нужно ли продолжать выполнение (executor) или переходить к ответу (responder).
    """
    next_route = state.get("next_route")

    if next_route == "executor":
        return "executor"

    if next_route == "responder":
        return "responder"

    tools_to_call = state.get("tools_to_call", [])
    if tools_to_call and len(tools_to_call) > 0:
        return "executor"

    return "responder"


def route_planner(state: OrchestratorState) -> str:
    """Маршрутизатор: Если есть шаги, идем выполнять, иначе - отвечаем."""
    if state.get("tools_to_call"):
        return "executor"
    return "responder"


def build_orchestrator_graph() -> StateGraph:
    """
    Создает и возвращает LangGraph для Оркестратора СЭД, используя фабрики узлов.
    """
    ALL_TOOLS = get_all_tools()
    LLM_FACTORY = get_chat_model

    planner_node = planner_node_factory(LLM_FACTORY, ALL_TOOLS)
    executor_node = tool_executor_node_factory(ALL_TOOLS)
    responder_node = responder_node_factory(LLM_FACTORY)

    workflow = StateGraph(OrchestratorState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("checker", tool_result_checker_node)
    workflow.add_node("responder", responder_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        route_planner,
        {
            "executor": "executor",
            "responder": "responder",
        }
    )

    workflow.add_edge("executor", "checker")

    workflow.add_conditional_edges(
        "checker",
        route_tool_result_checker,
        {
            "executor": "executor",
            "responder": "responder",
        }
    )

    workflow.add_edge("responder", END)

    return workflow
