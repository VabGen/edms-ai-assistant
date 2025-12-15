# Файл: edms_ai-assistant/graph.py

from langgraph.graph import StateGraph, END
from .models import OrchestratorState
from .nodes import (
    planner_node,
    tool_executor_node,
    tool_result_checker_node,
    responder_node,
    delete_history_node
)


# ----------------------------------------------------------------
# ФУНКЦИЯ ПРОВЕРКИ, ВОЗВРАЩАЮЩАЯ НЕЙТРАЛЬНЫЕ КЛЮЧИ
# ----------------------------------------------------------------
def check_result_router(state: OrchestratorState) -> str:
    """Маршрутизатор для Checker, возвращает нейтральные ключи."""
    # Мы вызываем node-функцию, но интерпретируем ее результат как нейтральный ключ
    result = tool_result_checker_node(state)

    if result == "planner":
        return "continue_plan"  # Новое имя ключа (для Executor)
    elif result == "human_in_the_loop":
        return "clarification_needed"  # Новое имя ключа (для Responder)
    else:  # "responder"
        return "final_response"  # Новое имя ключа (для Responder)


def build_orchestrator_graph() -> StateGraph:
    """
    Создает и возвращает LangGraph для Оркестратора СЭД.
    """
    workflow = StateGraph(OrchestratorState)

    # 1. ДОБАВЛЕНИЕ УЗЛОВ (Nodes)
    # Здесь используются импортированные узлы:
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", tool_executor_node)
    workflow.add_node("checker", tool_result_checker_node)
    workflow.add_node("responder", responder_node)
    workflow.add_node("deleter", delete_history_node)

    # 2. ОПРЕДЕЛЕНИЕ ТОЧКИ ВХОДА (Entry Point)
    workflow.set_entry_point("planner")

    # 3. ПЕРЕХОДЫ (Edges)

    # --- A. Переходы из Планировщика (planner) ---
    def route_planner(state: OrchestratorState) -> str:
        """Маршрутизатор: Если есть шаги, идем выполнять, иначе - отвечаем."""
        if state.get("tools_to_call"):
            return "executor"
        return "responder"

    # Этот переход корректен, так как planner_node возвращает Dict, а route_planner - String
    workflow.add_conditional_edges(
        "planner",
        route_planner,
        {
            "executor": "executor",
            "responder": "responder",
        }
    )

    # --- B. Переход из Исполнителя (executor) ---
    workflow.add_edge("executor", "checker")

    # --- C. УСЛОВНЫЕ ПЕРЕХОДЫ из Проверщика (checker) ---
    # checker теперь возвращает Dict, содержащий ключ 'next_step'.
    # Мы используем LangGraph.app.get_next_step для маршрутизации.
    workflow.add_conditional_edges(
        "checker",
        lambda state: state["next_step"],  # Функция, которая извлекает значение перехода из состояния
        {
            "planner": "executor",  # 'planner' из checker'а ведет к следующему инструменту (Executor)
            "responder": "responder",
            "human_in_the_loop": "responder",
        }
    )

    # --- D/E. Переходы Responder -> Deleter -> END ---
    workflow.add_edge("responder", "deleter")
    workflow.add_edge("deleter", END)

    return workflow
