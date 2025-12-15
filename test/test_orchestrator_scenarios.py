# test/test_orchestrator_scenarios.py

import asyncio
import logging
import sys
import os
from typing import List
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from edms_ai_assistant.models import OrchestratorState
from edms_ai_assistant.graph import build_orchestrator_graph

logger = logging.getLogger(__name__)


def run_tests():
    """
    Запускает тестовые сценарии для проверки функциональности LangGraph.
    """
    memory = InMemorySaver()
    graph = build_orchestrator_graph()
    app = graph.compile(checkpointer=memory)

    print("\n" + "=" * 50)
    print("=== ЗАПУСК ТЕСТОВ ОРКЕСТРАТОРА СЭД ===")
    print("=" * 50)

    llm_ready = True
    if not llm_ready:
        print("Тесты пропущены, так как LLM недоступен.")
        return

    # ----------------------------------------------------------------------
    # ТЕСТ 1: Сводка документа (Полный цикл)
    # ----------------------------------------------------------------------
    print("\n--- ТЕСТ 1: ЗАПРОС СВОДКИ ДОКУМЕНТА (Injected State) ---")

    thread_id_1 = "test-full-cycle-1"
    config_1 = {"configurable": {"thread_id": thread_id_1}}

    initial_state_1: OrchestratorState = {
        "messages": [HumanMessage(content="Сделай сводку по контракту.")],
        "context_ui_id": "doc-a1b2-c3d4",
        "tools_to_call": [],
        "tool_results_history": [],
        "user_context": {"role": "Администратор", "permissions": ["Read", "Write", "Summarize"]}
    }

    try:
        print(f"Running Graph с контекстом: {initial_state_1['user_context']}")
        final_state_1 = asyncio.run(app.ainvoke(initial_state_1, config=config_1))

        print("\nФинальный ответ ТЕСТА 1:")
        response_1 = final_state_1['messages'][-1].content
        print(response_1)

        if "Контракт" in response_1 and "2025/СМ-123" in response_1:
            print("✅ Тест 1 Успех: Ответ содержит ключевые данные контракта.")
        else:
            print("❌ Тест 1 Провал: Ответ не содержит ожидаемых данных.")

    except Exception as e:
        logger.error(f"Ошибка в Тесте 1: {e}")
        print(f"❌ Тест 1 завершился с ошибкой: {e}")

    # ----------------------------------------------------------------------
    # ТЕСТ 4: Проверка памяти и удаления сообщений (Два шага)
    # ----------------------------------------------------------------------
    print("\n\n--- ТЕСТ 4: ПРОВЕРКА ПАМЯТИ И УДАЛЕНИЯ СООБЩЕНИЙ ---")

    thread_id_4 = "test-memory-4"
    config_4 = {"configurable": {"thread_id": thread_id_4}}

    print("\n--- Шаг 1: Запрос с именем (Заполняем историю) ---")

    initial_state_4_step1: OrchestratorState = {
        "messages": [HumanMessage(content="Кто такой Сидоров?")],
        "context_ui_id": None,
        "tools_to_call": [],
        "tool_results_history": [],
        "user_context": None
    }

    final_state_step1 = asyncio.run(app.ainvoke(initial_state_4_step1, config=config_4))
    print(f"Ответ Шага 1: {final_state_step1['messages'][-1].content}")

    print("\n--- Шаг 2: Последующий запрос, использующий контекст (email) ---")

    follow_up_message = HumanMessage(content="А какой у него email?")

    final_state_step2 = asyncio.run(app.ainvoke({"messages": [follow_up_message]}, config=config_4))
    response_2 = final_state_step2['messages'][-1].content
    print(f"Ответ Шага 2: {response_2}")

    if "a.petrov@org.com" in response_2:
        print("✅ Тест 4 Успех: LLM использовал контекст и выдал корректный email (из мока).")
    else:
        print("❌ Тест 4 Провал: Не получен ожидаемый ответ с email.")

    print("\n--- Проверка состояния памяти после работы узла DELETER (Ожидается 0 или 1 сообщение) ---")
    history_state = memory.get(config_4)

    if history_state:
        history_messages: List[BaseMessage] = history_state.get('messages', [])
        print(f"Сообщений в истории потока {thread_id_4}: {len(history_messages)}")

        if len(history_messages) in [0, 1]:
            print(f"✅ Успех: История очищена. Осталось {len(history_messages)} сообщение(й).")
        else:
            print(
                f"❌ Провал: Ожидалось 0 или 1 сообщение, но осталось {len(history_messages)}. Очистка не сработала.")
    else:
        print("❌ Провал: Состояние памяти потока test-memory-4 не найдено.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tests()