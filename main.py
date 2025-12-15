# Файл: main.py (ОБНОВЛЕННЫЙ)

import asyncio
import os
import sys
import logging
from langgraph.checkpoint.memory import InMemorySaver

# Обратите внимание, импорты BaseMessage, HumanMessage больше не нужны
# для чистого запуска.

# Добавляем корневой каталог в путь для корректных относительных импортов
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорт графа
from edms_ai_assistant.graph import build_orchestrator_graph

logger = logging.getLogger(__name__)


# --- 1. ИНИЦИАЛИЗАЦИЯ И КОМПИЛЯЦИЯ ---
def run_orchestrator_app():
    """
    Инициализирует LangGraph с памятью и возвращает скомпилированное приложение.
    """
    # 1. Инициализация Памяти (Checkpointer)
    memory = InMemorySaver()

    # 2. Получаем скомпилированный граф
    graph = build_orchestrator_graph()

    # 3. Компиляция с Checkpointer
    app = graph.compile(checkpointer=memory)

    print("--- Оркестратор СЭД LangGraph запущен и скомпилирован с Памятью (InMemorySaver) ---")
    return app, memory


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    app, memory = run_orchestrator_app()

    # Здесь можно было бы запустить цикл интерактивного ввода,
    # но пока оставляем его как инициализатор.