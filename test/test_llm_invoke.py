# test/test_llm_invoke.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage
from edms_ai_assistant.config import settings
from edms_ai_assistant.llm import get_chat_model

print("=== Тест вызова LLM ===\n")

print("--- Конфигурация ---")
print(f"LLM Endpoint: {settings.LLM_GENERATIVE_URL}")
print(f"LLM Model: {settings.LLM_GENERATIVE_MODEL}")
print(f"Temperature: {settings.LLM_TEMPERATURE}\n")

try:
    print("Инициализация ChatModel...")
    llm = get_chat_model()
    print(f"ChatModel инициализирован: {type(llm).__name__}\n")

    test_message = "Привет! кто ты?"
    print(f"Отправка: '{test_message}'")
    print("Вызов LLM...")

    response = llm.invoke([HumanMessage(content=test_message)])

    print(f"\nОтвет получен:")
    print(f"   {response.content}")

except Exception as e:
    print(f"\nОшибка: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Тест завершён ===")