# test/test_llm_invoke.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from langchain_core.messages import HumanMessage

from edms_ai_assistant.config import settings
from edms_ai_assistant.llm import get_chat_model

headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
response = requests.get("https://api.proxyapi.ru/openai/v1/models", headers=headers)

if response.status_code == 200:
    models = [m['id'] for m in response.json()['data']]
    print("Доступные модели:", "\n".join(models))
else:
    print(f"Ошибка: {response.status_code}, {response.text}")


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

    print("\nОтвет получен:")
    print(f"   {response.content}")

except Exception as e:
    print(f"\nОшибка: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Тест завершён ===")
