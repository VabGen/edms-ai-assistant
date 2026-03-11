# debug_all_models.py
import httpx
from edms_ai_assistant.config import settings

servers = {
    "generative": settings.LLM_GENERATIVE_URL,
    "embedding": settings.LLM_EMBEDDING_URL,
}

for name, endpoint in servers.items():
    url = f"{endpoint}/models"
    print(f"\n{'=' * 60}")
    print(f"{name.upper()}: {endpoint}")
    print('=' * 60)

    try:
        resp = httpx.get(url, timeout=10)
        data = resp.json()

        if "data" in data:
            for model in data["data"]:
                print(f"Модель: {model['id']}")
                print(f"Root: {model.get('root', 'N/A')}")
                print(f"Max len: {model.get('max_model_len', 'N/A')}")
        else:
            print(f"Неожиданный ответ: {data}")
    except Exception as e:
        print(f"Ошибка: {e}")


# ------------------------------------------------------------------------------

import logging
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def get_weather(city: str) -> str:
    """Возвращает текущую погоду в указанном городе."""
    return f"В городе {city} сейчас 20°C."


@tool
def get_current_time(location: str) -> str:
    """Возвращает текущее время в указанной локации."""
    return f"В локации {location} сейчас 14:30."


def test_parallel_tool_calling():
    llm_params = {
        "base_url": settings.LLM_ENDPOINT.rstrip("/"),
        "api_key": settings.LLM_API_KEY or "placeholder-key",
        "model": settings.LLM_MODEL_NAME,
        "temperature": settings.LLM_TEMPERATURE,
        "extra_body": {"skip_special_tokens": False}
    }

    try:
        logger.info("Инициализация модели...")
        llm = ChatOpenAI(**llm_params)

        tools = [get_weather, get_current_time]
        llm_with_tools = llm.bind_tools(tools)

        query = "Какая сейчас погода в Минске и который там час?"
        logger.info(f"Отправка запроса: {query}")

        response = llm_with_tools.invoke([HumanMessage(content=query)])

        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТ ПРОВЕРКИ ПАРАЛЛЕЛЬНОГО ВЫЗОВА:")

        if response.tool_calls:
            print(f"УСПЕХ: Найдено вызовов инструментов: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls, 1):
                print(f"\nИнструмент #{i}:")
                print(f"  - Функция: {tool_call['name']}")
                print(f"  - Аргументы: {tool_call['args']}")
        else:
            print("ОШИБКА: Модель не вызвала инструменты.")
            print(f"Ответ модели: {response.content}")

        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Сбой при тесте: {e}", exc_info=True)


if __name__ == "__main__":
    test_parallel_tool_calling()