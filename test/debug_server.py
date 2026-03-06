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