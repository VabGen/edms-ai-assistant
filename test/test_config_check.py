# test_config_check.py
from edms_ai_assistant.config import settings
print(f"LLM_ENDPOINT: {settings.LLM_ENDPOINT}")
print(f"LLM_MODEL_NAME: {settings.LLM_MODEL_NAME}")
assert "model-embedding" in settings.LLM_ENDPOINT, "Endpoint не исправлен!"
assert settings.LLM_MODEL_NAME == "generative-model", "Модель не исправлена!"
print("Конфигурация верная!")