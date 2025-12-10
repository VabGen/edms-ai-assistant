# test_embeddings.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Импорт конфигурации и LLM...")
from edms_ai_assistant.config import settings
from edms_ai_assistant.llm import get_embedding_model_to_use

print("--- Конфигурация эмбеддингов ---")
print(f"Embedding Endpoint: {settings.EMBEDDING_ENDPOINT}")
print(f"Embedding Model: {settings.EMBEDDING_MODEL_NAME}")

print("\n--- Инициализация и вызов EmbeddingModel ---")

try:
    print("Инициализация EmbeddingModel...")
    embedding_model = get_embedding_model_to_use()
    print(f"EmbeddingModel успешно инициализирован: {type(embedding_model).__name__}")

    test_text = ["Тестовая строка для получения эмбеддинга."]
    print(f"\nОтправка текста для эмбеддинга: {test_text}")

    print("Выполняется вызов модели эмбеддингов...")
    embeddings = embedding_model.embed_documents(test_text)

    print(f"\nПолучены эмбеддинги:")
    print(f"Количество векторов: {len(embeddings)}")
    if embeddings and len(embeddings[0]) > 0:
        print(f"Размер одного вектора (эмбеддинга): {len(embeddings[0])}")
        print(f"Первые 5 компонентов первого эмбеддинга: {embeddings[0][:5]}")
    else:
        print("Эмбеддинги пусты или имеют нулевой размер.")

    query_text = "Какой-то вопрос?"
    print(f"\nТест embed_query для: '{query_text}'")
    query_embedding = embedding_model.embed_query(query_text)
    print(f"Размер эмбеддинга запроса: {len(query_embedding) if query_embedding else 'N/A'}")

except Exception as e:
    print(f"Ошибка при инициализации или вызове EmbeddingModel: {e}")
    import traceback

    traceback.print_exc()

print("\n--- Тест вызова эмбеддингов завершен ---")
