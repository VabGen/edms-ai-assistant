import httpx
from langchain_ollama import ChatOllama


def quick_test():
    print("Проверка доступности Ollama...")
    try:
        r = httpx.get("http://127.0.0.1:11434/")
        print(f"Статус сервера: {r.status_code}, Ответ: {r.text}")
    except Exception as e:
        print(f"Сервер Ollama недоступен: {e}")
        return

    print("Попытка вызвать модель llama3.2...")
    llm = ChatOllama(
        model="llama3.2",
        base_url="http://127.0.0.1:11434",
        temperature=0.1
    )

    try:
        response = llm.invoke("Hi")
        print(f"Успех! Ответ модели: {response.content}")
    except Exception as e:
        print(f"Ошибка при вызове модели: {e}")


if __name__ == "__main__":
    quick_test()
