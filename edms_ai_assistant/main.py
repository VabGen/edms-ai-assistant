# main.py

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from core.orchestrator import create_edms_graph
from langchain_core.messages import HumanMessage
from edms_ai_assistant.config import settings
import logging

# Настройка логирования
logging.basicConfig(level=settings.LOGGING_LEVEL, format=settings.LOGGING_FORMAT)
logger = logging.getLogger(__name__)

app = FastAPI()
# Инициализация графа
graph = create_edms_graph()


class ChatRequest(BaseModel):
    message: str
    current_entity_id: str | None = None
    user_token: str | None = None


@app.post("/chat")
async def chat_endpoint(request: ChatRequest, authorization: str = Header(None)):
    logger.debug(f"Получен запрос: {request.message}. Контекст ID: {request.current_entity_id}")

    # --- 1. ОПРЕДЕЛЕНИЕ ТОКЕНА ---
    final_token = None

    if request.user_token:
        final_token = request.user_token
    elif authorization and "Bearer" in authorization:
        final_token = authorization.replace("Bearer ", "").strip()

    if not final_token:
        raise HTTPException(
            status_code=401,
            detail="Bearer Token required either in Authorization header or in the request body ('user_token' field)."
        )

    # --- 2. Инициализация состояния ---
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_token": final_token,
        "tools_to_call": [],
        "tool_results_history": []
    }

    # Если мы в контексте UI, добавляем это в историю для LLM-Планировщика
    if request.current_entity_id:
        context_message = (
            f"КОНТЕКСТ UI: Текущая активная сущность имеет чистый UUID: {request.current_entity_id}. "
            f"Используй этот ID для неявных запросов, таких как 'сделай сводку' или 'кто автор'."
        )
        initial_state["messages"].append(HumanMessage(content=context_message))
        logger.debug(f"Контекст UI добавлен в историю: {request.current_entity_id}")

    # --- 3. Запуск графа ---
    try:
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Ошибка выполнения графа: {e}")
        raise HTTPException(status_code=500, detail=f"Internal assistant error during processing: {e}")

    # Возврат последнего сообщения (ответа ассистента)
    final_response = result["messages"][-1].content if result["messages"] else "Извините, не удалось получить ответ."
    return {"response": final_response}


if __name__ == "__main__":
    logger.info(f"Запуск EDMS AI Assistant на http://0.0.0.0:{settings.API_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=settings.API_PORT)