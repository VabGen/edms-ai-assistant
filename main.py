# main.py

import asyncio
import os
import sys
import logging
import json
import uuid
import base64
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
import uvicorn
import tempfile
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from edms_ai_assistant.config import settings
from edms_ai_assistant.models import OrchestratorState, UserInput, AssistantResponse, FileUploadResponse
from edms_ai_assistant.graph import build_orchestrator_graph

logging.basicConfig(level=settings.LOGGING_LEVEL, format=settings.LOGGING_FORMAT)
logger = logging.getLogger(__name__)

orchestrator_app = None
orchestrator_memory: Optional[InMemorySaver] = None

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_agent_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# --- Утилиты для управления файлами (без изменений) ---
# NOTE: Эти утилиты теперь будут использоваться двумя разными эндпоинтами.
# _cleanup_file теперь должен принимать Path или str, чтобы быть гибким.
def _cleanup_file(file_path: Union[Path, str]):
    """ Фоновая задача для безопасного удаления временного файла. """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    try:
        if UPLOAD_DIR in path.parents:
            path.unlink(missing_ok=True)
            logger.debug(f"Cleaned up temporary file: {path}")
    except Exception as e:
        logger.warning(f"Failed to clean up {path}: {e}")


async def save_uploaded_file_async(upload_file: UploadFile, user_uuid: uuid.UUID) -> Optional[Path]:
    """ Сохраняет загруженный файл во временный файл. """
    if not upload_file.filename:
        return None

    file_extension = Path(upload_file.filename).suffix
    # Используем уникальный UUID для предотвращения конфликтов
    file_path = UPLOAD_DIR / f"{user_uuid}_{uuid.uuid4()}{file_extension}"

    try:
        content = await upload_file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"File save error: {e}")
        return None


def run_orchestrator_app():
    """ Инициализирует и компилирует LangGraph с Checkpointer. """
    global orchestrator_memory

    memory = InMemorySaver()
    orchestrator_memory = memory

    graph = build_orchestrator_graph()

    app_compiled = graph.compile(checkpointer=memory)

    return app_compiled


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Управление жизненным циклом приложения FastAPI. """
    global orchestrator_app
    logger.info("Инициализация оркестратора LangGraph...")
    orchestrator_app = run_orchestrator_app()
    logger.info("Оркестратор инициализирован и готов.")
    yield
    logger.info("Выключение ассистента.")


app = FastAPI(title="AI-Powered EDMS Orchestrator API", lifespan=lifespan)


# --- УТИЛИТА: Извлечение ID пользователя из JWT (без изменений) ---
def _extract_user_id_from_token(user_token: str) -> str:
    """ Декодирует JWT payload для извлечения ID пользователя (id или sub). """
    try:
        _, payload_encoded, _ = user_token.split(".")
        padding_needed = 4 - (len(payload_encoded) % 4)
        if padding_needed < 4:
            payload_encoded += "=" * padding_needed

        payload_decoded = base64.urlsafe_b64decode(payload_encoded.encode("utf-8"))
        payload = json.loads(payload_decoded)
        user_id_for_thread = str(payload.get("id") or payload.get("sub"))

        if not user_id_for_thread:
            raise ValueError("User ID ('id' or 'sub') not found in JWT payload.")

        return user_id_for_thread

    except Exception as e:
        raise ValueError(f"Ошибка декодирования/парсинга JWT: {e}")


# ----------------------------------------------------------------
# НОВЫЙ ЭНДПОИНТ: ЗАГРУЗКА ФАЙЛА
# ----------------------------------------------------------------

@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file_for_analysis(
        file: UploadFile = File(..., description="Файл для загрузки и последующего анализа."),
        user_token: str = Form(..., description="JWT токен пользователя для создания уникального пути."),
        background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Загружает файл и сохраняет его во временное хранилище.
    Возвращает временный путь, который клиент должен передать в /chat.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Отсутствует имя файла.")

    try:
        # Получаем ID пользователя для уникальности пути
        user_id_for_thread = _extract_user_id_from_token(user_token)
        user_uuid_for_path = uuid.UUID(user_id_for_thread)

        file_path = await save_uploaded_file_async(file, user_uuid_for_path)

        if not file_path:
            raise HTTPException(status_code=500, detail="Ошибка при сохранении файла")

        return FileUploadResponse(
            file_path=str(file_path),
            file_name=file.filename,
            # Добавляем задачу на очистку, если клиент не успеет вызвать /chat в течение N времени (опционально)
            # background_tasks.add_task(_cleanup_file, file_path)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Неверный токен: {e}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при обработке файла")


@app.post("/chat", response_model=AssistantResponse)
async def chat_with_assistant(
        user_input: UserInput,
        background_tasks: BackgroundTasks,
):
    """
    Основной эндпоинт для чата, принимающий JSON-запрос.
    """
    if not orchestrator_app:
        raise HTTPException(status_code=503, detail="Оркестратор не инициализирован.")

    file_path: Optional[str] = user_input.file_path

    # 1. Валидация токена
    try:
        user_id_for_thread = _extract_user_id_from_token(user_input.user_token)

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Ошибка токена: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Неверный формат токена: {e}",
        )

    # 2. Настройка графа и начального состояния
    thread_id = user_id_for_thread
    config = {"configurable": {"thread_id": thread_id}}

    user_context_dict = (
        user_input.context.model_dump() if user_input.context else None
    )

    initial_state: OrchestratorState = {
        "messages": [HumanMessage(content=user_input.message)],
        "context_ui_id": user_input.context_ui_id,
        "user_context": user_context_dict,
        "file_path": file_path,
        "user_token": user_input.user_token,

        "tools_to_call": [],
        "tool_results_history": [],
    }

    # 3. Запуск Оркестратора (Асинхронный запуск)
    try:
        # 1. Используем копию начального состояния для безопасного обновления
        final_state = initial_state.copy()

        # 2. Передача начального состояния и конфигурации
        async for output in orchestrator_app.astream(initial_state, config=config):

            # 🌟 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Извлекаем изменения из словаря {узел: изменения}
            if isinstance(output, dict):
                # output: {node_name: {state_key: new_value, ...}}

                # Мы должны объединить все изменения, полученные от всех узлов в одном шаге.
                all_changes = {}
                for node_name, changes in output.items():
                    if isinstance(changes, dict):
                        all_changes.update(changes)

                # Обновляем наше финальное состояние всеми изменениями
                final_state.update(all_changes)

                # 4. Извлечение финального ответа
        messages: List[BaseMessage] = final_state.get("messages", [])
        response_content = None

        if messages and isinstance(messages[-1], BaseMessage):
            # Проверяем, что ответ — это AIMessage, или, по крайней мере, BaseMessage
            response_content = messages[-1].content

        if not response_content:
            logger.error(
                "Не удалось извлечь финальный ответ из последнего сообщения."
            )
            response_content = "Извините, не удалось сформулировать ответ."

        # 5. Очистка временного файла
        if file_path:
            # Очистка происходит здесь, после того как граф закончил работу с файлом.
            background_tasks.add_task(_cleanup_file, file_path)

        return AssistantResponse(response=response_content)

    except Exception as e:
        logger.error(f"Ошибка в обработчике чата: {e}", exc_info=True)
        if file_path:
            background_tasks.add_task(_cleanup_file, file_path)
        raise HTTPException(
            status_code=500, detail="Внутренняя ошибка сервера при обработке запроса"
        )


@app.get("/health")
def health_check():
    """Проверка состояния сервиса."""
    return {"status": "ok", "orchestrator_status": "ready" if orchestrator_app else "initializing"}


if __name__ == "__main__":
    logger.info("Инициализация оркестратора LangGraph в блоке __main__...")
    orchestrator_app = run_orchestrator_app()
    logger.info("Оркестратор инициализирован и готов.")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.API_PORT,
        log_level=settings.LOGGING_LEVEL.lower(),
    )
