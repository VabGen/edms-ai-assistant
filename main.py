# edms_ai_assistant/main.py

import os
import sys
import logging
import json
import uuid
import base64
import functools
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Union, Annotated, Dict, Any
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks,
    Form,
    Depends,
    status,
)
import uvicorn
import tempfile
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import Runnable
from starlette.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from edms_ai_assistant.config import settings
from edms_ai_assistant.models import (
    OrchestratorState,
    UserInput,
    AssistantResponse,
    FileUploadResponse,
)
from edms_ai_assistant.graph import build_orchestrator_graph
from edms_ai_assistant.security import extract_user_id_from_token

logging.basicConfig(level=settings.LOGGING_LEVEL, format=settings.LOGGING_FORMAT)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"


@functools.lru_cache(maxsize=1)
def get_memory_checkpointer() -> InMemorySaver:
    """ Ленивая инициализация InMemorySaver. """
    logger.info("Инициализация InMemorySaver для Checkpointing.")
    return InMemorySaver()


@functools.lru_cache(maxsize=1)
def get_orchestrator_app() -> Runnable:
    checkpointer = get_memory_checkpointer()
    graph = build_orchestrator_graph()

    logger.info("Компиляция LangGraph с Checkpointer...")
    app_compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Оркестратор LangGraph скомпилирован.")

    return app_compiled


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Управление жизненным циклом приложения FastAPI. """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Директория для загрузки файлов: {UPLOAD_DIR}")

    try:
        get_orchestrator_app()
    except Exception as e:
        logger.error(f"Критическая ошибка при инициализации оркестратора: {e}", exc_info=True)

    yield

    try:
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            logger.info(f"Очищена временная директория: {UPLOAD_DIR}")
    except Exception as e:
        logger.warning(f"Не удалось очистить временную директорию: {e}")

    logger.info("Выключение ассистента завершено.")


app = FastAPI(
    title="AI-Powered EDMS Orchestrator API",
    version="1.0.0",
    description="API для гибкого AI-ассистента, интегрированного с EDMS.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:3001",
                   "http://localhost:8080",
                   "chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup_file(file_path: Union[Path, str]) -> None:
    """ Фоновая задача для безопасного удаления временного файла. """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    try:
        if path.resolve().parent.samefile(UPLOAD_DIR.resolve()):
            path.unlink(missing_ok=True)
            logger.debug(f"Cleaned up temporary file: {path}")
        else:
            logger.warning(f"Skipping cleanup for file outside UPLOAD_DIR: {path}")
    except Exception as e:
        logger.warning(f"Failed to clean up {path}: {e}")


async def save_uploaded_file_async(upload_file: UploadFile, user_id: str) -> Optional[Path]:
    """ Сохраняет загруженный файл в асинхронном режиме. """
    if not upload_file.filename:
        return None

    file_extension = Path(upload_file.filename).suffix
    file_path = UPLOAD_DIR / f"{user_id}_{uuid.uuid4()}{file_extension}"

    try:
        content = await upload_file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"File save error for {upload_file.filename}: {e}", exc_info=True)
        return None


# ----------------------------------------------------------------
# ЗАВИСИМОСТИ (DEPENDENCIES)
# ----------------------------------------------------------------

def get_orchestrator_dependency() -> Runnable:
    """Dependency для получения инициализированного оркестратора."""
    orchestrator = get_orchestrator_app()
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Оркестратор еще не инициализирован. Пожалуйста, подождите.",
        )
    return orchestrator


# ----------------------------------------------------------------
# ЭНДПОИНТЫ
# ----------------------------------------------------------------

@app.get("/health")
def health_check(
        orchestrator: Annotated[Runnable, Depends(get_orchestrator_dependency)]
):
    """Проверка состояния сервиса и доступности оркестратора."""
    return {"status": "ok", "orchestrator_status": "ready"}


@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file_for_analysis(
        file: Annotated[UploadFile, File(..., description="Файл для загрузки и анализа.")],
        user_token: Annotated[str, Form(..., description="JWT токен пользователя.")],
        background_tasks: BackgroundTasks,
):
    """ Загружает файл и сохраняет его во временное хранилище. """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Отсутствует имя файла.")

    try:
        user_id_for_thread = extract_user_id_from_token(user_token)

        file_path = await save_uploaded_file_async(file, user_id_for_thread)

        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ошибка при сохранении файла на сервере."
            )

        return FileUploadResponse(
            file_path=str(file_path),
            file_name=file.filename,
        )

    except ValueError as e:
        logger.warning(f"Неверный токен при загрузке: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Неверный токен: {e}")
    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке файла: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера при обработке файла"
        )


@app.post("/chat", response_model=AssistantResponse)
async def chat_with_assistant(
        user_input: UserInput,
        background_tasks: BackgroundTasks,
        orchestrator: Annotated[Runnable, Depends(get_orchestrator_dependency)],
):
    """ Основной эндпоинт для чата. """
    file_path: Optional[str] = user_input.file_path

    try:
        user_id_for_thread = extract_user_id_from_token(user_input.user_token)
    except ValueError as e:
        logger.error(f"Ошибка токена в /chat: {e}")
        if file_path:
            background_tasks.add_task(_cleanup_file, file_path)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Неверный/невалидный токен: {e}",
        )

    thread_id = user_id_for_thread
    config = {"configurable": {"thread_id": thread_id}}

    user_context_dict: Optional[Dict[str, Any]] = (
        user_input.context.model_dump(mode='json') if user_input.context else None
    )

    initial_state: OrchestratorState = {
        "messages": [HumanMessage(content=user_input.message)],
        "context_ui_id": user_input.context_ui_id,
        "user_context": user_context_dict,
        "file_path": file_path,
        "user_token": user_input.user_token,
        "tools_to_call": [],
        "tool_results_history": [],
        "required_file_name": None,
    }

    try:
        final_state: Dict[str, Any] = initial_state.copy()

        async for output in orchestrator.astream(initial_state, config=config):
            if isinstance(output, dict):
                all_changes = {}
                for changes in output.values():
                    if isinstance(changes, dict):
                        all_changes.update(changes)
                final_state.update(all_changes)

        messages: List[BaseMessage] = final_state.get("messages", [])
        response_content = (
            messages[-1].content
            if messages and isinstance(messages[-1], BaseMessage)
            else None
        )

        if not response_content:
            logger.error(
                f"Не удалось извлечь финальный ответ из LangGraph. Состояние: {final_state}"
            )
            response_content = "Извините, не удалось сформулировать ответ из-за внутренней ошибки оркестратора."

        if file_path:
            background_tasks.add_task(_cleanup_file, file_path)

        return AssistantResponse(response=response_content)

    except Exception as e:
        logger.error(f"Критическая ошибка в обработчике чата: {e}", exc_info=True)
        if file_path:
            background_tasks.add_task(_cleanup_file, file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера при обработке запроса"
        )


if __name__ == "__main__":
    try:
        orchestrator_app = get_orchestrator_app()
    except Exception as e:
        logger.error(f"Критическая ошибка при инициализации оркестратора: {e}", exc_info=True)
        sys.exit(1)

    logger.info("LangGraph Оркестратор инициализирован и готов к запуску сервера.")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        log_level=settings.LOGGING_LEVEL.lower(),
        reload=settings.ENVIRONMENT == "development",
    )
