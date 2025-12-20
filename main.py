# main.py
import os
import sys
import logging
import shutil
import uuid
import tempfile
import aiofiles
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Annotated, Union

import uvicorn
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
from starlette.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from edms_ai_assistant.config import settings
from edms_ai_assistant.models import (
    UserInput,
    AssistantResponse,
    FileUploadResponse,
)
from edms_ai_assistant.agent import EdmsDocumentAgent
from edms_ai_assistant.security import extract_user_id_from_token

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

_agent: Optional[EdmsDocumentAgent] = None


def get_agent() -> EdmsDocumentAgent:
    """Dependency для получения синглтона агента."""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ИИ-Агент еще не инициализирован."
        )
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global _agent

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Временное хранилище готово: {UPLOAD_DIR}")

    try:
        _agent = EdmsDocumentAgent()
        logger.info("Ассистент EDMS успешно инициализирован.")
    except Exception as e:
        logger.error(f"Критическая ошибка инициализации агента: {e}", exc_info=True)

    yield

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("Временные файлы удалены.")


app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.0.0",
    description="Профессиональный AI-сервис для работы с документами и данными EDMS.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup_file(file_path: str) -> None:
    """Фоновое удаление файла."""
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.debug(f"Файл удален: {file_path}")
    except Exception as e:
        logger.warning(f"Не удалось удалить файл {file_path}: {e}")


async def save_upload(file: UploadFile, user_id: str) -> Path:
    """Безопасное асинхронное сохранение файла."""
    ext = Path(file.filename).suffix
    if ext.lower() not in ['.pdf', '.docx', '.txt']:
        raise ValueError("Неподдерживаемый тип файла")

    dest = UPLOAD_DIR / f"{user_id}_{uuid.uuid4()}{ext}"

    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)
    return dest


@app.get("/health")
def health_check(agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]):
    """Проверка работоспособности системы."""
    return {"status": "ok", "agent_status": "ready"}


@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file(
        user_token: Annotated[str, Form(...)],
        file: Annotated[UploadFile, File(...)]
):
    """Загрузка файла перед анализом."""
    try:
        user_id = extract_user_id_from_token(user_token)
        saved_path = await save_upload(file, user_id)

        return FileUploadResponse(
            file_path=str(saved_path),
            file_name=file.filename
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла.")


@app.post("/chat", response_model=AssistantResponse)
async def chat_endpoint(
        user_input: UserInput,
        background_tasks: BackgroundTasks,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]
):
    """Основной эндпоинт взаимодействия с ассистентом."""
    current_file_path = user_input.file_path

    try:
        try:
            thread_id = extract_user_id_from_token(user_input.user_token)
        except Exception as e:
            logger.warning(f"Ошибка парсинга токена: {e}")
            raise HTTPException(status_code=401, detail="Невалидный токен")

        if current_file_path:
            requested_path = Path(current_file_path).resolve()
            base_path = UPLOAD_DIR.resolve()
            if not str(requested_path).startswith(str(base_path)):
                logger.warning(f"Попытка доступа вне разрешенной папки: {current_file_path}")
                raise HTTPException(status_code=403, detail="Доступ к указанному файлу запрещен")

            if not requested_path.exists():
                logger.error(f"Файл не найден: {current_file_path}")

        response_text = await agent.chat(
            message=user_input.message,
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
            thread_id=thread_id,
            user_context=user_input.context.model_dump() if user_input.context else None,
            file_path=current_file_path
        )

        if current_file_path:
            background_tasks.add_task(_cleanup_file, current_file_path)

        return AssistantResponse(
            response=response_text,
            thread_id=thread_id
        )

    except HTTPException:
        if current_file_path:
            _cleanup_file(current_file_path)
        raise
    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /chat: {e}", exc_info=True)
        if current_file_path:
            _cleanup_file(current_file_path)
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=(settings.ENVIRONMENT == "development")
    )
