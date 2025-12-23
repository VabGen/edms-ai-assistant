# main.py
import os
import re
import sys
import logging
import shutil
import uuid
import tempfile
import aiofiles
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Annotated

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
from langchain_core.messages import HumanMessage, AIMessage
from starlette.middleware.cors import CORSMiddleware

from edms_ai_assistant.clients.employee_client import EmployeeClient

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
    current_file_path = user_input.file_path
    doc_id = user_input.context_ui_id

    try:
        user_id = extract_user_id_from_token(user_input.user_token)
        thread_id = f"{user_id}_{doc_id or 'general'}"

        user_context = user_input.context.model_dump() if user_input.context else None
        if not user_context:
            try:
                async with EmployeeClient() as emp_client:
                    user_context = await emp_client.get_employee(user_input.user_token, user_id)
            except:
                logger.warning("Не удалось загрузить контекст сотрудника")

        if current_file_path and not Path(current_file_path).exists():
            current_file_path = None

        agent_result = await agent.chat(
            message=user_input.message,
            user_token=user_input.user_token,
            context_ui_id=doc_id,
            thread_id=thread_id,
            user_context=user_context,
            file_path=current_file_path,
            human_choice=user_input.human_choice
        )

        if current_file_path and agent_result.get("status") != "requires_action":
            background_tasks.add_task(_cleanup_file, current_file_path)

        if agent_result.get("status") == "requires_action":
            return AssistantResponse(
                status="requires_action",
                action_type=agent_result.get("action_type"),
                message=agent_result.get("message"),
                thread_id=thread_id
            )

        return AssistantResponse(
            status="success",
            response=agent_result.get("content"),
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        if current_file_path: _cleanup_file(current_file_path)
        raise HTTPException(status_code=500, detail="Ошибка сервера")


from fastapi import Header, HTTPException
from pydantic import BaseModel
from edms_ai_assistant.tools import all_tools


class DirectSummarizeRequest(BaseModel):
    attachment_id: str
    summary_type: str
    thread_id: str
    file_name: str


def get_tool_by_name(name: str):
    return next((t for t in all_tools if t.name == name), None)


@app.post("/actions/summarize", response_model=AssistantResponse)
async def api_direct_summarize(
        user_input: UserInput,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]
):
    try:
        user_token = user_input.user_token
        user_id = extract_user_id_from_token(user_token)
        doc_id = user_input.context_ui_id
        thread_id = f"{user_id}_{doc_id or 'general'}"
        attachment_id = user_input.file_path
        summary_type = user_input.human_choice or "abstractive"

        if not attachment_id:
            raise HTTPException(status_code=400, detail="Не указан ID файла")

        get_file_tool = get_tool_by_name("doc_get_file_content")
        summarize_tool = get_tool_by_name("doc_summarize_text")

        file_data = await get_file_tool.ainvoke({
            "document_id": doc_id,
            "attachment_id": attachment_id,
            "token": user_token
        })

        if isinstance(file_data, dict) and file_data.get("status") == "error":
            raise HTTPException(status_code=400, detail=file_data.get("message"))

        summary_result = await summarize_tool.ainvoke({
            "text": file_data.get("content", ""),
            "summary_type": summary_type
        })

        if isinstance(summary_result, dict):
            summary_text = summary_result.get("summary", "Не удалось извлечь резюме")
        else:
            summary_text = str(summary_result)

        config = {"configurable": {"thread_id": thread_id}}

        labels = {"extractive": "факты", "abstractive": "пересказ", "thesis": "тезисы"}
        display_label = labels.get(summary_type, "анализ")

        await agent.agent.aupdate_state(config, {
            "messages": [
                HumanMessage(content=f"Кнопка: Анализ файла ({display_label})"),
                AIMessage(content=summary_text)
            ]
        }, as_node="agent")

        return AssistantResponse(
            status="success",
            response=summary_text,
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(f"Action error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")


@app.get("/chat/history/{thread_id}")
async def get_history(
        thread_id: str,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]
):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = await agent.agent.aget_state(config)
        messages = state.values.get("messages", [])

        return {
            "messages": [
                {
                    "type": "human" if m.type == "human" else "ai",
                    "content": m.content
                } for m in messages
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Не удалось загрузить историю")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=(settings.ENVIRONMENT == "development")
    )
