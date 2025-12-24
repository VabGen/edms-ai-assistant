# main.py
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
from edms_ai_assistant.config import settings
from edms_ai_assistant.models import (
    UserInput,
    AssistantResponse,
    FileUploadResponse,
)
from edms_ai_assistant.agent import EdmsDocumentAgent
from edms_ai_assistant.security import extract_user_id_from_token
from edms_ai_assistant.tools import all_tools

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"
_agent: Optional[EdmsDocumentAgent] = None


def get_agent() -> EdmsDocumentAgent:
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ИИ-Агент еще не инициализирован."
        )
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Временное хранилище: {UPLOAD_DIR}")
    try:
        _agent = EdmsDocumentAgent()
        logger.info("Ассистент EDMS успешно инициализирован.")
    except Exception as e:
        logger.error(f"Ошибка инициализации агента: {e}", exc_info=True)
    yield
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(
    title="EDMS AI Assistant API",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup_file(file_path: Optional[str]) -> None:
    """Безопасное фоновое удаление файла."""
    if not file_path: return
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.debug(f"Файл удален: {file_path}")
    except Exception as e:
        logger.warning(f"Не удалось удалить файл {file_path}: {e}")


async def save_upload(file: UploadFile, user_id: str) -> Path:
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
    try:
        user_id = extract_user_id_from_token(user_token)
        saved_path = await save_upload(file, user_id)
        return FileUploadResponse(file_path=str(saved_path), file_name=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла")


@app.post("/chat", response_model=AssistantResponse)
async def chat_endpoint(
        user_input: UserInput,
        background_tasks: BackgroundTasks,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]
):
    try:
        user_id = extract_user_id_from_token(user_input.user_token)
        thread_id = f"{user_id}_{user_input.context_ui_id or 'general'}"
        current_file_path = user_input.file_path
        if current_file_path and not Path(current_file_path).exists():
            logger.warning(f"Файл {current_file_path} не найден на диске, сбрасываем путь.")
            current_file_path = None

        user_context = user_input.context.model_dump() if user_input.context else None
        if not user_context:
            try:
                async with EmployeeClient() as emp_client:
                    user_context = await emp_client.get_employee(user_input.user_token, user_id)
            except Exception:
                logger.warning("Контекст сотрудника не получен")

        # 2. ВЫЗОВ АГЕНТА
        agent_result = await agent.chat(
            message=user_input.message,
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
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
            response=agent_result.get("content") or "Готово",
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        if user_input.file_path: _cleanup_file(user_input.file_path)
        raise HTTPException(status_code=500, detail="Ошибка сервера")


@app.post("/actions/summarize", response_model=AssistantResponse)
async def api_direct_summarize(
        user_input: UserInput,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]
):
    """Прямой вызов суммаризации через инструменты (для кнопок)."""
    try:
        user_id = extract_user_id_from_token(user_input.user_token)
        thread_id = f"{user_id}_{user_input.context_ui_id or 'general'}"
        get_file_tool = next((t for t in all_tools if t.name == "doc_get_file_content"), None)
        summarize_tool = next((t for t in all_tools if t.name == "doc_summarize_text"), None)

        file_data = await get_file_tool.ainvoke({
            "document_id": user_input.context_ui_id,
            "attachment_id": user_input.file_path,
            "token": user_input.user_token
        })

        if isinstance(file_data, dict) and file_data.get("status") == "error":
            logger.error(f"File Fetch Error: {file_data.get('message')}")
            return AssistantResponse(status="success", response=f"Ошибка: {file_data.get('message')}", thread_id=thread_id)

        extracted_text = file_data.get("content") if isinstance(file_data, dict) else str(file_data)

        if not extracted_text or len(extracted_text.strip()) < 10:
             return AssistantResponse(status="success", response="Файл пуст или не содержит текста.", thread_id=thread_id)

        summary_result = await summarize_tool.ainvoke({
            "text": extracted_text,
            "summary_type": user_input.human_choice or "abstractive"
        })

        if isinstance(summary_result, dict) and summary_result.get("status") == "error":
             summary_text = summary_result.get("message", "Техническая ошибка суммаризации.")
        else:
             summary_text = summary_result.get("content") or summary_result.get("summary") or "Ошибка анализа."

        labels = {"extractive": "факты", "abstractive": "пересказ", "thesis": "тезисы"}
        display_label = labels.get(user_input.human_choice, "анализ")

        await agent.agent.aupdate_state({"configurable": {"thread_id": thread_id}}, {
            "messages": [
                HumanMessage(content=f"Запрос анализа: {display_label}"),
                AIMessage(content=summary_text)
            ]
        }, as_node="agent")

        return AssistantResponse(status="success", response=summary_text, thread_id=thread_id)
    except Exception as e:
        logger.error(f"Action error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка обработки файла")


@app.get("/chat/history/{thread_id}")
async def get_history(thread_id: str, agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]):
    try:
        state = await agent.agent.aget_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        return {
            "messages": [
                {"type": "human" if m.type == "human" else "ai", "content": m.content}
                for m in messages if m.content
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="История недоступна")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.API_PORT, reload=settings.DEBUG)
