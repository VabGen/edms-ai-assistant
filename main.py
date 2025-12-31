import logging
import re
import shutil
import uuid
import tempfile
import aiofiles
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form, Depends, status
from starlette.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.config import settings
from edms_ai_assistant.models import UserInput, AssistantResponse, FileUploadResponse, NewChatRequest
from edms_ai_assistant.agent import EdmsDocumentAgent
from edms_ai_assistant.security import extract_user_id_from_token

logging.basicConfig(level=settings.LOGGING_LEVEL, format=settings.LOGGING_FORMAT)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"
_agent: Optional[EdmsDocumentAgent] = None

UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)


def get_agent() -> EdmsDocumentAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="ИИ-Агент не инициализирован.")
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _agent = EdmsDocumentAgent()
        logger.info("Ассистент EDMS успешно запущен.")
    except Exception as e:
        logger.error(f"Критическая ошибка инициализации агента: {e}")
    yield
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(title="EDMS AI Assistant API", version="2.1.2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup_file(file_path: str):
    """Безопасное удаление временного файла."""
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            logger.debug(f"Файл удален: {file_path}")
    except Exception as e:
        logger.warning(f"Не удалось удалить временный файл {file_path}: {e}")


@app.post("/chat", response_model=AssistantResponse)
async def chat_endpoint(user_input: UserInput, background_tasks: BackgroundTasks,
                        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]):
    user_id = extract_user_id_from_token(user_input.user_token)
    thread_id = user_input.thread_id or f"user_{user_id}_doc_{user_input.context_ui_id or 'general'}"

    user_context = user_input.context.model_dump() if user_input.context else None
    if not user_context:
        try:
            async with EmployeeClient() as emp_client:
                user_context = await emp_client.get_employee(user_input.user_token, user_id)
        except:
            user_context = {"firstName": "Коллега"}

    result = await agent.chat(
        message=user_input.message,
        user_token=user_input.user_token,
        context_ui_id=user_input.context_ui_id,
        thread_id=thread_id,
        user_context=user_context,
        file_path=user_input.file_path,
        human_choice=user_input.human_choice
    )

    if user_input.file_path:
        is_system_attachment = bool(UUID_PATTERN.match(str(user_input.file_path)))

        if not is_system_attachment and result.get("status") not in ["requires_action", "requires_choice"]:
            logger.info(f"Запланировано удаление временного файла: {user_input.file_path}")
            background_tasks.add_task(_cleanup_file, user_input.file_path)

    return AssistantResponse(
        status=result.get("status", "success"),
        response=result.get("content"),
        action_type=result.get("action_type"),
        message=result.get("message"),
        thread_id=thread_id
    )


@app.post("/actions/summarize", response_model=AssistantResponse)
async def api_direct_summarize(
        user_input: UserInput,
        background_tasks: BackgroundTasks,
        agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]
):
    try:
        user_id = extract_user_id_from_token(user_input.user_token)
        new_thread_id = f"action_{user_id}_{uuid.uuid4().hex[:6]}"

        current_path = user_input.file_path

        is_uuid = False
        if current_path:
            is_uuid = bool(
                re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', str(current_path), re.I))

        if current_path and not is_uuid:
            if not Path(current_path).exists():
                logger.warning(f"Локальный файл {current_path} не найден, сброс.")
                current_path = None

        agent_result = await agent.chat(
            message=f"Проанализируй вложение: {user_input.message}",
            user_token=user_input.user_token,
            context_ui_id=user_input.context_ui_id,
            thread_id=new_thread_id,
            file_path=current_path,
            human_choice=user_input.human_choice
        )

        if current_path and not is_uuid:
            background_tasks.add_task(_cleanup_file, current_path)

        return AssistantResponse(
            status="success",
            response=agent_result.get("content") or "Анализ готов.",
            thread_id=new_thread_id
        )
    except Exception as e:
        logger.error(f"Ошибка в direct_summarize: {e}", exc_info=True)
        if user_input.file_path and not bool(re.match(r'^[0-9a-f]{8}-', str(user_input.file_path))):
            _cleanup_file(user_input.file_path)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------------------------------------

@app.get("/chat/history/{thread_id}")
async def get_history(thread_id: str, agent: Annotated[EdmsDocumentAgent, Depends(get_agent)]):
    try:
        state = await agent.agent.aget_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        filtered = []
        for m in messages:
            if not isinstance(m, (HumanMessage, AIMessage)): continue
            if isinstance(m, AIMessage) and not m.content: continue
            filtered.append({"type": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content})
        return {"messages": filtered}
    except Exception as e:
        logger.error(f"History error: {e}")
        return {"messages": []}


@app.post("/chat/new")
async def create_new_thread(request: NewChatRequest):
    try:
        user_id = extract_user_id_from_token(request.user_token)
        new_thread_id = f"chat_{user_id}_{uuid.uuid4().hex[:6]}"
        return {"status": "success", "thread_id": new_thread_id}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file(user_token: Annotated[str, Form(...)], file: Annotated[UploadFile, File(...)]):
    try:
        user_id = extract_user_id_from_token(user_token)
        suffix = Path(file.filename).suffix
        file_id = f"{user_id}_{uuid.uuid4().hex}{suffix}"
        dest_path = UPLOAD_DIR / file_id
        async with aiofiles.open(dest_path, "wb") as out_file:
            while content := await file.read(1024 * 1024):
                await out_file.write(content)
        return FileUploadResponse(file_path=str(dest_path), file_name=file.filename)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.API_PORT, reload=settings.DEBUG)
