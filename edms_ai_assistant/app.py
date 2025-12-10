# edms_ai_assistant/app.py
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
import tempfile
import uuid
from pathlib import Path
import base64
from edms_ai_assistant.config import settings
from edms_ai_assistant.core.orchestrator import create_orchestrator_graph
from edms_ai_assistant.models.orchestrator_models import UserInput, AssistantResponse
from langchain_core.messages import HumanMessage, BaseMessage

logging.basicConfig(level=settings.LOGGING_LEVEL, format=settings.LOGGING_FORMAT)
logger = logging.getLogger(__name__)

orchestrator_app = None

# --- Настройки для временных файлов ---
UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_agent_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# --------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения FastAPI.
    Инициализирует граф LangGraph (Оркестратор) при запуске.
    """
    global orchestrator_app
    logger.info("Инициализация оркестратора...")
    orchestrator_app = create_orchestrator_graph()
    logger.info("Оркестратор инициализирован.")
    yield
    logger.info("Выключение ассистента...")


app = FastAPI(title="AI-Powered EDMS Task Automation Assistant", lifespan=lifespan)


def _cleanup_file(file_path: Path):
    """
    Фоновая задача для безопасного удаления временного файла.
    """
    try:
        if UPLOAD_DIR in file_path.parents:
            file_path.unlink(missing_ok=True)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up {file_path}: {e}")


async def save_uploaded_file_async(upload_file: UploadFile, user_uuid: uuid.UUID) -> Optional[Path]:
    """
    Сохраняет загруженный файл во временный файл в UPLOAD_DIR.
    """
    if not upload_file.filename:
        return None

    safe_filename = Path(upload_file.filename).name
    if not safe_filename:
        return None

    file_path = UPLOAD_DIR / f"{user_uuid}_{safe_filename}"

    try:
        content = await upload_file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"File save error: {e}")
        return None


@app.post("/chat", response_model=AssistantResponse)
async def chat_with_assistant(
        background_tasks: BackgroundTasks,
        user_request: str = Form(..., description="JSON-строка с UserInput данными."),
        file: Optional[UploadFile] = File(None, description="Загруженный файл (опционально)"),
):
    if not orchestrator_app:
        raise HTTPException(status_code=503, detail="Оркестратор не инициализирован")

    file_path: Optional[Path] = None
    user_request_model: Optional[UserInput] = None

    try:
        user_input_data = json.loads(user_request)
        user_request_model = UserInput(**user_input_data)
        user_token = user_request_model.user_token

    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON в user_request: {e}")
        raise HTTPException(status_code=422, detail=f"Поле user_request должно быть валидной JSON-строкой: {e}")
    except Exception as e:
        logger.error(f"Ошибка валидации Pydantic модели UserInput: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Ошибка валидации Pydantic модели UserInput: {e}")

    user_uuid: uuid.UUID
    user_id_for_thread: str
    try:
        _, payload_encoded, _ = user_token.split('.')

        padding_needed = 4 - (len(payload_encoded) % 4)
        if padding_needed < 4:
            payload_encoded += '=' * padding_needed

        payload_decoded = base64.urlsafe_b64decode(payload_encoded.encode('utf-8'))
        payload = json.loads(payload_decoded)

        user_id_for_thread = payload.get('id') or payload.get('sub')

        if not user_id_for_thread:
            raise ValueError("User ID ('id' or 'sub') not found in JWT payload.")

        user_uuid = uuid.UUID(user_id_for_thread)

    except ValueError as e:
        logger.error(f"Ошибка декодирования JWT или валидации UUID: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user_token or extracted ID format. Expected valid JWT with a UUID in 'id' or 'sub' claim. Error: {e}"
        )
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при обработке токена: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Token processing failed due to an unexpected error. {e}"
        )

    if file and file.filename:
        file_path = await save_uploaded_file_async(file, user_uuid)
        if not file_path:
            raise HTTPException(status_code=500, detail="Ошибка при сохранении файла")

    thread_id = str(user_uuid)
    config = {"configurable": {"thread_id": thread_id}}

    context_dict = user_request_model.context.model_dump() if user_request_model.context else None

    initial_state = {
        "messages": [HumanMessage(content=user_request_model.message)],
        "user_token": user_request_model.user_token,
        "file_path": str(file_path) if file_path else None,
        "context": context_dict,
        "subagent_result": None,
        "called_subagent": None,
        "final_response": None,
        "agent_history": [],
    }

    try:
        final_state = initial_state

        async for output in orchestrator_app.astream(initial_state, config=config):
            final_state = output

        source_state = final_state

        if final_state.get("called_subagent") and isinstance(final_state.get(final_state["called_subagent"]), dict):
            source_state = final_state[final_state["called_subagent"]]

        elif len(final_state) == 1 and isinstance(list(final_state.values())[0], dict):
            source_state = list(final_state.values())[0]

        response_content = source_state.get("final_response")

        if not response_content:
            messages = source_state.get("messages", [])

            if messages:
                last_message = messages[-1]

                if isinstance(last_message, BaseMessage) and last_message.content:
                    response_content = last_message.content

                elif isinstance(last_message, dict) and 'content' in last_message:
                    if last_message.get('role') != 'user' and last_message.get('type') != 'human':
                        response_content = last_message['content']

        if not response_content:
            subagent_result = source_state.get("subagent_result")
            if subagent_result and not subagent_result.startswith("general_agent_error"):
                response_content = subagent_result

        if not response_content:
            response_content = "Извините, не удалось сформулировать ответ."
            logger.error("Не удалось извлечь final_response ни из поля, ни из последнего сообщения.")

        if file_path:
            background_tasks.add_task(_cleanup_file, file_path)

        return AssistantResponse(response=response_content)

    except Exception as e:
        logger.error(f"Ошибка в обработчике чата: {e}", exc_info=True)
        # Очистка файла при ошибке
        if file_path:
            background_tasks.add_task(_cleanup_file, file_path)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при обработке запроса")


@app.get("/health")
def health_check():
    """Проверка состояния сервиса."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.API_PORT,
        log_level=settings.LOGGING_LEVEL.lower(),
    )
