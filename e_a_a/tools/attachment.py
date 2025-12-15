# edms_ai_assistant/tools/attachment.py
from typing import Optional
from langchain_core.tools import tool
import json
import logging
import os
from uuid import UUID
import asyncio
from pathlib import Path

try:
    from edms_ai_assistant.utils.file_utils import extract_text_from_bytes
except ImportError:
    def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
        return f"Mock text for summary from {filename}."

from e_a_a.infrastructure.api_clients.attachment_client import (
    AttachmentClient,
)
from edms_ai_assistant.llm import get_chat_model

try:
    from edms_ai_assistant.constants import SUMMARY_TYPES
except ImportError:
    SUMMARY_TYPES = {}

logger = logging.getLogger(__name__)

MAX_LLM_TEXT_LENGTH = 8000
DEFAULT_SUMMARY_TYPE = "1"


def _get_summary_prompt_template(summary_type: Optional[str] = None) -> str:
    """
    Возвращает шаблон промпта на основе типа резюме.
    Использует DEFAULT_SUMMARY_TYPE, если переданный тип недействителен или отсутствует.
    """

    selected_type = summary_type

    if selected_type is None or selected_type not in SUMMARY_TYPES:
        selected_type = DEFAULT_SUMMARY_TYPE
        if selected_type not in SUMMARY_TYPES:
            logger.error(f"Константа SUMMARY_TYPES не содержит даже DEFAULT_SUMMARY_TYPE='{DEFAULT_SUMMARY_TYPE}'.")

    type_details = SUMMARY_TYPES.get(selected_type, {})
    # 📌 ИЗМЕНЕНИЕ: Безопасное извлечение промпта
    base_prompt = type_details.get("prompt")

    if not base_prompt:
        logger.warning(f"Промпт для типа '{selected_type}' не найден. Использование универсального промпта.")
        base_prompt = "Создай краткое содержание (3-5 предложений) на русском языке. Выдели суть, ключевые условия, стороны, суммы, даты."

    # 📌 ИЗМЕНЕНИЕ: Добавление инструкции в промпт для LLM
    instruction = f"Ваша задача - создать резюме. Инструкция: {base_prompt}. СТРОГО ОТВЕТЬТЕ ТОЛЬКО РЕЗЮМЕ, без пояснений или заголовков."

    return instruction + "\nТекст:\n{text_to_summarize}" # Используем обновленный промпт


async def _get_attachment_client(service_token: str) -> AttachmentClient:
    """Вспомогательная функция для получения клиента."""
    return AttachmentClient(service_token=service_token)


@tool
async def summarize_attachment_tool(
        document_id: str,
        attachment_id: str,
        attachment_name: str,
        service_token: str,
        summary_type: Optional[str] = None,
) -> str:
    """
    Инструмент для суммаризации вложения документа по его ID и ID вложения.
    """
    logger.info(
        f"Вызов summarize_attachment_tool для {attachment_name} в документе {document_id}. Тип: {summary_type}"
    )

    try:
        doc_uuid = UUID(document_id)
        att_uuid = UUID(attachment_id)
    except ValueError:
        return json.dumps(
            {"error": "Неверный формат ID документа или вложения (ожидается UUID)."},
            ensure_ascii=False,
        )

    async with await _get_attachment_client(service_token) as client:
        try:
            file_bytes = await client.download_attachment(doc_uuid, att_uuid)
            logger.info(f"Файл скачан: {attachment_name}")
            if not file_bytes:
                return json.dumps(
                    {"error": "Файл не найден или ошибка загрузки."}, ensure_ascii=False
                )

            text = extract_text_from_bytes(file_bytes, attachment_name)

            if not text or len(text) < 20:
                return json.dumps(
                    {
                        "error": f"Файл '{attachment_name}' содержит мало текста или не поддерживается."
                    },
                    ensure_ascii=False,
                )

            llm = get_chat_model()
            text_to_summarize = text[:MAX_LLM_TEXT_LENGTH]

            prompt_template = _get_summary_prompt_template(summary_type)
            prompt = prompt_template.format(text_to_summarize=text_to_summarize)

            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            summary = getattr(response, "content", str(response))

            return json.dumps(
                {
                    "summary": f"Краткое содержание файла '{attachment_name}':\n{summary}"
                },
                ensure_ascii=False,
            )

        except Exception as e:
            logger.error(f"Ошибка в summarize_attachment_tool: {e}", exc_info=True)
            return json.dumps(
                {
                    "error": f"Не удалось обработать вложение: {type(e).__name__}: {str(e)}"
                },
                ensure_ascii=False,
            )


async def _read_file_blocking(file_path: str) -> bytes:
    """Выполняет блокирующую операцию чтения файла в отдельном потоке."""
    filepath = Path(file_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    return await asyncio.to_thread(filepath.read_bytes)


@tool
async def extract_and_summarize_file_async_tool(
        file_path: str, service_token: str, summary_type: Optional[str] = None
) -> str:
    """
    [ASYNC] Инструмент для извлечения текста из файла пользователя (загруженного локально) и суммаризации.
    """
    logger.info(
        f"[ASYNC] Вызов extract_and_summarize_file_async_tool для файла: {file_path}. Тип: {summary_type}"
    )

    filename = os.path.basename(file_path)

    try:
        file_bytes = await _read_file_blocking(file_path)

        text = extract_text_from_bytes(file_bytes, filename)

        if not text or len(text) < 20:
            return json.dumps(
                {
                    "error": f"Файл '{filename}' содержит мало текста или не поддерживается."
                },
                ensure_ascii=False,
            )

        llm = get_chat_model()
        text_to_summarize = text[:MAX_LLM_TEXT_LENGTH]

        prompt_template = _get_summary_prompt_template(summary_type)
        prompt = prompt_template.format(text_to_summarize=text_to_summarize)

        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        summary = getattr(response, "content", str(response))

        return json.dumps(
            {"summary": f"Краткое содержание файла '{filename}':\n{summary}"},
            ensure_ascii=False,
        )

    except FileNotFoundError:
        return json.dumps({"error": f"Файл не найден: {file_path}"}, ensure_ascii=False)
    except Exception as e:
        logger.error(
            f"Ошибка в extract_and_summarize_file_async_tool: {e}", exc_info=True
        )
        return json.dumps(
            {"error": f"Не удалось обработать файл: {type(e).__name__}: {str(e)}"},
            ensure_ascii=False,
        )
