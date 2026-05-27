# edms_ai_assistant/tools/document_comparison.py
import logging
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from edms_ai_assistant.agent.runnable_utils import get_token_from_config
from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)


class DocumentComparisonInput(BaseModel):
    """Схема входных данных для сравнения документов."""

    document_id_1: str = Field(..., description="UUID первого документа/версии")
    document_id_2: str = Field(..., description="UUID второго документа/версии")
    comparison_focus: str | None = Field(
        None,
        description="Конкретный аспект сравнения (metadata, attachments, content, all)",
    )


def _compare_metadata(doc1: dict, doc2: dict) -> dict[str, Any]:
    """Сравнивает метаданные двух документов."""
    fields_to_compare = [
        "regNumber",
        "regDate",
        "status",
        "shortSummary",
        "author",
        "correspondentName",
        "outRegNumber",
        "outRegDate",
    ]

    differences = {}
    for field in fields_to_compare:
        val1 = doc1.get(field)
        val2 = doc2.get(field)

        if val1 != val2:
            differences[field] = {"document_1": val1, "document_2": val2}

    return differences


def _compare_attachments(doc1: dict, doc2: dict) -> dict[str, Any]:
    """Сравнивает списки вложений."""
    att1 = doc1.get("attachmentDocument") or []
    att2 = doc2.get("attachmentDocument") or []

    def _get_att_name(a: Any) -> str:
        if isinstance(a, dict):
            return a.get("name") or a.get("originalName") or a.get("fileName") or ""
        # Поддержка Pydantic DTO
        return (
            getattr(a, "name", None)
            or getattr(a, "original_name", None)
            or getattr(a, "originalName", None)
            or ""
        )

    att1_names = {_get_att_name(a) for a in att1 if _get_att_name(a)}
    att2_names = {_get_att_name(a) for a in att2 if _get_att_name(a)}

    return {
        "added_in_doc2": list(att2_names - att1_names),
        "removed_from_doc1": list(att1_names - att2_names),
        "common": list(att1_names & att2_names),
        "total_count_doc1": len(att1),
        "total_count_doc2": len(att2),
    }


# ─── Tool Factory ─────────────────────────────────────────────────────────────


def create_doc_compare_documents_tool(
    document_client: DocumentClient,
    chat_model: BaseChatModel,
) -> StructuredTool:
    """Фабрика для создания инструмента сравнения документов.

    Args:
        document_client: Клиент для работы с документами EDMS.
        chat_model: Языковая модель для генерации сводки различий.

    Returns:
        Настроенный StructuredTool, готовый к регистрации в агенте.
    """

    async def doc_compare_documents(
        document_id_1: str,
        document_id_2: str,
        comparison_focus: str | None = "all",
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """
        Сравнивает два документа или версии документа.

        Используй когда:
        - Пользователь спрашивает о различиях между версиями
        - Нужно найти изменения в документе
        - Требуется сравнить два документа

        comparison_focus может быть:
        - "metadata": только метаданные (рег.номер, дата, статус и т.д.)
        - "attachments": только вложения
        - "content": только текстовое содержимое вложений
        - "all": полное сравнение

        Args:
            document_id_1: UUID первого документа.
            document_id_2: UUID второго документа.
            comparison_focus: Фокус сравнения.
            config: Конфиг Runnable (инжектируется автоматически).
        """
        try:
            token = get_token_from_config(config)
        except Exception as e:
            logger.error("Failed to get token from config: %s", e)
            return {
                "status": "error",
                "message": f"Ошибка авторизации: токен не найден в конфигурации запроса. {e}",
            }

        try:
            doc1_dto = await document_client.get_document_metadata(token, document_id_1)
            doc2_dto = await document_client.get_document_metadata(token, document_id_2)

            if not doc1_dto or not doc2_dto:
                return {
                    "status": "error",
                    "message": "Один или оба документа не найдены",
                }

            # Конвертируем DTO в dict с camelCase ключами для переиспользования
            # логики сравнения, которая опирается на ключи API СЭД
            doc1 = doc1_dto.model_dump(by_alias=True, exclude_none=True)
            doc2 = doc2_dto.model_dump(by_alias=True, exclude_none=True)

            comparison_result = {
                "status": "success",
                "document_1_id": document_id_1,
                "document_2_id": document_id_2,
                "differences": {},
            }

            # Сравнение метаданных
            if comparison_focus in ["metadata", "all"]:
                metadata_diff = _compare_metadata(doc1, doc2)
                comparison_result["differences"]["metadata"] = metadata_diff

            # Сравнение вложений
            if comparison_focus in ["attachments", "all"]:
                attachments_diff = _compare_attachments(doc1, doc2)
                comparison_result["differences"]["attachments"] = attachments_diff

            summary_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Ты — аналитик СЭД. Проанализируй различия между двумя документами и составь краткий отчет на русском языке.",
                    ),
                    (
                        "user",
                        "Различия между документами:\n{differences}\n\nСоставь структурированный отчет об основных изменениях:",
                    ),
                ]
            )

            chain = summary_prompt | chat_model | StrOutputParser()
            summary = await chain.ainvoke(
                {"differences": str(comparison_result["differences"])}
            )

            comparison_result["summary"] = summary.strip()

            return comparison_result

        except Exception as e:
            logger.error("[DOC-COMPARE-TOOL] Error: %s", e, exc_info=True)
            return {"status": "error", "message": f"Ошибка сравнения: {e!s}"}

    return StructuredTool.from_function(
        coroutine=doc_compare_documents,
        name="doc_compare_documents",
        description="Сравнивает два документа или версии документа.",
        args_schema=DocumentComparisonInput,
    )
