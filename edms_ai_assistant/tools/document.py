import logging
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)


class DocDetailsInput(BaseModel):
    document_id: str = Field(..., description="UUID документа (context_ui_id)")
    token: str = Field(..., description="Токен авторизации пользователя")


@tool("doc_get_details", args_schema=DocDetailsInput)
async def doc_get_details(document_id: str, token: str) -> Dict[str, Any]:
    """
    Анализирует документ СЭД и все его вложенные сущности (поручения, процессы, обращения, договоры).
    Возвращает семантически структурированный контекст.
    """
    try:
        async with DocumentClient() as client:
            raw_data = await client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)

            nlp = EDMSNaturalLanguageService()
            context = nlp.process_document(doc)

            def clean(d):
                if isinstance(d, dict):
                    return {
                        k: clean(v) for k, v in d.items() if v not in [None, [], {}, ""]
                    }
                if isinstance(d, list):
                    return [clean(i) for i in d if i not in [None, [], {}, ""]]
                return d

            return {"status": "success", "document_analytics": clean(context)}

    except Exception as e:
        logger.error(f"NLP Error on doc {document_id}: {e}", exc_info=True)
        return {"error": f"Ошибка обработки структуры документа: {str(e)}"}
