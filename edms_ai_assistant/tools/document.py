import logging
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)


class DocDetailsInput(BaseModel):
    document_id: str = Field(..., description="UUID документа (context_ui_id)")
    token: str = Field(..., description="Токен авторизации пользователя")


@tool("doc_get_details", args_schema=DocDetailsInput)
async def doc_get_details(document_id: str, token: str) -> Dict[str, Any]:
    """
    Получает исчерпывающую информацию о документе из СЭД.
    """
    try:
        async with DocumentClient() as client:
            raw_data = await client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)

            def get_val(obj, attr, default=None):
                if obj is None: return default
                if isinstance(obj, dict):
                    val = obj.get(attr, default)
                else:
                    val = getattr(obj, attr, default)

                if hasattr(val, 'value'):
                    return val.value
                return val

            appeal = doc.documentAppeal

            info = {
                "регистрация_и_статус": {
                    "id": str(doc.id) if doc.id else None,
                    "заголовок": doc.profileName,
                    "рег_номер": doc.regNumber,
                    "дата_рег": doc.regDate.strftime("%d.%m.%Y") if doc.regDate else None,
                    "статус": get_val(doc, "status"),
                    "тип_документа": get_val(doc.documentType, "typeName"),
                    "категория": get_val(doc, "docCategoryConstant"),
                    "журнал": get_val(doc.registrationJournal, "name"),
                    "краткое_содержание": doc.shortSummary,
                },
                "участники": {
                    "автор": f"{doc.author.lastName} {doc.author.firstName}" if doc.author else None,
                    "исполнитель": f"{doc.responsibleExecutor.lastName} {doc.responsibleExecutor.firstName}" if doc.responsibleExecutor else None,
                    "подписанты": [f"{u.lastName} {u.firstName}" for u in
                                   (doc.whoAddressed or [])] if doc.whoAddressed else None,
                },
                "контроль": {
                    "на_контроле": doc.controlFlag,
                    "срок_исполнения_дней": doc.daysExecution,
                    "задач_всего": doc.countTask,
                    "задач_выполнено": doc.completedTaskCount,
                },
                "обращение_граждан": {
                    "заявитель": get_val(appeal, "fioApplicant"),
                    "тематика": get_val(get_val(appeal, "subject"), "name"),
                    "подтематика": get_val(get_val(get_val(appeal, "subject"), "parentSubject"), "name"),
                    "город": get_val(appeal, "cityName"),
                    "адрес": get_val(appeal, "fullAddress"),
                    "телефон": get_val(appeal, "phone"),
                    "email": get_val(appeal, "email"),
                    "тип_обращения": get_val(get_val(appeal, "citizenType"), "name"),
                },
                "совещание": {
                    "дата": doc.dateMeeting.strftime("%d.%m.%Y") if doc.dateMeeting else None,
                    "место": doc.placeMeeting,
                    "вопросы": [get_val(q, "questionName") for q in
                                (doc.documentQuestions or [])] if doc.documentQuestions else None,
                },
                "договор": {
                    "номер": doc.contractNumber,
                    "сумма": doc.contractSum,
                    "валюта": get_val(doc.currency, "currencyName"),
                    "период": f"{doc.contractStartDate.strftime('%d.%m.%Y')} - {doc.contractDurationEnd.strftime('%d.%m.%Y')}"
                    if doc.contractStartDate and doc.contractDurationEnd else None,
                },
                "вложения": [
                    {"id": str(a.id), "имя": a.name} for a in (doc.attachmentDocument or [])
                ]
            }

            def compact(data):
                if isinstance(data, dict):
                    return {k: compact(v) for k, v in data.items() if v is not None and v != [] and v != {}}
                return data

            return {
                "status": "success",
                "document_info": compact(info)
            }

    except Exception as e:
        logger.error(f"Ошибка doc_get_details: {e}", exc_info=True)
        return {"error": f"Ошибка доступа к данным: {str(e)}"}
