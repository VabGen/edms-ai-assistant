# edms_ai_assistant/tools/doc_update_field.py
"""
EDMS AI Assistant — Document Field Update Tool.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)

_ALLOWED_FIELDS: dict[str, str] = {
    "shortSummary": "Заголовок/краткое содержание (≤80 символов)",
    "note": "Примечание",
    "pages": "Количество листов документа",
    "additionalPages": "Количество листов приложений",
    "exemplarCount": "Количество экземпляров",
}

_ALLOWED_APPEAL_FIELDS: dict[str, str] = {
    "fullAddress": "Адрес заявителя",
    "phone": "Телефон",
    "email": "Email",
    "signed": "Кем подписано (ФИО)",
    "correspondentOrgNumber": "Исх.№ корреспондента",
    "organizationName": "Название организации",
    "fioApplicant": "ФИО заявителя",
    "reviewProgress": "Ход рассмотрения",
}


class UpdateDocumentFieldInput(BaseModel):
    """Validated input for updating a single document field."""

    document_id: str = Field(
        ...,
        description="UUID документа в СЭД",
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )
    token: str = Field(..., description="JWT токен авторизации")
    field_name: str = Field(
        ...,
        description=(
            "Имя поля для обновления. Допустимые значения:\n"
            "Основные: shortSummary (заголовок), note, pages, additionalPages, exemplarCount\n"
            "Обращение: fullAddress, phone, email, signed, correspondentOrgNumber, "
            "organizationName, fioApplicant, reviewProgress"
        ),
    )
    field_value: str = Field(
        ...,
        description="Новое значение поля",
        min_length=1,
        max_length=500,
    )

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        allowed = set(_ALLOWED_FIELDS) | set(_ALLOWED_APPEAL_FIELDS)
        if v not in allowed:
            raise ValueError(
                f"Поле '{v}' не поддерживается. "
                f"Допустимые: {', '.join(sorted(allowed))}"
            )
        return v

    @field_validator("field_value")
    @classmethod
    def clean_value(cls, v: str) -> str:
        return v.strip()


def _enum_value(val: Any) -> Any:
    """Safely extract .value from enum-like objects."""
    return val.value if hasattr(val, "value") else val


async def _fetch_main_required_fields(
        client: DocumentClient, token: str, document_id: str
) -> dict[str, Any]:
    """
    Загружает обязательные поля для DOCUMENT_MAIN_FIELDS_UPDATE.
    """
    raw = await client.get_document_metadata(token, document_id)
    if not raw:
        return {}

    doc = DocumentDto.model_validate(raw)
    result: dict[str, Any] = {}

    if doc.documentTypeId:
        result["documentTypeId"] = str(doc.documentTypeId)
    if doc.deliveryMethodId:
        result["deliveryMethodId"] = str(doc.deliveryMethodId)
    for attr in ("pages", "additionalPages", "exemplarCount"):
        val = getattr(doc, attr, None)
        if val is not None:
            result[attr] = val
    if doc.investProgramId:
        result["investProgramId"] = str(doc.investProgramId)
    if doc.note:
        result["note"] = doc.note
    if doc.shortSummary:
        result["shortSummary"] = doc.shortSummary

    return result


async def _fetch_appeal_required_fields(
        client: DocumentClient, token: str, document_id: str
) -> dict[str, Any]:
    """
    Загружает обязательные поля для DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE.
    """
    raw = await client.get_document_metadata(token, document_id)
    if not raw:
        return {"declarantType": "INDIVIDUAL", "submissionForm": "WRITTEN"}

    doc = DocumentDto.model_validate(raw)
    appeal = getattr(doc, "documentAppeal", None)
    result: dict[str, Any] = {}

    if not appeal:
        logger.warning(
            "documentAppeal is None for %s — using fallback INDIVIDUAL/WRITTEN",
            document_id[:8],
        )
        return {"declarantType": "INDIVIDUAL", "submissionForm": "WRITTEN"}

    # === Обязательные поля ===

    declarant = getattr(appeal, "declarantType", None)
    result["declarantType"] = _enum_value(declarant) if declarant is not None else "INDIVIDUAL"

    sub_form = getattr(appeal, "submissionForm", None)
    result["submissionForm"] = _enum_value(sub_form) if sub_form is not None else "WRITTEN"

    # === Сохраняем текущие значения ===

    _preserve_str = (
        "fioApplicant",
        "organizationName",
        "fullAddress",
        "phone",
        "email",
        "signed",
        "correspondentOrgNumber",
        "reviewProgress",
        "countryAppealName",
        "regionName",
        "districtName",
        "cityName",
        "index",
        "indexDateCoverLetter",
    )
    for attr in _preserve_str:
        val = getattr(appeal, attr, None)
        if val is not None and str(val).strip():
            result[attr] = val

    for attr in ("collective", "anonymous", "reasonably"):
        val = getattr(appeal, attr, None)
        if val is not None:
            result[attr] = val

    _id_fields = (
        "citizenTypeId",
        "subjectId",
        "countryAppealId",
        "cityId",
        "districtId",
        "regionId",
        "correspondentAppealId",
        "solutionResultId",
    )
    for attr in _id_fields:
        val = getattr(appeal, attr, None)
        if val is not None:
            result[attr] = str(val)

    # Даты
    for attr in ("receiptDate", "dateDocCorrespondentOrg"):
        val = getattr(appeal, attr, None)
        if val is not None:
            result[attr] = str(val) if not hasattr(val, "isoformat") else val.isoformat()

    return result


@tool("doc_update_field", args_schema=UpdateDocumentFieldInput)
async def doc_update_field(
        document_id: str,
        token: str,
        field_name: str,
        field_value: str,
) -> dict[str, Any]:
    """Обновляет одно поле документа через API EDMS.

    Используй когда пользователь просит:
    - «Поменяй заголовок на "Обращение о..."»  → field_name="shortSummary"
    - «Измени примечание»                        → field_name="note"
    - «Поправь адрес заявителя»                  → field_name="fullAddress"
    - «Обнови телефон в карточке»                → field_name="phone"
    - «Измени ФИО подписанта»                    → field_name="signed"

    Args:
        document_id: UUID документа.
        token: JWT токен авторизации.
        field_name: Имя поля (см. описание).
        field_value: Новое значение.

    Returns:
        Dict со статусом операции.
    """
    if field_name == "shortSummary" and len(field_value) > 80:
        field_value = field_value[:80]
        logger.warning("shortSummary обрезан до 80 символов: '%s'", field_value)

    is_appeal_field = field_name in _ALLOWED_APPEAL_FIELDS
    operation_type = (
        "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE"
        if is_appeal_field
        else "DOCUMENT_MAIN_FIELDS_UPDATE"
    )

    value: Any = field_value
    if field_name in ("pages", "additionalPages", "exemplarCount"):
        try:
            value = int(field_value)
        except ValueError:
            return {
                "status": "error",
                "message": f"Поле '{field_name}' должно быть числом, получено: '{field_value}'",
            }

    logger.info(
        "doc_update_field: %s.%s = %r (operation=%s)",
        document_id[:8],
        field_name,
        value,
        operation_type,
    )

    try:
        async with DocumentClient() as client:
            if is_appeal_field:
                existing = await _fetch_appeal_required_fields(client, token, document_id)
            else:
                existing = await _fetch_main_required_fields(client, token, document_id)

            body: dict[str, Any] = {**existing, field_name: value}

            payload = [{"operationType": operation_type, "body": body}]
            json_payload = json.loads(json.dumps(payload, cls=CustomJSONEncoder))

            logger.debug(
                "doc_update_field payload keys: %s", list(body.keys())
            )

            await client._make_request(
                "POST",
                f"api/document/{document_id}/execute",
                token=token,
                json=json_payload,
                is_json_response=False,
            )

        field_label = (
                _ALLOWED_FIELDS.get(field_name)
                or _ALLOWED_APPEAL_FIELDS.get(field_name)
                or field_name
        )

        logger.info("doc_update_field success: %s = %r", field_name, value)
        return {
            "status": "success",
            "message": f"✅ Поле «{field_label}» успешно обновлено: «{value}».",
            "field_name": field_name,
            "new_value": value,
            "requires_reload": True,
        }

    except Exception as exc:
        logger.error("doc_update_field failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"❌ Не удалось обновить поле «{field_name}»: {exc}",
        }
