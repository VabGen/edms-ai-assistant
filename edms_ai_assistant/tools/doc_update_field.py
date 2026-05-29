# edms_ai_assistant/tools/doc_update_field.py
"""
EDMS AI Assistant — Document Field Update Tool.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.agent.runnable_utils import (
    get_document_id_from_config,
    get_token_from_config,
)
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from edms_ai_assistant.clients.document_client import DocumentClient

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
    "index": "Почтовый индекс",
    "indexDateCoverLetter": "Индекс и дата сопроводительного письма",
    "signed": "Кем подписано (ФИО)",
    "correspondentOrgNumber": "Исх.№ корреспондента",
    "organizationName": "Название организации",
    "fioApplicant": "ФИО заявителя",
    "reviewProgress": "Ход рассмотрения",
}


class UpdateDocumentFieldInput(BaseModel):
    """Validated input for updating document fields."""

    updates: dict[str, str] = Field(
        ...,
        description=(
            "Словарь полей и их новых значений. Допустимые ключи:\n"
            "Основные: shortSummary (заголовок), note, pages, additionalPages, exemplarCount\n"
            "Обращение: fullAddress, phone, email, index, indexDateCoverLetter, signed, "
            "correspondentOrgNumber, organizationName, fioApplicant, reviewProgress"
        ),
    )

    @field_validator("updates")
    @classmethod
    def validate_updates(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("Список обновлений не может быть пустым")
        allowed = set(_ALLOWED_FIELDS) | set(_ALLOWED_APPEAL_FIELDS)
        for field_name in v:
            if field_name not in allowed:
                raise ValueError(
                    f"Поле '{field_name}' не поддерживается. "
                    f"Допустимые: {', '.join(sorted(allowed))}"
                )
        return {k: str(val).strip() for k, val in v.items()}


def _enum_value(val: Any) -> Any:
    """Safely extract .value from enum-like objects."""
    return val.value if hasattr(val, "value") else val


async def _fetch_main_required_fields(
    client: DocumentClient, token: str, document_id: str
) -> dict[str, Any]:
    """
    Загружает обязательные поля для DOCUMENT_MAIN_FIELDS_UPDATE.
    Читает из строгого DocumentDto (snake_case) и мапит в camelCase для API.
    """
    doc = await client.get_document_metadata(token, document_id)
    if not doc:
        return {}

    result: dict[str, Any] = {}

    if doc.document_type_id:
        result["documentTypeId"] = str(doc.document_type_id)
    if getattr(doc, "delivery_method_id", None):
        result["deliveryMethodId"] = str(doc.delivery_method_id)

    if doc.pages is not None:
        result["pages"] = doc.pages
    if getattr(doc, "additional_pages", None) is not None:
        result["additionalPages"] = doc.additional_pages
    if getattr(doc, "exemplar_count", None) is not None:
        result["exemplarCount"] = doc.exemplar_count

    if getattr(doc, "invest_program_id", None):
        result["investProgramId"] = str(doc.invest_program_id)
    if doc.note:
        result["note"] = doc.note
    if doc.short_summary:
        result["shortSummary"] = doc.short_summary

    return result


async def _fetch_appeal_required_fields(
    client: DocumentClient, token: str, document_id: str
) -> dict[str, Any]:
    """
    Загружает обязательные поля для DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE.
    Читает из вложенного объекта DocumentDto.document_appeal.
    """
    doc = await client.get_document_metadata(token, document_id)
    if not doc:
        return {"declarantType": "INDIVIDUAL", "submissionForm": "WRITTEN"}

    appeal = getattr(doc, "document_appeal", None)
    result: dict[str, Any] = {}

    if not appeal:
        logger.warning(
            "documentAppeal is None for %s — using fallback INDIVIDUAL/WRITTEN",
            document_id[:8],
        )
        return {"declarantType": "INDIVIDUAL", "submissionForm": "WRITTEN"}

    # === Обязательные поля ===
    declarant = getattr(appeal, "declarant_type", None)
    result["declarantType"] = (
        _enum_value(declarant) if declarant is not None else "INDIVIDUAL"
    )

    sub_form = getattr(appeal, "submission_form", None)
    result["submissionForm"] = (
        _enum_value(sub_form) if sub_form is not None else "WRITTEN"
    )

    # Маппинг snake_case атрибутов Pydantic в camelCase ключи для JSON пейлоада
    _preserve = {
        "fio_applicant": "fioApplicant",
        "organization_name": "organizationName",
        "full_address": "fullAddress",
        "phone": "phone",
        "email": "email",
        "signed": "signed",
        "correspondent_org_number": "correspondentOrgNumber",
        "review_progress": "reviewProgress",
        "country_appeal_name": "countryAppealName",
        "region_name": "regionName",
        "district_name": "districtName",
        "city_name": "cityName",
        "index": "index",
        "index_date_cover_letter": "indexDateCoverLetter",
    }
    for attr, key in _preserve.items():
        val = getattr(appeal, attr, None)
        if val is not None and str(val).strip():
            result[key] = val

    for attr in ("collective", "anonymous", "reasonably"):
        val = getattr(appeal, attr, None)
        if val is not None:
            result[attr] = val

    _id_fields = {
        "citizen_type_id": "citizenTypeId",
        "subject_id": "subjectId",
        "country_appeal_id": "countryAppealId",
        "city_id": "cityId",
        "district_id": "districtId",
        "region_id": "regionId",
        "correspondent_appeal_id": "correspondentAppealId",
        "solution_result_id": "solutionResultId",
    }
    for attr, key in _id_fields.items():
        val = getattr(appeal, attr, None)
        if val is not None:
            result[key] = str(val)

    # Даты
    _date_fields = {
        "receipt_date": "receiptDate",
        "date_doc_correspondent_org": "dateDocCorrespondentOrg",
    }
    for attr, key in _date_fields.items():
        val = getattr(appeal, attr, None)
        if val is not None:
            result[key] = str(val) if not hasattr(val, "isoformat") else val.isoformat()

    return result


# ─── Tool Factory ─────────────────────────────────────────────────────────────


def create_doc_update_field_tool(
    document_client: DocumentClient,
) -> StructuredTool:
    """Фабрика для создания инструмента обновления полей документа."""

    async def doc_update_field(
        updates: dict[str, str],
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Обновляет поля документа через API EDMS (поддерживает массовое обновление).

        Передавай словарь полей, которые нужно изменить. Это эффективнее, чем вызывать
        инструмент для каждого поля отдельно, так как вызывает одну перезагрузку страницы.

        Пример: updates={"phone": "+37529...", "email": "test@mail.ru", "index": "220000"}

        Допустимые поля:
        - Основные: shortSummary (заголовок), note, pages, additionalPages, exemplarCount
        - Обращение: fullAddress, phone, email, index, indexDateCoverLetter, signed,
          correspondentOrgNumber, organizationName, fioApplicant, reviewProgress

        ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ.
        """
        try:
            token = get_token_from_config(config)
            document_id = get_document_id_from_config(config)
        except RuntimeError as exc:
            logger.error("Missing context in tool call: %s", exc)
            return {"status": "error", "message": str(exc)}

        logger.info(
            "doc_update_field: processing batch update for %s: %s",
            document_id[:8],
            list(updates.keys()),
        )

        main_updates: dict[str, Any] = {}
        appeal_updates: dict[str, Any] = {}

        for name, val in updates.items():
            if name == "shortSummary" and len(val) > 80:
                val = val[:80]

            processed_val: Any = val
            if name in ("pages", "additionalPages", "exemplarCount"):
                try:
                    processed_val = int(val)
                except ValueError:
                    return {
                        "status": "error",
                        "message": f"Поле '{name}' должно быть числом, получено: '{val}'",
                    }

            if name in _ALLOWED_APPEAL_FIELDS:
                appeal_updates[name] = processed_val
            else:
                main_updates[name] = processed_val

        try:
            operations = []

            if main_updates:
                existing_main = await _fetch_main_required_fields(
                    document_client, token, document_id
                )
                operations.append({
                    "operationType": "DOCUMENT_MAIN_FIELDS_UPDATE",
                    "body": {**existing_main, **main_updates}
                })

            if appeal_updates:
                existing_appeal = await _fetch_appeal_required_fields(
                    document_client, token, document_id
                )
                operations.append({
                    "operationType": "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE",
                    "body": {**existing_appeal, **appeal_updates}
                })

            if not operations:
                return {"status": "info", "message": "Нет полей для обновления."}

            json_payload = json.loads(json.dumps(operations, cls=CustomJSONEncoder))

            success = await document_client.execute_document_operations(
                token=token,
                document_id=document_id,
                operations=json_payload,
            )

            if not success:
                return {
                    "status": "error",
                    "message": "❌ Не удалось обновить поля: API вернул ошибку",
                }

            labels = []
            for k in updates:
                label = _ALLOWED_FIELDS.get(k) or _ALLOWED_APPEAL_FIELDS.get(k) or k
                labels.append(f"«{label}»")

            return {
                "status": "success",
                "message": f"✅ Успешно обновлены поля: {', '.join(labels)}.",
                "updated_fields": list(updates.keys()),
                "requires_reload": True,
            }

        except Exception as exc:
            logger.error("doc_update_field batch failed: %s", exc, exc_info=True)
            return {
                "status": "error",
                "message": f"❌ Ошибка при обновлении полей: {exc}",
            }

    return StructuredTool.from_function(
        coroutine=doc_update_field,
        name="doc_update_field",
        description="Обновляет поля документа через API EDMS (поддерживает массовое обновление).",
        args_schema=UpdateDocumentFieldInput,
    )
