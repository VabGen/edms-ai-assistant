# edms_ai_assistant/tools/doc_update_field.py
"""
EDMS AI Assistant — Document Field Update Tool.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Annotated, TYPE_CHECKING

from langchain_core.tools import StructuredTool, InjectedToolArg
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.agent.runnable_utils import get_document_id_from_config, get_token_from_config
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder

if TYPE_CHECKING:
    from edms_ai_assistant.clients.document_client import DocumentClient
    from langchain_core.runnables import RunnableConfig

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
    """Validated input for updating a single document field."""
    field_name: str = Field(
        ...,
        description=(
            "Имя поля для обновления. Допустимые значения:\n"
            "Основные: shortSummary (заголовок), note, pages, additionalPages, exemplarCount\n"
            "Обращение: fullAddress, phone, email, index, indexDateCoverLetter, signed, "
            "correspondentOrgNumber, organizationName, fioApplicant, reviewProgress"
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
            result[key] = (
                str(val) if not hasattr(val, "isoformat") else val.isoformat()
            )

    return result


# ─── Tool Factory ─────────────────────────────────────────────────────────────


def create_doc_update_field_tool(
        document_client: DocumentClient,
) -> StructuredTool:
    """Фабрика для создания инструмента обновления поля документа.

    Args:
        document_client: Клиент для работы с документами EDMS.

    Returns:
        Настроенный StructuredTool, готовый к регистрации в агенте.
    """

    async def doc_update_field(
            field_name: str,
            field_value: str,
            config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Обновляет одно поле документа через API EDMS.

        Используй когда пользователь просит:
        - «Поменяй заголовок на "Обращение о..."»  -> field_name="shortSummary"
        - «Измени примечание»                        -> field_name="note"
        - «Поправь адрес заявителя»                  -> field_name="fullAddress"
        - «Обнови телефон в карточке»                -> field_name="phone"
        - «Исправь почтовый индекс»                  -> field_name="index"
        - «Измени ФИО подписанта»                    -> field_name="signed"

        ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ.
        Тебе НЕ НУЖНО запрашивать их у пользователя или передавать в аргументах.

        Args:
            field_name: Имя поля (см. описание).
            field_value: Новое значение.
            config: LangGraph RunnableConfig (инжектируется автоматически).

        Returns:
            Dict со статусом операции.
        """
        try:
            token = get_token_from_config(config)
            document_id = get_document_id_from_config(config)
        except RuntimeError as exc:
            logger.error("Missing context in tool call: %s", exc)
            return {"status": "error", "message": str(exc)}

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
            if is_appeal_field:
                existing = await _fetch_appeal_required_fields(
                    document_client, token, document_id
                )
            else:
                existing = await _fetch_main_required_fields(
                    document_client, token, document_id
                )

            body: dict[str, Any] = {**existing, field_name: value}

            payload = [{"operationType": operation_type, "body": body}]
            json_payload = json.loads(json.dumps(payload, cls=CustomJSONEncoder))

            logger.debug("doc_update_field payload keys: %s", list(body.keys()))

            success = await document_client.execute_document_operations(
                token=token,
                document_id=document_id,
                operations=json_payload,
            )

            if not success:
                return {
                    "status": "error",
                    "message": f"❌ Не удалось обновить поле «{field_name}»: API вернул ошибку",
                }

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

    return StructuredTool.from_function(
        coroutine=doc_update_field,
        name="doc_update_field",
        description="Обновляет одно поле документа через API EDMS.",
        args_schema=UpdateDocumentFieldInput,
    )
