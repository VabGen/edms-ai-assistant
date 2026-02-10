# edms_ai_assistant/tools/appeal_autofill.py
"""
Инструмент автоматического заполнения карточки обращения APPEAL.
- DocumentTool.updateDocumentAppealFields()
- DocumentMainFieldsAppealOperationExecutor.doAccept()
"""
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.reference_client import ReferenceClient
from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.services.appeal_extraction_service import AppealExtractionService
from edms_ai_assistant.utils.file_utils import extract_text_from_bytes
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder
from edms_ai_assistant.generated.resources_openapi import (
    DocumentDto,
    DeclarantType as GeneratedDeclarantType,
)

logger = logging.getLogger(__name__)


class AppealAutofillInput(BaseModel):

    document_id: str = Field(..., description="UUID документа категории APPEAL")
    token: str = Field(..., description="JWT токен авторизации пользователя")
    attachment_id: Optional[str] = Field(
        None, description="UUID конкретного вложения для анализа"
    )


def is_empty(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        trimmed = value.strip()

        if not trimmed:
            return True

        placeholders = {
            "none",
            "null",
            "nil",
            "n/a",
            "na",
            "unknown",
            "not specified",
            "no",
            "нет",
            "неизвестно",
            "н/д",
        }

        if trimmed.lower() in placeholders:
            return True

    return False


def sanitize_string(value: Optional[str]) -> Optional[str]:
    if is_empty(value):
        return None

    cleaned = (
        value.replace('"', '"')
        .replace('"', '"')
        .replace("„", '"')
        .replace("«", '"')
        .replace("»", '"')
        .strip()
    )

    return cleaned if cleaned else None


def fix_datetime_format(dt: Any) -> Optional[str]:
    if dt is None:
        return None

    if isinstance(dt, str):
        dt = dt.replace(" ", "T")
        if not dt.endswith("Z") and "+00:00" in dt:
            dt = dt.replace("+00:00", "Z")
        return dt

    if isinstance(dt, datetime):
        # Datetime → ISO 8601
        if dt.tzinfo is not None:
            return dt.isoformat()
        else:
            return dt.isoformat() + "Z"

    return None


async def execute_document_operation(
    client: DocumentClient,
    token: str,
    document_id: str,
    operation_type: str,
    body: Dict[str, Any],
) -> None:
    payload = [{"operationType": operation_type, "body": body}]
    json_safe_payload = json.loads(json.dumps(payload, cls=CustomJSONEncoder))

    logger.debug(f"[APPEAL-AUTOFILL] Операция: {operation_type}")
    logger.debug(
        f"[APPEAL-AUTOFILL] Payload:\n{json.dumps(json_safe_payload, indent=2, ensure_ascii=False)}"
    )

    try:
        await client._make_request(
            "POST",
            f"api/document/{document_id}/execute",
            token=token,
            json=json_safe_payload,
        )
        logger.info(f"[APPEAL-AUTOFILL] {operation_type} — успешно выполнена")
    except Exception as e:
        logger.error(f"[APPEAL-AUTOFILL] {operation_type} — ошибка: {e}", exc_info=True)
        raise


@tool("autofill_appeal_document", args_schema=AppealAutofillInput)
async def autofill_appeal_document(
    document_id: str, token: str, attachment_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Автоматически заполняет карточку обращения (APPEAL) на основе анализа вложенного файла.

    Процесс:
    1. Загрузка метаданных документа из EDMS
    2. Извлечение текста из вложения (PDF/DOCX/TXT)
    3. LLM-анализ текста → AppealFields
    4. Поиск ID в справочниках (страна, регион, город, корреспондент и т.д.)
    5. Выполнение операций:
       - DOCUMENT_MAIN_FIELDS_UPDATE (краткое содержание, способ доставки)
       - DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE (все поля обращения)

    Args:
        document_id: UUID документа категории APPEAL
        token: JWT токен авторизации
        attachment_id: UUID конкретного файла (опционально)

    Returns:
        Dict со статусом выполнения и сообщением
    """
    logger.info(f"[APPEAL-AUTOFILL] ========== СТАРТ ==========")
    logger.info(f"[APPEAL-AUTOFILL] DocumentID: {document_id}")
    warnings = []

    try:
        # ========== 1. ПОЛУЧЕНИЕ МЕТАДАННЫХ ДОКУМЕНТА ==========
        async with DocumentClient() as doc_client:
            raw_document = await doc_client.get_document_metadata(token, document_id)
            document = DocumentDto.model_validate(raw_document)

        logger.info(f"[APPEAL-AUTOFILL] Категория: {document.docCategoryConstant}")

        if document.docCategoryConstant != "APPEAL":
            return {
                "status": "error",
                "message": f"Документ должен быть категории APPEAL, а не {document.docCategoryConstant}",
            }

        if not document.attachmentDocument:
            return {"status": "error", "message": "В документе отсутствуют вложения"}

        # ========== 2. ВЫБОР ВЛОЖЕНИЯ ==========
        target = None
        if attachment_id:
            target = next(
                (a for a in document.attachmentDocument if str(a.id) == attachment_id),
                None,
            )
            if not target:
                warnings.append(
                    f"Вложение ID={attachment_id} не найдено, используется автоподбор"
                )

        if not target:
            supported_ext = (".pdf", ".docx", ".txt", ".doc", ".rtf")
            target = next(
                (
                    a
                    for a in document.attachmentDocument
                    if a.name.lower().endswith(supported_ext)
                ),
                document.attachmentDocument[0],
            )

        logger.info(f"[APPEAL-AUTOFILL] Вложение: {target.name}")

        # ========== 3. ИЗВЛЕЧЕНИЕ ТЕКСТА ==========
        async with EdmsAttachmentClient() as attach_client:
            file_bytes = await attach_client.get_attachment_content(
                token, document_id, str(target.id)
            )

        extracted_text = extract_text_from_bytes(file_bytes, target.name)
        if not extracted_text or len(extracted_text) < 50:
            return {
                "status": "error",
                "message": "Текст не извлечен или слишком короткий",
            }

        logger.info(f"[APPEAL-AUTOFILL] Извлечено: {len(extracted_text)} символов")

        # ========== 4. LLM АНАЛИЗ ==========
        extraction_service = AppealExtractionService()
        fields = await extraction_service.extract_appeal_fields(extracted_text)

        # logger.info(f"[APPEAL-AUTOFILL] ========== LLM РЕЗУЛЬТАТЫ ==========")
        # logger.info(f"[APPEAL-AUTOFILL]   - declarantType: {fields.declarantType or 'н/д'}")
        # logger.info(f"[APPEAL-AUTOFILL]   - fioApplicant: {fields.fioApplicant or 'н/д'}")
        # logger.info(f"[APPEAL-AUTOFILL]   - organizationName: {fields.organizationName or 'н/д'}")
        # logger.info(f"[APPEAL-AUTOFILL]   - signed: {fields.signed or 'н/д'}")
        # logger.info(
        #     f"[APPEAL-AUTOFILL]   - shortSummary: {(fields.shortSummary[:50] + '...') if fields.shortSummary else 'н/д'}")
        # logger.info(f"[APPEAL-AUTOFILL] ==========================================")

        # ========== 5. ПОДГОТОВКА ДАННЫХ ==========
        async with ReferenceClient() as ref_client:

            # ===== 5.1. ОПЕРАЦИЯ 1: DOCUMENT_MAIN_FIELDS_UPDATE =====
            delivery_id = document.deliveryMethodId
            if not delivery_id:
                delivery_method_name = (
                    fields.deliveryMethod
                    if not is_empty(fields.deliveryMethod)
                    else "Курьер"
                )
                delivery_id = await ref_client.find_delivery_method(
                    token, delivery_method_name
                )
                logger.info(
                    f"[APPEAL-AUTOFILL] Способ доставки: {delivery_method_name} → ID: {delivery_id}"
                )

            main_payload = {
                "shortSummary": (
                    sanitize_string(fields.shortSummary)
                    if not is_empty(fields.shortSummary)
                    else document.shortSummary
                ),
                "deliveryMethodId": delivery_id,
                "documentTypeId": (
                    str(document.documentTypeId) if document.documentTypeId else None
                ),
                "pages": document.pages,
                "note": document.note,
                "additionalPages": document.additionalPages,
                "exemplarCount": document.exemplarCount,
                "investProgramId": document.investProgramId,
            }

            main_payload = {k: v for k, v in main_payload.items() if v is not None}

            await execute_document_operation(
                doc_client,
                token,
                document_id,
                "DOCUMENT_MAIN_FIELDS_UPDATE",
                main_payload,
            )

            # ===== 5.2. ОПЕРАЦИЯ 2: DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE =====
            d = document.documentAppeal

            # logger.info(f"[APPEAL-AUTOFILL] ========== СОСТОЯНИЕ БД ==========")
            # if d:
            #     logger.info(f"[APPEAL-AUTOFILL]   - declarantType (БД): {d.declarantType or 'н/д'}")
            #     logger.info(f"[APPEAL-AUTOFILL]   - fioApplicant (БД): {d.fioApplicant or 'н/д'}")
            #     logger.info(f"[APPEAL-AUTOFILL]   - organizationName (БД): {d.organizationName or 'н/д'}")
            # else:
            #     logger.info(f"[APPEAL-AUTOFILL]   - documentAppeal: NULL (будет создан)")
            # logger.info(f"[APPEAL-AUTOFILL] ==========================================")

            if d is None:
                d = type(
                    "EmptyDocumentAppeal",
                    (),
                    {
                        "fioApplicant": None,
                        "countryAppealId": None,
                        "countryAppealName": None,
                        "regionId": None,
                        "regionName": None,
                        "districtId": None,
                        "districtName": None,
                        "cityId": None,
                        "cityName": None,
                        "correspondentAppealId": None,
                        "correspondentAppeal": None,
                        "citizenTypeId": None,
                        "declarantType": None,
                        "collective": None,
                        "anonymous": None,
                        "reasonably": None,
                        "receiptDate": None,
                        "dateDocCorrespondentOrg": None,
                        "organizationName": None,
                        "fullAddress": None,
                        "phone": None,
                        "email": None,
                        "index": None,
                        "signed": None,
                        "correspondentOrgNumber": None,
                        "indexDateCoverLetter": None,
                        "reviewProgress": None,
                        "subjectId": None,
                        "solutionResultId": None,
                        "nomenclatureAffairId": None,
                    },
                )()

            f = {}

            # ========== ГЕОГРАФИЯ ==========

            # ===== COUNTRY =====
            country_to_search = None

            if d.countryAppealName and not is_empty(d.countryAppealName):
                country_to_search = d.countryAppealName
            elif not is_empty(fields.country):
                country_to_search = fields.country

            if country_to_search:
                try:
                    country_data = await ref_client.find_country_with_name(
                        token, country_to_search
                    )
                    if country_data:
                        f["countryAppealId"] = country_data["id"]
                        f["countryAppealName"] = country_data["name"]
                        logger.info(
                            f"[APPEAL-AUTOFILL] Страна: '{country_to_search}' → '{country_data['name']}' (ID: {country_data['id']})"
                        )
                    else:
                        logger.warning(
                            f"[APPEAL-AUTOFILL] Страна '{country_to_search}' не найдена в справочнике"
                        )
                except Exception as e:
                    logger.error(f"[APPEAL-AUTOFILL] Ошибка поиска страны: {e}")

            elif d.countryAppealId:
                f["countryAppealId"] = str(d.countryAppealId)
                logger.warning(
                    f"[APPEAL-AUTOFILL] Страна: есть только ID={d.countryAppealId}, название не найдено"
                )

            # ===== CITY =====
            city_to_search = None

            if d.cityName and not is_empty(d.cityName):
                city_to_search = d.cityName
            elif not is_empty(fields.cityName):
                city_to_search = fields.cityName

            if city_to_search:
                try:
                    city_data = await ref_client.find_city_with_hierarchy(
                        token, city_to_search
                    )

                    if city_data:
                        # Сохраняем город
                        f["cityId"] = city_data["id"]
                        f["cityName"] = city_data["name"]
                        logger.info(
                            f"[APPEAL-AUTOFILL] Город: '{city_to_search}' → '{city_data['name']}' (ID: {city_data['id']})"
                        )

                        if city_data.get("regionId") and city_data.get("regionName"):
                            has_explicit_region = (
                                d.regionName and not is_empty(d.regionName)
                            ) or (fields.regionName and not is_empty(fields.regionName))

                            if not has_explicit_region:
                                # Автозаполнение из иерархии
                                f["regionId"] = city_data["regionId"]
                                f["regionName"] = city_data["regionName"]
                                logger.info(
                                    f"[APPEAL-AUTOFILL] ✅ Регион автоопределен из иерархии города: {city_data['regionName']}"
                                )
                            else:
                                logger.debug(
                                    f"[APPEAL-AUTOFILL] Регион явно указан в документе, пропускаем автозаполнение"
                                )

                        if city_data.get("districtId") and city_data.get(
                            "districtName"
                        ):
                            # Проверяем: есть ли уже явно указанный район?
                            has_explicit_district = (
                                d.districtName and not is_empty(d.districtName)
                            ) or (
                                fields.districtName
                                and not is_empty(fields.districtName)
                            )

                            if not has_explicit_district:
                                # Автозаполнение из иерархии
                                f["districtId"] = city_data["districtId"]
                                f["districtName"] = city_data["districtName"]
                                logger.info(
                                    f"[APPEAL-AUTOFILL] Район автоопределен из иерархии города: {city_data['districtName']}"
                                )
                            else:
                                logger.debug(
                                    f"[APPEAL-AUTOFILL] Район явно указан в документе, пропускаем автозаполнение"
                                )
                    else:
                        logger.warning(
                            f"[APPEAL-AUTOFILL] Город '{city_to_search}' не найден в справочнике"
                        )
                except Exception as e:
                    logger.error(f"[APPEAL-AUTOFILL] Ошибка поиска города: {e}")

            elif d.cityId:
                f["cityId"] = str(d.cityId)
                logger.warning(
                    f"[APPEAL-AUTOFILL] Город: есть только ID={d.cityId}, название не найдено"
                )

            # ===== REGION =====
            if "regionId" not in f:
                region_to_search = None

                if d.regionName and not is_empty(d.regionName):
                    region_to_search = d.regionName
                elif not is_empty(fields.regionName):
                    region_to_search = fields.regionName

                if region_to_search:
                    try:
                        region_data = await ref_client.find_region_with_name(
                            token, region_to_search
                        )
                        if region_data:
                            f["regionId"] = region_data["id"]
                            f["regionName"] = region_data["name"]
                            logger.info(
                                f"[APPEAL-AUTOFILL] Регион (явно указан): '{region_to_search}' → '{region_data['name']}' (ID: {region_data['id']})"
                            )
                        else:
                            logger.warning(
                                f"[APPEAL-AUTOFILL] Регион '{region_to_search}' не найден в справочнике"
                            )
                    except Exception as e:
                        logger.error(f"[APPEAL-AUTOFILL] Ошибка поиска региона: {e}")

                elif d.regionId:
                    f["regionId"] = str(d.regionId)
                    logger.warning(
                        f"[APPEAL-AUTOFILL] Регион: есть только ID={d.regionId}, название не найдено"
                    )

            # ===== DISTRICT =====
            if "districtId" not in f:
                district_to_search = None

                if d.districtName and not is_empty(d.districtName):
                    district_to_search = d.districtName
                elif not is_empty(fields.districtName):
                    district_to_search = fields.districtName

                if district_to_search:
                    try:
                        district_data = await ref_client.find_district_with_name(
                            token, district_to_search
                        )
                        if district_data:
                            f["districtId"] = district_data["id"]
                            f["districtName"] = district_data["name"]
                            logger.info(
                                f"[APPEAL-AUTOFILL] Район (явно указан): '{district_to_search}' → '{district_data['name']}' (ID: {district_data['id']})"
                            )
                        else:
                            logger.warning(
                                f"[APPEAL-AUTOFILL] Район '{district_to_search}' не найден в справочнике"
                            )
                    except Exception as e:
                        logger.error(f"[APPEAL-AUTOFILL] Ошибка поиска района: {e}")

                elif d.districtId:
                    f["districtId"] = str(d.districtId)
                    logger.warning(
                        f"[APPEAL-AUTOFILL] Район: есть только ID={d.districtId}, название не найдено"
                    )

            # ========== КОРРЕСПОНДЕНТ ==========
            if d.correspondentAppealId:
                f["correspondentAppealId"] = str(d.correspondentAppealId)
                f["correspondentAppeal"] = sanitize_string(d.correspondentAppeal)
            elif not is_empty(d.correspondentAppeal):
                corr_id = await ref_client.find_correspondent(
                    token, d.correspondentAppeal
                )
                if corr_id:
                    f["correspondentAppealId"] = corr_id
                    f["correspondentAppeal"] = sanitize_string(d.correspondentAppeal)
            elif not is_empty(fields.correspondentAppeal):
                corr_id = await ref_client.find_correspondent(
                    token, fields.correspondentAppeal
                )
                if corr_id:
                    f["correspondentAppealId"] = corr_id
                    f["correspondentAppeal"] = sanitize_string(
                        fields.correspondentAppeal
                    )

            if (
                "correspondentAppealId" not in f
                or f.get("correspondentAppealId") is None
            ):
                f["correspondentAppealId"] = None
                f["correspondentAppeal"] = None

            # ========== ОСНОВНЫЕ ПОЛЯ ==========
            # fioApplicant
            if not is_empty(d.fioApplicant):
                f["fioApplicant"] = sanitize_string(d.fioApplicant)
            elif not is_empty(fields.fioApplicant):
                f["fioApplicant"] = sanitize_string(fields.fioApplicant)

            # citizenTypeId
            if d.citizenTypeId:
                f["citizenTypeId"] = str(d.citizenTypeId)
            elif not is_empty(fields.citizenType):
                citizen_type_id = await ref_client.find_citizen_type(
                    token, fields.citizenType
                )
                if citizen_type_id:
                    f["citizenTypeId"] = citizen_type_id

            # dateDocCorrespondentOrg (для ENTITY)
            if d.dateDocCorrespondentOrg:
                f["dateDocCorrespondentOrg"] = fix_datetime_format(
                    d.dateDocCorrespondentOrg
                )
            elif not is_empty(fields.dateDocCorrespondentOrg):
                f["dateDocCorrespondentOrg"] = fix_datetime_format(
                    fields.dateDocCorrespondentOrg
                )

            # ========== ТЕМАТИКА (SUBJECT) ==========
            if d.subjectId:
                f["subjectId"] = str(d.subjectId)
            else:
                subject_id = await ref_client.find_best_subject(token, extracted_text)
                if subject_id:
                    f["subjectId"] = subject_id
                    logger.info(f"[APPEAL-AUTOFILL] Тема определена LLM: {subject_id}")
                else:
                    logger.warning("[APPEAL-AUTOFILL] Тема не определена LLM")

            # ========== КАТЕГОРИЯ ЗАЯВИТЕЛЯ (declarantType) ==========
            if fields.declarantType:
                declarant_value = fields.declarantType

                if isinstance(declarant_value, str):
                    try:
                        f["declarantType"] = GeneratedDeclarantType[
                            declarant_value.upper()
                        ]
                        logger.info(
                            f"[APPEAL-AUTOFILL] declarantType из LLM (str→enum): {declarant_value} → {f['declarantType']}"
                        )
                    except KeyError:
                        logger.warning(
                            f"[APPEAL-AUTOFILL] Неизвестное значение declarantType от LLM: '{declarant_value}', используем fallback"
                        )
                        f["declarantType"] = GeneratedDeclarantType.INDIVIDUAL
                else:
                    f["declarantType"] = declarant_value
                    logger.info(
                        f"[APPEAL-AUTOFILL] declarantType из LLM (enum): {f['declarantType']}"
                    )

            elif d.declarantType:
                # Берем из БД только если LLM не определил
                f["declarantType"] = d.declarantType
                logger.info(f"[APPEAL-AUTOFILL] declarantType из БД: {d.declarantType}")

            else:
                # FALLBACK: если ни LLM, ни БД не определили
                f["declarantType"] = GeneratedDeclarantType.INDIVIDUAL
                warnings.append(
                    "declarantType не определен автоматически, установлен INDIVIDUAL по умолчанию"
                )
                logger.warning(
                    "[APPEAL-AUTOFILL] declarantType установлен INDIVIDUAL (fallback)"
                )

            # ========== CONDITIONAL FIELDS ==========
            if f.get("declarantType"):
                current_declarant_type = f["declarantType"]

                if current_declarant_type == GeneratedDeclarantType.ENTITY:
                    f["organizationName"] = sanitize_string(
                        d.organizationName
                        if not is_empty(d.organizationName)
                        else fields.organizationName
                    )
                    f["signed"] = sanitize_string(
                        d.signed if not is_empty(d.signed) else fields.signed
                    )
                    f["correspondentOrgNumber"] = sanitize_string(
                        d.correspondentOrgNumber
                        if not is_empty(d.correspondentOrgNumber)
                        else fields.correspondentOrgNumber
                    )

                elif current_declarant_type == GeneratedDeclarantType.INDIVIDUAL:
                    f["organizationName"] = None
                    f["signed"] = None
                    f["correspondentOrgNumber"] = None

            # Boolean
            f["collective"] = (
                True
                if (fields.collective is True)
                else (d.collective if d.collective is not None else fields.collective)
            )
            f["anonymous"] = (
                True
                if (fields.anonymous is True)
                else (d.anonymous if d.anonymous is not None else fields.anonymous)
            )
            f["reasonably"] = (
                True
                if (fields.reasonably is True)
                else (d.reasonably if d.reasonably is not None else fields.reasonably)
            )

            # Даты
            f["receiptDate"] = fix_datetime_format(
                d.receiptDate if d.receiptDate else fields.receiptDate
            )

            # Строковые поля (общие для INDIVIDUAL и ENTITY)
            f["fullAddress"] = sanitize_string(
                d.fullAddress if not is_empty(d.fullAddress) else fields.fullAddress
            )
            f["phone"] = sanitize_string(
                d.phone if not is_empty(d.phone) else fields.phone
            )
            f["email"] = sanitize_string(
                d.email if not is_empty(d.email) else fields.email
            )
            f["index"] = sanitize_string(
                d.index if not is_empty(d.index) else fields.index
            )
            f["indexDateCoverLetter"] = sanitize_string(
                d.indexDateCoverLetter
                if not is_empty(d.indexDateCoverLetter)
                else fields.indexDateCoverLetter
            )
            f["reviewProgress"] = sanitize_string(
                d.reviewProgress
                if not is_empty(d.reviewProgress)
                else fields.reviewProgress
            )

            # Поля только из БД
            if d.subjectId:
                f["subjectId"] = str(d.subjectId)
            if d.solutionResultId:
                f["solutionResultId"] = str(d.solutionResultId)
            if d.nomenclatureAffairId:
                f["nomenclatureAffairId"] = str(d.nomenclatureAffairId)

            # ========== ВАЛИДАЦИЯ ПЕРЕД ОТПРАВКОЙ ==========
            if not f.get("declarantType"):
                return {
                    "status": "error",
                    "message": "КРИТИЧЕСКАЯ ОШИБКА: declarantType обязателен, но не установлен",
                }

            # ========== ФИЛЬТРАЦИЯ PAYLOAD ==========
            appeal_payload = {}
            for k, v in f.items():
                if k in ["correspondentAppeal", "correspondentAppealId"]:
                    appeal_payload[k] = v
                elif v is not None and not is_empty(v):
                    appeal_payload[k] = v

            logger.info(
                f"[APPEAL-AUTOFILL] Финальный payload DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE:"
            )
            logger.info(
                json.dumps(appeal_payload, indent=2, ensure_ascii=False, default=str)
            )

            await execute_document_operation(
                doc_client,
                token,
                document_id,
                "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE",
                appeal_payload,
            )

        logger.info(f"[APPEAL-AUTOFILL] ========== УСПЕХ ==========")

        return {
            "status": "success",
            "message": "Документ успешно заполнен",
            "warnings": warnings,
            "attachment_used": target.name,
        }

    except Exception as e:
        logger.error(f"[APPEAL-AUTOFILL] ========== ОШИБКА ==========", exc_info=True)
        return {"status": "error", "message": f"Ошибка автозаполнения: {str(e)}"}
