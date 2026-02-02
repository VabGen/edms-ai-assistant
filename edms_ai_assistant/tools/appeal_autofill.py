# edms_ai_assistant/tools/appeal_autofill.py
import logging
import json
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.reference_client import ReferenceClient
from edms_ai_assistant.services.appeal_extraction_service import (
    AppealExtractionService,
)
from edms_ai_assistant.generated.resources_openapi import (
    DocumentDto,
    DeclarantType as GeneratedDeclarantType,
)
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)


class AppealAutofillInput(BaseModel):
    document_id: str = Field(..., description="UUID документа категории APPEAL")
    token: str = Field(..., description="JWT токен авторизации пользователя")
    attachment_id: Optional[str] = Field(
        None,
        description="UUID конкретного вложения для анализа. Если не указан, tool автоматически выберет первый поддерживаемый файл (.docx, .pdf, .txt) из списка вложений документа.",
    )


async def execute_document_operation(
        client: DocumentClient,
        token: str,
        document_id: str,
        operation_type: str,
        body: Dict[str, Any],
) -> None:
    """
    Выполняет операцию над документом через API execute endpoint.
    Использует CustomJSONEncoder для сериализации datetime/UUID/Enum.
    """
    logger.info(
        f"Выполнение операции {operation_type} для документа {document_id}"
    )

    payload = [{"operationType": operation_type, "body": body}]

    json_safe_payload = json.loads(
        json.dumps(payload, cls=CustomJSONEncoder)
    )

    try:
        await client._make_request(
            "POST",
            f"api/document/{document_id}/execute",
            token=token,
            json=json_safe_payload,
        )
        logger.info(f"Операция {operation_type} успешно выполнена")
    except Exception as e:
        logger.error(f"Ошибка выполнения операции {operation_type}: {e}")
        raise


@tool("autofill_appeal_document", args_schema=AppealAutofillInput)
async def autofill_appeal_document(
        document_id: str, token: str, attachment_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Автоматически заполняет поля документа категории APPEAL на основе анализа вложенного файла.

    Процесс:
    1. Проверяет категорию документа (должен быть APPEAL)
    2. Находит подходящее вложение (.docx, .pdf, .txt)
    3. Извлекает текст из вложения
    4. Использует LLM для извлечения структурированных данных
    5. Валидирует данные через справочники EDMS
    6. Обновляет поля документа

    Возвращает детальный отчет о заполненных полях.
    """
    logger.info(
        f"[APPEAL-AUTOFILL] Запуск автозаполнения. Документ: {document_id}"
    )

    filled_fields = []
    warnings = []

    try:
        # ==================== ШАГ 1: Получение метаданных документа ====================
        logger.info(f"[APPEAL-AUTOFILL] Запрос метаданных документа {document_id}")

        async with DocumentClient() as doc_client:
            raw_document = await doc_client.get_document_metadata(token, document_id)
            document = DocumentDto.model_validate(raw_document)

        logger.info(
            f"[APPEAL-AUTOFILL] Документ получен. Категория: {document.docCategoryConstant}"
        )

        if document.docCategoryConstant != "APPEAL":
            return {
                "status": "error",
                "message": f"Документ должен быть категории APPEAL, получен: {document.docCategoryConstant}",
            }

        # ==================== ШАГ 2: Выбор вложения для анализа ====================
        if not document.attachmentDocument:
            return {
                "status": "error",
                "message": "В документе отсутствуют вложения для анализа.",
            }

        if attachment_id:
            target_attachment = next(
                (a for a in document.attachmentDocument if str(a.id) == attachment_id),
                None,
            )
            if not target_attachment:
                return {
                    "status": "error",
                    "message": f"Вложение с ID {attachment_id} не найдено.",
                }
        else:
            supported_extensions = [".docx", ".pdf", ".txt"]
            target_attachment = next(
                (
                    a
                    for a in document.attachmentDocument
                    if any(a.name.lower().endswith(ext) for ext in supported_extensions)
                ),
                None,
            )

            if not target_attachment:
                return {
                    "status": "error",
                    "message": f"Не найдено вложение с поддерживаемым форматом ({', '.join(supported_extensions)})",
                }

        logger.info(f"[APPEAL-AUTOFILL] Выбрано вложение: {target_attachment.name}")

        # ==================== ШАГ 3: Извлечение текста из вложения ====================
        logger.info(
            f"[APPEAL-AUTOFILL] Извлечение текста из: {target_attachment.name}"
        )

        from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
        from edms_ai_assistant.utils.file_utils import extract_text_from_bytes

        async with EdmsAttachmentClient() as attach_client:
            file_bytes = await attach_client.get_attachment_content(
                token, document_id, str(target_attachment.id)
            )

        extracted_text = extract_text_from_bytes(file_bytes, target_attachment.name)

        if not extracted_text or len(extracted_text) < 50:
            return {
                "status": "error",
                "message": "Не удалось извлечь текст из вложения или текст слишком короткий.",
            }

        logger.info(f"[APPEAL-AUTOFILL] Извлечено {len(extracted_text)} символов")

        # ==================== ШАГ 4: Анализ текста с помощью LLM ====================
        logger.info("[APPEAL-AUTOFILL] Запуск LLM анализа...")

        extraction_service = AppealExtractionService()
        appeal_fields = await extraction_service.extract_appeal_fields(extracted_text)

        logger.info(
            f"[APPEAL-AUTOFILL] LLM извлек данные: {appeal_fields.model_dump()}"
        )

        # ==================== ШАГ 5: Валидация через справочники EDMS ====================
        logger.info("[APPEAL-AUTOFILL] Валидация данных через справочники EDMS...")

        validated_ids = {}

        async with ReferenceClient() as ref_client:
            # Валидация вида обращения
            if appeal_fields.citizenType:
                citizen_type_id = await ref_client.find_citizen_type(
                    token, appeal_fields.citizenType
                )
                if citizen_type_id:
                    validated_ids["citizenTypeId"] = citizen_type_id
                    filled_fields.append("citizenType")
                else:
                    warnings.append(
                        f"Вид обращения '{appeal_fields.citizenType}' не найден в справочнике"
                    )

            # Валидация страны
            if appeal_fields.country:
                country_id = await ref_client.find_country(token, appeal_fields.country)
                if country_id:
                    validated_ids["countryId"] = country_id
                    filled_fields.append("country")
                else:
                    warnings.append(
                        f"Страна '{appeal_fields.country}' не найдена в справочнике"
                    )

            # Валидация города
            if appeal_fields.cityName:
                city_id = await ref_client.find_city(token, appeal_fields.cityName)
                if city_id:
                    validated_ids["cityId"] = city_id
                    filled_fields.append("cityName")
                else:
                    warnings.append(
                        f"Город '{appeal_fields.cityName}' не найден в справочнике"
                    )

            # Валидация корреспондента
            if appeal_fields.correspondentAppeal:
                correspondent_id = await ref_client.find_correspondent(
                    token, appeal_fields.correspondentAppeal
                )
                if correspondent_id:
                    validated_ids["correspondentAppealId"] = correspondent_id
                    filled_fields.append("correspondentAppeal")
                else:
                    warnings.append(
                        f"Корреспондент '{appeal_fields.correspondentAppeal}' не найден"
                    )

        # ==================== ШАГ 6: deliveryMethodId (3-level fallback) ====================
        delivery_method_id = None

        if hasattr(document, 'deliveryMethodId') and document.deliveryMethodId:
            delivery_method_id = document.deliveryMethodId
            logger.info(
                f"[APPEAL-AUTOFILL] Используется deliveryMethodId из документа: {delivery_method_id}"
            )
        elif appeal_fields.deliveryMethod:
            async with ReferenceClient() as ref_client:
                delivery_method_id = await ref_client.find_delivery_method(
                    token, appeal_fields.deliveryMethod
                )
                if delivery_method_id:
                    logger.info(
                        f"[APPEAL-AUTOFILL] LLM извлек deliveryMethod: {delivery_method_id}"
                    )
                else:
                    warnings.append(
                        f"Способ доставки '{appeal_fields.deliveryMethod}' не найден"
                    )

        if not delivery_method_id:
            async with ReferenceClient() as ref_client:
                delivery_method_id = await ref_client.find_delivery_method(
                    token, "Курьер"
                )
                if delivery_method_id:
                    logger.info(
                        "[APPEAL-AUTOFILL] Использован fallback deliveryMethodId: Курьер"
                    )
                    warnings.append("Способ доставки установлен по умолчанию: Курьер")

        if not delivery_method_id:
            return {
                "status": "error",
                "error_code": "CRITICAL_ERROR",
                "message": "Критическая ошибка: не удалось определить deliveryMethodId",
                "filled_fields": filled_fields,
                "warnings": warnings,
            }

        # ==================== ШАГ 7: Обновление полей документа ====================
        logger.info("[APPEAL-AUTOFILL] Обновление полей документа...")

        async with DocumentClient() as doc_client:
            main_fields_payload = {
                "shortSummary": appeal_fields.shortSummary,
                "deliveryMethodId": delivery_method_id,
                "documentTypeId": str(document.documentTypeId) if document.documentTypeId else None,
            }

            if appeal_fields.shortSummary:
                filled_fields.append("shortSummary")

            await execute_document_operation(
                doc_client,
                token,
                document_id,
                "DOCUMENT_MAIN_FIELDS_UPDATE",
                main_fields_payload,
            )

            # 7.2 Конвертация DeclarantType
            converted_declarant_type = None
            if appeal_fields.declarantType:
                try:
                    converted_declarant_type = GeneratedDeclarantType(
                        appeal_fields.declarantType.value
                    )
                except ValueError:
                    try:
                        converted_declarant_type = GeneratedDeclarantType[
                            appeal_fields.declarantType.value
                        ]
                    except (KeyError, ValueError):
                        logger.warning(
                            f"Не удалось конвертировать declarantType: {appeal_fields.declarantType.value}"
                        )
                        warnings.append(
                            "Тип заявителя не распознан. Заполните вручную."
                        )

            # 7.3 Обновление полей обращения
            appeal_payload = {
                "declarantType": converted_declarant_type.value if converted_declarant_type else None,
                "receiptDate": appeal_fields.receiptDate,
                "citizenTypeId": validated_ids.get("citizenTypeId"),
                "collective": appeal_fields.collective,
                "anonymous": appeal_fields.anonymous,
                "fioApplicant": appeal_fields.fioApplicant,
                "organizationName": appeal_fields.organizationName,
                "countryAppealId": validated_ids.get("countryId"),
                "countryAppealName": appeal_fields.country,
                "regionName": appeal_fields.regionName,
                "districtName": appeal_fields.districtName,
                "cityId": validated_ids.get("cityId"),
                "cityName": appeal_fields.cityName,
                "index": appeal_fields.index,
                "fullAddress": appeal_fields.fullAddress,
                "phone": appeal_fields.phone,
                "email": appeal_fields.email,
                "signed": appeal_fields.signed,
                "correspondentOrgNumber": appeal_fields.correspondentOrgNumber,
                "dateDocCorrespondentOrg": appeal_fields.dateDocCorrespondentOrg,
                "correspondentAppeal": appeal_fields.correspondentAppeal,
                "correspondentAppealId": validated_ids.get("correspondentAppealId"),
                "indexDateCoverLetter": appeal_fields.indexDateCoverLetter,
                "reviewProgress": appeal_fields.reviewProgress,
                "reasonably": appeal_fields.reasonably,
            }

            appeal_payload = {k: v for k, v in appeal_payload.items() if v is not None}

            await execute_document_operation(
                doc_client,
                token,
                document_id,
                "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE",
                appeal_payload,
            )

        # ==================== ШАГ 8: Формирование результата ====================
        status = "success" if not warnings else "partial_success"

        result = {
            "status": status,
            "message": f"Автозаполнение завершено. Обработано полей: {len(filled_fields)}",
            "filled_count": len(filled_fields),
            "filled_fields": filled_fields,
            "warnings": warnings,
            "extracted_data": {
                "fio": appeal_fields.fioApplicant,
                "organization": appeal_fields.organizationName,
                "city": appeal_fields.cityName,
                "summary": appeal_fields.shortSummary,
            },
        }

        logger.info(f"[APPEAL-AUTOFILL] ✅ Успешно завершено: {result}")
        return result

    except Exception as e:
        logger.error(
            f"[APPEAL-AUTOFILL] Критическая ошибка: {e}", exc_info=True
        )
        return {
            "status": "error",
            "message": f"Критическая ошибка: {str(e)}",
            "error_code": "CRITICAL_ERROR",
            "filled_fields": filled_fields,
            "warnings": warnings,
        }