"""
EDMS AI Assistant - Appeal Autofill Tool

Architecture:
- AppealAutofillOrchestrator: Главный координатор процесса
- GeographyResolver: Резолвинг географических сущностей
- AppealFieldsBuilder: Построение payload для API
- ValueSanitizer: Очистка и валидация данных
- DocumentOperationExecutor: Выполнение операций над документами
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.reference_client import ReferenceClient
from edms_ai_assistant.generated.resources_openapi import (
    DeclarantType as GeneratedDeclarantType,
)
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.appeal_extraction_service import AppealExtractionService
from edms_ai_assistant.utils.file_utils import extract_text_from_bytes
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)


class AppealAutofillInput(BaseModel):
    """Валидированная схема входных данных для автозаполнения."""

    document_id: str = Field(
        ...,
        description="UUID документа категории APPEAL",
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )
    token: str = Field(..., description="JWT токен авторизации пользователя")
    attachment_id: Optional[str] = Field(
        None,
        description="UUID конкретного вложения для анализа",
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )

    @field_validator("attachment_id")
    @classmethod
    def validate_attachment_id(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.strip():
            return None
        return v


@dataclass(frozen=True)
class AutofillResult:
    """Immutable результат автозаполнения."""

    status: str
    message: str
    warnings: Optional[List[str]] = None
    attachment_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"status": self.status, "message": self.message}
        if self.warnings:
            result["warnings"] = self.warnings
        if self.attachment_used:
            result["attachment_used"] = self.attachment_used
        return result


class ValueSanitizer:
    """Утилиты для очистки и валидации данных."""

    EMPTY_PLACEHOLDERS = {
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

    @classmethod
    def is_empty(cls, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed or trimmed.lower() in cls.EMPTY_PLACEHOLDERS:
                return True
        return False

    @classmethod
    def sanitize_string(cls, value: Optional[str]) -> Optional[str]:
        if cls.is_empty(value):
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

    @classmethod
    def fix_datetime_format(cls, dt: Any) -> Optional[str]:
        if dt is None:
            return None
        if isinstance(dt, str):
            dt = dt.replace(" ", "T")
            if not dt.endswith("Z") and "+00:00" in dt:
                dt = dt.replace("+00:00", "Z")
            return dt
        if isinstance(dt, datetime):
            return dt.isoformat() if dt.tzinfo else dt.isoformat() + "Z"
        return None


class DocumentOperationExecutor:
    """Executor для выполнения операций над документами."""

    @staticmethod
    async def execute(
        client: DocumentClient,
        token: str,
        document_id: str,
        operation_type: str,
        body: Dict[str, Any],
    ) -> None:
        payload = [{"operationType": operation_type, "body": body}]
        json_safe_payload = json.loads(json.dumps(payload, cls=CustomJSONEncoder))

        logger.debug(
            f"Executing {operation_type}",
            extra={"document_id": document_id, "operation": operation_type},
        )

        try:
            await client._make_request(
                "POST",
                f"api/document/{document_id}/execute",
                token=token,
                json=json_safe_payload,
            )
            logger.info(f"{operation_type} executed successfully")
        except Exception as e:
            logger.error(f"{operation_type} failed: {e}", exc_info=True)
            raise


class AttachmentSelector:
    """Селектор для выбора подходящего вложения."""

    SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".doc", ".rtf")

    @classmethod
    def select(cls, document: DocumentDto, attachment_id: Optional[str]) -> tuple:
        warnings = []

        if not document.attachmentDocument:
            raise ValueError("В документе отсутствуют вложения")

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
            target = next(
                (
                    a
                    for a in document.attachmentDocument
                    if a.name.lower().endswith(cls.SUPPORTED_EXTENSIONS)
                ),
                document.attachmentDocument[0],
            )

        logger.info(f"Attachment selected: {target.name}")
        return target, warnings


class GeographyResolver:
    """Резолвер географических сущностей с поддержкой иерархии."""

    def __init__(self, ref_client: ReferenceClient, token: str):
        self.ref_client = ref_client
        self.token = token

    async def resolve_geography(self, document, fields) -> Dict[str, Any]:
        geo_data = {}

        await self._resolve_country(document, fields, geo_data)
        await self._resolve_city_with_hierarchy(document, fields, geo_data)

        if "regionId" not in geo_data:
            await self._resolve_region(document, fields, geo_data)
        if "districtId" not in geo_data:
            await self._resolve_district(document, fields, geo_data)

        return geo_data

    async def _resolve_country(self, document, fields, geo_data: Dict) -> None:
        d = document.documentAppeal
        country_name = None

        if (
            d
            and d.countryAppealName
            and not ValueSanitizer.is_empty(d.countryAppealName)
        ):
            country_name = d.countryAppealName
        elif not ValueSanitizer.is_empty(fields.country):
            country_name = fields.country

        if country_name:
            try:
                country_data = await self.ref_client.find_country_with_name(
                    self.token, country_name
                )
                if country_data:
                    geo_data["countryAppealId"] = country_data["id"]
                    geo_data["countryAppealName"] = country_data["name"]
                    logger.info(
                        f"Country resolved: {country_name} → {country_data['name']}"
                    )
                else:
                    logger.warning(f"Country not found: {country_name}")
            except Exception as e:
                logger.error(f"Country resolution error: {e}")
        elif d and d.countryAppealId:
            geo_data["countryAppealId"] = str(d.countryAppealId)

    async def _resolve_city_with_hierarchy(
        self, document, fields, geo_data: Dict
    ) -> None:
        d = document.documentAppeal
        city_name = None

        if d and d.cityName and not ValueSanitizer.is_empty(d.cityName):
            city_name = d.cityName
            logger.debug(f"City name from DB: {city_name}")
        elif not ValueSanitizer.is_empty(fields.cityName):
            city_name = fields.cityName
            logger.debug(f"City name from LLM: {city_name}")

        if city_name:
            try:
                logger.info(f"Calling find_city_with_hierarchy API for: {city_name}")
                city_data = await self.ref_client.find_city_with_hierarchy(
                    self.token, city_name
                )

                if city_data:
                    geo_data["cityId"] = city_data["id"]
                    geo_data["cityName"] = city_data["name"]
                    logger.info(
                        f"City resolved: {city_name} → {city_data['name']} (ID: {city_data['id']})"
                    )

                    api_has_region = city_data.get("regionId") and city_data.get(
                        "regionName"
                    )

                    logger.debug(
                        f"Region sources: "
                        f"DB_ID={bool(d and d.regionId)}, "
                        f"DB_NAME={bool(d and d.regionName and not ValueSanitizer.is_empty(d.regionName))}, "
                        f"LLM_NAME={bool(fields.regionName and not ValueSanitizer.is_empty(fields.regionName))}, "
                        f"API={api_has_region}"
                    )

                    if api_has_region:
                        geo_data["regionId"] = city_data["regionId"]
                        geo_data["regionName"] = city_data["regionName"]
                        logger.info(
                            f"Region AUTO-RESOLVED from city API: "
                            f"{city_data['regionName']} (ID: {city_data['regionId']})"
                        )
                    else:
                        logger.warning(
                            f"City API response does not contain regionId/regionName"
                        )

                    api_has_district = city_data.get("districtId") and city_data.get(
                        "districtName"
                    )

                    logger.debug(
                        f"District sources: "
                        f"DB_ID={bool(d and d.districtId)}, "
                        f"DB_NAME={bool(d and d.districtName and not ValueSanitizer.is_empty(d.districtName))}, "
                        f"LLM_NAME={bool(fields.districtName and not ValueSanitizer.is_empty(fields.districtName))}, "
                        f"API={api_has_district}"
                    )

                    if api_has_district:
                        geo_data["districtId"] = city_data["districtId"]
                        geo_data["districtName"] = city_data["districtName"]
                        logger.info(
                            f"District AUTO-RESOLVED from city API: "
                            f"{city_data['districtName']} (ID: {city_data['districtId']})"
                        )
                    else:
                        logger.debug(
                            f"City does not have district (normal for some cities)"
                        )
                else:
                    logger.warning(f"City not found in reference: {city_name}")

            except Exception as e:
                logger.error(f"City hierarchy resolution error: {e}", exc_info=True)

        # Fallback: Если есть только cityId в БД без имени
        elif d and d.cityId:
            geo_data["cityId"] = str(d.cityId)
            logger.warning(
                f"Using existing cityId from DB without API call: {d.cityId}"
            )

    async def _resolve_region(self, document, fields, geo_data: Dict) -> None:
        d = document.documentAppeal
        region_name = None

        if d and d.regionName and not ValueSanitizer.is_empty(d.regionName):
            region_name = d.regionName
        elif not ValueSanitizer.is_empty(fields.regionName):
            region_name = fields.regionName

        if region_name:
            try:
                region_data = await self.ref_client.find_region_with_name(
                    self.token, region_name
                )
                if region_data:
                    geo_data["regionId"] = region_data["id"]
                    geo_data["regionName"] = region_data["name"]
                    logger.info(f"Region resolved explicitly: {region_name}")
                else:
                    logger.warning(f"Region not found: {region_name}")
            except Exception as e:
                logger.error(f"Region resolution error: {e}")
        elif d and d.regionId:
            geo_data["regionId"] = str(d.regionId)

    async def _resolve_district(self, document, fields, geo_data: Dict) -> None:
        d = document.documentAppeal
        district_name = None

        if d and d.districtName and not ValueSanitizer.is_empty(d.districtName):
            district_name = d.districtName
        elif not ValueSanitizer.is_empty(fields.districtName):
            district_name = fields.districtName

        if district_name:
            try:
                district_data = await self.ref_client.find_district_with_name(
                    self.token, district_name
                )
                if district_data:
                    geo_data["districtId"] = district_data["id"]
                    geo_data["districtName"] = district_data["name"]
                    logger.info(f"District resolved explicitly: {district_name}")
                else:
                    logger.warning(f"District not found: {district_name}")
            except Exception as e:
                logger.error(f"District resolution error: {e}")
        elif d and d.districtId:
            geo_data["districtId"] = str(d.districtId)


class AppealFieldsBuilder:
    """Построитель payload для DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE."""

    def __init__(self, ref_client: ReferenceClient, token: str):
        self.ref_client = ref_client
        self.token = token
        self.warnings = []

    async def build(
        self, document: DocumentDto, fields, extracted_text: str, geo_data: Dict
    ) -> Dict[str, Any]:
        d = document.documentAppeal or self._create_empty_appeal()
        payload = {}

        payload.update(geo_data)
        await self._add_correspondent(d, fields, payload)
        self._add_personal_data(d, fields, payload)
        await self._add_classification(d, fields, extracted_text, payload)
        await self._add_declarant_type(d, fields, payload)
        self._add_conditional_fields(d, fields, payload)
        self._add_common_fields(d, fields, payload)
        self._add_db_only_fields(d, payload)

        return self._filter_payload(payload)

    def _create_empty_appeal(self):
        return type(
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

    async def _add_correspondent(self, d, fields, payload: Dict) -> None:
        if d.correspondentAppealId:
            payload["correspondentAppealId"] = str(d.correspondentAppealId)
            payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(
                d.correspondentAppeal
            )
        elif not ValueSanitizer.is_empty(d.correspondentAppeal):
            corr_id = await self.ref_client.find_correspondent(
                self.token, d.correspondentAppeal
            )
            if corr_id:
                payload["correspondentAppealId"] = corr_id
                payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(
                    d.correspondentAppeal
                )
        elif not ValueSanitizer.is_empty(fields.correspondentAppeal):
            corr_id = await self.ref_client.find_correspondent(
                self.token, fields.correspondentAppeal
            )
            if corr_id:
                payload["correspondentAppealId"] = corr_id
                payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(
                    fields.correspondentAppeal
                )

        if (
            "correspondentAppealId" not in payload
            or payload.get("correspondentAppealId") is None
        ):
            payload["correspondentAppealId"] = None
            payload["correspondentAppeal"] = None

    def _add_personal_data(self, d, fields, payload: Dict) -> None:
        if not ValueSanitizer.is_empty(d.fioApplicant):
            payload["fioApplicant"] = ValueSanitizer.sanitize_string(d.fioApplicant)
        elif not ValueSanitizer.is_empty(fields.fioApplicant):
            payload["fioApplicant"] = ValueSanitizer.sanitize_string(
                fields.fioApplicant
            )

        if d.dateDocCorrespondentOrg:
            payload["dateDocCorrespondentOrg"] = ValueSanitizer.fix_datetime_format(
                d.dateDocCorrespondentOrg
            )
        elif not ValueSanitizer.is_empty(fields.dateDocCorrespondentOrg):
            payload["dateDocCorrespondentOrg"] = ValueSanitizer.fix_datetime_format(
                fields.dateDocCorrespondentOrg
            )

    async def _add_classification(
        self, d, fields, extracted_text, payload: Dict
    ) -> None:
        if d.citizenTypeId:
            payload["citizenTypeId"] = str(d.citizenTypeId)
        elif not ValueSanitizer.is_empty(fields.citizenType):
            citizen_type_id = await self.ref_client.find_citizen_type(
                self.token, fields.citizenType
            )
            if citizen_type_id:
                payload["citizenTypeId"] = citizen_type_id

        if d.subjectId:
            payload["subjectId"] = str(d.subjectId)
        else:
            subject_id = await self.ref_client.find_best_subject(
                self.token, extracted_text
            )
            if subject_id:
                payload["subjectId"] = subject_id
                logger.info(f"Subject determined by LLM: {subject_id}")

    async def _add_declarant_type(self, d, fields, payload: Dict) -> None:
        if fields.declarantType:
            if isinstance(fields.declarantType, str):
                try:
                    payload["declarantType"] = GeneratedDeclarantType[
                        fields.declarantType.upper()
                    ]
                    logger.info(f"declarantType from LLM: {fields.declarantType}")
                except KeyError:
                    logger.warning(
                        f"Unknown declarantType: {fields.declarantType}, using INDIVIDUAL"
                    )
                    payload["declarantType"] = GeneratedDeclarantType.INDIVIDUAL
            else:
                payload["declarantType"] = fields.declarantType
        elif d.declarantType:
            payload["declarantType"] = d.declarantType
        else:
            payload["declarantType"] = GeneratedDeclarantType.INDIVIDUAL
            self.warnings.append("declarantType установлен INDIVIDUAL по умолчанию")
            logger.warning("declarantType set to INDIVIDUAL (fallback)")

    def _add_conditional_fields(self, d, fields, payload: Dict) -> None:
        if payload.get("declarantType") == GeneratedDeclarantType.ENTITY:
            payload["organizationName"] = ValueSanitizer.sanitize_string(
                d.organizationName
                if not ValueSanitizer.is_empty(d.organizationName)
                else fields.organizationName
            )
            payload["signed"] = ValueSanitizer.sanitize_string(
                d.signed if not ValueSanitizer.is_empty(d.signed) else fields.signed
            )
            payload["correspondentOrgNumber"] = ValueSanitizer.sanitize_string(
                d.correspondentOrgNumber
                if not ValueSanitizer.is_empty(d.correspondentOrgNumber)
                else fields.correspondentOrgNumber
            )
        elif payload.get("declarantType") == GeneratedDeclarantType.INDIVIDUAL:
            payload["organizationName"] = None
            payload["signed"] = None
            payload["correspondentOrgNumber"] = None

    def _add_common_fields(self, d, fields, payload: Dict) -> None:
        payload["collective"] = (
            True
            if fields.collective is True
            else (d.collective if d.collective is not None else fields.collective)
        )
        payload["anonymous"] = (
            True
            if fields.anonymous is True
            else (d.anonymous if d.anonymous is not None else fields.anonymous)
        )
        payload["reasonably"] = (
            True
            if fields.reasonably is True
            else (d.reasonably if d.reasonably is not None else fields.reasonably)
        )
        payload["receiptDate"] = ValueSanitizer.fix_datetime_format(
            d.receiptDate if d.receiptDate else fields.receiptDate
        )
        payload["fullAddress"] = ValueSanitizer.sanitize_string(
            d.fullAddress
            if not ValueSanitizer.is_empty(d.fullAddress)
            else fields.fullAddress
        )
        payload["phone"] = ValueSanitizer.sanitize_string(
            d.phone if not ValueSanitizer.is_empty(d.phone) else fields.phone
        )
        payload["email"] = ValueSanitizer.sanitize_string(
            d.email if not ValueSanitizer.is_empty(d.email) else fields.email
        )
        payload["index"] = ValueSanitizer.sanitize_string(
            d.index if not ValueSanitizer.is_empty(d.index) else fields.index
        )
        payload["indexDateCoverLetter"] = ValueSanitizer.sanitize_string(
            d.indexDateCoverLetter
            if not ValueSanitizer.is_empty(d.indexDateCoverLetter)
            else fields.indexDateCoverLetter
        )
        payload["reviewProgress"] = ValueSanitizer.sanitize_string(
            d.reviewProgress
            if not ValueSanitizer.is_empty(d.reviewProgress)
            else fields.reviewProgress
        )

    def _add_db_only_fields(self, d, payload: Dict) -> None:
        if d.subjectId:
            payload["subjectId"] = str(d.subjectId)
        if d.solutionResultId:
            payload["solutionResultId"] = str(d.solutionResultId)
        if d.nomenclatureAffairId:
            payload["nomenclatureAffairId"] = str(d.nomenclatureAffairId)

    def _filter_payload(self, payload: Dict) -> Dict[str, Any]:
        filtered = {}
        for k, v in payload.items():
            if k in ["correspondentAppeal", "correspondentAppealId"]:
                filtered[k] = v
            elif v is not None and not ValueSanitizer.is_empty(v):
                filtered[k] = v

        if not filtered.get("declarantType"):
            raise ValueError("declarantType обязателен, но не установлен")

        logger.info(
            f"Final payload: {json.dumps(filtered, indent=2, ensure_ascii=False, default=str)}"
        )
        return filtered


class AppealAutofillOrchestrator:
    """Главный оркестратор процесса автозаполнения."""

    MIN_TEXT_LENGTH = 50

    def __init__(self, document_id: str, token: str, attachment_id: Optional[str]):
        self.document_id = document_id
        self.token = token
        self.attachment_id = attachment_id
        self.warnings: List[str] = []

    async def execute(self) -> AutofillResult:
        document = await self._load_document()
        self._validate_document_category(document)

        target_attachment, warnings = AttachmentSelector.select(
            document, self.attachment_id
        )
        self.warnings.extend(warnings)

        extracted_text = await self._extract_text(target_attachment)
        self._validate_text_length(extracted_text)

        fields = await self._analyze_text(extracted_text)

        await self._update_document(document, fields, extracted_text)

        return AutofillResult(
            status="success",
            message="Документ успешно заполнен",
            warnings=self.warnings if self.warnings else None,
            attachment_used=target_attachment.name,
        )

    async def _load_document(self) -> DocumentDto:
        async with DocumentClient() as client:
            raw_document = await client.get_document_metadata(
                self.token, self.document_id
            )
            return DocumentDto.model_validate(raw_document)

    def _validate_document_category(self, document: DocumentDto) -> None:
        if document.docCategoryConstant != "APPEAL":
            raise ValueError(
                f"Документ должен быть категории APPEAL, а не {document.docCategoryConstant}"
            )

    async def _extract_text(self, attachment: Any) -> str:
        async with EdmsAttachmentClient() as client:
            file_bytes = await client.get_attachment_content(
                self.token, self.document_id, str(attachment.id)
            )

        extracted_text = extract_text_from_bytes(file_bytes, attachment.name)
        logger.info(f"Text extracted: {len(extracted_text)} chars")
        return extracted_text

    def _validate_text_length(self, text: str) -> None:
        if not text or len(text) < self.MIN_TEXT_LENGTH:
            raise ValueError("Текст не извлечен или слишком короткий")

    async def _analyze_text(self, text: str):
        extraction_service = AppealExtractionService()
        fields = await extraction_service.extract_appeal_fields(text)
        logger.info("LLM analysis complete")
        return fields

    async def _update_document(
        self, document: DocumentDto, fields, extracted_text: str
    ) -> None:
        async with DocumentClient() as doc_client, ReferenceClient() as ref_client:
            await self._execute_main_fields_update(
                doc_client, ref_client, document, fields
            )
            await self._execute_appeal_fields_update(
                doc_client, ref_client, document, fields, extracted_text
            )

    async def _execute_main_fields_update(
        self, doc_client, ref_client, document, fields
    ) -> None:
        delivery_id = document.deliveryMethodId
        if not delivery_id:
            delivery_method_name = (
                fields.deliveryMethod
                if not ValueSanitizer.is_empty(fields.deliveryMethod)
                else "Курьер"
            )
            delivery_id = await ref_client.find_delivery_method(
                self.token, delivery_method_name
            )

        main_payload = {
            "shortSummary": (
                ValueSanitizer.sanitize_string(fields.shortSummary)
                if not ValueSanitizer.is_empty(fields.shortSummary)
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

        await DocumentOperationExecutor.execute(
            doc_client,
            self.token,
            self.document_id,
            "DOCUMENT_MAIN_FIELDS_UPDATE",
            main_payload,
        )

    async def _execute_appeal_fields_update(
        self, doc_client, ref_client, document, fields, extracted_text
    ) -> None:
        geo_resolver = GeographyResolver(ref_client, self.token)
        geo_data = await geo_resolver.resolve_geography(document, fields)

        fields_builder = AppealFieldsBuilder(ref_client, self.token)
        appeal_payload = await fields_builder.build(
            document, fields, extracted_text, geo_data
        )
        self.warnings.extend(fields_builder.warnings)

        await DocumentOperationExecutor.execute(
            doc_client,
            self.token,
            self.document_id,
            "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE",
            appeal_payload,
        )


@tool("autofill_appeal_document", args_schema=AppealAutofillInput)
async def autofill_appeal_document(
    document_id: str, token: str, attachment_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Автоматически заполняет карточку обращения (APPEAL) через LLM-анализ.

    Args:
        document_id: UUID документа категории APPEAL
        token: JWT токен авторизации
        attachment_id: UUID конкретного файла (опционально)

    Returns:
        Dict с результатом операции
    """
    logger.info(
        "========== APPEAL AUTOFILL START ==========",
        extra={"document_id": document_id},
    )

    try:
        orchestrator = AppealAutofillOrchestrator(document_id, token, attachment_id)
        result = await orchestrator.execute()
        logger.info("========== APPEAL AUTOFILL SUCCESS ==========")
        return result.to_dict()

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return AutofillResult(status="error", message=str(e)).to_dict()
    except Exception as e:
        logger.error("========== APPEAL AUTOFILL ERROR ==========", exc_info=True)
        return AutofillResult(
            status="error", message=f"Ошибка автозаполнения: {str(e)}"
        ).to_dict()
