"""
EDMS AI Assistant - Appeal Autofill Tool

Geography fill order (required by EDMS UI):

  Обычный город (напр. Гомель):
    1. country  — страна (из письма или "Республика Беларусь" по умолчанию)
    2. region   — область (только если ЯВНО указана в письме: "Гомельская область")
    3. district — административный район города/области (только если ЯВНО указан:
                  "Октябрьский район", "Советский район" и т.п.)
                  ❗ Район улицы (часть почтового адреса) — НЕ является районом EDMS
    4. city     — город (последним, после region и district)

  Столица (напр. Минск):
    1. country  — страна
    2. city     — город
    Область и район EDMS заполняет автоматически — мы их НЕ отправляем.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.reference_client import ReferenceClient
from edms_ai_assistant.generated.resources_openapi import (
    DeclarantType as GeneratedDeclarantType,
)
from edms_ai_assistant.generated.resources_openapi import (
    DocumentDto,
)
from edms_ai_assistant.services.appeal_extraction_service import AppealExtractionService
from edms_ai_assistant.utils.file_utils import extract_text_from_bytes
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)

_CAPITAL_CITIES: set[str] = {"минск"}


class AppealAutofillInput(BaseModel):
    """Валидированная схема входных данных для автозаполнения."""

    document_id: str = Field(
        ...,
        description="UUID документа категории APPEAL",
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )
    token: str = Field(..., description="JWT токен авторизации пользователя")
    attachment_id: str | None = Field(
        None,
        description="UUID конкретного вложения для анализа",
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )

    @field_validator("attachment_id")
    @classmethod
    def validate_attachment_id(cls, v: str | None) -> str | None:
        if v and not v.strip():
            return None
        return v


@dataclass(frozen=True)
class AutofillResult:
    """Immutable результат автозаполнения."""

    status: str
    message: str
    warnings: list[str] | None = None
    attachment_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
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
    def sanitize_string(cls, value: str | None) -> str | None:
        if cls.is_empty(value):
            return None
        cleaned = (
            value.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u201e", '"')
            .replace("\u00ab", '"')
            .replace("\u00bb", '"')
            .strip()
        )
        return cleaned if cleaned else None

    @classmethod
    def fix_datetime_format(cls, dt: Any) -> str | None:
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
        body: dict[str, Any],
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
    def select(cls, document: DocumentDto, attachment_id: str | None) -> tuple:
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
    """
    Резолвер географических сущностей.

    Правило заполнения:
      - ВСЕГДА: country → city
      - ТОЛЬКО если явно указаны в документе: region, district

    Если область/район НЕ указаны явно в тексте письма — НЕ отправляем их.
    EDMS заполняет область и район автоматически на основании города.

    Это исключает путаницу между:
      - административным районом области (Октябрьский район Гомельской области)
      - внутригородским районом/микрорайоном (Октябрьский р-н г. Гомель)
      - районом в названии организации ("суд Октябрьского района")
    """

    def __init__(self, ref_client: ReferenceClient, token: str) -> None:
        self.ref_client = ref_client
        self.token = token

    async def resolve_geography(self, document: Any, fields: Any) -> dict[str, Any]:
        """
        Resolves geo fields with minimal API calls.

        Strategy:
          1. country  — always resolved (from document or LLM)
          2. region   — only if EXPLICITLY stated in the source document
          3. district — only if EXPLICITLY stated in the source document
          4. city     — always resolved (from document or LLM)

        If region/district are absent → only country + city are sent.
        EDMS auto-fills region and district based on the selected city.

        Args:
            document: DocumentDto with optional documentAppeal sub-object.
            fields: AppealFields extracted by LLM.

        Returns:
            Ordered dict: country* → [region*] → [district*] → city*
            Region and district are included only when explicitly present.
        """
        geo_data: dict[str, Any] = {}

        city_name = self._get_field(document, fields, "cityName")
        region_name = self._get_field(document, fields, "regionName")
        district_name = self._get_field(document, fields, "districtName")

        # ── 1. Страна ─────────────────────────────────────────────────────────
        await self._resolve_country(document, fields, geo_data)

        # ── 2. Область — только если явно указана ────────────────────────────
        if region_name:
            await self._lookup(
                "region", region_name, "regionId", "regionName", geo_data
            )

        # ── 3. Район — только если явно указан ───────────────────────────────
        if district_name:
            await self._lookup(
                "district", district_name, "districtId", "districtName", geo_data
            )

        # ── 4. Город ─────────────────────────────────────────────────────────
        await self._resolve_city(city_name, document, geo_data)

        logger.info(
            "Geography resolved: country=%s region=%s district=%s city=%s",
            geo_data.get("countryAppealName"),
            geo_data.get("regionName", "—(auto)"),
            geo_data.get("districtName", "—(auto)"),
            geo_data.get("cityName"),
        )
        return geo_data

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_field(document: Any, fields: Any, attr: str) -> str | None:
        """
        Returns first non-empty value for attr: DB record → LLM extraction.

        Args:
            document: DocumentDto.
            fields: AppealFields (LLM output).
            attr: Attribute name to look up.

        Returns:
            Non-empty string value or None.
        """
        d = document.documentAppeal
        db_val = getattr(d, attr, None) if d else None
        if not ValueSanitizer.is_empty(db_val):
            return db_val
        llm_val = getattr(fields, attr, None)
        if not ValueSanitizer.is_empty(llm_val):
            return llm_val
        return None

    # ── Country ───────────────────────────────────────────────────────────────

    async def _resolve_country(
        self, document: Any, fields: Any, geo_data: dict
    ) -> None:
        """Resolves country via reference lookup or falls back to existing DB ID."""
        d = document.documentAppeal
        country_name: str | None = None

        if d and not ValueSanitizer.is_empty(d.countryAppealName):
            country_name = d.countryAppealName
        elif not ValueSanitizer.is_empty(fields.country):
            country_name = fields.country

        if country_name:
            try:
                data = await self.ref_client.find_country_with_name(
                    self.token, country_name
                )
                if data:
                    geo_data["countryAppealId"] = data["id"]
                    geo_data["countryAppealName"] = data["name"]
                    logger.info("Country: %s → %s", country_name, data["name"])
                else:
                    logger.warning("Country not found: %s", country_name)
            except Exception as exc:
                logger.error("Country resolution error: %s", exc)
        elif d and d.countryAppealId:
            geo_data["countryAppealId"] = str(d.countryAppealId)

    # ── Generic reference lookup (region / district) ──────────────────────────

    async def _lookup(
        self,
        endpoint: str,
        name: str,
        id_key: str,
        name_key: str,
        geo_data: dict,
    ) -> None:
        """
        Two-step reference lookup: fts-name → GET /{id} → canonical name.

        Args:
            endpoint: API endpoint ("region" or "district").
            name: Name to search for.
            id_key: Key to store resolved ID in geo_data.
            name_key: Key to store canonical name in geo_data.
            geo_data: Mutable geo dict to update in place.
        """
        try:
            method = getattr(self.ref_client, f"find_{endpoint}_with_name")
            data = await method(self.token, name)
            if data:
                geo_data[id_key] = data["id"]
                geo_data[name_key] = data["name"]
                logger.info(
                    "%s: %s → %s (ID: %s)", endpoint, name, data["name"], data["id"]
                )
            else:
                logger.warning("%s not found in reference: %s", endpoint, name)
        except Exception as exc:
            logger.error("%s lookup error for '%s': %s", endpoint, name, exc)

    # ── City ──────────────────────────────────────────────────────────────────

    async def _resolve_city(
        self,
        city_name: str | None,
        document: Any,
        geo_data: dict,
    ) -> None:
        """
        Resolves city via find_city_with_hierarchy (fts-name → GET /{id}?includes=DISTRICT_WITH_REGION).

        Hierarchy data (districtId/districtName/regionId/regionName) is written
        into geo_data ONLY if those keys are not already set — i.e. only when
        region/district were NOT explicitly stated in the source document and
        _lookup() has not already resolved them.

        This covers the most common case: applicant writes only city name
        (e.g. "г. Минск") and EDMS needs all four geo levels filled.

        Args:
            city_name: City name from document or LLM.
            document: DocumentDto (for DB cityId fallback).
            geo_data: Mutable geo dict to update in place.
        """
        d = document.documentAppeal
        if city_name:
            try:
                data = await self.ref_client.find_city_with_hierarchy(
                    self.token, city_name
                )
                if data:
                    geo_data["cityId"] = data["id"]
                    geo_data["cityName"] = data["name"]
                    logger.info(
                        "City resolved: %s → '%s' (ID: %s)",
                        city_name,
                        data["name"],
                        data["id"],
                    )

                    if "districtId" not in geo_data and data.get("districtId"):
                        geo_data["districtId"] = data["districtId"]
                        logger.info(
                            "District from hierarchy: id=%s", data["districtId"]
                        )
                    if "districtName" not in geo_data and data.get("districtName"):
                        geo_data["districtName"] = data["districtName"]
                        logger.info(
                            "District name from hierarchy: '%s'", data["districtName"]
                        )
                    if "regionId" not in geo_data and data.get("regionId"):
                        geo_data["regionId"] = data["regionId"]
                        logger.info("Region from hierarchy: id=%s", data["regionId"])
                    if "regionName" not in geo_data and data.get("regionName"):
                        geo_data["regionName"] = data["regionName"]
                        logger.info(
                            "Region name from hierarchy: '%s'", data["regionName"]
                        )
                else:
                    logger.warning("City not found in reference: '%s'", city_name)
            except Exception as exc:
                logger.error(
                    "City resolution error for '%s': %s", city_name, exc, exc_info=True
                )
        elif d and d.cityId:
            geo_data["cityId"] = str(d.cityId)
            logger.debug("City fallback: DB cityId=%s", d.cityId)


class AppealFieldsBuilder:
    """Построитель payload для DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE."""

    def __init__(self, ref_client: ReferenceClient, token: str) -> None:
        self.ref_client = ref_client
        self.token = token
        self.warnings: list[str] = []

    async def build(
        self,
        document: DocumentDto,
        fields: Any,
        extracted_text: str,
        geo_data: dict,
    ) -> dict[str, Any]:
        """
        Builds the full appeal fields payload by merging geo data with
        personal, classification, and conditional fields.

        Args:
            document: DocumentDto with current DB values.
            fields: AppealFields extracted by LLM.
            extracted_text: Raw text used for subject classification.
            geo_data: Pre-resolved geography dict.

        Returns:
            Filtered payload dict ready for DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE.
        """
        d = document.documentAppeal or self._create_empty_appeal()
        payload: dict[str, Any] = {}

        GEO_KEY_ORDER = [
            "countryAppealId",
            "countryAppealName",
            "regionId",
            "regionName",
            "districtId",
            "districtName",
            "cityId",
            "cityName",
        ]
        for key in GEO_KEY_ORDER:
            if key in geo_data:
                payload[key] = geo_data[key]
        await self._add_correspondent(d, fields, payload)
        self._add_personal_data(d, fields, payload)
        await self._add_classification(d, fields, extracted_text, payload)
        await self._add_declarant_type(d, fields, payload)
        self._add_conditional_fields(d, fields, payload)
        self._add_common_fields(d, fields, payload)
        self._add_db_only_fields(d, payload)

        return self._filter_payload(payload)

    def _create_empty_appeal(self) -> Any:
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

    async def _add_correspondent(self, d: Any, fields: Any, payload: dict) -> None:
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

        if not payload.get("correspondentAppealId"):
            payload["correspondentAppealId"] = None
            payload["correspondentAppeal"] = None

    def _add_personal_data(self, d: Any, fields: Any, payload: dict) -> None:
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
        self, d: Any, fields: Any, extracted_text: str, payload: dict
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

    async def _add_declarant_type(self, d: Any, fields: Any, payload: dict) -> None:
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

    def _add_conditional_fields(self, d: Any, fields: Any, payload: dict) -> None:
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

    def _add_common_fields(self, d: Any, fields: Any, payload: dict) -> None:
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

    def _add_db_only_fields(self, d: Any, payload: dict) -> None:
        if d.subjectId:
            payload["subjectId"] = str(d.subjectId)
        if d.solutionResultId:
            payload["solutionResultId"] = str(d.solutionResultId)
        if d.nomenclatureAffairId:
            payload["nomenclatureAffairId"] = str(d.nomenclatureAffairId)

    def _filter_payload(self, payload: dict) -> dict[str, Any]:
        filtered = {}
        for k, v in payload.items():
            if k in ("correspondentAppeal", "correspondentAppealId") or (
                v is not None and not ValueSanitizer.is_empty(v)
            ):
                filtered[k] = v

        if not filtered.get("declarantType"):
            raise ValueError("declarantType обязателен, но не установлен")

        logger.info(
            "Final payload: %s",
            json.dumps(filtered, indent=2, ensure_ascii=False, default=str),
        )
        return filtered


class AppealAutofillOrchestrator:
    """Главный оркестратор процесса автозаполнения."""

    MIN_TEXT_LENGTH = 50

    def __init__(self, document_id: str, token: str, attachment_id: str | None) -> None:
        self.document_id = document_id
        self.token = token
        self.attachment_id = attachment_id
        self.warnings: list[str] = []

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

    async def _analyze_text(self, text: str) -> Any:
        extraction_service = AppealExtractionService()
        fields = await extraction_service.extract_appeal_fields(text)
        logger.info("LLM analysis complete")
        return fields

    async def _update_document(
        self, document: DocumentDto, fields: Any, extracted_text: str
    ) -> None:
        async with DocumentClient() as doc_client, ReferenceClient() as ref_client:
            await self._execute_main_fields_update(
                doc_client, ref_client, document, fields
            )
            await self._execute_appeal_fields_update(
                doc_client, ref_client, document, fields, extracted_text
            )

    async def _execute_main_fields_update(
        self, doc_client: Any, ref_client: Any, document: DocumentDto, fields: Any
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
        self,
        doc_client: Any,
        ref_client: Any,
        document: DocumentDto,
        fields: Any,
        extracted_text: str,
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
    document_id: str,
    token: str,
    attachment_id: str | None = None,
) -> dict[str, Any]:
    """
    Автоматически заполняет карточку обращения (APPEAL) через LLM-анализ.

    Args:
        document_id: UUID документа категории APPEAL.
        token: JWT токен авторизации.
        attachment_id: UUID конкретного файла (опционально).

    Returns:
        Dict с результатом операции.
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
            status="error", message=f"Ошибка автозаполнения: {e!s}"
        ).to_dict()