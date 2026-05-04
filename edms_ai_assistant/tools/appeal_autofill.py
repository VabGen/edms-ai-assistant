# edms_ai_assistant/clients/appeal_autofill_tool.py

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
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
    DocumentAppealDto,
    DocumentDto,
)
from edms_ai_assistant.models.appeal_fields import AppealFields, SubmissionFormAppeal
from edms_ai_assistant.services.appeal_extraction_service import AppealExtractionService
from edms_ai_assistant.utils.file_utils import extract_text_from_bytes
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)


class AppealAutofillInput(BaseModel):
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
    generate_summary_choices: bool = Field(
        False,
        description=(
            "Если True — после заполнения возвращает 3 варианта заголовка (shortSummary) "
            "для выбора пользователем. Используй когда пользователь просит "
            "выбрать или изменить заголовок документа."
        ),
    )

    @field_validator("attachment_id")
    @classmethod
    def validate_attachment_id(cls, v: str | None) -> str | None:
        if v and not v.strip():
            return None
        return v


@dataclass(frozen=True)
class AutofillResult:
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
            if not dt.endswith("Z") and "+00:00" not in dt:
                dt += "Z"
            elif dt.endswith("+00:00"):
                dt = dt.replace("+00:00", "Z")
            return dt
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        return None


class DocumentOperationExecutor:
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
            "Executing %s",
            operation_type,
            extra={"document_id": document_id, "operation": operation_type},
        )

        try:
            await client.execute_document_operations(
                token, document_id, json_safe_payload
            )
            logger.info("%s executed successfully", operation_type)
        except Exception as e:
            logger.error("%s failed: %s", operation_type, e, exc_info=True)
            raise


class AttachmentSelector:
    SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".doc", ".rtf", ".odt", ".xlsx")

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
                    if a.name and a.name.lower().endswith(cls.SUPPORTED_EXTENSIONS)
                ),
                document.attachmentDocument[0],
            )

        logger.info("Attachment selected: %s", target.name)
        return target, warnings


class GeographyResolver:
    def __init__(self, ref_client: ReferenceClient, token: str) -> None:
        self.ref_client = ref_client
        self.token = token

    async def resolve_geography(self, document: Any, fields: Any) -> dict[str, Any]:
        geo_data: dict[str, Any] = {}

        city_name = self._get_field(document, fields, "cityName")
        region_name = self._get_field(document, fields, "regionName")
        district_name = self._get_field(document, fields, "districtName")

        await self._resolve_country(document, fields, geo_data)

        if region_name:
            await self._lookup(
                "region", region_name, "regionId", "regionName", geo_data
            )

        if district_name:
            await self._lookup(
                "district", district_name, "districtId", "districtName", geo_data
            )

        await self._resolve_city(city_name, document, geo_data)

        logger.info(
            "Geography resolved: country=%s region=%s district=%s city=%s",
            geo_data.get("countryAppealName"),
            geo_data.get("regionName", "—(auto)"),
            geo_data.get("districtName", "—(auto)"),
            geo_data.get("cityName"),
        )
        return geo_data

    @staticmethod
    def _get_field(document: Any, fields: Any, attr: str) -> str | None:
        d = document.documentAppeal
        db_val = getattr(d, attr, None) if d else None
        if not ValueSanitizer.is_empty(db_val):
            return db_val
        llm_val = getattr(fields, attr, None)
        if not ValueSanitizer.is_empty(llm_val):
            return llm_val
        return None

    async def _resolve_country(
        self, document: Any, fields: Any, geo_data: dict
    ) -> None:
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

    async def _lookup(
        self, endpoint: str, name: str, id_key: str, name_key: str, geo_data: dict
    ) -> None:
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

    async def _resolve_city(
        self, city_name: str | None, document: Any, geo_data: dict
    ) -> None:
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


def _is_good_correspondent_match(query: str, canonical: str) -> bool:
    import re as _re
    from difflib import SequenceMatcher

    q = query.lower().strip()
    c = canonical.lower().strip()

    if c in q or q in c:
        return True
    if SequenceMatcher(None, q, c).ratio() >= 0.60:
        return True

    q_words = {w for w in q.split() if len(w) > 3}
    c_words = {w for w in c.split() if len(w) > 3}
    if q_words and c_words and len(q_words & c_words) / len(q_words) >= 0.40:
        return True

    orig_words = [w for w in query.strip().split() if len(w) > 3]
    if len(orig_words) >= 2:
        abbrev = "".join(w[0].upper() for w in orig_words)
        if _re.search(r"\b" + _re.escape(abbrev) + r"\b", canonical, _re.IGNORECASE):
            return True

    return False


class AppealFieldsBuilder:
    def __init__(self, ref_client: ReferenceClient, token: str) -> None:
        self.ref_client = ref_client
        self.token = token
        self.warnings: list[str] = []

    async def build(
        self,
        document: DocumentDto,
        fields: AppealFields,
        extracted_text: str,
        geo_data: dict,
    ) -> dict[str, Any]:
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
        await self._add_conditional_fields(d, fields, payload)
        self._add_common_fields(d, fields, payload)
        self._add_db_only_fields(d, payload)

        return self._filter_payload(payload)

    def _create_empty_appeal(self) -> DocumentAppealDto:
        return DocumentAppealDto()

    async def _add_correspondent(
        self, d: DocumentAppealDto, fields: AppealFields, payload: dict
    ) -> None:
        if d.correspondentAppealId:
            payload["correspondentAppealId"] = str(d.correspondentAppealId)
            payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(
                d.correspondentAppeal
            )
            return

        corr_name: str | None = None
        if not ValueSanitizer.is_empty(d.correspondentAppeal):
            corr_name = d.correspondentAppeal
        elif not ValueSanitizer.is_empty(fields.correspondentAppeal):
            corr_name = fields.correspondentAppeal

        if corr_name:
            canonical = await self.ref_client._find_entity_with_name(
                self.token, "correspondent", corr_name, "Корреспондент"
            )
            if canonical and _is_good_correspondent_match(corr_name, canonical["name"]):
                payload["correspondentAppealId"] = canonical["id"]
                payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(
                    corr_name
                )
                logger.info(
                    "correspondentAppeal matched: '%s' → '%s'",
                    corr_name[:50],
                    canonical["name"][:50],
                )
            else:
                if canonical:
                    logger.info(
                        "correspondentAppeal registry rejected: '%s' ≉ '%s' — free-text",
                        corr_name[:50],
                        canonical["name"][:50],
                    )
                payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(
                    corr_name
                )

        if not payload.get("correspondentAppealId"):
            payload["correspondentAppealId"] = None
            if "correspondentAppeal" not in payload:
                payload["correspondentAppeal"] = None

    def _add_personal_data(
        self, d: DocumentAppealDto, fields: AppealFields, payload: dict
    ) -> None:
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
        self,
        d: DocumentAppealDto,
        fields: AppealFields,
        extracted_text: str,
        payload: dict,
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
                logger.info("Subject determined by LLM: %s", subject_id)

    async def _add_declarant_type(
        self, d: DocumentAppealDto, fields: AppealFields, payload: dict
    ) -> None:
        if fields.declarantType:
            if isinstance(fields.declarantType, str):
                try:
                    payload["declarantType"] = GeneratedDeclarantType[
                        fields.declarantType.upper()
                    ]
                    logger.info("declarantType from LLM: %s", fields.declarantType)
                except KeyError:
                    logger.warning(
                        "Unknown declarantType: %s, using INDIVIDUAL",
                        fields.declarantType,
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

    async def _add_conditional_fields(
        self, d: DocumentAppealDto, fields: AppealFields, payload: dict
    ) -> None:
        if payload.get("declarantType") == GeneratedDeclarantType.ENTITY:
            raw_org: str | None = (
                d.organizationName
                if not ValueSanitizer.is_empty(d.organizationName)
                else fields.organizationName
            )
            if not ValueSanitizer.is_empty(raw_org):
                corr_data = await self.ref_client._find_entity_with_name(
                    self.token, "correspondent", raw_org, "Организация"
                )
                if corr_data and _is_good_correspondent_match(
                    raw_org, corr_data.get("name", "")
                ):
                    payload["organizationName"] = corr_data["name"]
                    logger.info(
                        "organizationName from registry: '%s' → '%s'",
                        raw_org[:50],
                        corr_data["name"][:50],
                    )
                else:
                    if corr_data:
                        logger.info(
                            "organizationName registry rejected (low quality): '%s' ≉ '%s' — free-text",
                            raw_org[:50],
                            corr_data.get("name", "")[:50],
                        )
                    payload["organizationName"] = ValueSanitizer.sanitize_string(
                        raw_org
                    )
            else:
                payload["organizationName"] = None

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

    def _add_common_fields(
        self, d: DocumentAppealDto, fields: AppealFields, payload: dict
    ) -> None:
        payload["collective"] = (
            fields.collective if fields.collective is not None else d.collective
        )
        payload["anonymous"] = (
            fields.anonymous if fields.anonymous is not None else d.anonymous
        )
        payload["reasonably"] = (
            fields.reasonably if fields.reasonably is not None else d.reasonably
        )

        raw_receipt = d.receiptDate if d.receiptDate else fields.receiptDate
        if raw_receipt:
            _receipt_date = raw_receipt if isinstance(raw_receipt, datetime) else None
            if (
                _receipt_date
                and abs((_receipt_date.date() - datetime.now(UTC).date()).days) <= 1
            ):
                logger.info(
                    "receiptDate=%s выглядит как today → пропускаем",
                    _receipt_date.date(),
                )
                payload["receiptDate"] = None
            else:
                payload["receiptDate"] = ValueSanitizer.fix_datetime_format(raw_receipt)
        else:
            payload["receiptDate"] = None

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

        submission_form = getattr(d, "submissionForm", None)
        if not submission_form and fields.submissionForm:
            submission_form = fields.submissionForm

        payload["submissionForm"] = (
            submission_form if submission_form else SubmissionFormAppeal.WRITTEN
        )

    def _add_db_only_fields(self, d: DocumentAppealDto, payload: dict) -> None:
        if d.subjectId:
            payload["subjectId"] = str(d.subjectId)
        if d.solutionResultId:
            payload["solutionResultId"] = str(d.solutionResultId)
        if d.nomenclatureAffairId:
            payload["nomenclatureAffairId"] = str(d.nomenclatureAffairId)

    def _filter_payload(self, payload: dict) -> dict[str, Any]:
        _ALLOW_NULL = {
            "correspondentAppeal",
            "correspondentAppealId",
            "submissionForm",
            "organizationName",
            "signed",
            "correspondentOrgNumber",
            "fioApplicant",
            "fullAddress",
            "phone",
            "email",
            "index",
            "receiptDate",
            "dateDocCorrespondentOrg",
            "collective",
            "anonymous",
            "reasonably",
        }

        filtered = {}
        for k, v in payload.items():
            if v is not None or k in _ALLOW_NULL:
                filtered[k] = v

        if not filtered.get("declarantType"):
            raise ValueError("declarantType обязателен, но не установлен")

        if not filtered.get("submissionForm"):
            filtered["submissionForm"] = SubmissionFormAppeal.WRITTEN

        logger.info(
            "Final payload: %s",
            json.dumps(filtered, indent=2, ensure_ascii=False, default=str),
        )
        return filtered


class AppealAutofillOrchestrator:
    MIN_TEXT_LENGTH = 50

    def __init__(self, document_id: str, token: str, attachment_id: str | None) -> None:
        self.document_id = document_id
        self.token = token
        self.attachment_id = attachment_id
        self.warnings: list[str] = []
        self._last_extracted_text: str | None = None
        self._last_short_summary: str | None = None

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
            doc = DocumentDto.model_validate(raw_document)

            appeal = getattr(doc, "documentAppeal", None)
            logger.info(
                "documentAppeal.submissionForm = %s",
                getattr(appeal, "submissionForm", "ATTR_NOT_FOUND"),
            )
            return doc

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
        logger.info("Text extracted: %d chars", len(extracted_text))
        return extracted_text

    def _validate_text_length(self, text: str) -> None:
        if not text or len(text) < self.MIN_TEXT_LENGTH:
            raise ValueError("Текст не извлечен или слишком короткий")

    async def _analyze_text(self, text: str) -> AppealFields:
        extraction_service = AppealExtractionService()
        fields = await extraction_service.extract_appeal_fields(text)
        self._last_extracted_text = text
        self._last_short_summary = getattr(fields, "shortSummary", None)
        logger.info("LLM analysis complete")
        return fields

    async def _update_document(
        self, document: DocumentDto, fields: AppealFields, extracted_text: str
    ) -> None:
        async with DocumentClient() as doc_client, ReferenceClient() as ref_client:
            await self._execute_main_fields_update(
                doc_client, ref_client, document, fields
            )
            await self._execute_appeal_fields_update(
                doc_client, ref_client, document, fields, extracted_text
            )

    async def _execute_main_fields_update(
        self,
        doc_client: DocumentClient,
        ref_client: ReferenceClient,
        document: DocumentDto,
        fields: AppealFields,
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

        raw_summary = (
            ValueSanitizer.sanitize_string(fields.shortSummary)
            if not ValueSanitizer.is_empty(fields.shortSummary)
            else document.shortSummary
        )
        if raw_summary and len(raw_summary) > 80:
            raw_summary = raw_summary[:80]
            logger.warning("shortSummary обрезан до 80 символов: '%s'", raw_summary)

        main_payload = {
            "shortSummary": raw_summary,
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
        doc_client: DocumentClient,
        ref_client: ReferenceClient,
        document: DocumentDto,
        fields: AppealFields,
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


async def _generate_summary_variants(
    text: str, current_summary: str | None, token: str | None = None
) -> list[str]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from edms_ai_assistant.llm import get_chat_model

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Ты — эксперт по делопроизводству. "
                    "Сформулируй РОВНО 3 варианта краткого содержания (заголовка) обращения. "
                    "Каждый вариант — отдельная строка, максимум 80 символов. "
                    "Стили: 1) краткий (суть в 5-8 словах), "
                    "2) официальный (с указанием типа обращения), "
                    "3) описательный (с ключевыми деталями). "
                    "Отвечай ТОЛЬКО тремя строками без нумерации и без лишних слов."
                ),
            ),
            ("user", "Текст обращения:\n{text}"),
        ]
    )

    llm = get_chat_model()
    chain = prompt | llm | StrOutputParser()

    try:
        result = await chain.ainvoke({"text": text[:2000]})
        variants = [v.strip() for v in result.strip().split("\n") if v.strip()]
        variants = [v[:77] + "..." if len(v) > 80 else v for v in variants[:3]]
        if current_summary and current_summary not in variants:
            variants = [
                (
                    current_summary[:77] + "..."
                    if len(current_summary) > 80
                    else current_summary
                )
            ] + variants[:2]
        return variants[:3]
    except Exception as exc:
        logger.warning("Failed to generate summary variants: %s", exc)
        return [current_summary] if current_summary else []


@tool("autofill_appeal_document", args_schema=AppealAutofillInput)
async def autofill_appeal_document(
    document_id: str,
    token: str,
    attachment_id: str | None = None,
    generate_summary_choices: bool = False,
) -> dict[str, Any]:
    """
    Автоматически заполняет карточку обращения (APPEAL) через LLM-анализ.

    Используй generate_summary_choices=True когда пользователь хочет
    выбрать или изменить заголовок (краткое содержание) документа.

    Args:
        document_id: UUID документа категории APPEAL.
        token: JWT токен авторизации.
        attachment_id: UUID конкретного файла (опционально).
        generate_summary_choices: Вернуть 3 варианта заголовка для выбора.

    Returns:
        Dict с результатом операции. Если generate_summary_choices=True —
        дополнительно содержит summary_choices: list[str].
    """
    logger.info(
        "========== APPEAL AUTOFILL START ==========",
        extra={"document_id": document_id},
    )

    try:
        orchestrator = AppealAutofillOrchestrator(document_id, token, attachment_id)
        result = await orchestrator.execute()
        logger.info("========== APPEAL AUTOFILL SUCCESS ==========")

        result_dict = result.to_dict()

        if generate_summary_choices and orchestrator._last_extracted_text:
            current_summary = orchestrator._last_short_summary
            variants = await _generate_summary_variants(
                orchestrator._last_extracted_text, current_summary
            )
            if variants:
                result_dict["summary_choices"] = variants
                result_dict["summary_choices_hint"] = (
                    "Выберите заголовок или предложите свой вариант. "
                    "Скажите 'Установи заголовок: <текст>' чтобы применить."
                )
                logger.info("Generated %d summary variants", len(variants))

        return result_dict

    except ValueError as e:
        logger.error("Validation error: %s", e)
        return AutofillResult(status="error", message=str(e)).to_dict()
    except Exception as e:
        logger.error("========== APPEAL AUTOFILL ERROR ==========", exc_info=True)
        return AutofillResult(
            status="error", message=f"Ошибка автозаполнения: {e!s}"
        ).to_dict()
