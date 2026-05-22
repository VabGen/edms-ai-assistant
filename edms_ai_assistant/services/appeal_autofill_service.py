# edms_ai_assistant/services/appeal_autofill_service.py
"""
Сервис оркестрации автозаполнения обращений.
"""

from __future__ import annotations

import json
import logging
import re as _re
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any, TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from edms_ai_assistant.domain.appeal_fields import AppealFields, SubmissionFormAppeal
from edms_ai_assistant.domain.document import DocumentDto, DocumentAppealDto
from edms_ai_assistant.domain.enums import DeclarantType
from edms_ai_assistant.utils.file_utils import extract_text_from_bytes
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder

if TYPE_CHECKING:
    from edms_ai_assistant.services.appeal_extraction_service import AppealExtractionService
    from edms_ai_assistant.clients.attachment_client import AttachmentClient
    from langchain_core.language_models.chat_models import BaseChatModel
    from edms_ai_assistant.clients.reference_client import ReferenceClient
    from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutofillResult:
    status: str
    message: str
    warnings: list[str] | None = None
    attachment_used: str | None = None
    last_extracted_text: str | None = None
    last_short_summary: str | None = None
    summary_choices: list[str] | None = None
    summary_choices_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"status": self.status, "message": self.message}
        if self.warnings:
            result["warnings"] = self.warnings
        if self.attachment_used:
            result["attachment_used"] = self.attachment_used
        if self.summary_choices:
            result["summary_choices"] = self.summary_choices
        if self.summary_choices_hint:
            result["summary_choices_hint"] = self.summary_choices_hint
        return result


class ValueSanitizer:
    EMPTY_PLACEHOLDERS = {
        "none", "null", "nil", "n/a", "na", "unknown", "not specified", "no",
        "нет", "неизвестно", "н/д",
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
            value.replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u201e", '"').replace("\u00ab", '"')
            .replace("\u00bb", '"').strip()
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
                dt = dt.replace(tzinfo=UTC)
            return dt.isoformat().replace("+00:00", "Z")
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

        try:
            await client.execute_document_operations(token, document_id, json_safe_payload)
            logger.info("%s executed successfully", operation_type)
        except Exception as e:
            logger.error("%s failed: %s", operation_type, e, exc_info=True)
            raise


class AttachmentSelector:
    SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".doc", ".rtf", ".odt", ".xlsx")

    @classmethod
    def select(cls, document: DocumentDto, attachment_id: str | None) -> tuple[Any, list[str]]:
        warnings = []
        # attachments в DocumentDto нет явно, но может прийти от Enricher или getattr
        attachments = getattr(document, "attachment_document", None) or []

        if not attachments:
            raise ValueError("В документе отсутствуют вложения")

        target = None
        if attachment_id:
            target = next(
                (a for a in attachments if str(getattr(a, "id", "")) == attachment_id), None,
            )
            if not target:
                warnings.append(f"Вложение ID={attachment_id} не найдено, используется автоподбор")

        if not target:
            target = next(
                (a for a in attachments if
                 getattr(a, "name", "") and str(getattr(a, "name", "")).lower().endswith(cls.SUPPORTED_EXTENSIONS)),
                attachments[0],
            )

        logger.info("Attachment selected: %s", getattr(target, "name", "unknown"))
        return target, warnings


def _is_good_correspondent_match(query: str, canonical: str) -> bool:
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


class GeographyResolver:
    def __init__(self, ref_client: ReferenceClient, token: str) -> None:
        self.ref_client = ref_client
        self.token = token

    async def resolve_geography(self, document: DocumentDto, fields: AppealFields) -> dict[str, Any]:
        geo_data: dict[str, Any] = {}

        city_name = self._get_field(document, fields, "city_name", "cityName")
        region_name = self._get_field(document, fields, "region_name", "regionName")
        district_name = self._get_field(document, fields, "district_name", "districtName")

        await self._resolve_country(document, fields, geo_data)

        if region_name:
            await self._lookup("region", region_name, "regionId", "regionName", geo_data)

        if district_name:
            await self._lookup("district", district_name, "districtId", "districtName", geo_data)

        await self._resolve_city(city_name, document, geo_data)

        return geo_data

    @staticmethod
    def _get_field(document: DocumentDto, fields: AppealFields, snake_attr: str, camel_attr: str) -> str | None:
        d = document.document_appeal
        db_val = getattr(d, snake_attr, None) if d else None
        if not ValueSanitizer.is_empty(db_val):
            return str(db_val)

        llm_val = getattr(fields, camel_attr, None)
        if not ValueSanitizer.is_empty(llm_val):
            return str(llm_val)
        return None

    async def _resolve_country(self, document: DocumentDto, fields: AppealFields, geo_data: dict[str, Any]) -> None:
        d = document.document_appeal
        country_name: str | None = None

        if d and not ValueSanitizer.is_empty(d.country_appeal_name):
            country_name = str(d.country_appeal_name)
        elif not ValueSanitizer.is_empty(fields.country):
            country_name = fields.country

        if country_name:
            try:
                data = await self.ref_client.find_country_with_name(self.token, country_name)
                if data:
                    geo_data["countryAppealId"] = str(data.id) if data.id else None
                    geo_data["countryAppealName"] = data.name
            except Exception:
                logger.error("Country resolution error", exc_info=True)
        elif d and d.country_appeal_id:
            geo_data["countryAppealId"] = str(d.country_appeal_id)

    async def _lookup(self, endpoint: str, name: str, id_key: str, name_key: str, geo_data: dict[str, Any]) -> None:
        try:
            method = getattr(self.ref_client, f"find_{endpoint}_with_name")
            data = await method(self.token, name)
            if data:
                geo_data[id_key] = str(data.id) if data.id else None
                geo_data[name_key] = data.name
        except Exception:
            logger.error("%s lookup error", endpoint, exc_info=True)

    async def _resolve_city(self, city_name: str | None, document: DocumentDto, geo_data: dict[str, Any]) -> None:
        d = document.document_appeal
        if city_name:
            try:
                data = await self.ref_client.find_city_with_hierarchy(self.token, city_name)
                if data:
                    geo_data["cityId"] = str(data.id) if data.id else None
                    geo_data["cityName"] = data.name

                    if "districtId" not in geo_data and data.district_id:
                        geo_data["districtId"] = str(data.district_id)
                    if "districtName" not in geo_data and data.district_name:
                        geo_data["districtName"] = data.district_name
                    if "regionId" not in geo_data and data.region_id:
                        geo_data["regionId"] = str(data.region_id)
                    if "regionName" not in geo_data and data.region_name:
                        geo_data["regionName"] = data.region_name
            except Exception:
                logger.error("City resolution error", exc_info=True)
        elif d and d.city_id:
            geo_data["cityId"] = str(d.city_id)


class AppealFieldsBuilder:
    def __init__(self, ref_client: ReferenceClient, token: str) -> None:
        self.ref_client = ref_client
        self.token = token
        self.warnings: list[str] = []

    async def build(
            self, document: DocumentDto, fields: AppealFields, extracted_text: str, geo_data: dict[str, Any],
    ) -> dict[str, Any]:
        d = document.document_appeal or DocumentAppealDto()
        payload: dict[str, Any] = {}

        geo_keys = ["countryAppealId", "countryAppealName", "regionId", "regionName",
                    "districtId", "districtName", "cityId", "cityName"]
        for key in geo_keys:
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

    async def _add_correspondent(self, d: DocumentAppealDto, fields: AppealFields, payload: dict[str, Any]) -> None:
        if d.correspondent_appeal_id:
            payload["correspondentAppealId"] = str(d.correspondent_appeal_id)
            payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(d.correspondent_appeal)
            return

        corr_name: str | None = None
        if not ValueSanitizer.is_empty(d.correspondent_appeal):
            corr_name = str(d.correspondent_appeal)
        elif not ValueSanitizer.is_empty(fields.correspondentAppeal):
            corr_name = fields.correspondentAppeal

        if corr_name:
            canonical = await self.ref_client._find_entity_with_name(
                self.token, "correspondent", corr_name, "Корреспондент"
            )
            if canonical and _is_good_correspondent_match(corr_name, canonical.name or ""):
                payload["correspondentAppealId"] = str(canonical.id) if canonical.id else None
                payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(corr_name)
            else:
                payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(corr_name)

    @staticmethod
    def _add_personal_data(d: DocumentAppealDto, fields: AppealFields, payload: dict[str, Any]) -> None:
        if not ValueSanitizer.is_empty(d.fio_applicant):
            payload["fioApplicant"] = ValueSanitizer.sanitize_string(d.fio_applicant)
        elif not ValueSanitizer.is_empty(fields.fioApplicant):
            payload["fioApplicant"] = ValueSanitizer.sanitize_string(fields.fioApplicant)

        if d.date_doc_correspondent_org:
            payload["dateDocCorrespondentOrg"] = ValueSanitizer.fix_datetime_format(d.date_doc_correspondent_org)
        elif not ValueSanitizer.is_empty(fields.dateDocCorrespondentOrg):
            payload["dateDocCorrespondentOrg"] = ValueSanitizer.fix_datetime_format(fields.dateDocCorrespondentOrg)

    async def _add_classification(self, d: DocumentAppealDto, fields: AppealFields, extracted_text: str, payload: dict[str, Any]) -> None:
        if d.citizen_type_id:
            payload["citizenTypeId"] = str(d.citizen_type_id)
        elif not ValueSanitizer.is_empty(fields.citizenType):
            citizen_type_id = await self.ref_client.find_citizen_type(self.token, fields.citizenType)
            if citizen_type_id:
                payload["citizenTypeId"] = citizen_type_id

        if d.subject_id:
            payload["subjectId"] = str(d.subject_id)
        else:
            subject_id = await self.ref_client.find_best_subject(self.token, extracted_text)
            if subject_id:
                payload["subjectId"] = subject_id

    async def _add_declarant_type(self, d: DocumentAppealDto, fields: AppealFields, payload: dict[str, Any]) -> None:
        if fields.declarantType:
            try:
                payload["declarantType"] = DeclarantType(fields.declarantType.upper())
            except (ValueError, AttributeError):
                payload["declarantType"] = DeclarantType.INDIVIDUAL
        elif d.declarant_type:
            payload["declarantType"] = d.declarant_type
        else:
            payload["declarantType"] = DeclarantType.INDIVIDUAL

    async def _add_conditional_fields(self, d: DocumentAppealDto, fields: AppealFields, payload: dict[str, Any]) -> None:
        if payload.get("declarantType") == DeclarantType.ENTITY:
            raw_org: str | None = (
                str(d.organization_name) if not ValueSanitizer.is_empty(d.organization_name) else fields.organizationName
            )
            if not ValueSanitizer.is_empty(raw_org):
                corr_data = await self.ref_client._find_entity_with_name(self.token, "correspondent", str(raw_org), "Организация")
                if corr_data and _is_good_correspondent_match(str(raw_org), corr_data.name or ""):
                    payload["organizationName"] = corr_data.name
                else:
                    payload["organizationName"] = ValueSanitizer.sanitize_string(raw_org)

            payload["signed"] = ValueSanitizer.sanitize_string(d.signed or fields.signed)
            payload["correspondentOrgNumber"] = ValueSanitizer.sanitize_string(d.correspondent_org_number or fields.correspondentOrgNumber)
        elif payload.get("declarantType") == DeclarantType.INDIVIDUAL:
            payload["organizationName"] = None
            payload["signed"] = None
            payload["correspondentOrgNumber"] = None

    @staticmethod
    def _add_common_fields(d: DocumentAppealDto, fields: AppealFields, payload: dict[str, Any]) -> None:
        payload["collective"] = fields.collective if fields.collective is not None else (d.collective or False)
        payload["anonymous"] = fields.anonymous if fields.anonymous is not None else (d.anonymous or False)
        payload["reasonably"] = fields.reasonably if fields.reasonably is not None else (d.reasonably or False)

        raw_receipt = d.receipt_date or fields.receiptDate
        payload["receiptDate"] = ValueSanitizer.fix_datetime_format(raw_receipt) or datetime.now(UTC).strftime("%Y-%m-%dT00:00:00Z")

        payload["fullAddress"] = ValueSanitizer.sanitize_string(d.full_address or fields.fullAddress)
        payload["phone"] = ValueSanitizer.sanitize_string(d.phone or fields.phone)
        payload["email"] = ValueSanitizer.sanitize_string(d.email or fields.email)
        payload["index"] = ValueSanitizer.sanitize_string(d.index or fields.index)

        sub_form = d.submission_form or fields.submissionForm
        payload["submissionForm"] = sub_form if sub_form else SubmissionFormAppeal.WRITTEN

    @staticmethod
    def _add_db_only_fields(d: DocumentAppealDto, payload: dict[str, Any]) -> None:
        if d.subject_id:
            payload["subjectId"] = str(d.subject_id)
        if d.solution_result_id:
            payload["solutionResultId"] = str(d.solution_result_id)
        if d.nomenclature_affair_id:
            payload["nomenclatureAffairId"] = str(d.nomenclature_affair_id)

    def _filter_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ALLOW_NULL = {
            "correspondentAppeal", "correspondentAppealId", "submissionForm",
            "organizationName", "signed", "correspondentOrgNumber", "fioApplicant",
            "fullAddress", "phone", "email", "index", "receiptDate",
            "dateDocCorrespondentOrg", "collective", "anonymous", "reasonably",
        }
        filtered = {k: v for k, v in payload.items() if v is not None or k in _ALLOW_NULL}
        if not filtered.get("declarantType"):
            filtered["declarantType"] = DeclarantType.INDIVIDUAL
        return filtered


class AppealAutofillService:
    MIN_TEXT_LENGTH = 50

    def __init__(
            self,
            doc_client: DocumentClient,
            attach_client: AttachmentClient,
            ref_client: ReferenceClient,
            extraction_service: AppealExtractionService,
            chat_model: BaseChatModel,
    ) -> None:
        self.doc_client = doc_client
        self.attach_client = attach_client
        self.ref_client = ref_client
        self.extraction_service = extraction_service
        self.chat_model = chat_model

    async def process_and_fill(
            self, token: str, document_id: str, attachment_id: str | None, generate_summary_choices: bool = False,
    ) -> AutofillResult:
        document = await self._load_document(token, document_id)
        if document.doc_category_const != "APPEAL":
            raise ValueError(f"Документ должен быть категории APPEAL, а не {document.doc_category_const}")

        target_attachment, warnings = AttachmentSelector.select(document, attachment_id)
        extracted_text = await self._extract_text(token, document_id, target_attachment)
        if not extracted_text or len(extracted_text) < self.MIN_TEXT_LENGTH:
            raise ValueError("Текст не извлечен или слишком короткий")

        fields = await self.extraction_service.extract_appeal_fields(extracted_text)
        await self._update_document(token, document_id, document, fields, extracted_text)

        summary_choices = None
        if generate_summary_choices:
            summary_choices = await self.generate_summary_variants(extracted_text, fields.shortSummary)

        return AutofillResult(
            status="success",
            message="Документ успешно заполнен",
            warnings=warnings if warnings else None,
            attachment_used=getattr(target_attachment, "name", "unknown"),
            last_extracted_text=extracted_text,
            last_short_summary=fields.shortSummary,
            summary_choices=summary_choices,
        )

    async def _load_document(self, token: str, document_id: str) -> DocumentDto:
        doc = await self.doc_client.get_document_metadata(token, document_id)
        if not doc:
            raise ValueError(f"Документ {document_id} не найден")
        return doc

    async def _extract_text(self, token: str, document_id: str, attachment: Any) -> str:
        file_bytes = await self.attach_client.get_attachment_content(token, document_id, str(getattr(attachment, "id", "")))
        if not file_bytes: return ""
        return extract_text_from_bytes(file_bytes, getattr(attachment, "name", ""))

    async def _update_document(self, token: str, document_id: str, document: DocumentDto, fields: AppealFields, extracted_text: str) -> None:
        await self._execute_main_fields_update(token, document_id, document, fields)

        geo_resolver = GeographyResolver(self.ref_client, token)
        geo_data = await geo_resolver.resolve_geography(document, fields)
        fields_builder = AppealFieldsBuilder(self.ref_client, token)
        appeal_payload = await fields_builder.build(document, fields, extracted_text, geo_data)

        await DocumentOperationExecutor.execute(self.doc_client, token, document_id, "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE", appeal_payload)

    async def _execute_main_fields_update(self, token: str, document_id: str, document: DocumentDto, fields: AppealFields) -> None:
        delivery_id = document.delivery_method_id
        if not delivery_id:
            name = fields.deliveryMethod or "Электронно"
            delivery_id = await self.ref_client.find_delivery_method(token, name)

        raw_summary = fields.shortSummary or document.short_summary
        if raw_summary and len(raw_summary) > 80:
            raw_summary = raw_summary[:80]

        main_payload = {
            "shortSummary": raw_summary,
            "deliveryMethodId": delivery_id,
            "documentTypeId": str(document.document_type.id) if document.document_type else None,
        }
        await DocumentOperationExecutor.execute(self.doc_client, token, document_id, "DOCUMENT_MAIN_FIELDS_UPDATE", {k:v for k,v in main_payload.items() if v is not None})

    async def generate_summary_variants(self, text: str, current_summary: str | None) -> list[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Сформулируй РОВНО 3 варианта краткого содержания обращения (до 80 симв)."),
            ("user", "Текст:\n{text}"),
        ])
        chain = prompt | self.chat_model | StrOutputParser()
        try:
            result = await chain.ainvoke({"text": text[:2000]})
            return [v.strip() for v in result.strip().split("\n") if v.strip()][:3]
        except Exception:
            return [current_summary] if current_summary else []
