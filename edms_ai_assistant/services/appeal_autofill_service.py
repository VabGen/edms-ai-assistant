# edms_ai_assistant/services/appeal_autofill_service.py
"""
Сервис оркестрации автозаполнения обращений.
"""

from __future__ import annotations

import json
import logging
import re as _re
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from edms_ai_assistant.clients.attachment_client import AttachmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.reference_client import ReferenceClient
from edms_ai_assistant.domain.appeal_fields import AppealFields, SubmissionFormAppeal
from edms_ai_assistant.generated.resources_openapi import (
    DeclarantType as GeneratedDeclarantType,
)
from edms_ai_assistant.generated.resources_openapi import (
    DocumentAppealDto,
    DocumentDto,
)
from edms_ai_assistant.services.appeal_extraction_service import AppealExtractionService
from edms_ai_assistant.utils.file_utils import extract_text_from_bytes
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Models & Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════════════
# Utility Classes
# ══════════════════════════════════════════════════════════════════════════════


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
                dt = dt.replace(tzinfo=timezone.utc)
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

        logger.debug(
            "Executing %s", operation_type,
            extra={"document_id": document_id, "operation": operation_type},
        )

        try:
            await client.execute_document_operations(token, document_id, json_safe_payload)
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
                (a for a in document.attachmentDocument if str(a.id) == attachment_id), None,
            )
            if not target:
                warnings.append(f"Вложение ID={attachment_id} не найдено, используется автоподбор")

        if not target:
            target = next(
                (a for a in document.attachmentDocument if
                 a.name and a.name.lower().endswith(cls.SUPPORTED_EXTENSIONS)),
                document.attachmentDocument[0],
            )

        logger.info("Attachment selected: %s", target.name)
        return target, warnings


# ══════════════════════════════════════════════════════════════════════════════
# Resolvers & Builders
# ══════════════════════════════════════════════════════════════════════════════


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

    async def resolve_geography(self, document: Any, fields: Any) -> dict[str, Any]:
        geo_data: dict[str, Any] = {}

        city_name = self._get_field(document, fields, "cityName")
        region_name = self._get_field(document, fields, "regionName")
        district_name = self._get_field(document, fields, "districtName")

        await self._resolve_country(document, fields, geo_data)

        if region_name:
            await self._lookup("region", region_name, "regionId", "regionName", geo_data)

        if district_name:
            await self._lookup("district", district_name, "districtId", "districtName", geo_data)

        await self._resolve_city(city_name, document, geo_data)

        logger.info(
            "Geography resolved: country=%s region=%s district=%s city=%s",
            geo_data.get("countryAppealName"), geo_data.get("regionName", "—(auto)"),
            geo_data.get("districtName", "—(auto)"), geo_data.get("cityName"),
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

    async def _resolve_country(self, document: Any, fields: Any, geo_data: dict) -> None:
        d = document.documentAppeal
        country_name: str | None = None

        if d and not ValueSanitizer.is_empty(d.countryAppealName):
            country_name = d.countryAppealName
        elif not ValueSanitizer.is_empty(fields.country):
            country_name = fields.country

        if country_name:
            try:
                data = await self.ref_client.find_country_with_name(self.token, country_name)
                if data:
                    geo_data["countryAppealId"] = str(data.id) if data.id else None
                    geo_data["countryAppealName"] = data.name
                else:
                    logger.warning("Country not found: %s", country_name)
            except Exception as exc:
                logger.error("Country resolution error: %s", exc)
        elif d and d.countryAppealId:
            geo_data["countryAppealId"] = str(d.countryAppealId)

    async def _lookup(self, endpoint: str, name: str, id_key: str, name_key: str, geo_data: dict) -> None:
        try:
            method = getattr(self.ref_client, f"find_{endpoint}_with_name")
            data = await method(self.token, name)
            if data:
                geo_data[id_key] = str(data.id) if data.id else None
                geo_data[name_key] = data.name
            else:
                logger.warning("%s not found in reference: %s", endpoint, name)
        except Exception as exc:
            logger.error("%s lookup error for '%s': %s", endpoint, name, exc)

    async def _resolve_city(self, city_name: str | None, document: Any, geo_data: dict) -> None:
        d = document.documentAppeal
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
                else:
                    logger.warning("City not found in reference: '%s'", city_name)
            except Exception as exc:
                logger.error("City resolution error for '%s': %s", city_name, exc, exc_info=True)
        elif d and d.cityId:
            geo_data["cityId"] = str(d.cityId)


class AppealFieldsBuilder:
    def __init__(self, ref_client: ReferenceClient, token: str) -> None:
        self.ref_client = ref_client
        self.token = token
        self.warnings: list[str] = []

    async def build(
            self, document: DocumentDto, fields: AppealFields, extracted_text: str, geo_data: dict,
    ) -> dict[str, Any]:
        d = document.documentAppeal or DocumentAppealDto()
        payload: dict[str, Any] = {}

        geo_key_order = [
            "countryAppealId", "countryAppealName", "regionId", "regionName",
            "districtId", "districtName", "cityId", "cityName",
        ]
        for key in geo_key_order:
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

    async def _add_correspondent(self, d: DocumentAppealDto, fields: AppealFields, payload: dict) -> None:
        if d.correspondentAppealId:
            payload["correspondentAppealId"] = str(d.correspondentAppealId)
            payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(d.correspondentAppeal)
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
            if canonical and _is_good_correspondent_match(corr_name, canonical.name or ""):
                payload["correspondentAppealId"] = str(canonical.id) if canonical.id else None
                payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(corr_name)
            else:
                payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(corr_name)

        if not payload.get("correspondentAppealId"):
            payload["correspondentAppealId"] = None
            if "correspondentAppeal" not in payload:
                payload["correspondentAppeal"] = None

    @staticmethod
    def _add_personal_data(d: DocumentAppealDto, fields: AppealFields, payload: dict) -> None:
        if not ValueSanitizer.is_empty(d.fioApplicant):
            payload["fioApplicant"] = ValueSanitizer.sanitize_string(d.fioApplicant)
        elif not ValueSanitizer.is_empty(fields.fioApplicant):
            payload["fioApplicant"] = ValueSanitizer.sanitize_string(fields.fioApplicant)

        if d.dateDocCorrespondentOrg:
            payload["dateDocCorrespondentOrg"] = ValueSanitizer.fix_datetime_format(d.dateDocCorrespondentOrg)
        elif not ValueSanitizer.is_empty(fields.dateDocCorrespondentOrg):
            payload["dateDocCorrespondentOrg"] = ValueSanitizer.fix_datetime_format(fields.dateDocCorrespondentOrg)

    async def _add_classification(self, d: DocumentAppealDto, fields: AppealFields, extracted_text: str,
                                  payload: dict) -> None:
        if d.citizenTypeId:
            payload["citizenTypeId"] = str(d.citizenTypeId)
        elif not ValueSanitizer.is_empty(fields.citizenType):
            citizen_type_id = await self.ref_client.find_citizen_type(self.token, fields.citizenType)
            if citizen_type_id:
                payload["citizenTypeId"] = citizen_type_id

        if d.subjectId:
            payload["subjectId"] = str(d.subjectId)
        else:
            subject_id = await self.ref_client.find_best_subject(self.token, extracted_text)
            if subject_id:
                payload["subjectId"] = subject_id

    async def _add_declarant_type(self, d: DocumentAppealDto, fields: AppealFields, payload: dict) -> None:
        if fields.declarantType:
            if isinstance(fields.declarantType, str):
                try:
                    payload["declarantType"] = getattr(GeneratedDeclarantType, fields.declarantType.upper())
                except AttributeError:
                    payload["declarantType"] = GeneratedDeclarantType.INDIVIDUAL
            else:
                payload["declarantType"] = fields.declarantType
        elif d.declarantType:
            payload["declarantType"] = d.declarantType
        else:
            payload["declarantType"] = GeneratedDeclarantType.INDIVIDUAL
            self.warnings.append("declarantType установлен INDIVIDUAL по умолчанию")

    async def _add_conditional_fields(self, d: DocumentAppealDto, fields: AppealFields, payload: dict) -> None:
        if payload.get("declarantType") == GeneratedDeclarantType.ENTITY:
            raw_org: str | None = (
                d.organizationName if not ValueSanitizer.is_empty(d.organizationName) else fields.organizationName
            )
            if not ValueSanitizer.is_empty(raw_org):
                # ИСПРАВЛЕНИЕ: Убрано подчеркивание
                corr_data = await self.ref_client._find_entity_with_name(
                    self.token, "correspondent", raw_org, "Организация"
                )
                if corr_data and _is_good_correspondent_match(raw_org, corr_data.name or ""):
                    payload["organizationName"] = corr_data.name
                else:
                    payload["organizationName"] = ValueSanitizer.sanitize_string(raw_org)
            else:
                payload["organizationName"] = None

            payload["signed"] = ValueSanitizer.sanitize_string(
                d.signed if not ValueSanitizer.is_empty(d.signed) else fields.signed
            )
            payload["correspondentOrgNumber"] = ValueSanitizer.sanitize_string(
                d.correspondentOrgNumber if not ValueSanitizer.is_empty(
                    d.correspondentOrgNumber) else fields.correspondentOrgNumber
            )
        elif payload.get("declarantType") == GeneratedDeclarantType.INDIVIDUAL:
            payload["organizationName"] = None
            payload["signed"] = None
            payload["correspondentOrgNumber"] = None

    # ИСПРАВЛЕНИЕ: Добавлен декоратор @staticmethod
    @staticmethod
    def _add_common_fields(d: DocumentAppealDto, fields: AppealFields, payload: dict) -> None:
        payload["collective"] = fields.collective if fields.collective is not None else d.collective
        payload["anonymous"] = fields.anonymous if fields.anonymous is not None else d.anonymous
        payload["reasonably"] = fields.reasonably if fields.reasonably is not None else d.reasonably

        raw_receipt = d.receiptDate if d.receiptDate else fields.receiptDate
        if raw_receipt:
            payload["receiptDate"] = ValueSanitizer.fix_datetime_format(raw_receipt)
        else:
            today = datetime.now(UTC)
            payload["receiptDate"] = today.strftime("%Y-%m-%dT00:00:00Z")

        payload["fullAddress"] = ValueSanitizer.sanitize_string(
            d.fullAddress if not ValueSanitizer.is_empty(d.fullAddress) else fields.fullAddress)
        payload["phone"] = ValueSanitizer.sanitize_string(
            d.phone if not ValueSanitizer.is_empty(d.phone) else fields.phone)
        payload["email"] = ValueSanitizer.sanitize_string(
            d.email if not ValueSanitizer.is_empty(d.email) else fields.email)
        payload["index"] = ValueSanitizer.sanitize_string(
            d.index if not ValueSanitizer.is_empty(d.index) else fields.index)
        payload["indexDateCoverLetter"] = ValueSanitizer.sanitize_string(
            d.indexDateCoverLetter if not ValueSanitizer.is_empty(
                d.indexDateCoverLetter) else fields.indexDateCoverLetter
        )
        payload["reviewProgress"] = ValueSanitizer.sanitize_string(
            d.reviewProgress if not ValueSanitizer.is_empty(d.reviewProgress) else fields.reviewProgress
        )

        submission_form = getattr(d, "submissionForm", None)
        if not submission_form and fields.submissionForm:
            submission_form = fields.submissionForm
        payload["submissionForm"] = submission_form if submission_form else SubmissionFormAppeal.WRITTEN

    # ИСПРАВЛЕНИЕ: Добавлен декоратор @staticmethod
    @staticmethod
    def _add_db_only_fields(d: DocumentAppealDto, payload: dict) -> None:
        if d.subjectId:
            payload["subjectId"] = str(d.subjectId)
        if d.solutionResultId:
            payload["solutionResultId"] = str(d.solutionResultId)
        if d.nomenclatureAffairId:
            payload["nomenclatureAffairId"] = str(d.nomenclatureAffairId)

    def _filter_payload(self, payload: dict) -> dict[str, Any]:
        _ALLOW_NULL = {
            "correspondentAppeal", "correspondentAppealId", "submissionForm",
            "organizationName", "signed", "correspondentOrgNumber", "fioApplicant",
            "fullAddress", "phone", "email", "index", "receiptDate",
            "dateDocCorrespondentOrg", "collective", "anonymous", "reasonably",
        }

        filtered = {}
        for k, v in payload.items():
            if v is not None or k in _ALLOW_NULL:
                filtered[k] = v

        if not filtered.get("declarantType"):
            raise ValueError("declarantType обязателен, но не установлен")
        if not filtered.get("submissionForm"):
            filtered["submissionForm"] = SubmissionFormAppeal.WRITTEN

        return filtered


# ══════════════════════════════════════════════════════════════════════════════
# Main Service Class
# ══════════════════════════════════════════════════════════════════════════════


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
            self,
            token: str,
            document_id: str,
            attachment_id: str | None,
            generate_summary_choices: bool = False,
    ) -> AutofillResult:
        """Основной метод оркестрации автозаполнения."""

        document = await self._load_document(token, document_id)
        self._validate_document_category(document)

        target_attachment, warnings = AttachmentSelector.select(document, attachment_id)

        extracted_text = await self._extract_text(token, document_id, target_attachment)
        self._validate_text_length(extracted_text)

        fields = await self.extraction_service.extract_appeal_fields(extracted_text)

        await self._update_document(token, document_id, document, fields, extracted_text)

        summary_choices = None
        summary_hint = None
        current_summary = getattr(fields, "shortSummary", None)

        if generate_summary_choices:
            summary_choices = await self.generate_summary_variants(extracted_text, current_summary)
            if summary_choices:
                summary_hint = (
                    "Выберите заголовок или предложите свой вариант. "
                    "Скажите 'Установи заголовок: <текст>' чтобы применить."
                )

        return AutofillResult(
            status="success",
            message="Документ успешно заполнен",
            warnings=warnings if warnings else None,
            attachment_used=target_attachment.name,
            last_extracted_text=extracted_text,
            last_short_summary=current_summary,
            summary_choices=summary_choices,
            summary_choices_hint=summary_hint,
        )

    async def _load_document(self, token: str, document_id: str) -> DocumentDto:
        raw_document = await self.doc_client.get_document_metadata(token, document_id)
        doc = DocumentDto.model_validate(raw_document)
        return doc

    def _validate_document_category(self, document: DocumentDto) -> None:
        if document.docCategoryConstant != "APPEAL":
            raise ValueError(f"Документ должен быть категории APPEAL, а не {document.docCategoryConstant}")

    async def _extract_text(self, token: str, document_id: str, attachment: Any) -> str:
        file_bytes = await self.attach_client.get_attachment_content(
            token, document_id, str(attachment.id)
        )
        extracted_text = extract_text_from_bytes(file_bytes, attachment.name)
        logger.info("Text extracted: %d chars", len(extracted_text))
        return extracted_text

    def _validate_text_length(self, text: str) -> None:
        if not text or len(text) < self.MIN_TEXT_LENGTH:
            raise ValueError("Текст не извлечен или слишком короткий")

    async def _update_document(
            self, token: str, document_id: str, document: DocumentDto, fields: AppealFields, extracted_text: str
    ) -> None:
        await self._execute_main_fields_update(token, document_id, document, fields)
        await self._execute_appeal_fields_update(token, document_id, document, fields, extracted_text)

    async def _execute_main_fields_update(
            self, token: str, document_id: str, document: DocumentDto, fields: AppealFields
    ) -> None:
        delivery_id = document.deliveryMethodId
        if not delivery_id:
            delivery_method_name = (
                fields.deliveryMethod if not ValueSanitizer.is_empty(fields.deliveryMethod) else "Электронно"
            )
            delivery_id = await self.ref_client.find_delivery_method(token, delivery_method_name)

        raw_summary = (
            ValueSanitizer.sanitize_string(fields.shortSummary)
            if not ValueSanitizer.is_empty(fields.shortSummary)
            else document.shortSummary
        )
        if raw_summary and len(raw_summary) > 80:
            raw_summary = raw_summary[:80]

        main_payload = {
            "shortSummary": raw_summary,
            "deliveryMethodId": delivery_id,
            "documentTypeId": str(document.documentTypeId) if document.documentTypeId else None,
            "pages": document.pages,
            "note": document.note,
            "additionalPages": document.additionalPages,
            "exemplarCount": document.exemplarCount,
            "investProgramId": document.investProgramId,
        }
        main_payload = {k: v for k, v in main_payload.items() if v is not None}

        await DocumentOperationExecutor.execute(
            self.doc_client, token, document_id, "DOCUMENT_MAIN_FIELDS_UPDATE", main_payload
        )

    async def _execute_appeal_fields_update(
            self, token: str, document_id: str, document: DocumentDto, fields: AppealFields, extracted_text: str
    ) -> None:
        geo_resolver = GeographyResolver(self.ref_client, token)
        geo_data = await geo_resolver.resolve_geography(document, fields)

        fields_builder = AppealFieldsBuilder(self.ref_client, token)
        appeal_payload = await fields_builder.build(document, fields, extracted_text, geo_data)

        await DocumentOperationExecutor.execute(
            self.doc_client, token, document_id, "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE", appeal_payload
        )

    async def generate_summary_variants(self, text: str, current_summary: str | None) -> list[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Ты — эксперт по делопроизводству. Сформулируй РОВНО 3 варианта краткого содержания (заголовка) обращения. "
                    "Каждый вариант — отдельная строка, максимум 80 символов. "
                    "Стили: 1) краткий (суть в 5-8 словах), 2) официальный (с указанием типа обращения), "
                    "3) описательный (с ключевыми деталями). Отвечай ТОЛЬКО тремя строками без нумерации.",
                ),
                ("user", "Текст обращения:\n{text}"),
            ]
        )
        chain = prompt | self.chat_model | StrOutputParser()

        try:
            result = await chain.ainvoke({"text": text[:2000]})
            variants = [v.strip() for v in result.strip().split("\n") if v.strip()]
            variants = [v[:77] + "..." if len(v) > 80 else v for v in variants[:3]]
            if current_summary and current_summary not in variants:
                variants = [(current_summary[:77] + "..." if len(
                    current_summary) > 80 else current_summary)] + variants[:2]
            return variants[:3]
        except Exception as exc:
            logger.warning("Failed to generate summary variants: %s", exc)
            return [current_summary] if current_summary else []
