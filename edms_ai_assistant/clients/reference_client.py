# edms_ai_assistant/clients/reference_client.py
import logging
from typing import Any

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.reference import (
    CityHierarchyDto,
    ReferenceItemDto,
    SubjectDto,
)

logger = logging.getLogger(__name__)


class ReferenceClient:
    """
    Client for EDMS reference book (справочники) API.
    """

    _CANONICAL_NAME_FIELDS: dict[str, tuple[str, ...]] = {
        "country": ("fullName", "name", "shortName"),
        "region": ("nameRegion", "name", "shortName"),
        "district": ("nameDistrict", "name", "shortName"),
        "city": ("nameCity", "name", "shortName"),
        "citizen-type": ("name", "shortName"),
        "correspondent": ("name", "fullName", "shortName"),
        "delivery-method": ("name", "shortName"),
        "department": ("name", "fullName", "shortName"),
        "group": ("name", "shortName"),
    }

    def __init__(self, base_client: EdmsBaseClient):
        self._client = base_client

    async def _find_entity_with_name(
            self,
            token: str,
            endpoint: str,
            search_name: str,
            entity_label: str,
    ) -> ReferenceItemDto | None:
        """
        Two-step reference lookup: fts-name → GET /{id} → canonical name.

        Args:
            token: JWT authorization token.
            endpoint: API endpoint prefix (e.g. "region", "district", "city").
            search_name: Human-readable name to search for.
            entity_label: Русское название для логов (e.g. "Район").

        Returns:
            ReferenceItemDto из справочника, или None.
        """
        if not search_name or not search_name.strip():
            logger.debug("Пропуск поиска %s: пустое значение", entity_label)
            return None

        search_query = search_name.strip()

        # ── Шаг 1: FTS-поиск → получаем id ──────────────────────────────────
        try:
            logger.info("Поиск %s: '%s'", entity_label, search_query)
            fts_result = await self._client._make_request(
                "GET",
                f"api/{endpoint}/fts-name",
                token=token,
                params={"fts": search_query},
            )
        except EdmsNotFoundError:
            logger.warning("%s не найден в справочнике: '%s'", entity_label, search_query)
            return None

        if not fts_result:
            logger.warning("%s не найден по FTS: '%s'", entity_label, search_query)
            return None

        fts_data = (
            fts_result[0]
            if isinstance(fts_result, list) and fts_result
            else (fts_result if isinstance(fts_result, dict) else None)
        )
        if not fts_data or not isinstance(fts_data, dict):
            return None

        entity_id = fts_data.get("id")
        if not entity_id:
            return None

        # ── Шаг 2: GET /{id} → канонический name из справочника ─────────────
        try:
            record = await self._client._make_request(
                "GET",
                f"api/{endpoint}/{entity_id}",
                token=token,
            )
            if isinstance(record, dict) and record:
                name = self._extract_canonical_name(record, endpoint) or search_query
                return ReferenceItemDto(id=entity_id, name=name)
        except EdmsNotFoundError:
            pass

        # Fallback если GET /{id} упал с 404 или вернул пустой ответ
        name = self._extract_canonical_name(fts_data, endpoint) or search_query
        return ReferenceItemDto(id=entity_id, name=name)

    def _extract_canonical_name(
            self, record: dict[str, Any], endpoint: str
    ) -> str | None:
        """
        Extracts the canonical name from a DB record using endpoint-specific
        field priority from _CANONICAL_NAME_FIELDS.
        """
        priority_fields = self._CANONICAL_NAME_FIELDS.get(
            endpoint, ("name", "shortName", "fullName")
        )
        for field in priority_fields:
            val = record.get(field)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return None

    async def _find_entity_id(
            self, token: str, endpoint: str, name: str, entity_label: str
    ) -> str | None:
        """Legacy helper: возвращает только id (без name)."""
        result = await self._find_entity_with_name(token, endpoint, name, entity_label)
        return str(result.id) if result and result.id else None

    # ─────────────────────────────────────────────────────────────────
    # Гео-справочники: {id, canonical name}
    # ─────────────────────────────────────────────────────────────────

    async def find_country_with_name(
            self, token: str, name: str
    ) -> ReferenceItemDto | None:
        """Two-step country lookup → ReferenceItemDto."""
        return await self._find_entity_with_name(token, "country", name, "Страна")

    async def find_region_with_name(
            self, token: str, name: str
    ) -> ReferenceItemDto | None:
        """Two-step region lookup → ReferenceItemDto."""
        return await self._find_entity_with_name(token, "region", name, "Регион")

    async def find_district_with_name(
            self, token: str, name: str
    ) -> ReferenceItemDto | None:
        """Two-step district lookup → ReferenceItemDto."""
        return await self._find_entity_with_name(token, "district", name, "Район")

    async def find_city_with_name(self, token: str, name: str) -> ReferenceItemDto | None:
        """Two-step city lookup → ReferenceItemDto."""
        return await self._find_entity_with_name(token, "city", name, "Город")

    async def find_city_with_hierarchy(
            self, token: str, city_name: str
    ) -> CityHierarchyDto | None:
        if not city_name or not city_name.strip():
            return None

        try:
            fts_result = await self._client._make_request(
                "GET", "api/city/fts-name", token=token, params={"fts": city_name.strip()}
            )
        except EdmsNotFoundError:
            return None

        if not fts_result:
            return None

        fts_city = fts_result[0] if isinstance(fts_result, list) else fts_result
        if not isinstance(fts_city, dict) or not fts_city.get("id"):
            return None

        city_id = fts_city["id"]

        try:
            city_dto = await self._client._make_request(
                "GET", f"api/city/{city_id}", token=token, params={"includes": "DISTRICT_WITH_REGION"}
            )
        except EdmsNotFoundError:
            return None

        if not isinstance(city_dto, dict):
            return None

        # Безопасное извлечение вложенных структур
        district = city_dto.get("district") or {}
        region = district.get("region") or {}

        # Создаем DTO иммутабельно за один вызов
        return CityHierarchyDto(
            id=city_id,
            name=city_dto.get("nameCity") or city_name,
            district_id=district.get("id"),
            district_name=district.get("nameDistrict"),
            region_id=region.get("id"),
            region_name=region.get("nameRegion")
        )

    async def find_best_subject(self, token: str, text: str) -> str | None:
        """
        Ищет тематика обращения через FTS по тексту.
        Возвращает UUID тематики.
        """
        if not text or not text.strip():
            return None

        try:
            search_query = text.strip()[:200]

            fts_result = await self._client._make_request(
                "GET",
                "api/subject/fts-name",
                token=token,
                params={"fts": search_query},
            )

            if isinstance(fts_result, list) and fts_result:
                subject_id = fts_result[0].get("id")
                return str(subject_id) if subject_id else None

        except EdmsNotFoundError:
            logger.warning("Subject not found for text: '%s...'", search_query[:50])
        except Exception as exc:
            logger.error("Subject FTS lookup error: %s", exc)

        return None

    # ─────────────────────────────────────────────────────────────────
    # Legacy API (только id, обратная совместимость)
    # ─────────────────────────────────────────────────────────────────

    async def find_country(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "country", name, "Страна")

    async def find_region(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "region", name, "Регион")

    async def find_district(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "district", name, "Район")

    async def find_city(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "city", name, "Город")

    async def find_citizen_type(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "citizen-type", name, "Вид обращения")

    async def find_correspondent(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "correspondent", name, "Корреспондент")

    async def find_delivery_method(self, token: str, name: str) -> str | None:
        result = await self._find_entity_id(token, "delivery-method", name, "Способ доставки")
        if not result and name != "Курьер":
            logger.info("Fallback: используем 'Курьер'")
            return await self._find_entity_id(token, "delivery-method", "Курьер", "Способ доставки")
        return result

    async def find_department(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "department", name, "Подразделение")

    async def find_group(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "group", name, "Группа")

    # ─────────────────────────────────────────────────────────────────
    # Тематики (Только API вызовы)
    # ─────────────────────────────────────────────────────────────────

    async def get_parent_subjects(self, token: str) -> list[SubjectDto]:
        try:
            result = await self._client._make_request(
                "GET", "api/subject/parents", token=token, params={"listAttribute": "true"}
            )
            if isinstance(result, list):
                return [SubjectDto.model_validate(s) for s in result]
        except EdmsNotFoundError:
            pass
        return []

    async def get_child_subjects(self, token: str, parent_id: str) -> list[SubjectDto]:
        try:
            result = await self._client._make_request(
                "GET", f"api/subject/parent/{parent_id}", token=token
            )
            if isinstance(result, list):
                return [SubjectDto.model_validate(s) for s in result]
        except EdmsNotFoundError:
            pass
        return []
