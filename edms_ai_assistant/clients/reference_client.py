# edms_ai_assistant/clients/reference_client.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.reference import (
    BasicSearchRequest,
    CitizenTypeDto,
    CitizenTypeRequest,
    CityDto,
    CityFilter,
    CityHierarchyDto,
    ReferenceItemDto,
    SubjectDto,
)

if TYPE_CHECKING:
    from uuid import UUID

    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)


class ReferenceClient(EdmsBaseClient):
    """Client for EDMS reference book API."""

    _CANONICAL_NAME_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
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

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def find_entity_with_name(
        self,
        token: str,
        endpoint: str,
        search_name: str,
        entity_label: str | None = None,
    ) -> ReferenceItemDto | None:
        if not search_name or not search_name.strip():
            return None

        search_query = search_name.strip()
        logger.info(
            "Searching %s with name: %s", entity_label or endpoint, search_query
        )

        try:
            fts_result = await self.make_request(
                "GET",
                f"api/{endpoint}/fts-name",
                token=token,
                params={"fts": search_query},
            )
        except EdmsNotFoundError:
            return None

        if not fts_result:
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

        try:
            record = await self.make_request(
                "GET",
                f"api/{endpoint}/{entity_id}",
                token=token,
            )
            if isinstance(record, dict) and record:
                name = self._extract_canonical_name(record, endpoint) or search_query
                return ReferenceItemDto(id=entity_id, name=name)
        except EdmsNotFoundError:
            pass

        name = self._extract_canonical_name(fts_data, endpoint) or search_query
        return ReferenceItemDto(id=entity_id, name=name)

    def _extract_canonical_name(
        self, record: dict[str, Any], endpoint: str
    ) -> str | None:
        priority_fields = self._CANONICAL_NAME_FIELDS.get(
            endpoint, ("name", "shortName", "fullName")
        )
        for field in priority_fields:
            val = record.get(field)
            if val and isinstance(val, str) and val.strip():
                return cast("str", val.strip())
        return None

    async def find_country_with_name(
        self, token: str, name: str
    ) -> ReferenceItemDto | None:
        return await self.find_entity_with_name(token, "country", name, "Страна")

    async def find_region_with_name(
        self, token: str, name: str
    ) -> ReferenceItemDto | None:
        return await self.find_entity_with_name(token, "region", name, "Регион")

    async def find_city_with_hierarchy(
        self, token: str, city_name: str
    ) -> CityHierarchyDto | None:
        if not city_name or not city_name.strip():
            return None

        try:
            fts_result = await self.make_request(
                "GET",
                "api/city/fts-name",
                token=token,
                params={"fts": city_name.strip()},
            )
            if not fts_result:
                return None

            fts_city = fts_result[0] if isinstance(fts_result, list) else fts_result
            if not isinstance(fts_city, dict) or not fts_city.get("id"):
                return None

            city_id = fts_city["id"]
            city_dto_raw = await self.make_request(
                "GET",
                f"api/city/{city_id}",
                token=token,
                params={"includes": "DISTRICT_WITH_REGION"},
            )

            if not isinstance(city_dto_raw, dict):
                return None

            district = city_dto_raw.get("district") or {}
            region = district.get("region") or {}

            return CityHierarchyDto(
                id=city_id,
                name=city_dto_raw.get("nameCity") or city_name,
                district_id=district.get("id"),
                district_name=district.get("nameDistrict"),
                region_id=region.get("id"),
                region_name=region.get("nameRegion"),
            )
        except EdmsNotFoundError:
            return None

    async def get_parent_subjects(self, token: str) -> list[SubjectDto]:
        try:
            return await self._request_list(
                "GET",
                "api/subject/parents",
                token,
                SubjectDto,
                params={"listAttribute": "true"},
            )
        except EdmsNotFoundError:
            return []

    # ══════════════════════════════════════════════════════════════════════════════
    # Citizen Types
    # ══════════════════════════════════════════════════════════════════════════════

    async def get_citizen_types(
        self, token: str, search: BasicSearchRequest | None = None
    ) -> list[CitizenTypeDto]:
        """GET api/citizen-type"""
        params = search.model_dump(exclude_none=True) if search else {}
        result = await self.make_request(
            "GET", "api/citizen-type", token=token, params=params
        )
        # Assuming SliceDto structure but returning content as list for simplicity if used in tools
        if isinstance(result, dict) and "content" in result:
            return [CitizenTypeDto.model_validate(item) for item in result["content"]]
        return [CitizenTypeDto.model_validate(item) for item in result]

    async def get_citizen_type(
        self, token: str, citizen_type_id: UUID
    ) -> CitizenTypeDto:
        """GET api/citizen-type/{id}"""
        result = await self.make_request(
            "GET", f"api/citizen-type/{citizen_type_id}", token=token
        )
        return CitizenTypeDto.model_validate(result)

    async def create_citizen_type(
        self, token: str, request: CitizenTypeRequest
    ) -> CitizenTypeDto:
        """POST api/citizen-type"""
        result = await self.make_request(
            "POST",
            "api/citizen-type",
            token=token,
            json_data=request.model_dump(exclude_none=True),
        )
        return CitizenTypeDto.model_validate(result)

    async def update_citizen_type(
        self, token: str, request: CitizenTypeRequest
    ) -> CitizenTypeDto:
        """PUT api/citizen-type"""
        result = await self.make_request(
            "PUT",
            "api/citizen-type",
            token=token,
            json_data=request.model_dump(exclude_none=True),
        )
        return CitizenTypeDto.model_validate(result)

    async def delete_citizen_types(self, token: str, ids: list[UUID]):
        """DELETE api/citizen-type"""
        await self.make_request(
            "DELETE",
            "api/citizen-type",
            token=token,
            json_data={"ids": [str(i) for i in ids]},
        )

    async def search_citizen_type_fts(self, token: str, fts: str) -> CitizenTypeDto:
        """GET api/citizen-type/fts-name"""
        result = await self.make_request(
            "GET", "api/citizen-type/fts-name", token=token, params={"fts": fts}
        )
        return CitizenTypeDto.model_validate(result)

    # ══════════════════════════════════════════════════════════════════════════════
    # Cities
    # ══════════════════════════════════════════════════════════════════════════════

    async def get_cities_by_region(
        self, token: str, region_id: UUID, filter: CityFilter | None = None
    ) -> list[CityDto]:
        """GET api/city/{regionId}/region"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        result = await self.make_request(
            "GET", f"api/city/{region_id}/region", token=token, params=params
        )
        return [CityDto.model_validate(item) for item in result]

    async def get_cities(
        self, token: str, filter: CityFilter | None = None, list_attribute: bool = True
    ) -> list[CityDto]:
        """GET api/city"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        params["listAttribute"] = str(list_attribute).lower()
        result = await self.make_request("GET", "api/city", token=token, params=params)
        if isinstance(result, dict) and "content" in result:
            return [CityDto.model_validate(item) for item in result["content"]]
        return [CityDto.model_validate(item) for item in result]

    async def get_cities_by_district(
        self,
        token: str,
        district_id: UUID,
        filter: CityFilter | None = None,
        list_attribute: bool = True,
    ) -> list[CityDto]:
        """GET api/city/{districtId}/district"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        params["listAttribute"] = str(list_attribute).lower()
        result = await self.make_request(
            "GET", f"api/city/{district_id}/district", token=token, params=params
        )
        if isinstance(result, dict) and "content" in result:
            return [CityDto.model_validate(item) for item in result["content"]]
        return [CityDto.model_validate(item) for item in result]

    async def get_city(
        self, token: str, city_id: UUID, filter: CityFilter | None = None
    ) -> CityDto:
        """GET api/city/{id}"""
        params = filter.model_dump(exclude_none=True) if filter else {}
        result = await self.make_request(
            "GET", f"api/city/{city_id}", token=token, params=params
        )
        return CityDto.model_validate(result)

    async def create_city(self, token: str, city: CityDto) -> CityDto:
        """POST api/city"""
        result = await self.make_request(
            "POST",
            "api/city",
            token=token,
            json_data=city.model_dump(exclude_none=True),
        )
        return CityDto.model_validate(result)

    async def update_city(self, token: str, city: CityDto) -> CityDto:
        """PUT api/city"""
        result = await self.make_request(
            "PUT", "api/city", token=token, json_data=city.model_dump(exclude_none=True)
        )
        return CityDto.model_validate(result)

    async def delete_cities(self, token: str, ids: list[UUID]):
        """DELETE api/city"""
        await self.make_request(
            "DELETE", "api/city", token=token, json_data={"ids": [str(i) for i in ids]}
        )

    async def search_city_fts(self, token: str, fts: str) -> CityDto:
        """GET api/city/fts-name"""
        result = await self.make_request(
            "GET", "api/city/fts-name", token=token, params={"fts": fts}
        )
        return CityDto.model_validate(result)

    async def find_citizen_type(self, token: str, name: str) -> str | None:
        """Legacy helper: возвращает только id."""
        result = await self.find_entity_with_name(
            token, "citizen-type", name, "Вид обращения"
        )
        return str(result.id) if result and result.id else None

    async def find_delivery_method(self, token: str, name: str) -> str | None:
        """Legacy helper: возвращает только id."""
        result = await self.find_entity_with_name(
            token, "delivery-method", name, "Способ доставки"
        )
        if not result and name != "Курьер":
            result = await self.find_entity_with_name(
                token, "delivery-method", "Курьер", "Способ доставки"
            )
        return str(result.id) if result and result.id else None

    async def find_best_subject(self, token: str, text: str) -> str | None:
        """Ищет тематику обращения через FTS по тексту."""
        if not text or not text.strip():
            return None
        try:
            search_query = text.strip()[:200]
            fts_result = await self.make_request(
                "GET", "api/subject/fts-name", token=token, params={"fts": search_query}
            )
            if isinstance(fts_result, list) and fts_result:
                subject_id = fts_result[0].get("id")
                return str(subject_id) if subject_id else None
        except (EdmsNotFoundError, Exception):
            pass
        return None

    async def get_child_subjects(self, token: str, parent_id: str) -> list[SubjectDto]:
        try:
            return await self._request_list(
                "GET", f"api/subject/parent/{parent_id}", token, SubjectDto
            )
        except EdmsNotFoundError:
            return []
