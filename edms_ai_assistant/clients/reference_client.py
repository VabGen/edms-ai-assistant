# edms_ai_assistant/clients/reference_client.py
import logging
from typing import Optional, Dict, Any
from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class ReferenceClient(EdmsHttpClient):

    async def _find_entity_with_name(
        self,
        token: str,
        endpoint: str,
        search_name: str,
        entity_label: str,
        name_field: str = "shortName",
    ) -> Optional[Dict[str, str]]:
        if not search_name or not search_name.strip():
            logger.debug(
                f"[REFERENCE-CLIENT] Пропуск поиска {entity_label}: пустое значение"
            )
            return None

        search_query = search_name.strip()

        try:
            logger.info(f"[REFERENCE-CLIENT] Поиск {entity_label}: '{search_query}'")

            result = await self._make_request(
                "GET",
                f"api/{endpoint}/fts-name",
                token=token,
                params={"fts": search_query},
            )

            if not result:
                logger.warning(
                    f"[REFERENCE-CLIENT] {entity_label} не найден: '{search_query}'"
                )
                return None

            data = (
                result[0]
                if isinstance(result, list) and len(result) > 0
                else (result if isinstance(result, dict) else None)
            )

            if not data:
                logger.warning(
                    f"[REFERENCE-CLIENT] Пустой ответ для {entity_label}: '{search_query}'"
                )
                return None

            entity_id = str(data.get("id"))

            entity_name = (
                data.get(name_field)
                or data.get("fullName")
                or data.get("name")
                or data.get("shortName")
                or search_query
            )

            logger.info(
                f"[REFERENCE-CLIENT] {entity_label}: '{search_query}' → "
                f"ID: {entity_id}, Name: '{entity_name}'"
            )

            return {"id": entity_id, "name": entity_name}

        except Exception as e:
            logger.error(
                f"[REFERENCE-CLIENT] Ошибка поиска {entity_label} '{search_query}': {e}",
                exc_info=True,
            )
            return None

    async def _find_entity_id(
        self, token: str, endpoint: str, name: str, entity_label: str
    ) -> Optional[str]:
        result = await self._find_entity_with_name(token, endpoint, name, entity_label)
        return result["id"] if result else None

    # ══════════════════════════════════════════════════════════════════
    # СПРАВОЧНИКИ С ПОДДЕРЖКОЙ {ID, NAME}
    # ══════════════════════════════════════════════════════════════════

    async def find_country_with_name(
        self, token: str, name: str
    ) -> Optional[Dict[str, str]]:
        return await self._find_entity_with_name(
            token, "country", name, "Страна", name_field="fullName"
        )

    async def find_region_with_name(
        self, token: str, name: str
    ) -> Optional[Dict[str, str]]:
        """Поиск региона с возвратом {id, name}."""
        return await self._find_entity_with_name(
            token, "region", name, "Регион", name_field="regionName"
        )

    async def find_district_with_name(
        self, token: str, name: str
    ) -> Optional[Dict[str, str]]:
        """Поиск района с возвратом {id, name}."""
        return await self._find_entity_with_name(
            token, "district", name, "Район", name_field="districtName"
        )

    async def find_city_with_name(
        self, token: str, name: str
    ) -> Optional[Dict[str, str]]:
        """Поиск города с возвратом {id, name}."""
        return await self._find_entity_with_name(
            token, "city", name, "Город", name_field="cityName"
        )

    async def find_city_with_hierarchy(
        self, token: str, city_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск города с автоматическим извлечением региона/района.
        """
        result = await self._make_request(
            "GET", f"api/city/fts-name", token=token, params={"fts": city_name.strip()}
        )

        if not result:
            return None

        city_dto = result[0] if isinstance(result, list) else result

        response = {
            "id": str(city_dto.get("id")),
            "name": city_dto.get("cityName") or city_dto.get("name") or city_name,
        }

        region_id = city_dto.get("regionId")
        if region_id:
            try:
                region_dto = await self._make_request(
                    "GET", f"api/region/{region_id}", token=token
                )
                response["regionId"] = str(region_id)
                response["regionName"] = region_dto.get("regionName") or region_dto.get(
                    "name"
                )
                logger.info(
                    f"[REFERENCE-CLIENT] Регион для '{city_name}': {response['regionName']}"
                )
            except Exception as e:
                logger.warning(f"[REFERENCE-CLIENT] Не удалось получить регион: {e}")

        district_id = city_dto.get("districtId")
        if district_id:
            try:
                district_dto = await self._make_request(
                    "GET", f"api/district/{district_id}", token=token
                )
                response["districtId"] = str(district_id)
                response["districtName"] = district_dto.get(
                    "districtName"
                ) or district_dto.get("name")
                logger.info(
                    f"[REFERENCE-CLIENT] Район для '{city_name}': {response['districtName']}"
                )
            except Exception as e:
                logger.warning(f"[REFERENCE-CLIENT] Не удалось получить район: {e}")

        return response

    # ══════════════════════════════════════════════════════════════════
    # LEGACY API (только ID, для обратной совместимости)
    # ══════════════════════════════════════════════════════════════════

    async def find_country(self, token: str, name: str) -> Optional[str]:
        """Поиск страны (только ID)."""
        return await self._find_entity_id(token, "country", name, "Страна")

    async def find_region(self, token: str, name: str) -> Optional[str]:
        """Поиск региона (только ID)."""
        return await self._find_entity_id(token, "region", name, "Регион")

    async def find_district(self, token: str, name: str) -> Optional[str]:
        """Поиск района (только ID)."""
        return await self._find_entity_id(token, "district", name, "Район")

    async def find_city(self, token: str, name: str) -> Optional[str]:
        """Поиск города (только ID)."""
        return await self._find_entity_id(token, "city", name, "Город")

    async def find_citizen_type(self, token: str, name: str) -> Optional[str]:
        """Поиск вида обращения (только ID)."""
        return await self._find_entity_id(token, "citizen-type", name, "Вид обращения")

    async def find_correspondent(self, token: str, name: str) -> Optional[str]:
        """Поиск корреспондента (только ID)."""
        return await self._find_entity_id(token, "correspondent", name, "Корреспондент")

    async def find_delivery_method(self, token: str, name: str) -> Optional[str]:
        """Поиск способа доставки с fallback на "Курьер"."""
        result = await self._find_entity_id(
            token, "delivery-method", name, "Способ доставки"
        )

        if not result and name != "Курьер":
            logger.info("[REFERENCE-CLIENT] Fallback: используем 'Курьер'")
            return await self._find_entity_id(
                token, "delivery-method", "Курьер", "Способ доставки"
            )

        return result

    async def find_department(self, token: str, name: str) -> Optional[str]:
        """Поиск подразделения (только ID)."""
        return await self._find_entity_id(token, "department", name, "Подразделение")

    async def find_group(self, token: str, name: str) -> Optional[str]:
        """Поиск группы (только ID)."""
        return await self._find_entity_id(token, "group", name, "Группа")

    # ══════════════════════════════════════════════════════════════════
    # ТЕМАТИКИ (SUBJECT)
    # ══════════════════════════════════════════════════════════════════

    async def get_parent_subjects(self, token: str):
        """Получить список родительских тем."""
        try:
            result = await self._make_request(
                "GET",
                "api/subject/parents",
                token=token,
                params={"listAttribute": "true"},
            )
            logger.info(
                f"[REFERENCE-CLIENT] Загружено родительских тем: {len(result) if isinstance(result, list) else 0}"
            )
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"[REFERENCE-CLIENT] Ошибка получения родительских тем: {e}")
            return []

    async def get_child_subjects(self, token: str, parent_id: str):
        """Получить дочерние темы."""
        try:
            result = await self._make_request(
                "GET", f"api/subject/parent/{parent_id}", token=token
            )
            logger.info(
                f"[REFERENCE-CLIENT] Дочерних тем для {parent_id}: {len(result) if isinstance(result, list) else 0}"
            )
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"[REFERENCE-CLIENT] Ошибка получения дочерних тем: {e}")
            return []

    async def find_best_subject(self, token: str, text: str):
        """Поиск подходящей темы через LLM."""
        from edms_ai_assistant.llm import get_chat_model

        parents = await self.get_parent_subjects(token)
        if not parents:
            logger.warning("[REFERENCE-CLIENT] Родительские темы не загружены")
            return None

        themes_text = "\n".join(
            [f"{i + 1}. {s['name']}" for i, s in enumerate(parents)]
        )
        llm = get_chat_model()

        prompt = f"""Выбери ОДНУ наиболее подходящую тему для обращения.

СПИСОК ТЕМ:
{themes_text}

ТЕКСТ ОБРАЩЕНИЯ (фрагмент):
{text[:800]}

Ответь ТОЛЬКО номером (например: 3)"""

        try:
            response = await llm.ainvoke(prompt)
            choice_text = response.content.strip()

            import re

            match = re.search(r"\d+", choice_text)
            if not match:
                logger.warning(
                    f"[REFERENCE-CLIENT] LLM не вернула номер: {choice_text}"
                )
                return None

            index = int(match.group(0)) - 1
            if index < 0 or index >= len(parents):
                logger.warning(f"[REFERENCE-CLIENT] Неверный индекс: {index}")
                return None

            parent = parents[index]
            parent_id = str(parent["id"])
            logger.info(
                f"[REFERENCE-CLIENT] Родительская тема: {parent['name']} (ID: {parent_id})"
            )

            children = await self.get_child_subjects(token, parent_id)
            if not children:
                return parent_id

            children_text = "\n".join(
                [f"{i + 1}. {c['name']}" for i, c in enumerate(children)]
            )

            prompt2 = f"""Выбери ОДНУ наиболее подходящую подтему.

СПИСОК ПОДТЕМ:
{children_text}

ТЕКСТ ОБРАЩЕНИЯ (фрагмент):
{text[:800]}

Ответь ТОЛЬКО номером (например: 2)"""

            response2 = await llm.ainvoke(prompt2)
            choice2_text = response2.content.strip()

            match2 = re.search(r"\d+", choice2_text)
            if not match2:
                logger.warning(
                    f"[REFERENCE-CLIENT] LLM не вернула номер подтемы: {choice2_text}"
                )
                return parent_id

            child_index = int(match2.group(0)) - 1
            if child_index < 0 or child_index >= len(children):
                logger.warning(
                    f"[REFERENCE-CLIENT] Неверный индекс подтемы: {child_index}"
                )
                return parent_id

            child = children[child_index]
            child_id = str(child["id"])
            logger.info(
                f"[REFERENCE-CLIENT] Дочерняя тема: {child['name']} (ID: {child_id})"
            )

            return child_id

        except Exception as e:
            logger.error(f"[REFERENCE-CLIENT] Ошибка выбора темы: {e}", exc_info=True)
            return None
