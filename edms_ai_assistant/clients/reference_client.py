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

        Workflow:
        1. FTS поиск города → получаем ID
        2. GET /api/city/{id}?district=true → получаем city + embedded district
        3. district.regionId → GET /api/region/{id} → получаем область

        Returns:
            Dict: {id, name, regionId, regionName, districtId, districtName}
        """
        logger.info(f"[REFERENCE-CLIENT] 🔍 Finding city with hierarchy: '{city_name}'")

        # ШАГ 1: FTS поиск города
        try:
            fts_result = await self._make_request(
                "GET",
                "api/city/fts-name",
                token=token,
                params={"fts": city_name.strip()},
            )

            if not fts_result:
                logger.warning(
                    f"[REFERENCE-CLIENT] ⚠️ City not found via FTS: '{city_name}'"
                )
                return None

            fts_city = fts_result[0] if isinstance(fts_result, list) else fts_result
            city_id = str(fts_city.get("id"))

            logger.debug(f"[REFERENCE-CLIENT] City FTS result: ID={city_id}")

        except Exception as e:
            logger.error(
                f"[REFERENCE-CLIENT] City FTS search error: {e}", exc_info=True
            )
            return None

        # ШАГ 2: Получаем ПОЛНЫЕ данные города с district И region
        try:
            logger.debug(
                f"[REFERENCE-CLIENT] 📡 Fetching city data: GET /api/city/{city_id}?district=true"
            )

            city_dto = await self._make_request(
                "GET",
                f"api/city/{city_id}",
                token=token,
                params={"district": "true"},  # Загружаем district с region
            )

            if not city_dto:
                logger.warning(
                    f"[REFERENCE-CLIENT] City GET returned empty: {city_id}"
                )
                return None

            logger.debug(f"[REFERENCE-CLIENT] City DTO keys: {list(city_dto.keys())}")

        except Exception as e:
            logger.error(f"[REFERENCE-CLIENT] City GET error: {e}", exc_info=True)
            return None

        # ШАГ 3: Формируем базовый response (API использует nameCity!)
        response = {
            "id": city_id,
            "name": city_dto.get("nameCity") or city_dto.get("cityName") or city_name,
        }

        # ШАГ 4: Извлекаем DISTRICT и REGION
        district_id = city_dto.get("districtId")
        district_obj = city_dto.get("district")

        if district_id:
            try:
                # Если district уже загружен как объект - используем его
                if district_obj and isinstance(district_obj, dict):
                    response["districtId"] = str(district_id)
                    response["districtName"] = (
                        district_obj.get("nameDistrict")
                        or district_obj.get("districtName")
                        or district_obj.get("name")
                    )
                    logger.info(
                        f"[REFERENCE-CLIENT] District from embedded object: "
                        f"{response['districtName']} (ID: {district_id})"
                    )

                    # Получаем REGION из district (может быть объект или ID)
                    district_region_obj = district_obj.get(
                        "region"
                    )  # Embedded region object?
                    district_region_id = district_obj.get("regionId")

                    if district_region_obj and isinstance(district_region_obj, dict):
                        # Region уже загружен как объект внутри district
                        response["regionId"] = str(district_region_obj.get("id"))
                        response["regionName"] = district_region_obj.get(
                            "regionName"
                        ) or district_region_obj.get("name")
                        logger.info(
                            f"[REFERENCE-CLIENT] Region from district.region embedded object: "
                            f"{response['regionName']} (ID: {response['regionId']})"
                        )
                    elif district_region_id:
                        # Fallback: запрашиваем region
                        response["regionId"] = str(district_region_id)

                        # Делаем запрос на получение полного имени региона
                        try:
                            region_dto = await self._make_request(
                                "GET", f"api/region/{district_region_id}", token=token
                            )
                            if region_dto:
                                response["regionName"] = region_dto.get(
                                    "regionName"
                                ) or region_dto.get("name")
                                logger.info(
                                    f"[REFERENCE-CLIENT] Region via fallback request: "
                                    f"{response['regionName']} (ID: {district_region_id})"
                                )
                        except Exception as e:
                            logger.warning(
                                f"[REFERENCE-CLIENT] ️ Failed to fetch region: {e}"
                            )
                else:
                    # Fallback: делаем отдельный запрос на district С REGION
                    logger.debug(
                        f"[REFERENCE-CLIENT] 📡 Fetching district with region: GET /api/district/{district_id}?region=true"
                    )

                    district_dto = await self._make_request(
                        "GET",
                        f"api/district/{district_id}",
                        token=token,
                        params={"region": "true"},
                    )

                    if district_dto:
                        response["districtId"] = str(district_id)
                        response["districtName"] = (
                            district_dto.get("nameDistrict")
                            or district_dto.get("districtName")
                            or district_dto.get("name")
                        )
                        logger.info(
                            f"[REFERENCE-CLIENT] District via separate request: "
                            f"{response['districtName']} (ID: {district_id})"
                        )

                        # Получаем REGION из district (может быть объект или ID)
                        district_region_obj = district_dto.get(
                            "region"
                        )  # Embedded object?
                        district_region_id = district_dto.get("regionId")

                        if district_region_obj and isinstance(
                            district_region_obj, dict
                        ):
                            # Region уже загружен как объект
                            response["regionId"] = str(district_region_obj.get("id"))
                            response["regionName"] = district_region_obj.get(
                                "regionName"
                            ) or district_region_obj.get("name")
                            logger.info(
                                f"[REFERENCE-CLIENT] ✅ Region from district.region embedded object: "
                                f"{response['regionName']} (ID: {response['regionId']})"
                            )
                        elif district_region_id:
                            # Fallback: запрашиваем region отдельно
                            response["regionId"] = str(district_region_id)
                            try:
                                region_dto = await self._make_request(
                                    "GET",
                                    f"api/region/{district_region_id}",
                                    token=token,
                                )
                                if region_dto:
                                    response["regionName"] = region_dto.get(
                                        "regionName"
                                    ) or region_dto.get("name")
                                    logger.info(
                                        f"[REFERENCE-CLIENT] ✅ Region via fallback request: "
                                        f"{response['regionName']} (ID: {district_region_id})"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"[REFERENCE-CLIENT] ⚠️ Failed to fetch region: {e}"
                                )

            except Exception as e:
                logger.warning(f"[REFERENCE-CLIENT] ⚠️ District resolution error: {e}")
        else:
            logger.debug(f"[REFERENCE-CLIENT] ℹ️ City has no districtId")

        logger.info(
            f"[REFERENCE-CLIENT] ✅ City hierarchy complete: "
            f"city={response.get('name')}, "
            f"region={response.get('regionName', 'N/A')}, "
            f"district={response.get('districtName', 'N/A')}"
        )

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
