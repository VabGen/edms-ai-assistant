# edms_ai_assistant/clients/reference_client.py
"""
Унифицированный клиент для работы со справочниками EDMS через REST API.

Все справочники используют единый паттерн поиска: GET /api/{entity}/fts-name?fts=...
Возвращают либо DTO объект, либо 404 ResourceNotFoundException.
"""
import logging
from typing import Optional
from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class ReferenceClient(EdmsHttpClient):
    """
    Универсальный клиент для поиска записей в справочниках СЭД.

    Реализует unified interface для всех справочников, используя
    полнотекстовый поиск (FTS - Full Text Search) по имени.

    Поддерживаемые справочники:
    - Категории граждан (citizen-type)
    - География: страна, регион, район, город
    - Корреспонденты
    - Способы доставки
    - Организационная структура: подразделения, группы

    Examples:
        ...     async with ReferenceClient() as client:
        ...     country_id = await client.find_country(token, "Беларусь")
        ...     print(country_id)  # "5bf864db-113d-435d-99ef-b6858323791f"
    """

    async def _find_entity_id(
        self, token: str, endpoint: str, name: str, entity_label: str
    ) -> Optional[str]:
        """
        Универсальный метод поиска ID сущности в справочнике.

        Выполняет GET-запрос к эндпоинту /api/{endpoint}/fts-name?fts={name}
        и извлекает ID из первого найденного элемента.

        Args:
            token: JWT токен авторизации
            endpoint: Название эндпоинта справочника (напр. 'city', 'country')
            name: Текстовое наименование для поиска
            entity_label: Человекочитаемое название для логирования

        Returns:
            UUID сущности в виде строки или None, если не найдено

        Raises:
            Exception: При технических ошибках HTTP-запроса

        Note:
            API может возвращать как одиночный объект, так и массив объектов.
            Метод обрабатывает оба варианта.
        """
        if not name or not name.strip():
            logger.debug(f"Пропуск поиска {entity_label}: пустое значение")
            return None

        search_query = name.strip()

        try:
            logger.debug(f"🔍 Поиск {entity_label} в СЭД: '{search_query}'")

            # Выполнение GET-запроса к справочнику
            result = await self._make_request(
                "GET",
                f"api/{endpoint}/fts-name",
                token=token,
                params={"fts": search_query},
            )

            if not result:
                logger.warning(
                    f"❌ {entity_label} по запросу '{search_query}' не найден (пустой ответ)"
                )
                return None

            # Обработка ответа (может быть List[DTO] или одиночный DTO)
            data = None
            if isinstance(result, list):
                if len(result) > 0:
                    data = result[0]
                    if len(result) > 1:
                        logger.debug(
                            f"ℹ️ Найдено несколько совпадений для {entity_label} '{search_query}', "
                            f"используется первое"
                        )
            elif isinstance(result, dict):
                data = result

            if data and data.get("id"):
                entity_id = str(data.get("id"))
                logger.info(
                    f"✅ Успешное сопоставление {entity_label}: '{search_query}' → ID: {entity_id}"
                )
                return entity_id

            logger.warning(
                f"⚠️ ID для {entity_label} '{search_query}' отсутствует в теле ответа"
            )
            return None

        except Exception as e:
            logger.error(
                f"❌ Техническая ошибка при поиске {entity_label} '{search_query}': "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # СПРАВОЧНИКИ ДЛЯ КАРТОЧКИ ОБРАЩЕНИЯ (APPEAL)
    # ══════════════════════════════════════════════════════════════════════════

    async def find_citizen_type(self, token: str, name: str) -> Optional[str]:
        """
        Поиск категории/вида обращения гражданина.

        Args:
            token: JWT токен
            name: Название категории (например, "Жалоба", "Заявление")

        Returns:
            UUID категории или None

        Examples:
            id = await client.find_citizen_type(token, "Жалоба")
        """
        return await self._find_entity_id(
            token, "citizen-type", name, "Категория гражданина"
        )

    async def find_country(self, token: str, name: str) -> Optional[str]:
        """
        Поиск страны.

        Args:
            token: JWT токен
            name: Название страны (например, "Беларусь", "Россия")

        Returns:
            UUID страны или None
        """
        return await self._find_entity_id(token, "country", name, "Страна")

    async def find_region(self, token: str, name: str) -> Optional[str]:
        """
        Поиск региона/области.

        Args:
            token: JWT токен
            name: Название региона (например, "Минская область")

        Returns:
            UUID региона или None
        """
        return await self._find_entity_id(token, "region", name, "Регион")

    async def find_district(self, token: str, name: str) -> Optional[str]:
        """
        Поиск района.

        Args:
            token: JWT токен
            name: Название района (например, "Октябрьский район")

        Returns:
            UUID района или None
        """
        return await self._find_entity_id(token, "district", name, "Район")

    async def find_city(self, token: str, name: str) -> Optional[str]:
        """
        Поиск города/населенного пункта.

        Args:
            token: JWT токен
            name: Название города (например, "Минск")

        Returns:
            UUID города или None
        """
        return await self._find_entity_id(token, "city", name, "Город")

    async def find_correspondent(self, token: str, name: str) -> Optional[str]:
        """
        Поиск корреспондента (организации или лица).

        Args:
            token: JWT токен
            name: Название корреспондента

        Returns:
            UUID корреспондента или None
        """
        return await self._find_entity_id(token, "correspondent", name, "Корреспондент")

    async def find_delivery_method(self, token: str, name: str) -> Optional[str]:
        """
        Поиск способа доставки с fallback на значение по умолчанию.

        Если указанный способ доставки не найден, автоматически ищет "Курьер"
        в качестве дефолтного варианта.

        Args:
            token: JWT токен
            name: Название способа доставки (например, "Почта", "Email")

        Returns:
            UUID способа доставки или None

        Note:
            Это единственный справочник с fallback-логикой, так как
            deliveryMethodId является обязательным полем в DocMainFields.
        """
        result = await self._find_entity_id(
            token, "delivery-method", name, "Способ доставки"
        )

        # Fallback
        if not result and name != "Курьер":
            logger.info(
                "⚠️ Специфичный способ доставки не найден. "
                "Попытка применить 'Курьер' по умолчанию."
            )
            return await self._find_entity_id(
                token, "delivery-method", "Курьер", "Способ доставки (Default)"
            )

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # СПРАВОЧНИКИ ОРГАНИЗАЦИОННОЙ СТРУКТУРЫ
    # ══════════════════════════════════════════════════════════════════════════

    async def find_department(self, token: str, name: str) -> Optional[str]:
        """
        Поиск подразделения организации.

        Args:
            token: JWT токен
            name: Название подразделения

        Returns:
            UUID подразделения или None
        """
        return await self._find_entity_id(token, "department", name, "Подразделение")

    async def find_group(self, token: str, name: str) -> Optional[str]:
        """
        Поиск рабочей группы или группы рассылки.

        Args:
            token: JWT токен
            name: Название группы

        Returns:
            UUID группы или None
        """
        return await self._find_entity_id(token, "group", name, "Группа")
