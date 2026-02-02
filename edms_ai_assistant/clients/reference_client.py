# edms_ai_assistant/clients/reference_client.py
import logging
from typing import Optional, Dict, Any, List
from abc import abstractmethod

from .base_client import EdmsHttpClient, EdmsBaseClient

logger = logging.getLogger(__name__)


class EdmsReferenceClient(EdmsBaseClient):
    """Абстрактный интерфейс для работы со справочниками EDMS"""

    @abstractmethod
    async def find_by_name(
            self, token: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """Поиск справочного значения по имени"""
        raise NotImplementedError


# ==================== КЛИЕНТЫ СПРАВОЧНИКОВ ====================


class CitizenTypeClient(EdmsReferenceClient, EdmsHttpClient):
    """Клиент для работы со справочником видов обращений граждан"""

    async def find_by_name(
            self, token: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск вида обращения по наименованию.
        """
        try:
            logger.debug(f"Поиск вида обращения: {name}")

            result = await self._make_request(
                "GET",
                "api/citizen-type/fts-name",
                token=token,
                params={"fts": name}
            )

            if result:
                logger.info(f"Найден вид обращения: {name} → ID: {result.get('id')}")
                return result

            logger.warning(f"Вид обращения '{name}' не найден в справочнике")
            return None

        except Exception as e:
            logger.error(f"Ошибка поиска вида обращения '{name}': {e}")
            return None


class CountryClient(EdmsReferenceClient, EdmsHttpClient):
    """Клиент для работы со справочником стран"""

    async def find_by_name(
            self, token: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск страны по наименованию.
        """
        try:
            logger.debug(f"Поиск страны: {name}")

            # Прямой поиск
            result = await self._make_request(
                "GET",
                "api/country/fts-name",
                token=token,
                params={"fts": name}
            )

            if result:
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

                logger.info(f"Найдена страна: {name} → ID: {result.get('id')}")
                return result

            logger.warning(f"Страна '{name}' не найдена в справочнике")
            return None

        except Exception as e:
            logger.error(f"Ошибка поиска страны '{name}': {e}")
            return None


class RegionClient(EdmsReferenceClient, EdmsHttpClient):
    """Клиент для работы со справочником областей"""

    async def find_by_name(
            self, token: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск области по наименованию.
        """
        try:
            logger.debug(f"Поиск области: {name}")

            result = await self._make_request(
                "GET",
                "api/region/fts-name",
                token=token,
                params={"fts": name}
            )

            if result:
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

                logger.info(f"Найдена область: {name} → ID: {result.get('id')}")
                return result

            logger.warning(f"Область '{name}' не найдена в справочнике")
            return None

        except Exception as e:
            logger.error(f"Ошибка поиска области '{name}': {e}")
            return None


class DistrictClient(EdmsReferenceClient, EdmsHttpClient):
    """Клиент для работы со справочником районов"""

    async def find_by_name(
            self, token: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск района по наименованию.
        """
        try:
            logger.debug(f"Поиск района: {name}")

            result = await self._make_request(
                "GET",
                "api/district/fts-name",
                token=token,
                params={"fts": name}
            )

            if result:
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

                logger.info(f"Найден район: {name} → ID: {result.get('id')}")
                return result

            logger.warning(f"Район '{name}' не найден в справочнике")
            return None

        except Exception as e:
            logger.error(f"Ошибка поиска района '{name}': {e}")
            return None


class CityClient(EdmsReferenceClient, EdmsHttpClient):
    """Клиент для работы со справочником городов"""

    async def find_by_name(
            self, token: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск города по наименованию.
        """
        try:
            logger.debug(f"Поиск города: {name}")

            result = await self._make_request(
                "GET",
                "api/city/fts-name",
                token=token,
                params={"fts": name}
            )

            if result:
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

                logger.info(f"Найден город: {name} → ID: {result.get('id')}")
                return result

            logger.warning(f"Город '{name}' не найден в справочнике")
            return None

        except Exception as e:
            logger.error(f"Ошибка поиска города '{name}': {e}")
            return None


class CorrespondentClient(EdmsReferenceClient, EdmsHttpClient):
    """Клиент для работы со справочником корреспондентов"""

    async def find_by_name(
            self, token: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск корреспондента по наименованию.
        """
        try:
            logger.debug(f"Поиск корреспондента: {name}")

            result = await self._make_request(
                "GET",
                "api/correspondent/fts-name",
                token=token,
                params={"fts": name}
            )

            if result:
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

                logger.info(f"Найден корреспондент: {name} → ID: {result.get('id')}")
                return result

            logger.warning(f"Корреспондент '{name}' не найден в справочнике")
            return None

        except Exception as e:
            logger.error(f"Ошибка поиска корреспондента '{name}': {e}")
            return None


class DeliveryMethodClient(EdmsReferenceClient, EdmsHttpClient):
    """Клиент для работы со справочником способов доставки"""

    async def find_by_name(
            self, token: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск способа доставки по наименованию.
        """
        try:
            logger.debug(f"Поиск способа доставки: {name}")

            result = await self._make_request(
                "GET",
                "api/delivery-method/fts-name",
                token=token,
                params={"fts": name}
            )

            if result:
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

                logger.info(f"Найден способ доставки: {name} → ID: {result.get('id')}")
                return result

            logger.warning(f"Способ доставки '{name}' не найден в справочнике")
            return None

        except Exception as e:
            logger.error(f"Ошибка поиска способа доставки '{name}': {e}")
            # Fallback на значение по умолчанию
            return await self.get_default_delivery_method(token)

    async def get_default_delivery_method(
            self, token: str
    ) -> Optional[Dict[str, Any]]:
        """Получить способ доставки по умолчанию (Курьер)"""
        try:
            return await self.find_by_name(token, "Курьер")
        except Exception as e:
            logger.error(f"Не удалось получить способ доставки по умолчанию: {e}")
            return None


class ReferenceClient(EdmsHttpClient):
    """
    wrapper для работы со всеми справочниками.

    использование в appeal_autofill.py:

    async with ReferenceClient() as ref_client:
        citizen_type_id = await ref_client.find_citizen_type(token, name)
        country_id = await ref_client.find_country(token, name)
    """

    def __init__(self):
        super().__init__()
        self._citizen_type = None
        self._country = None
        self._region = None
        self._district = None
        self._city = None
        self._correspondent = None
        self._delivery_method = None

    async def __aenter__(self):
        """Инициализация всех специализированных клиентов"""
        await super().__aenter__()

        self._citizen_type = CitizenTypeClient()
        self._country = CountryClient()
        self._region = RegionClient()
        self._district = DistrictClient()
        self._city = CityClient()
        self._correspondent = CorrespondentClient()
        self._delivery_method = DeliveryMethodClient()

        # Открываем все клиенты
        await self._citizen_type.__aenter__()
        await self._country.__aenter__()
        await self._region.__aenter__()
        await self._district.__aenter__()
        await self._city.__aenter__()
        await self._correspondent.__aenter__()
        await self._delivery_method.__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Закрытие всех специализированных клиентов"""
        if self._citizen_type:
            await self._citizen_type.__aexit__(exc_type, exc_val, exc_tb)
        if self._country:
            await self._country.__aexit__(exc_type, exc_val, exc_tb)
        if self._region:
            await self._region.__aexit__(exc_type, exc_val, exc_tb)
        if self._district:
            await self._district.__aexit__(exc_type, exc_val, exc_tb)
        if self._city:
            await self._city.__aexit__(exc_type, exc_val, exc_tb)
        if self._correspondent:
            await self._correspondent.__aexit__(exc_type, exc_val, exc_tb)
        if self._delivery_method:
            await self._delivery_method.__aexit__(exc_type, exc_val, exc_tb)

        await super().__aexit__(exc_type, exc_val, exc_tb)

    # ========== МЕТОДЫ ДЛЯ APPEAL AUTOFILL ==========

    async def find_citizen_type(self, token: str, name: str) -> Optional[str]:
        """Возвращает ID вида обращения"""
        result = await self._citizen_type.find_by_name(token, name)
        return result.get("id") if result else None

    async def find_country(self, token: str, name: str) -> Optional[str]:
        """Возвращает ID страны"""
        result = await self._country.find_by_name(token, name)
        return result.get("id") if result else None

    async def find_region(self, token: str, name: str) -> Optional[str]:
        """Возвращает ID области"""
        result = await self._region.find_by_name(token, name)
        return result.get("id") if result else None

    async def find_district(self, token: str, name: str) -> Optional[str]:
        """Возвращает ID района"""
        result = await self._district.find_by_name(token, name)
        return result.get("id") if result else None

    async def find_city(self, token: str, name: str) -> Optional[str]:
        """Возвращает ID города"""
        result = await self._city.find_by_name(token, name)
        return result.get("id") if result else None

    async def find_correspondent(self, token: str, name: str) -> Optional[str]:
        """Возвращает ID корреспондента"""
        result = await self._correspondent.find_by_name(token, name)
        return result.get("id") if result else None

    async def find_delivery_method(self, token: str, name: str) -> Optional[str]:
        """Возвращает ID способа доставки (с fallback на "Курьер")"""
        result = await self._delivery_method.find_by_name(token, name)
        return result.get("id") if result else None
