# edms_ai_assistant/core/sub_agents.py

import os
import importlib
import logging
from typing import Dict, Any, List, Callable, Optional

logger = logging.getLogger(__name__)

# --- Глобальный Реестр Агентов ---
AGENT_REGISTRY: Dict[str, Callable[[], Any]] = {}


def register_agent(name: str):
    """Декоратор для регистрации под-агента в глобальном реестре."""

    def decorator(agent_factory: Callable[[], Any]):
        if name in AGENT_REGISTRY:
            logger.warning(f"Агент '{name}' уже зарегистрирован. Перезапись.")
        AGENT_REGISTRY[name] = agent_factory
        logger.info(f"Агент '{name}' зарегистрирован в реестре.")
        return agent_factory

    return decorator


# --- Функция Автоматического Обнаружения (Auto-Discovery) ---
def discover_sub_agents(package_name: str = "edms_ai_assistant.sub_agents"):
    """Сканирует указанный пакет и импортирует все модули, чтобы запустить декораторы @register_agent."""
    logger.info("Запуск Auto-Discovery под-агентов...")
    try:
        package = importlib.import_module(package_name)
        package_dir = os.path.dirname(package.__file__)
        for filename in os.listdir(package_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                full_module_name = f"{package_name}.{module_name}"
                try:
                    importlib.import_module(full_module_name)
                    logger.debug(
                        f"Успешно обнаружен и импортирован модуль: {module_name}"
                    )
                except Exception as e:
                    logger.error(f"Ошибка при импорте модуля {module_name}: {e}")
    except ImportError as e:
        logger.error(f"Ошибка при импорте пакета {package_name}: {e}")


# --- Фабрика для получения агента по имени ---
def get_sub_agent_executor(agent_name: str) -> Optional[Any]:
    factory = AGENT_REGISTRY.get(agent_name)
    if factory:
        return factory()
    return None


def get_available_agent_names() -> List[str]:
    logger.info(f"Возвращаю список имеющихся агентов. {len(AGENT_REGISTRY)}")
    return list(AGENT_REGISTRY.keys())


_DISCOVERY_RUN = False


def run_discovery_if_needed():
    """Запускает discovery, если он ещё не был запущен."""
    global _DISCOVERY_RUN
    if not _DISCOVERY_RUN:
        logger.info("Запуск discovery из run_discovery_if_needed.")
        discover_sub_agents()
        _DISCOVERY_RUN = True
        logger.info(
            f"Auto-Discovery завершено. Обнаружено агентов: {len(get_available_agent_names())}"
        )
