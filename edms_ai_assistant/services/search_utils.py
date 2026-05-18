# edms_ai_assistant/services/search_utils.py
"""
Сервисный слой для поиска — маппинг доменных моделей на API-контракты.

Этот модуль знает о:
- Java EmployeeFilter (camelCase поля)
- OR-логике EmployeeFilter (отправляем только lastName)
- Константах API (DEFAULT_INCLUDES, DEFAULT_PAGEABLE)

Этот модуль НЕ знает о:
- HTTP-клиентах
- Конкретных endpoint'ах
- Бизнес-логике создания задач/ознакомлений

Зависимости: domain.search → services.search_utils → services/task_service, tools/
"""

from __future__ import annotations

import logging
from typing import Any

from edms_ai_assistant.domain.search import NameParts, merge_name_parts

logger = logging.getLogger(__name__)


DEFAULT_INCLUDES: list[str] = ["POST", "DEPARTMENT"]
DEFAULT_PAGE: int = 0
DEFAULT_SIZE: int = 20
DEFAULT_PAGEABLE: dict[str, Any] = {
    "page": DEFAULT_PAGE,
    "size": DEFAULT_SIZE,
    "sort": "lastName,ASC",
}


# ---------------------------------------------------------------------------
# build_employee_filter — маппинг домена → API
# ---------------------------------------------------------------------------


def build_employee_filter(
    *,
    name_query: str | None = None,
    last_name: str | None = None,
    first_name: str | None = None,
    middle_name: str | None = None,
    full_post_name: str | None = None,
    post_id: int | None = None,
    active: bool | None = None,
    fired: bool | None = None,
    includes: list[str] | None = None,
) -> dict[str, Any]:
    """Строит EmployeeFilter для POST /api/employee/search.

    КРИТИЧЕСКОЕ ПРАВИЛО: Java EmployeeFilter использует OR-логику
    между lastName, firstName и middleName. Поэтому отправляем
    ТОЛЬКО lastName (если задан). Фильтрация по остальным полям
    выполняется scoring engine на стороне Python (AND-логика).

    Returns:
        Dict с camelCase ключами, совместимый с EmployeeFilter.java
    """
    merged = merge_name_parts(
        name_query=name_query,
        last_name=last_name,
        first_name=first_name,
        middle_name=middle_name,
    )

    result: dict[str, Any] = {}

    if merged.last_name:
        result["lastName"] = merged.last_name
    elif merged.first_name:
        result["firstName"] = merged.first_name
    elif merged.middle_name:
        result["middleName"] = merged.middle_name

    if full_post_name and full_post_name.strip():
        result["fullPostName"] = full_post_name.strip()
    if post_id is not None:
        result["postId"] = post_id
    if active is not None:
        result["active"] = active
    if fired is not None:
        result["fired"] = fired

    result["includes"] = includes if includes is not None else DEFAULT_INCLUDES

    return result


def get_merged_name_parts(
    *,
    name_query: str | None = None,
    last_name: str | None = None,
    first_name: str | None = None,
    middle_name: str | None = None,
) -> NameParts:
    """Возвращает объединённые NameParts для scoring engine.

    Вызывается отдельно от build_employee_filter, потому что scoring
    нужен доступ ко ВСЕМ компонентам ФИО (включая те, что не отправлены в API).

    Usage::

        merged = get_merged_name_parts(name_query="Костин Леонид")
        # NameParts(last_name="Костин", first_name="Леонид")

        search_filter = build_employee_filter(name_query="Костин Леонид")
        # {"lastName": "Костин", ...}
        # firstName НЕ включён (OR-логика!)

        results = await client.search_employees_post(employee_filter=search_filter)
        best = _find_best_match(results, merged.last_name, merged.first_name, ...)
    """
    return merge_name_parts(
        name_query=name_query,
        last_name=last_name,
        first_name=first_name,
        middle_name=middle_name,
    )
