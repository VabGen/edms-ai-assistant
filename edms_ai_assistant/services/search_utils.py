# edms_ai_assistant/services/search_utils.py
"""
Сервисный слой для поиска — маппинг доменных моделей на API-контракты.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from edms_ai_assistant.domain.search import NameParts, merge_name_parts

if TYPE_CHECKING:
    from edms_ai_assistant.domain.employee import EmployeeDto

logger = logging.getLogger(__name__)

_SCORE_GAP_THRESHOLD: int = 5

DEFAULT_INCLUDES: list[str] = ["POST", "DEPARTMENT"]
DEFAULT_PAGE: int = 0
DEFAULT_SIZE: int = 20
DEFAULT_PAGEABLE: dict[str, Any] = {
    "page": DEFAULT_PAGE,
    "size": DEFAULT_SIZE,
    "sort": "lastName,ASC",
}


# ---------------------------------------------------------------------------
# build_employee_filter — маппинг домена -> API
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


def score_employee_result(
    result: EmployeeDto,
    last_name: str | None,
    first_name: str | None,
    middle_name: str | None,
    full_post_name: str | None = None,
) -> int:
    """Оценивает совпадение сотрудника с заданными критериями."""
    score = 0
    if last_name:
        val = (result.last_name or "").lower()
        term = last_name.lower()
        if val == term:
            score += 10
        elif val.startswith(term):
            score += 5
    if first_name:
        val = (result.first_name or "").lower()
        term = first_name.lower()
        if val == term:
            score += 10
        elif val.startswith(term):
            score += 5
    if middle_name:
        val = (result.middle_name or "").lower()
        term = middle_name.lower()
        if val == term:
            score += 10
        elif val.startswith(term):
            score += 5
    if full_post_name:
        post_name = (result.post.post_name if result.post else "").lower()
        term = full_post_name.lower()
        if post_name == term:
            score += 10
        elif post_name.startswith(term):
            score += 7
        elif term in post_name:
            score += 3
    return score


def find_best_employee_match(
    results: list[EmployeeDto],
    last_name: str | None,
    first_name: str | None,
    middle_name: str | None,
    full_post_name: str | None = None,
) -> EmployeeDto | None:
    """Ищет лучший результат среди списка сотрудников на основе весов."""
    has_criteria = any((last_name, first_name, middle_name, full_post_name))
    if not has_criteria:
        return None
    scored = [
        (
            r,
            score_employee_result(
                r, last_name, first_name, middle_name, full_post_name
            ),
        )
        for r in results
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return None
    top_result, top_score = scored[0]
    if top_score < 5:
        return None
    if len(scored) == 1:
        return top_result
    _, second_score = scored[1]
    if top_score - second_score >= _SCORE_GAP_THRESHOLD:
        logger.info(
            "Best match selected: score=%d vs %d (gap=%d)",
            top_score,
            second_score,
            top_score - second_score,
        )
        return top_result
    return None


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
