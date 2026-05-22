# edms_ai_assistant/domain/search.py
"""
Доменные модели и правила для поиска сущностей.

Этот модуль не имеет внешних зависимостей и не знает
о конкретных API (EDMS, camelCase, EmployeeFilter).

Содержит:
- NameParts — Value Object для распарсенного ФИО
- parse_name_query — чистая функция разбора строки ФИО
- merge_name_parts — слияние с приоритетом явных полей
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NameParts — Value Object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NameParts:
    """Неизменяемый Value Object — результат парсинга строки ФИО.

    Инварианты:
    - Каждое поле либо None, либо непустая строка (без ведущих/хвостовых пробелов)
    - first_name и middle_name не могут быть заданы без last_name
      (логически невозможно иметь имя без фамилии)
    """

    last_name: str | None = None
    first_name: str | None = None
    middle_name: str | None = None

    def __post_init__(self) -> None:
        """Гарантирует, что непустые строки не содержат пробелов по краям."""
        if self.last_name is not None:
            stripped = self.last_name.strip()
            object.__setattr__(self, "last_name", stripped if stripped else None)
        if self.first_name is not None:
            stripped = self.first_name.strip()
            object.__setattr__(self, "first_name", stripped if stripped else None)
        if self.middle_name is not None:
            stripped = self.middle_name.strip()
            object.__setattr__(self, "middle_name", stripped if stripped else None)

    @property
    def has_any(self) -> bool:
        """True если хотя бы один компонент задан."""
        return bool(self.last_name or self.first_name or self.middle_name)

    @property
    def has_first_name(self) -> bool:
        """True если имя задано (используется scoring engine)."""
        return bool(self.first_name)

    @property
    def has_middle_name(self) -> bool:
        """True если отчество задано (используется scoring engine)."""
        return bool(self.middle_name)

    def to_display(self) -> str:
        """Читаемое представление для логов."""
        parts = []
        if self.last_name:
            parts.append(f"last_name={self.last_name!r}")
        if self.first_name:
            parts.append(f"first_name={self.first_name!r}")
        if self.middle_name:
            parts.append(f"middle_name={self.middle_name!r}")
        return ", ".join(parts) if parts else "empty"

    def to_full_name(self) -> str:
        """Полное ФИО одной строкой."""
        parts = [p for p in (self.last_name, self.first_name, self.middle_name) if p]
        return " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# parse_name_query — чистая функция
# ---------------------------------------------------------------------------


def parse_name_query(name_query: str | None) -> NameParts:
    """Разбивает произвольную строку ФИО на структурированные компоненты.

    Правила разбора:
    - Первое слово -> lastName (фамилия)
    - Второе слово -> firstName (имя)
    - Третье слово и далее (maxsplit=2) -> middleName (отчество)
    - Пустая / None строка -> NameParts()

    Это **чистая функция** без побочных эффектов.
    """
    if not name_query or not isinstance(name_query, str) or not name_query.strip():
        return NameParts()

    parts = name_query.strip().split(maxsplit=2)

    return NameParts(
        last_name=parts[0] if len(parts) >= 1 and parts[0] else None,
        first_name=parts[1] if len(parts) >= 2 and parts[1] else None,
        middle_name=parts[2] if len(parts) >= 3 and parts[2] else None,
    )


def merge_name_parts(
    *,
    name_query: str | None = None,
    last_name: str | None = None,
    first_name: str | None = None,
    middle_name: str | None = None,
) -> NameParts:
    """Сливает name_query с явными полями. Приоритет у явных полей.

    Когда LLM может передать и query, и last_name,
    явный last_name побеждает над разобранным из query.

    Это **чистая функция** без побочных эффектов.
    """
    parsed = parse_name_query(name_query)

    def _effective(explicit: str | None, from_parsed: str | None) -> str | None:
        if explicit and explicit.strip():
            return explicit.strip()
        return from_parsed

    return NameParts(
        last_name=_effective(last_name, parsed.last_name),
        first_name=_effective(first_name, parsed.first_name),
        middle_name=_effective(middle_name, parsed.middle_name),
    )
