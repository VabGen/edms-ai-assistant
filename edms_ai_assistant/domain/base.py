from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class EdmsBaseDto(BaseModel):
    """Базовая модель для всех DTO из EDMS API."""
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        extra="ignore",
        frozen=True,
    )


T = TypeVar("T")


class SpringPage(EdmsBaseDto, Generic[T]):
    """Модель Spring Data Page<T>."""
    content: list[T]
    total_elements: int = 0
    total_pages: int = 0


class SpringSlice(EdmsBaseDto, Generic[T]):
    """Модель Spring Data Slice<T>."""
    content: list[T]
    has_next: bool = False
