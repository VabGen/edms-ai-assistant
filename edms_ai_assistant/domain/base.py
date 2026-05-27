from __future__ import annotations

from typing import TypeVar

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


class SpringPage[T](EdmsBaseDto):
    """Модель Spring Data Page<T>."""
    content: list[T]
    total_elements: int = 0
    total_pages: int = 0


class SpringSlice[T](EdmsBaseDto):
    """Модель Spring Data Slice<T>."""
    content: list[T]
    number: int = 0
    size: int = 20
    number_of_elements: int = 0
    has_next: bool = False

    @property
    def hasNext(self) -> bool:
        return self.has_next


SliceDto = SpringSlice
