# edms_ai_assistant/services/subject_service.py
from __future__ import annotations

import logging
import re

from edms_ai_assistant.llm import get_chat_model
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edms_ai_assistant.clients.reference_client import ReferenceClient

logger = logging.getLogger(__name__)


class SubjectService:
    """Бизнес-логика: выбор тематики обращения с помощью LLM."""

    def __init__(self, reference_client: ReferenceClient):
        self._ref_client = reference_client

    async def find_best_subject(self, token: str, text: str) -> str | None:
        """LLM-based two-level subject selection (parent → child)."""
        parents = await self._ref_client.get_parent_subjects(token)
        if not parents:
            logger.warning("Родительские темы не загружены")
            return None

        themes_text = "\n".join(f"{i + 1}. {s.name}" for i, s in enumerate(parents) if s.name)
        llm = get_chat_model()

        prompt = (
            f"Выбери ОДНУ наиболее подходящую тему для обращения.\n\n"
            f"СПИСОК ТЕМ:\n{themes_text}\n\n"
            f"ТЕКСТ ОБРАЩЕНИЯ (фрагмент):\n{text[:800]}\n\n"
            f"Ответь ТОЛЬКО номером (например: 3)"
        )

        try:
            response = await llm.ainvoke(prompt)
            match = re.search(r"\d+", response.content.strip())
            if not match:
                return None

            index = int(match.group(0)) - 1
            if not (0 <= index < len(parents)):
                return None

            parent = parents[index]
            parent_id = str(parent.id)
            logger.info("Родительская тема: %s (ID: %s)", parent.name, parent_id)

            children = await self._ref_client.get_child_subjects(token, parent_id)
            if not children:
                return parent_id

            children_text = "\n".join(f"{i + 1}. {c.name}" for i, c in enumerate(children) if c.name)
            prompt2 = (
                f"Выбери ОДНУ наиболее подходящую подтему.\n\n"
                f"СПИСОК ПОДТЕМ:\n{children_text}\n\n"
                f"ТЕКСТ ОБРАЩЕНИЯ (фрагмент):\n{text[:800]}\n\n"
                f"Ответь ТОЛЬКО номером (например: 2)"
            )

            response2 = await llm.ainvoke(prompt2)
            match2 = re.search(r"\d+", response2.content.strip())
            if not match2:
                return parent_id

            child_index = int(match2.group(0)) - 1
            if not (0 <= child_index < len(children)):
                return parent_id

            child = children[child_index]
            logger.info("Дочерняя тема: %s (ID: %s)", child.name, child.id)
            return str(child.id)

        except Exception as exc:
            logger.error("Ошибка выбора темы: %s", exc, exc_info=True)
            return None
