import asyncio
import logging
import hashlib
import time
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from edms_ai_assistant.utils.token_counter import token_counter
from edms_ai_assistant.schemas.summarization import SummaryFormat, SummarizationResult
from edms_ai_assistant.utils.async_utils import spawn_background_task
from edms_ai_assistant.llm import get_chat_model
import json_repair
from sqlalchemy import select
from edms_ai_assistant.db.database import AsyncSessionLocal
from edms_ai_assistant.db.generated.models.summarization_cache import SummarizationCache

logger = logging.getLogger(__name__)

# ВАЖНО: Подняли версию, чтобы инвалидировать старый кэш с плохими промптами!
PROMPT_VERSION = "v3"


# Реальное хэширование, как описано в документации кэша
def _make_cache_key(file_identifier: str, summary_type: str) -> str:
    raw = f"{file_identifier}::{summary_type}::{PROMPT_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:48]


def _token_chunks(text: str, max_tokens: int = 1000) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=int(max_tokens * 0.1),
        length_function=token_counter.count,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


_DIRECT_PROMPTS = {
    SummaryFormat.EXTRACTIVE: (
        "ВНИМАНИЕ: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ!\n\n"
        "Ты — строгий деловой аналитик. Извлеки ключевые смысловые факты из документа.\n"
        "ЖЕСТКИЕ ПРАВИЛА:\n"
        "1. Формат: Обычный маркированный список (символ '-'). ЗАПРЕЩЕНО использовать Markdown-таблицы (символы |) **Короткий заголовок с пробелами**: Значение'.\n"
        "2. Пиши заголовки с пробелами, как обычный текст! НЕ пиши слитно (неправильно: ОбязательнаяАнтивируснаяЗащита, правильно: **Антивирусная защита**).\n"
        "3. НЕ пиши слово 'Суть'! Используй конкретные названия: **Тема**, **Организация**, **Сумма**, **Срок**, **Требование**.\n"
        "4. Краткость: Максимум 15 слов на пункт. Без воды и деталей.\n"
        "5. ИГНОРИРУЙ технические данные: UUID, идентификаторы, типы вложений (ATTACHMENT), размеры файлов.\n"
        "6. Лимит: Максимум 10 самых важных фактов.\n\n"
        "ПРИМЕР ПРАВИЛЬНОГО ВЫВОДА:\n"
        "- **Документ**: Стандарт СOVT 9.03-2025 «Банковская деятельность»\n"
        "- **Внедрение**: С января 2027 года для банков\n"
        "- **Антивирусная защита**: Обязательна для всех рабочих станций\n"
        "- **Криптографическая защита**: Требуется для передачи данных\n\n"
        "Текст документа:\n<text>\n{text}\n</text>\n\nКлючевые факты:"
    ),
    SummaryFormat.ABSTRACTIVE: (
        "ВНИМАНИЕ: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ!\n\n"
        "Ты — эксперт по деловой коммуникации. Напиши краткое резюме документа.\n"
        "ПРАВИЛА:\n"
        "1) Формат: Markdown (**Главная цель**, **Ключевые детали**, **Итог**). \n"
        "2) Максимум 100 слов. Без канцеляризмов.\n"
        "3) ИГНОРИРУЙ технические метаданные: UUID, типы вложений (ATTACHMENT), размеры файлов.\n\n"
        "Текст:\n<text>\n{text}\n</text>\n\nРезюме:"
    ),
    SummaryFormat.THESIS: (
        "ВНИМАНИЕ: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ!\n\n"
        "Ты — методолог. Построй иерархический тезисный план документа.\n"
        "ПРАВИЛА:\n"
        "1) Формат: Markdown (## Раздел, - Подтезис). \n"
        "2) Максимум 4 раздела, до 2 тезисов в каждом.\n"
        "3) Тезисы максимально короткие (до 10 слов).\n"
        "4) ИГНОРИРУЙ технические данные (UUID, размеры файлов, типы вложений).\n\n"
        "Текст:\n<text>\n{text}\n</text>\n\nТезисный план:"
    ),
}

_MAP_PROMPTS = {
    SummaryFormat.EXTRACTIVE: (
        "ВНИМАНИЕ: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ!\n\n"
        "Ты — аналитик. Извлеки ключевые смысловые факты ИЗ ЧАСТИ документа.\n"
        "ПРАВИЛА:\n"
        "- Формат: '- **Короткий заголовок с пробелами**: Значение'.\n"
        "- НЕ пиши заголовки слитно! Разделяй слова пробелами.\n"
        "- Максимум 15 слов на пункт.\n"
        "- ИГНОРИРУЙ UUID, типы вложений, размеры файлов.\n\n"
        "Часть текста:\n<chunk>\n{chunk}\n</chunk>\n\nФакты:"
    ),
    SummaryFormat.ABSTRACTIVE: (
        "ВНИМАНИЕ: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ!\n\n"
        "Ты — эксперт. Напиши краткое саммари ЧАСТИ документа.\n"
        "ПРАВИЛА: Максимум 50 слов. Сохрани смысл. Игнорируй технические метаданные.\n\n"
        "Часть текста:\n<chunk>\n{chunk}\n</chunk>\n\nСаммари:"
    ),
    SummaryFormat.THESIS: (
        "ВНИМАНИЕ: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ!\n\n"
        "Ты — методолог. Выдели основные тезисы ЧАСТИ документа.\n"
        "ПРАВИЛА: Формат: Markdown-список. До 10 слов на тезис. Игнорируй системные ID.\n\n"
        "Часть текста:\n<chunk>\n{chunk}\n</chunk>\n\nТезисы:"
    ),
}

_REDUCE_PROMPTS = {
    SummaryFormat.EXTRACTIVE: (
        "ВНИМАНИЕ: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ!\n\n"
        "Ты — главный аналитик. Тебе предоставлены факты из разных частей документа.\n"
        "ПРАВИЛА: \n"
        "- Объедини дубликаты. Оставь 10 самых важных фактов.\n"
        "- Формат: '- **Короткий заголовок с пробелами**: Значение'.\n"
        "- НЕ пиши заголовки слитно (неправильно: СрокДействия, правильно: **Срок действия**).\n"
        "- Максимум 15 слов на пункт.\n"
        "- Удали всю техническую информацию (UUID, ATTACHMENT, логи).\n\n"
        "Сборные факты:\n<summaries>\n{summaries}\n</summaries>\n\nИтоговые факты:"
    ),
    SummaryFormat.ABSTRACTIVE: (
        "ВНИМАНИЕ: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ!\n\n"
        "Ты — главный эксперт. На основе саммари частей напиши итоговое резюме.\n"
        "ПРАВИЛА: Формат: Markdown. Максимум 100 слов. Выдели главную суть. Без технических ID.\n\n"
        "Части саммари:\n<summaries>\n{summaries}\n</summaries>\n\nРезюме:"
    ),
    SummaryFormat.THESIS: (
        "ВНИМАНИЕ: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ!\n\n"
        "Ты — главный методолог. На основе тезисов построй итоговый план.\n"
        "ПРАВИЛА: Формат: Markdown (## Раздел, - Подтезис). Максимум 4 раздела. Убери повторы и технические данные.\n\n"
        "Тезисы частей:\n<summaries>\n{summaries}\n</summaries>\n\nТезисный план:"
    ),
}


async def _self_critique_async(summary: str, file_identifier: str, summary_type: str) -> None:
    if len(summary) < 50:
        return

    critique_prompt = (
        "Оцени качество саммари ниже по шкале от 0 до 1 (где 1 - идеально). "
        "Ответь ТОЛЬКО в формате JSON: {\"score\": 0.85, \"confidence\": \"high\"}\n"
        f"Саммари:\n{summary[:2000]}"
    )

    try:
        llm = get_chat_model()
        raw_response = await llm.ainvoke([HumanMessage(content=critique_prompt)])
        data = json_repair.loads(raw_response.content)
        score = round(min(max(float(data.get("score", 0.7)), 0.0), 1.0), 2)
        confidence = data.get("confidence", "medium")

        async with AsyncSessionLocal() as db:
            cache_key = _make_cache_key(file_identifier, summary_type)
            stmt = select(SummarizationCache).where(
                SummarizationCache.file_identifier == cache_key,
                SummarizationCache.summary_type == summary_type
            )
            row = await db.scalar(stmt)
            if row:
                payload = SummarizationResult.model_validate_json(row.content)
                payload.quality_score = score
                payload.confidence = confidence
                row.content = payload.model_dump_json()
                await db.commit()
    except Exception as exc:
        logger.debug("Async self-critique failed safely: %s", exc)


async def _load_from_cache(file_identifier: str, summary_type: str) -> SummarizationResult | None:
    async with AsyncSessionLocal() as db:
        cache_key = _make_cache_key(file_identifier, summary_type)
        stmt = select(SummarizationCache).where(
            SummarizationCache.file_identifier == cache_key,
            SummarizationCache.summary_type == summary_type
        )
        result = await db.scalar(stmt)
        if result:
            return SummarizationResult.model_validate_json(result.content)
    return None


async def _save_to_cache(file_identifier: str, summary_type: str, data: SummarizationResult) -> None:
    async with AsyncSessionLocal() as db, db.begin():
        cache_key = _make_cache_key(file_identifier, summary_type)
        stmt = select(SummarizationCache).where(
            SummarizationCache.file_identifier == cache_key,
            SummarizationCache.summary_type == summary_type
        )
        existing = await db.scalar(stmt)
        if existing:
            existing.content = data.model_dump_json()
        else:
            db.add(SummarizationCache(file_identifier=cache_key, summary_type=summary_type,
                                      content=data.model_dump_json()))


async def stream_summarize(
        text: str,
        fmt: SummaryFormat,
        file_identifier: str | None = None,
        map_reduce_threshold_tokens: int = 3000
) -> AsyncGenerator[str, None]:
    llm = get_chat_model()
    text_tokens = token_counter.count(text)
    prompt = None

    if text_tokens <= map_reduce_threshold_tokens:
        prompt = _DIRECT_PROMPTS[fmt].format(text=text)
    else:
        chunks = _token_chunks(text, max_tokens=1000)
        sem = asyncio.Semaphore(4)

        async def _map_chunk(c: str, i: int) -> str:
            async with sem:
                m_prompt = _MAP_PROMPTS[fmt].format(chunk=c)
                res = await llm.ainvoke([HumanMessage(content=m_prompt)])
                return f"Фрагмент {i + 1}:\n{res.content.strip()}"

        map_res = await asyncio.gather(
            *[_map_chunk(c, i) for i, c in enumerate(chunks)],
            return_exceptions=True
        )
        valid = [r for r in map_res if isinstance(r, str)]
        if not valid:
            yield "Ошибка: не удалось обработать фрагменты документа."
            return

        prompt = _REDUCE_PROMPTS[fmt].format(summaries="\n\n---\n\n".join(valid))

    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        yield chunk.content

    if file_identifier:
        spawn_background_task(
            _self_critique_async(prompt, file_identifier, fmt.value)
        )


class SummarizationOrchestrator:
    def __init__(self, enable_cache: bool = True):
        self.enable_cache = enable_cache

    async def check_cache(self, file_identifier: str, summary_type: str) -> SummarizationResult | None:
        return await _load_from_cache(file_identifier, summary_type)

    async def check_llm_health(self) -> bool:
        try:
            llm = get_chat_model()
            await llm.ainvoke([HumanMessage(content="test")], max_tokens=1)
            return True
        except Exception:
            return False

    async def summarize(
            self,
            text: str,
            summary_type: str | SummaryFormat,
            file_identifier: str | None = None,
    ) -> SummarizationResult:
        start = time.monotonic()
        try:
            fmt = SummaryFormat(str(summary_type).strip().lower())
        except ValueError:
            fmt = SummaryFormat.EXTRACTIVE

        text = (text or "").strip()
        if len(text) < 30:
            return SummarizationResult(status="error", content="Текст слишком короткий.", format_used=fmt,
                                       text_length=len(text))

        content = []
        try:
            async for token in stream_summarize(text, fmt, file_identifier):
                content.append(token)
            summary = "".join(content)
        except Exception as exc:
            return SummarizationResult(status="error", content=f"Ошибка генерации: {exc}", format_used=fmt,
                                       text_length=len(text))

        elapsed_ms = int((time.monotonic() - start) * 1000)
        result = SummarizationResult(
            status="success", content=summary, format_used=fmt,
            processing_time_ms=elapsed_ms, chunks_processed=1,
            text_length=len(text), pipeline="stream", warnings=[]
        )

        if self.enable_cache and file_identifier:
            await _save_to_cache(file_identifier, fmt.value, result)

        return result


def get_orchestrator() -> SummarizationOrchestrator:
    return SummarizationOrchestrator(enable_cache=True)
