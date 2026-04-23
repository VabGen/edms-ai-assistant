"""
EDMS AI Assistant — Summarization Orchestrator (Enterprise Edition).
...
"""

from __future__ import annotations

import asyncio
import httpx
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import delete, select

from edms_ai_assistant.db.database import AsyncSessionLocal, SummarizationCache
from edms_ai_assistant.llm import get_chat_model

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_DIRECT_LLM_THRESHOLD: int = 4_000
_MAX_CHUNK_CHARS: int = 3_500
_DEFAULT_CONCURRENCY: int = 4
_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 1.5
_PROMPT_VERSION: str = "v3"
_llm_healthy: bool | None = None
_last_health_check: float = 0.0
_HEALTH_CHECK_TTL: float = 30.0


class SummaryFormat(StrEnum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


class SummarizationResult(BaseModel):
    status: str = "success"
    content: str = ""
    format_used: SummaryFormat = SummaryFormat.EXTRACTIVE
    quality_score: float | None = Field(None, ge=0.0, le=1.0)
    confidence: str | None = None
    processing_time_ms: int = 0
    chunks_processed: int = 0
    was_truncated: bool = False
    text_length: int = 0
    pipeline: str = "direct"
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    from_cache: bool = False
    cache_key: str | None = None


def _semantic_chunks(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    paragraphs: list[str] = []

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            paragraphs.append(block)
            continue

        current_lines: list[str] = []
        current_len = 0
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            line_len = len(line) + 1
            if current_len + line_len > max_chars and current_lines:
                paragraphs.append("\n".join(current_lines))
                current_lines = []
                current_len = 0
            current_lines.append(line)
            current_len += line_len
        if current_lines:
            paragraphs.append("\n".join(current_lines))

    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        if not buffer:
            buffer = para
            continue
        candidate = buffer + "\n\n" + para
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            merged.append(buffer)
            buffer = para
    if buffer:
        merged.append(buffer)

    result = [c for c in merged if c.strip()]
    return result if result else [text[:max_chars]]


_MAP_PROMPTS: dict[SummaryFormat, str] = {
    SummaryFormat.EXTRACTIVE: (
        "Ты — аналитик СЭД. Из фрагмента документа извлеки ВСЕ конкретные факты: "
        "даты, суммы, имена, сроки, обязательства. "
        "Формат: нумерованный список. Только факты из текста, без домыслов.\n\n"
        "<fragment>\n{chunk}\n</fragment>\n\nФакты:"
    ),
    SummaryFormat.ABSTRACTIVE: (
        "Ты — аналитик СЭД. Перескажи фрагмент своими словами (2–4 предложения). "
        "Сохрани суть, убери канцелярские обороты.\n\n"
        "<fragment>\n{chunk}\n</fragment>\n\nПересказ:"
    ),
    SummaryFormat.THESIS: (
        "Ты — аналитик СЭД. Сформулируй 2–5 тезисов фрагмента. "
        "Нумерованный список, каждый тезис — завершённое предложение.\n\n"
        "<fragment>\n{chunk}\n</fragment>\n\nТезисы:"
    ),
}

_REDUCE_PROMPTS: dict[SummaryFormat, str] = {
    SummaryFormat.EXTRACTIVE: (
        "Ты — ведущий аналитик СЭД. Объедини факты из всех фрагментов в единый "
        "структурированный список. Убери дубликаты, сохрани все уникальные данные. "
        "Формат: нумерованный список по смысловым блокам.\n\n"
        "<fragments_summaries>\n{summaries}\n</fragments_summaries>\n\nИтоговый список фактов:"
    ),
    SummaryFormat.ABSTRACTIVE: (
        "Ты — ведущий аналитик СЭД. Составь связный Executive Summary (3–5 абзацев) "
        "на основе пересказов фрагментов. Структура: проблема → решение → последствия.\n\n"
        "<fragments_summaries>\n{summaries}\n</fragments_summaries>\n\nExecutive Summary:"
    ),
    SummaryFormat.THESIS: (
        "Ты — ведущий аналитик СЭД. Построй иерархический тезисный план документа "
        "на основе тезисов фрагментов. Нумерация: 1., 1.1., 2., 2.1. и т.д.\n\n"
        "<fragments_summaries>\n{summaries}\n</fragments_summaries>\n\nИтоговый тезисный план:"
    ),
}

_DIRECT_PROMPTS: dict[SummaryFormat, str] = {
    SummaryFormat.EXTRACTIVE: (
        "Ты — ведущий AI-аналитик СЭД. Извлеки ВСЕ конкретные факты: "
        "даты, суммы, имена, сроки, обязательства. "
        "Нумерованный список. Только то, что есть в тексте.\n"
        "<constraints>Не выдумывай. Только русский язык. Без вводных фраз.</constraints>\n\n"
        "<document>\n{text}\n</document>\n\nКлючевые факты:"
    ),
    SummaryFormat.ABSTRACTIVE: (
        "Ты — ведущий AI-аналитик СЭД. Напиши Executive Summary (2–3 абзаца): "
        "проблема, решение, ключевые последствия.\n"
        "<constraints>Не выдумывай. Только русский язык. Без вводных фраз.</constraints>\n\n"
        "<document>\n{text}\n</document>\n\nExecutive Summary:"
    ),
    SummaryFormat.THESIS: (
        "Ты — ведущий AI-аналитик СЭД. Построй иерархический тезисный план. "
        "Нумерация: 1., 1.1., 2., 2.1.\n"
        "<constraints>Не выдумывай. Только русский язык. Без вводных фраз.</constraints>\n\n"
        "<document>\n{text}\n</document>\n\nТезисный план:"
    ),
}

_SELF_CRITIQUE_PROMPT = (
    "Ты — эксперт по качеству аналитических текстов. "
    "Оцени итоговую суммаризацию по 4 критериям (каждый 0–10):\n"
    "1. completeness — охват ключевых данных\n"
    "2. accuracy — соответствие исходному тексту (не выдуманные факты)\n"
    "3. clarity — понятность и структурированность\n"
    "4. conciseness — отсутствие воды\n\n"
    "Верни ТОЛЬКО JSON без markdown-блоков и без комментариев:\n"
    '{"completeness": 8, "accuracy": 9, "clarity": 7, "conciseness": 8, '
    '"overall": 8.0, "confidence": "high", "issues": []}\n\n'
    "<summary>\n{summary}\n</summary>\n\nJSON оценки:"
)

_MD_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _make_cache_key(file_identifier: str, summary_type: str) -> str:
    raw = f"{file_identifier}::{summary_type}::{_PROMPT_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:48]


async def _load_from_cache(
    file_identifier: str,
    summary_type: str,
) -> SummarizationResult | None:
    cache_key = _make_cache_key(file_identifier, summary_type)
    try:
        async with AsyncSessionLocal() as db:
            row = await db.scalar(
                select(SummarizationCache).where(
                    SummarizationCache.file_identifier == cache_key,
                    SummarizationCache.summary_type == summary_type,
                )
            )
        if row:
            base = SummarizationResult.model_validate_json(row.content)
            cached = base.model_copy(
                update={"from_cache": True, "cache_key": cache_key}
            )
            logger.info(
                "Cache HIT",
                extra={"cache_key": cache_key[:12], "summary_type": summary_type},
            )
            return cached
    except Exception as exc:
        logger.warning("Cache read error: %s", exc)
    return None


async def _save_to_cache(
    file_identifier: str,
    summary_type: str,
    result: SummarizationResult,
) -> None:
    cache_key = _make_cache_key(file_identifier, summary_type)
    try:
        async with AsyncSessionLocal() as db, db.begin():
            await db.execute(
                delete(SummarizationCache).where(
                    SummarizationCache.file_identifier == cache_key,
                    SummarizationCache.summary_type == summary_type,
                )
            )
            db.add(
                SummarizationCache(
                    id=str(uuid.uuid4()),
                    file_identifier=cache_key,
                    summary_type=summary_type,
                    content=result.model_dump_json(),
                )
            )
        logger.info(
            "Cache SAVE",
            extra={"cache_key": cache_key[:12], "summary_type": summary_type},
        )
    except Exception as exc:
        logger.warning("Cache write error (non-fatal): %s", exc)


async def _check_llm_health(base_url: str = "http://127.0.0.1:11434") -> bool:
    global _llm_healthy, _last_health_check

    now = time.monotonic()
    if _llm_healthy is not None and (now - _last_health_check) < _HEALTH_CHECK_TTL:
        return _llm_healthy

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            _llm_healthy = resp.status_code == 200
    except Exception:
        _llm_healthy = False

    _last_health_check = now
    logger.info("LLM health check: %s", "OK" if _llm_healthy else "UNREACHABLE")
    return _llm_healthy


async def _llm_call_with_retry(
    prompt: str,
    *,
    retries: int = _MAX_RETRIES,
    base_delay: float = _RETRY_BASE_DELAY,
    label: str = "llm_call",
) -> str:
    from langchain_core.messages import HumanMessage

    if not await _check_llm_health():
        raise RuntimeError("LLM service unreachable — skipping retry cycle")

    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            llm = get_chat_model()
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = str(response.content).strip()
            if content:
                return content
            raise ValueError("Empty LLM response")
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "LLM retry %d/%d [%s]: %s — waiting %.1fs",
                    attempt,
                    retries,
                    label,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "LLM exhausted %d retries [%s]: %s",
                    retries,
                    label,
                    exc,
                    exc_info=True,
                )

    raise RuntimeError(
        f"LLM failed after {retries} retries [{label}]: {last_exc}"
    ) from last_exc


@dataclass
class CritiqueResult:
    overall: float = 0.7
    confidence: str = "medium"
    issues: list[str] = field(default_factory=list)


async def _self_critique(summary: str) -> CritiqueResult:
    if len(summary) < 50:
        return CritiqueResult(overall=0.4, confidence="low", issues=["too_short"])

    prompt = _SELF_CRITIQUE_PROMPT.format(summary=summary[:3_000])
    try:
        raw = await _llm_call_with_retry(prompt, retries=2, label="self_critique")
        clean = _MD_FENCE_RE.sub("", raw).strip()
        data: dict[str, Any] = json.loads(clean)
        raw_score = float(data.get("overall", 7.0))
        overall = raw_score / 10.0 if raw_score > 1.0 else raw_score
        confidence = str(data.get("confidence", "medium"))
        issues: list[str] = [str(i) for i in data.get("issues", [])]
        return CritiqueResult(
            overall=round(min(max(overall, 0.0), 1.0), 2),
            confidence=(
                confidence if confidence in ("high", "medium", "low") else "medium"
            ),
            issues=issues,
        )
    except Exception as exc:
        logger.debug("Self-critique failed (non-fatal): %s", exc)
        return CritiqueResult(overall=0.7, confidence="medium")


async def _direct_pipeline(text: str, fmt: SummaryFormat) -> tuple[str, int]:
    prompt = _DIRECT_PROMPTS[fmt].format(text=text[:8_000])
    summary = await _llm_call_with_retry(prompt, label=f"direct_{fmt}")
    return summary, 1


async def _map_chunk(
    chunk: str,
    idx: int,
    fmt: SummaryFormat,
    semaphore: asyncio.Semaphore,
) -> str | None:
    async with semaphore:
        prompt = _MAP_PROMPTS[fmt].format(chunk=chunk)
        try:
            result = await _llm_call_with_retry(prompt, label=f"map_chunk_{idx}")
            logger.debug("Map chunk %d done (%d chars out)", idx, len(result))
            return result
        except Exception as exc:
            logger.error("Map chunk %d failed permanently: %s", idx, exc)
            return None


async def _reduce_stage(chunk_summaries: list[str], fmt: SummaryFormat) -> str:
    numbered = "\n\n---\n\n".join(
        f"[Фрагмент {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    prompt = _REDUCE_PROMPTS[fmt].format(summaries=numbered[:10_000])
    return await _llm_call_with_retry(prompt, label=f"reduce_{fmt}")


async def _map_reduce_pipeline(
    text: str,
    fmt: SummaryFormat,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> tuple[str, int, bool]:
    chunks = _semantic_chunks(text, max_chars=_MAX_CHUNK_CHARS)
    total_chunks = len(chunks)
    logger.info(
        "Map-Reduce start: %d chunks, concurrency=%d, format=%s",
        total_chunks,
        concurrency,
        fmt,
    )

    semaphore = asyncio.Semaphore(concurrency)
    map_results = await asyncio.gather(
        *[_map_chunk(chunk, idx, fmt, semaphore) for idx, chunk in enumerate(chunks)],
        return_exceptions=True,
    )

    successful: list[str] = []
    failed_count = 0
    for r in map_results:
        if isinstance(r, Exception) or r is None:
            failed_count += 1
        elif r:
            successful.append(r)

    degraded = failed_count > 0
    if degraded:
        logger.warning(
            "Map stage: %d/%d chunks failed — continuing in degraded mode",
            failed_count,
            total_chunks,
        )

    if not successful:
        logger.error(
            "Map stage: all %d chunks failed — raw text fallback", total_chunks
        )
        return "\n\n".join(chunks[:3])[:3_000], 0, True

    if len(successful) == 1:
        return successful[0], total_chunks, degraded

    try:
        final = await _reduce_stage(successful, fmt)
        return final, total_chunks, degraded
    except Exception as exc:
        logger.error(
            "Reduce stage failed: %s — returning concatenated Map results", exc
        )
        return "\n\n---\n\n".join(successful)[:5_000], total_chunks, True


class SummarizationOrchestrator:
    def __init__(
        self,
        direct_threshold: int = _DIRECT_LLM_THRESHOLD,
        concurrency: int = _DEFAULT_CONCURRENCY,
        enable_cache: bool = True,
        enable_self_critique: bool = True,
    ) -> None:
        self.direct_threshold = direct_threshold
        self.concurrency = concurrency
        self.enable_cache = enable_cache
        self.enable_self_critique = enable_self_critique

    async def check_llm_health(self) -> bool:
        """Public method to check LLM health without breaking encapsulation."""
        return await _check_llm_health()

    async def check_cache(
        self,
        file_identifier: str,
        summary_type: str,
    ) -> SummarizationResult | None:
        if not self.enable_cache or not file_identifier:
            return None
        return await _load_from_cache(file_identifier, summary_type)

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
            logger.warning("Unknown summary_type '%s' → extractive", summary_type)
            fmt = SummaryFormat.EXTRACTIVE

        text = (text or "").strip()
        if len(text) < 30:
            return SummarizationResult(
                status="error",
                content="Текст слишком короткий для анализа.",
                format_used=fmt,
                text_length=len(text),
            )

        if self.enable_cache and file_identifier:
            cached = await _load_from_cache(file_identifier, str(fmt))
            if cached is not None:
                return cached

        text_length = len(text)
        pipeline: str = "direct"
        degraded: bool = False
        chunks_processed: int = 1

        try:
            if text_length < self.direct_threshold:
                pipeline = "direct"
                summary, chunks_processed = await _direct_pipeline(text, fmt)
            else:
                pipeline = "map_reduce"
                summary, chunks_processed, degraded = await _map_reduce_pipeline(
                    text, fmt, self.concurrency
                )
        except Exception as exc:
            logger.error("Summarization pipeline failed: %s", exc, exc_info=True)
            return SummarizationResult(
                status="error",
                content=f"Ошибка генерации: {exc}",
                format_used=fmt,
                text_length=text_length,
                pipeline=pipeline,
            )

        quality_score: float | None = None
        confidence: str | None = None
        warnings: list[str] = []

        if self.enable_self_critique and not degraded:
            try:
                critique = await _self_critique(summary)
                quality_score = critique.overall
                confidence = critique.confidence
                if critique.issues:
                    warnings.extend(critique.issues)
            except Exception as exc:
                logger.debug("Self-critique skipped (non-fatal): %s", exc)

        if pipeline == "direct":
            was_truncated = text_length > 8_000
        else:
            was_truncated = False

        elapsed_ms = int((time.monotonic() - start) * 1_000)
        result = SummarizationResult(
            status="partial" if degraded else "success",
            content=summary,
            format_used=fmt,
            quality_score=quality_score,
            confidence=confidence,
            processing_time_ms=elapsed_ms,
            chunks_processed=chunks_processed,
            was_truncated=was_truncated,
            text_length=text_length,
            pipeline=pipeline,
            degraded=degraded,
            warnings=warnings,
        )

        if self.enable_cache and file_identifier and not degraded:
            await _save_to_cache(file_identifier, str(fmt), result)

        logger.info(
            "Summarization complete",
            extra={
                "format": str(fmt),
                "pipeline": pipeline,
                "chunks": chunks_processed,
                "elapsed_ms": elapsed_ms,
                "quality": quality_score,
                "confidence": confidence,
                "degraded": degraded,
                "text_length": text_length,
                "was_truncated": was_truncated,
            },
        )

        return result


_orchestrator_instance: SummarizationOrchestrator | None = None


def get_orchestrator() -> SummarizationOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SummarizationOrchestrator()
        logger.info("SummarizationOrchestrator singleton created")
    return _orchestrator_instance
