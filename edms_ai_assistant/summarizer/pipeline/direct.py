"""
DirectSummarizationPipeline — single-pass LLM call.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import httpx
from pydantic import ValidationError

from edms_ai_assistant.summarizer.chunking.token_aware import count_tokens
from edms_ai_assistant.summarizer.errors import (
    LLMClientError,
    LLMError,
    LLMRateLimitedError,
    LLMResponseError,
    LLMServerError,
    LLMTransportError,
    PipelineError,
)
from edms_ai_assistant.summarizer.errors import ValidationError as SummarizerValidationError

# Lazy probe для опционального json_repair
try:
    from json_repair import repair_json as _repair_json  # type: ignore[import]

    _HAS_JSON_REPAIR = True
except ImportError:  # pragma: no cover
    _repair_json = None  # type: ignore[assignment]
    _HAS_JSON_REPAIR = False
from edms_ai_assistant.summarizer.observability.tracing import (
    Stopwatch,
    record_llm_call,
    trace_stage,
)
from edms_ai_assistant.summarizer.prompts.registry import PromptRegistry
from edms_ai_assistant.summarizer.structured.models import (
    MODE_OUTPUT_MODEL,
    AbstractiveOutput,
    ExtractiveOutput,
    LLMBaseModel,
    SummaryMode,
)

logger = logging.getLogger(__name__)

_OLLAMA_BACKENDS = ("11434", "ollama")


def _is_ollama(base_url: str) -> bool:
    return any(marker in base_url for marker in _OLLAMA_BACKENDS)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> tuple[str, int, int]:
        """Return (response_text, input_tokens, output_tokens)."""
        ...

    @abstractmethod
    async def complete_plain(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str: ...

    @abstractmethod
    def complete_stream(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncIterator["StreamEvent"]:
        """Стрим OpenAI-совместимых дельт.

        Yields:
            StreamEvent — последовательность token-дельт + финальный usage.
        """
        ...

    async def aclose(self) -> None:
        pass


@dataclasses.dataclass(frozen=True, slots=True)
class StreamEvent:
    """Событие стрима: текстовая дельта или финальный usage."""

    kind: str  # "delta" | "done" | "error"
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    error: str | None = None


class OpenAICompatibleClient(LLMClient):
    """Async client for any OpenAI-compatible API (OpenAI, Ollama, vLLM)."""

    _MAX_RETRIES = 3
    _BASE_BACKOFF = 1.0

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._is_ollama = _is_ollama(base_url)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> tuple[str, int, int]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format and not self._is_ollama:
            payload["response_format"] = response_format

        return await self._post_with_retry(payload)

    async def complete_plain(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        text, _, _ = await self.complete(
            system,
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return text

    async def complete_stream(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        """Стрим OpenAI SSE: парсит `data: {...}` строки и эмитит StreamEvent'ы."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        prompt_tokens_estimate = count_tokens(system) + count_tokens(user)
        accumulated_text = ""
        finish_reason: str | None = None
        usage_in: int | None = None
        usage_out: int | None = None

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    yield StreamEvent(
                        kind="error",
                        error=f"HTTP {response.status_code}: {body[:300]}",
                    )
                    return

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content") or ""
                        if content:
                            accumulated_text += content
                            yield StreamEvent(kind="delta", text=content)
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr

                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        usage_in = usage.get("prompt_tokens", usage_in)
                        usage_out = usage.get("completion_tokens", usage_out)

        except (httpx.TransportError, httpx.TimeoutException) as exc:
            yield StreamEvent(kind="error", error=f"transport: {exc}")
            return

        yield StreamEvent(
            kind="done",
            text=accumulated_text,
            input_tokens=usage_in if usage_in is not None else prompt_tokens_estimate,
            output_tokens=(
                usage_out if usage_out is not None else count_tokens(accumulated_text)
            ),
            finish_reason=finish_reason,
        )

    async def _post_with_retry(self, payload: dict) -> tuple[str, int, int]:
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    wait = self._BASE_BACKOFF * (2**attempt)
                    logger.warning(
                        "LLM transport error (attempt %d/%d): %s — retry in %.1fs",
                        attempt + 1,
                        self._MAX_RETRIES,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                break

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2**attempt))
                logger.warning("Rate limited (429) — retrying after %.1fs", retry_after)
                last_exc = LLMRateLimitedError(
                    "HTTP 429 Too Many Requests", status_code=429
                )
                await asyncio.sleep(retry_after)
                continue

            if response.status_code >= 500:
                last_exc = LLMServerError(
                    f"HTTP {response.status_code}: {response.text[:300]}",
                    status_code=response.status_code,
                )
                if attempt < self._MAX_RETRIES - 1:
                    wait = self._BASE_BACKOFF * (2**attempt)
                    logger.warning(
                        "LLM server %d (attempt %d/%d) — retry in %.1fs",
                        response.status_code,
                        attempt + 1,
                        self._MAX_RETRIES,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                break

            if 400 <= response.status_code < 500:
                raise LLMClientError(
                    f"LLM client error {response.status_code}: {response.text[:300]}",
                    status_code=response.status_code,
                )

            try:
                data = response.json()
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
            except (ValueError, KeyError, IndexError) as exc:
                raise LLMResponseError(
                    f"Malformed LLM response: {exc}; body={response.text[:300]}",
                    status_code=response.status_code,
                ) from exc

            finish_reason = choice.get("finish_reason", "stop")
            if finish_reason == "length":
                logger.warning(
                    "LLM response truncated (finish_reason=length, max_tokens=%s)",
                    payload.get("max_tokens", "?"),
                )

            usage = data.get("usage", {})
            in_t = usage.get("prompt_tokens") or count_tokens(
                "".join(m.get("content", "") for m in payload.get("messages", []))
            )
            out_t = usage.get("completion_tokens") or count_tokens(text)
            return text, in_t, out_t

        if isinstance(last_exc, (httpx.TransportError, httpx.TimeoutException)):
            raise LLMTransportError(
                f"LLM transport failed after {self._MAX_RETRIES} attempts: {last_exc}"
            ) from last_exc
        if isinstance(last_exc, LLMError):
            raise last_exc
        raise LLMServerError(
            f"LLM call failed after {self._MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# JSON Repair utilities
# ---------------------------------------------------------------------------


def _try_repair_truncated_json(raw: str) -> str | None:
    """
    Пытается починить обрезанный JSON от Ollama.

    Стратегии:
    1. json_repair если установлен
    2. Подсчёт незакрытых { [ и дозакрытие
    3. Удаление неполной последней строки и попытка json.loads
    """
    text = raw.strip()
    if not text:
        return None

    if _HAS_JSON_REPAIR and _repair_json is not None:
        try:
            repaired = _repair_json(text)
            if repaired and repaired != "null":
                return repaired
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug("json_repair failed: %s", exc)

    try:
        lines = text.split("\n")
        for i in range(len(lines), 0, -1):
            candidate = "\n".join(lines[:i])
            open_braces = candidate.count("{") - candidate.count("}")
            open_brackets = candidate.count("[") - candidate.count("]")

            if open_braces <= 0 and open_brackets <= 0:
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

            suffix = "]" * open_brackets + "}" * open_braces
            repaired = candidate.rstrip(",").rstrip() + "\n" + suffix
            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                continue

    except Exception as e:
        logger.debug("JSON repair failed: %s", e)

    return None


def _extract_json_from_text(raw: str, mode: SummaryMode) -> str | None:
    """
    Если модель вернула текст вместо JSON — пробуем найти JSON внутри.
    Или создаём минимальный валидный объект из текста.
    """
    text = raw.strip()

    # Ищем JSON-блок в тексте
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        candidate = json_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            repaired = _try_repair_truncated_json(candidate)
            if repaired:
                return repaired

    truncated = text[:2000]
    short = text[:200]

    fallback_map: dict[SummaryMode, dict] = {
        SummaryMode.ABSTRACTIVE: {"summary": truncated, "key_themes": []},
        SummaryMode.EXTRACTIVE: {"facts": [], "document_summary": short},
        SummaryMode.EXECUTIVE: {
            "headline": short,
            "bullets": [],
            "recommendation": None,
        },
        SummaryMode.THESIS: {
            "main_argument": short,
            "sections": [],
            "conclusion": "",
        },
        SummaryMode.ACTION_ITEMS: {
            "action_items": [],
            "document_context": short,
        },
        SummaryMode.DETAILED_NOTES: {
            "document_type": "ДОКУМЕНТ",
            "sections": [
                {"title": "Содержание", "content": truncated, "subsections": []}
            ],
            "key_entities": [],
            "date_range": None,
        },
        SummaryMode.MULTILINGUAL: {
            "detected_language": "ru",
            "summary_language": "ru",
            "summary": truncated,
            "translation_notes": None,
        },
    }

    payload = fallback_map.get(mode)
    return json.dumps(payload, ensure_ascii=False) if payload else None


# ---------------------------------------------------------------------------
# Pipeline Result
# ---------------------------------------------------------------------------


class PipelineResult:
    __slots__ = (
        "mode",
        "output",
        "raw_json",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "model",
        "chunking_strategy",
        "chunk_count",
    )

    def __init__(
        self,
        mode: SummaryMode,
        output: LLMBaseModel,
        raw_json: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model: str,
        chunking_strategy: str = "direct",
        chunk_count: int = 1,
    ) -> None:
        self.mode = mode
        self.output = output
        self.raw_json = raw_json
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.model = model
        self.chunking_strategy = chunking_strategy
        self.chunk_count = chunk_count


# ---------------------------------------------------------------------------
# JSON Schema builder
# ---------------------------------------------------------------------------


def build_response_format(mode: SummaryMode) -> dict:
    model_cls = MODE_OUTPUT_MODEL[mode]
    schema = model_cls.model_json_schema()
    _clean_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{mode.value}_output",
            "strict": False,
            "schema": schema,
        },
    }


def _clean_schema(schema: dict) -> None:
    for key in ("title", "description"):
        schema.pop(key, None)
    for value in schema.values():
        if isinstance(value, dict):
            _clean_schema(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _clean_schema(item)


# ---------------------------------------------------------------------------
# Direct Pipeline
# ---------------------------------------------------------------------------


class DirectSummarizationPipeline:
    """Single-pass pipeline for documents fitting in context window."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_registry: PromptRegistry,
        model: str,
        max_output_tokens: int = 4096,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_registry
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._is_ollama = _is_ollama(getattr(llm_client, "_base_url", ""))

    async def run(
        self,
        text: str,
        mode: SummaryMode,
        *,
        language: str = "ru",
        span: Any = None,
    ) -> PipelineResult:
        template = self._prompts.get(mode)
        system, user = template.render(text, language=language)

        # Дополнительное усиление языкового требования
        if language == "ru":
            system = (
                system
                + "\n\nВАЖНО: Все значения в JSON должны быть написаны на РУССКОМ языке."
            )

        response_format = None
        if not self._is_ollama:
            response_format = build_response_format(mode)

        sw = Stopwatch()

        async with trace_stage(
            "direct.llm_call", {"mode": mode.value, "model": self._model}
        ):
            raw_text, in_t, out_t = await self._llm.complete(
                system,
                user,
                model=self._model,
                temperature=0.1,
                max_tokens=self._max_output_tokens,
                response_format=response_format,
            )

        latency_ms = sw.elapsed_ms()

        if span:
            record_llm_call(span, "direct", self._model, in_t, out_t, latency_ms)

        output = self._validate_output(mode, raw_text)

        return PipelineResult(
            mode=mode,
            output=output,
            raw_json=raw_text,
            input_tokens=in_t,
            output_tokens=out_t,
            latency_ms=latency_ms,
            model=self._model,
            chunking_strategy="direct",
            chunk_count=1,
        )

    async def run_stream(
        self,
        text: str,
        mode: SummaryMode,
        *,
        language: str = "ru",
        span: Any = None,
    ) -> AsyncIterator["StreamEvent | PipelineResult"]:
        """
        Стрим-версия: yields StreamEvent (kind='delta') во время генерации
        и финальный PipelineResult со структурированным выводом в конце.

        Применимо для всех режимов: для структурированных режимов токены LLM
        идут как сырой JSON, который потом валидируется в Pydantic-модель.
        """
        template = self._prompts.get(mode)
        system, user = template.render(text, language=language)
        if language == "ru":
            system = (
                system + "\n\nВАЖНО: Все значения в JSON должны быть на русском языке."
            )

        sw = Stopwatch()
        accumulated = ""
        in_t = 0
        out_t = 0

        async for event in self._llm.complete_stream(
            system,
            user,
            model=self._model,
            temperature=0.1,
            max_tokens=self._max_output_tokens,
        ):
            if event.kind == "delta":
                accumulated += event.text
                yield event
            elif event.kind == "done":
                in_t = event.input_tokens
                out_t = event.output_tokens
                if event.finish_reason == "length":
                    logger.warning("Stream truncated (finish_reason=length)")
            elif event.kind == "error":
                logger.error("LLM stream error: %s", event.error)
                raise LLMTransportError(f"LLM streaming failed: {event.error}")

        latency_ms = sw.elapsed_ms()
        if span:
            record_llm_call(span, "direct.stream", self._model, in_t, out_t, latency_ms)

        output = self._validate_output(mode, accumulated)
        yield PipelineResult(
            mode=mode,
            output=output,
            raw_json=accumulated,
            input_tokens=in_t,
            output_tokens=out_t,
            latency_ms=latency_ms,
            model=self._model,
            chunking_strategy="direct.stream",
            chunk_count=1,
        )

    def _validate_output(self, mode: SummaryMode, raw_text: str) -> LLMBaseModel:
        """
        Разбирает и валидирует вывод LLM.

        Стратегии восстановления (в порядке приоритета):
        1. Прямой JSON parse
        2. Убираем markdown-блоки
        3. JSON repair (починка обрезанного JSON)
        4. Извлечение JSON из текста
        5. Fallback объект с текстом внутри
        """
        model_cls = MODE_OUTPUT_MODEL[mode]

        # Кандидаты для парсинга
        candidates = self._build_candidates(raw_text)

        last_error: Exception | None = None
        for i, candidate in enumerate(candidates):
            try:
                data = json.loads(candidate)
                result = model_cls.model_validate(data)
                if i > 0:
                    logger.info(
                        "JSON parsed using recovery strategy %d for mode=%s",
                        i,
                        mode.value,
                    )
                return result
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                continue

        # Финальный fallback — оборачиваем текст
        logger.warning(
            "All JSON parse strategies failed for mode=%s, using text fallback. "
            "Last error: %s. Raw (first 200 chars): %.200s",
            mode.value,
            last_error,
            raw_text,
        )
        fallback_json = _extract_json_from_text(raw_text, mode)
        if fallback_json:
            try:
                data = json.loads(fallback_json)
                return model_cls.model_validate(data)
            except Exception as exc:
                logger.error("Fallback JSON also failed: %s", exc)

        raise SummarizerValidationError(
            f"Не удалось разобрать ответ LLM для режима {mode.value}. "
            f"Последняя ошибка: {last_error}"
        )

    def _build_candidates(self, raw: str) -> list[str]:
        """Генерирует кандидатов для JSON-парсинга."""
        candidates: list[str] = []
        text = raw.strip()

        # 1. Как есть
        candidates.append(text)

        # 2. Убираем markdown-блоки (```json ... ```)
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", text)
        stripped = re.sub(r"\n?```\s*$", "", stripped).strip()
        if stripped != text:
            candidates.append(stripped)

        # 3. JSON repair для обрезанного JSON
        repaired = _try_repair_truncated_json(stripped or text)
        if repaired and repaired not in candidates:
            candidates.append(repaired)

        # 4. Извлекаем первый JSON-объект из текста
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            extracted = json_match.group(0)
            if extracted not in candidates:
                candidates.append(extracted)
            # Repair извлечённого
            repaired2 = _try_repair_truncated_json(extracted)
            if repaired2 and repaired2 not in candidates:
                candidates.append(repaired2)

        return candidates
