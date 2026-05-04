"""
DirectSummarizationPipeline — single-pass LLM call for documents
that fit within the model's context window.

Threshold: document_tokens < context_window * 0.70
(30% headroom for system prompt + output)

2025 Best Practices:
- Structured Outputs via JSON Schema enforcement
- Prompt Caching prefix for repeated system prompts
- Full async — no blocking calls
- Retry with exponential backoff on rate limit / 5xx
- Token counting BEFORE the call (not after) to prevent context overflow
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from edms_ai_assistant.summarizer.chunking.token_aware import count_tokens
from edms_ai_assistant.summarizer.observability.tracing import (
    Stopwatch,
    record_llm_call,
    trace_stage,
)
from edms_ai_assistant.summarizer.prompts.registry import PromptRegistry
from edms_ai_assistant.summarizer.structured.models import (
    MODE_OUTPUT_MODEL,
    LLMBaseModel,
    SummaryMode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Client Abstraction
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract LLM client — swap between OpenAI, Ollama, Anthropic."""

    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
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
        max_tokens: int = 1024,
    ) -> str:
        """Return plain text response (no JSON enforcement)."""
        ...


class OpenAICompatibleClient(LLMClient):
    """Async client for any OpenAI-compatible API (OpenAI, Ollama, vLLM, etc.)

    Uses httpx directly — no LangChain dependency, full control.
    Supports:
      - Structured Outputs (response_format with JSON Schema)
      - Prompt Caching (automatic for compatible backends)
      - Retry with exponential backoff
    """

    _MAX_RETRIES = 3
    _BASE_BACKOFF = 1.0  # seconds

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        import httpx
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
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

        if response_format:
            payload["response_format"] = response_format

        text, in_t, out_t = await self._post_with_retry(payload)
        return text, in_t, out_t

    async def complete_plain(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        text, _, _ = await self.complete(
            system, user, model=model, temperature=temperature, max_tokens=max_tokens
        )
        return text

    async def _post_with_retry(self, payload: dict) -> tuple[str, int, int]:
        """POST with exponential backoff retry on 429 / 5xx."""
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = await self._client.post("/chat/completions", json=payload)

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning("Rate limited (429) — retrying after %.1fs", retry_after)
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code >= 500:
                    wait = self._BASE_BACKOFF * (2 ** attempt)
                    logger.warning("Server error %d — retrying after %.1fs",
                                   response.status_code, wait)
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
                usage = data.get("usage", {})
                in_t = usage.get("prompt_tokens", count_tokens(str(payload.get("messages", ""))))
                out_t = usage.get("completion_tokens", count_tokens(text))
                return text, in_t, out_t

            except Exception as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    wait = self._BASE_BACKOFF * (2 ** attempt)
                    logger.warning("LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                                   attempt + 1, self._MAX_RETRIES, exc, wait)
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {self._MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# JSON Schema builder for Structured Outputs
# ---------------------------------------------------------------------------


def build_response_format(mode: SummaryMode) -> dict:
    """Build OpenAI Structured Outputs response_format from Pydantic v2 model schema."""
    model_cls = MODE_OUTPUT_MODEL[mode]
    schema = model_cls.model_json_schema()
    # Remove Pydantic-specific keys not valid in JSON Schema Draft-7
    _clean_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{mode.value}_output",
            "strict": True,
            "schema": schema,
        },
    }


def _clean_schema(schema: dict) -> None:
    """Recursively remove Pydantic-specific keys from JSON Schema."""
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
# Pipeline Result
# ---------------------------------------------------------------------------


class PipelineResult:
    """Result of a summarization pipeline run."""

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
# Direct Pipeline
# ---------------------------------------------------------------------------


class DirectSummarizationPipeline:
    """Single-pass pipeline for documents fitting in context window.

    Flow: text → render prompt → LLM call → validate JSON → PipelineResult
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_registry: PromptRegistry,
        model: str,
    ) -> None:
        self._llm = llm_client
        self._prompts = prompt_registry
        self._model = model

    async def run(
        self,
        text: str,
        mode: SummaryMode,
        *,
        language: str = "ru",
        span: Any = None,
    ) -> PipelineResult:
        """Execute direct summarization pipeline.

        Args:
            text: Document text (already extracted).
            mode: Summarization mode.
            language: BCP-47 output language tag.
            span: Optional OTel span from parent.

        Returns:
            PipelineResult with typed output.
        """
        template = self._prompts.get(mode)
        system, user = template.render(text, language=language)
        if language and language not in ("en", "auto"):
            system += (
                f"\n\nCRITICAL INSTRUCTION: You MUST generate ALL text content, "
                f"analysis, and reasoning strictly in '{language}' language. "
                f"Do NOT use English for any generated text values."
            )
        response_format = build_response_format(mode)

        sw = Stopwatch()

        async with trace_stage("direct.llm_call", {"mode": mode.value, "model": self._model}):
            raw_text, in_t, out_t = await self._llm.complete(
                system,
                user,
                model=self._model,
                temperature=0.1,
                max_tokens=2048,
                response_format=response_format,
            )

        latency_ms = sw.elapsed_ms()

        if span:
            record_llm_call(span, "direct", self._model, in_t, out_t, latency_ms)

        # Validate against Pydantic model
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

    # def _validate_output(self, mode: SummaryMode, raw_text: str) -> LLMBaseModel:
    #     """Parse and validate LLM JSON output against Pydantic model.
    #
    #     Attempts three recovery strategies on malformed JSON:
    #     1. Direct JSON parse
    #     2. Strip markdown fences
    #     3. Extract first JSON object literal
    #     """
    #     model_cls = MODE_OUTPUT_MODEL[mode]
    #
    #     for attempt, text in enumerate(self._candidate_texts(raw_text)):
    #         try:
    #             data = json.loads(text)
    #             return model_cls.model_validate(data)
    #         except (json.JSONDecodeError, ValidationError) as exc:
    #             if attempt == 2:
    #                 logger.error(
    #                     "Failed to validate LLM output for mode=%s after 3 attempts: %s\n"
    #                     "Raw: %.200s",
    #                     mode.value, exc, raw_text,
    #                 )
    #                 raise ValueError(
    #                     f"LLM output validation failed for mode {mode.value}: {exc}"
    #                 ) from exc
    #     # Unreachable, but satisfies mypy
    #     raise RuntimeError("Unreachable")

    def _validate_output(self, mode: SummaryMode, raw_text: str) -> "LLMBaseModel":
        """Parse and validate LLM output against the expected Pydantic model."""
        import json
        import re
        from edms_ai_assistant.summarizer.structured.models import MODE_OUTPUT_MODEL

        # 1. Очистка маркдауна (LLM часто оборачивают JSON в ```json ... ```)
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            # Удаляем первую строку (```json) и последнюю (```)
            clean_text = re.sub(r"^```(?:json)?\s*\n?", "", clean_text)
            clean_text = re.sub(r"\n?```\s*$", "", clean_text)
            clean_text = clean_text.strip()

        # 2. Парсинг JSON
        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse LLM output as JSON: %s\nOutput: %s",
                e, raw_text[:500]
            )
            raise ValueError(f"LLM output is not valid JSON: {e}") from e

        # 3. Валидация через Pydantic модель на основе режима
        model_cls = MODE_OUTPUT_MODEL.get(mode)

        if model_cls is None:
            raise ValueError(f"No output model defined for mode: {mode.value}")

        try:
            return model_cls.model_validate(data)
        except Exception as e:
            logger.error(
                "Pydantic validation failed for mode %s: %s\nData: %s",
                mode.value, e, json.dumps(data, ensure_ascii=False, indent=2)[:500]
            )
            raise ValueError(
                f"LLM output does not match expected schema for {mode.value}: {e}"
            ) from e

    @staticmethod
    def _candidate_texts(raw: str) -> list[str]:
        """Generate candidate cleaned texts from raw LLM output."""
        candidates = [raw.strip()]
        # Strip markdown fences
        stripped = raw.strip()
        if stripped.startswith("```"):
            inner = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
            candidates.append(inner.strip())
        # Extract first JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start : end + 1])
        return candidates