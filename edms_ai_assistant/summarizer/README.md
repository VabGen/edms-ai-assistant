# `summarizer/` — Document Summarization Module

Production-ready пайплайн суммаризации документов EDMS с двухуровневым кэшем,
streaming, idempotency, OpenTelemetry и LLM-as-judge оценкой качества.

## Содержание

- [Архитектура](#архитектура)
- [Public API](#public-api)
- [HTTP-эндпоинты](#http-эндпоинты)
- [Конфигурация](#конфигурация)
- [Режимы суммаризации](#режимы-суммаризации)
- [Точки расширения](#точки-расширения)
- [Observability](#observability)
- [Обработка ошибок](#обработка-ошибок)
- [Тестирование](#тестирование)

---

## Архитектура

```
┌──────────────┐
│ FastAPI      │  api/router.py — HTTP, SSE-стриминг, маппинг ошибок→HTTP
└──────┬───────┘
       ▼
┌──────────────┐
│ Service      │  service.py — фасад: cache, in-flight dedup,
│ (facade)     │               quality scoring, streaming
└──────┬───────┘
       ▼
┌──────────────────────────┐    ┌─────────────────────┐
│ DirectPipeline           │    │ MapReducePipeline    │
│ (≤ context_window)        │    │ (> context_window)   │
│ run() / run_stream()      │    │ run() / run_stream() │
└──────────┬───────────────┘    └──────────┬──────────┘
           ▼                                ▼
       ┌──────────┐                ┌──────────────────┐
       │ LLMClient│                │ SmartChunker     │
       │ + retry  │                │ (structural →    │
       │ + stream │                │  fallback split) │
       └──────────┘                └──────────────────┘
```

Layers:

- **`api/`** — FastAPI router и DTO. Зависит только от Service-фасада.
- **`service.py`** — оркестрация: cache → text extraction → pipeline → response. Idempotency через in-flight registry.
- **`pipeline/`** — `DirectSummarizationPipeline` (один LLM-вызов), `MapReducePipeline` (параллельный map + одиночный reduce). Оба умеют `run()` и `run_stream()`.
- **`chunking/`** — `SmartChunker` авто-выбирает между структурным (по заголовкам) и token-aware fallback chunking.
- **`prompts/registry.py`** — версионированные промпты для всех режимов + map/reduce/judge.
- **`structured/models.py`** — typed Pydantic-модели для каждого режима. `MODE_OUTPUT_MODEL` mapping.
- **`cache/cache.py`** — TwoLevelCache: Redis L1 (TTL 1h) + Postgres L2 (TTL 30d). Cache key включает prompt_version → bump промптов инвалидирует всё.
- **`observability/`** — OpenTelemetry (GenAI semantic conventions), cost accumulator, request_id propagation в логи.
- **`errors.py`** — типизированная иерархия исключений.
- **`config.py`** — `SummarizerConfig` (frozen Pydantic), фабрика из app settings.
- **`container.py`** — DI: собирает Service из конфига.

---

## Public API

```python
from edms_ai_assistant.summarizer import (
    SummarizationService,
    SummarizationRequest,
    SummarizationResponse,
    SummaryMode,
    build_summarization_service,
    format_output_as_markdown,
)

# Bootstrap
service = await build_summarization_service(settings)

# Non-stream
req = SummarizationRequest(
    file_content=open("doc.pdf", "rb").read(),
    file_name="doc.pdf",
    mode=SummaryMode.EXTRACTIVE,
    request_id=str(uuid.uuid4()),
    enable_quality_score=True,  # +1 LLM-вызов на judge
)
resp = await service.summarize(req)
text = format_output_as_markdown(resp)

# Streaming
async for event in service.summarize_stream(req):
    if hasattr(event, "kind"):  # StreamEvent
        if event.kind == "delta":
            print(event.text, end="", flush=True)
    else:  # SummarizationResponse (финал)
        print(format_output_as_markdown(event))

# Lifecycle
await service.aclose()
```

---

## HTTP-эндпоинты

| Method | Path | Описание |
|---|---|---|
| `GET` | `/summarize/modes` | Список режимов суммаризации с описаниями |
| `POST` | `/summarize` | Multipart upload → синхронный `SummarizationResponse` |
| `POST` | `/summarize/stream` | SSE: events `delta` / `result` / `error` / `[DONE]` |
| `DELETE` | `/summarize/cache/{file_hash}/{mode}` | Инвалидация кэша |
| `GET` | `/summarize/health` | Cache layers + cache stats |

### SSE event format

```
data: {"event":"delta","text":"Это связный..."}

data: {"event":"delta","text":" пересказ..."}

data: {"event":"result","formatted":"...","cache_hit":false,"latency_ms":2300,"cost_usd":0.0042,"input_tokens":1200,"output_tokens":450,"chunking_strategy":"direct.stream","request_id":"..."}

data: [DONE]
```

При ошибке:
```
data: {"event":"error","message":"Не удалось извлечь текст из 'doc.pdf'"}
data: [DONE]
```

---

## Конфигурация

`SummarizerConfig.from_app_settings(settings)` извлекает:

| Field | Settings key | Default |
|---|---|---|
| `llm_base_url` | `LLM_GENERATIVE_URL` | `http://localhost:11434/v1` |
| `llm_model` | `LLM_GENERATIVE_MODEL` | required |
| `llm_api_key` | `LLM_API_KEY` / `OPENAI_API_KEY` | `"ollama"` |
| `llm_timeout_s` | `LLM_TIMEOUT` | 120 |
| `context_window_tokens` | `SUMMARIZER_CONTEXT_WINDOW` | 4096 |
| `max_output_tokens` | `SUMMARIZER_MAX_OUTPUT_TOKENS` | 4096 |
| `max_concurrent_map` | `SUMMARIZER_MAX_CONCURRENT_MAP` | 6 |
| `otlp_endpoint` | `TELEMETRY_ENDPOINT` | `None` |

Cache key:
```
SHA-256(file_content)::mode::language::prompt_version
```
→ хеш-32 hex → `smz:abcdef…` (36 chars).

Bump `PROMPT_REGISTRY_VERSION` инвалидирует ВСЁ автоматически.

---

## Режимы суммаризации

| Mode | Output model | Use case |
|---|---|---|
| `executive` | `ExecutiveSummaryOutput` | Headline + 3-5 bullets + recommendation для руководителей |
| `detailed_notes` | `DetailedNotesOutput` | Структурированный конспект с типом документа и сущностями |
| `action_items` | `ActionItemsOutput` | Задачи с owner / deadline / priority |
| `thesis` | `ThesisPlanOutput` | Иерархический тезисный план для аналитики |
| `extractive` | `ExtractiveOutput` | Категоризированные факты (ДАТА / ПЕРСОНА / СУММА…) |
| `abstractive` | `AbstractiveOutput` | Связный пересказ + темы |
| `multilingual` | `MultilingualOutput` | Авто-определение языка + изложение на целевом |

---

## Точки расширения

### Новый режим суммаризации

1. Добавить значение в `SummaryMode` (`structured/models.py`).
2. Описать выходную модель: `class FooOutput(LLMBaseModel): ...`.
3. Зарегистрировать в `MODE_OUTPUT_MODEL`.
4. Создать `PROMPT_FOO` в `prompts/registry.py` и добавить в `PromptRegistry._DIRECT`.
5. Расширить `format_output_as_markdown` в `service.py`.
6. Bump `PROMPT_REGISTRY_VERSION`.
7. Добавить unit-тест в `tests/summarizer/test_format_output.py`.

### Новый LLM-провайдер

Реализовать абстрактный `LLMClient` (`pipeline/direct.py`):
- `complete(system, user, *, model, temperature, max_tokens, response_format) -> (text, in_tokens, out_tokens)`
- `complete_plain(...) -> str`
- `complete_stream(...) -> AsyncIterator[StreamEvent]`
- `aclose() -> None`

Затем добавить ветку в `container.py` или передать DI напрямую.

### Новая chunking-стратегия

Реализовать `ChunkingStrategy` (`chunking/structural.py`):
- `chunk(text, *, max_tokens, overlap_tokens) -> list[TextChunk]`
- `can_handle(text) -> bool`

Подключить в `SmartChunker.__init__`.

### Кастомный pricing

`tracing._COST_TABLE` — добавить запись `"my-model": (input_per_1k, output_per_1k)`. Lookup делается longest-match по подстроке.

---

## Observability

### Spans

```
summarizer.extract_text
summarizer.pipeline | pipeline.stream
  ├── summarizer.direct.llm_call          (Direct path)
  └── summarizer.map_reduce.pipeline      (Map-Reduce path)
      ├── summarizer.map_reduce.chunking
      ├── summarizer.map_reduce.map
      └── summarizer.map_reduce.reduce
summarizer.quality.judge                  (опционально)
```

### Span-атрибуты (OTel GenAI semantic conventions)

- `gen_ai.system = "openai"`
- `gen_ai.request.model`, `gen_ai.response.model`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- `gen_ai.operation.name`

Параллельно сохранены legacy-атрибуты для обратной совместимости:
- `llm.model`, `llm.input_tokens`, `llm.output_tokens`, `llm.cost_usd`, `llm.latency_ms`

### Cost accumulator

`RequestCostAccumulator` хранится в contextvar и суммирует все LLM-вызовы внутри запроса:

```python
from edms_ai_assistant.summarizer.observability.tracing import get_current_accumulator

acc = get_current_accumulator()
print(acc.to_dict())  # включает stages, total tokens, total cost
```

### Structured logging

`request_id_var` (contextvar) автоматически пробрасывается в каждый log-record через `RequestIdFilter`. Чтобы увидеть в выводе — добавьте `%(request_id)s` в `LOGGING_FORMAT`.

```
2026-01-15 10:23:45 INFO [req-abc123] Cache L1 hit: key=smz:c4f2...
```

---

## Обработка ошибок

Иерархия `errors.py` → HTTP-маппинг:

| Exception | HTTP | Случай |
|---|---|---|
| `TextExtractionError` | 422 | Битый PDF/DOCX, пустой текст |
| `ValidationError` | 422 | LLM вернул JSON, не прошедший pydantic |
| `LLMRateLimitedError` | 503 | 429 от провайдера |
| `LLMClientError` | 502 | 4xx (битый prompt, плохой токен) |
| `LLMServerError` | 504 | 5xx после исчерпания ретраев |
| `LLMTransportError` | 504 | Network / timeout |
| `LLMResponseError` | 500 | Невалидный 2xx body |
| `MapStageError` / `PipelineError` | 500 | Все Map-чанки упали |
| `CacheError` | — | Best-effort, логируется, не пробрасывается |

4xx → детали ошибки попадают в response. 5xx → generic `"Ошибка суммаризации. См. логи сервиса."` (без leak внутренних строк).

---

## Тестирование

```bash
# Unit-тесты (в памяти, без внешних сервисов, ~6s)
pytest tests/summarizer/

# С coverage
pytest tests/summarizer/ --cov=edms_ai_assistant.summarizer --cov-report=term-missing
```

Тесты разбиты по модулям и **не требуют ни Docker, ни LLM**: используют `httpx.MockTransport` для SSE и fake `LLMClient` для service-уровневых тестов.

| File | Coverage |
|---|---|
| `test_models.py` | Pydantic models + computed_field |
| `test_chunking.py` | SmartChunker + sentence splitter (с русскими аббревиатурами) + token counting |
| `test_json_repair.py` | JSON repair / extract / fallback для всех режимов |
| `test_format_output.py` | `format_output_as_markdown` для всех режимов |
| `test_prompts_and_cost.py` | PromptRegistry + pricing table (longest-match) |
| `test_cache.py` | CacheEntry key derivation + serialization |
| `test_streaming.py` | SSE-парсинг с моковым httpx (deltas, errors, malformed lines) |
| `test_config.py` | `SummarizerConfig.from_app_settings`, frozen, normalized URL |
| `test_quality_judge.py` | LLM-as-judge: confidence levels, clamping, error handling |
| `test_map_reduce_stream.py` | `MapReducePipeline.run_stream` |
| `test_idempotency.py` | In-flight dedup для конкурентных запросов |
| `test_errors.py` | Иерархия исключений + HTTP-маппинг |
| `test_logging_ctx.py` | request_id contextvar propagation |

### Что **не** покрывают unit-тесты (требуют интеграции)

- Реальный Postgres + Redis L1/L2 — TwoLevelCache cache hit/miss flow на боевых хранилищах. Решение: testcontainers.
- Реальный LLM (Ollama/OpenAI) — end-to-end summarization. Решение: e2e тест с feature flag.
- Concurrency stress на cache layer.

---

## Соглашения

- **Никогда** не ловите `Exception` без причины — используйте конкретные классы из `errors.py`.
- **Cache** — best-effort: любая ошибка кэша → log + продолжаем без кэша.
- **Background tasks** трекаем через `service._bg_tasks` set + `add_done_callback(set.discard)`. На shutdown `service.aclose()` дожидается их с timeout.
- **Streaming**: для структурированных режимов клиент видит сырые JSON-токены; финальный `result` events содержит уже отформатированный markdown.
- **Idempotency**: запросы с одинаковым `cache_key` (= file_hash + mode + language + prompt_version) ждут одну общую генерацию. `force_refresh=True` обходит.
