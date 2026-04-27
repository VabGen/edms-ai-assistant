# EDMS AI Assistant

> AI-powered assistant for electronic document management systems (EDMS/СЭД).  
> Automates document workflows, analysis, and task routing via LangGraph agents.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-0.5+-blueviolet)](https://docs.astral.sh/uv/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running](#running)
- [API](#api)
- [Development](#development)
- [Code Quality](#code-quality)
- [Database](#database)
- [Redis](#redis)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Overview

EDMS AI Assistant integrates with your EDMS platform via a FastAPI backend and a
LangGraph agent loop. It reads documents, resolves employee and geography references,
creates tasks and familiarity lists, and fills appeal cards — all through a chat interface.

**Key capabilities:**

- Document analysis and summarization (extractive / abstractive / thesis)
- Appeal card autofill from attached letters
- Task and familiarity list creation with employee disambiguation
- Resolution and notification workflows
- Human-in-the-loop for disambiguation and format selection
- Local file upload and comparison with EDMS attachments

---

## Requirements

| Tool      | Version | Install                                                                       |
|-----------|---------|-------------------------------------------------------------------------------|
| Python    | 3.13+   | [python.org](https://www.python.org/downloads/)                               |
| uv        | 0.5.0+  | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/)  |
| Redis     | 7.0+    | [redis.io](https://redis.io/docs/install/)                                    |
| PostgreSQL| 18+     | [postgresql.org](https://www.postgresql.org/download/)                        |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/edms-ai-assistant.git
cd edms-ai-assistant
```

### 2. Create virtual environment

```bash
uv venv --python 3.13
```

### 3. Install dependencies

```bash
# Development — runtime + dev + lint
uv sync --all-groups

# Production — runtime only
uv sync --no-dev

ollama serve

https://www.libreoffice.org/download/download/

Get-ChildItem -Path . -Directory -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
```

### Dependency groups

| Command                  | Installs              |
|--------------------------|-----------------------|
| `uv sync --all-groups`   | Runtime + dev + lint  |
| `uv sync --group dev`    | Runtime + tests       |
| `uv sync --group lint`   | Runtime + linters     |
| `uv sync --no-dev`       | Runtime only          |

### Activate virtual environment

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## Configuration

### 1. Create `.env` from template

```bash
cp .env.example .env
```
---

## Running

### Development

```bash
uv run uvicorn edms_ai_assistant.main:app --reload --reload-exclude ".venv"
```

### Production

```bash
uv run uvicorn edms_ai_assistant.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (если есть)

```bash
docker compose up --build
docker compose up -d          # фоновый режим
docker compose logs -f        # логи в реальном времени
docker compose down           # остановить
```

### PyCharm Run Configuration

`Run → Edit Configurations → + → Python`

| Field                  | Value                                                         |
|------------------------|---------------------------------------------------------------|
| **Module name**        | `uvicorn`                                                     |
| **Parameters**         | `edms_ai_assistant.main:app --reload --reload-exclude .venv`  |
| **Working directory**  | `D:\project\edms-ai-assistant`                                |
| **Interpreter**        | `.venv\Scripts\python.exe`                                    |

> ⚠️ Use **Module name**, not Script path — otherwise the package won't resolve correctly.

---

## API

После запуска документация доступна по адресам:

| UI        | URL                                |
|-----------|------------------------------------|
| Swagger   | http://localhost:8000/docs         |
| ReDoc     | http://localhost:8000/redoc        |
| OpenAPI   | http://localhost:8000/openapi.json |

### Основные эндпоинты

| Method | Path                      | Description                        |
|--------|---------------------------|------------------------------------|
| POST   | `/chat`                   | Отправить сообщение агенту         |
| POST   | `/upload-file`            | Загрузить файл для анализа         |
| GET    | `/chat/history/{thread_id}` | История диалога                  |
| POST   | `/chat/new`               | Создать новый тред                 |
| POST   | `/actions/summarize`      | Прямая суммаризация вложения       |
| GET    | `/health`                 | Статус компонентов агента          |

---

## Development

### Update lock file

```bash
# После редактирования pyproject.toml
uv lock

# Обновить конкретный пакет
uv lock --upgrade-package langchain

# Обновить все пакеты
uv lock --upgrade
```

> `uv.lock` должен быть закоммичен — он обеспечивает воспроизводимые сборки
> во всех окружениях и CI/CD (аналог `package-lock.json`).

### Добавить зависимость

```bash
# Runtime
uv add httpx

# Dev-only
uv add --group dev pytest-mock

# Lint-only
uv add --group lint pylint
```

### Удалить зависимость

```bash
uv remove httpx
```

### Показать дерево зависимостей

```bash
uv tree
uv tree --package langchain   # только для конкретного пакета
```

### Run tests

```bash
# Все тесты
uv run pytest

# С отчётом покрытия
uv run pytest --cov=edms_ai_assistant --cov-report=term-missing

# Конкретный файл или тест
uv run pytest tests/test_agent.py
uv run pytest tests/test_agent.py::test_chat_basic -v

# Только быстрые тесты (без интеграционных)
uv run pytest -m "not integration"

# Параллельно (если установлен pytest-xdist)
uv run pytest -n auto
```

---

## Code Quality

### Format

```bash
uv run black edms_ai_assistant/

# Только проверка без изменений
uv run black --check edms_ai_assistant/
```

### Lint

```bash
# Только проверка
uv run ruff check .

# Проверка + автофикс
uv run ruff check --fix .

# Показать все правила
uv run ruff rule --all
```

### Sort imports

```bash
uv run isort edms_ai_assistant/

# Только проверка
uv run isort --check-only edms_ai_assistant/
```

### Type check

```bash
uv run mypy .

# Строгий режим
uv run mypy . --strict

# Конкретный файл
uv run mypy edms_ai_assistant/agent.py
```

### Run all checks

```bash
uv run black edms_ai_assistant/ && uv run ruff check --fix . && uv run mypy .
```

### Pre-commit (рекомендуется)

```bash
# Установить хуки
uv run pre-commit install

# Запустить вручную на всех файлах
uv run pre-commit run --all-files
```

---

## Database

### PostgreSQL — основные команды

```bash
# Подключиться к БД
psql -h localhost -U postgres -d postgres

# Проверить подключение
psql -h localhost -U postgres -c "SELECT version();"
```

### Alembic миграции

```bash
# Инициализировать (только первый раз)
uv run alembic init alembic

# Создать новую миграцию
uv run alembic revision --autogenerate -m "add_table_name"

# Применить все миграции
uv run alembic upgrade head

# Откатить последнюю миграцию
uv run alembic downgrade -1

# Откатить все
uv run alembic downgrade base

# Показать текущую версию
uv run alembic current

# История миграций
uv run alembic history --verbose
```

---

## Redis

### Windows

```powershell
# Установить
winget install Redis.Redis

# http://localhost:5540
winget install RedisInsight.RedisInsight

# Запустить сервер
redis-server
# или
& "C:\Program Files\Redis\redis-server.exe"

net start Redis   # один раз после установки

# Проверить
redis-cli ping   # → PONG

# Управление службой
Start-Service Redis
Stop-Service Redis
Restart-Service Redis
Get-Service Redis
```

### Linux / macOS

```bash
# Установить (Ubuntu/Debian)
sudo apt install redis-server

# Установить (macOS)
brew install redis

# Запустить
sudo systemctl start redis
sudo systemctl enable redis   # автозапуск

# Проверить
redis-cli ping   # → PONG
redis-cli info server
```

### Отладка кэша

```bash
# Все ключи агента
redis-cli keys "edms:doc:*"

redis-cli get "edms:doc_analysis:<UUID>" | python -m json.tool

# TTL конкретного документа
redis-cli ttl "edms:doc:<UUID>"

# Получить значение ключа
redis-cli get "edms:doc:<UUID>"

# Удалить конкретный ключ
redis-cli del "edms:doc:<UUID>"

# Очистить все ключи (осторожно!)
redis-cli flushdb

# Мониторинг команд в реальном времени
redis-cli monitor

# Статистика памяти
redis-cli info memory
```

---

## Project Structure

```
edms-ai-assistant/
├── edms_ai_assistant/
│   ├── main.py               # FastAPI entry point
│   ├── agent.py              # LangGraph orchestration
│   ├── config.py             # Settings (pydantic-settings)
│   ├── model.py              # Pydantic models (public contracts)
│   ├── llm.py                # LLM factory
│   ├── security.py           # JWT token extraction
│   ├── tools/                # LangChain tools
│   │   ├── attachment.py     # EDMS attachment analysis
│   │   ├── local_file_tool.py# Local file reading
│   │   ├── file_compare_tool.py # File vs attachment comparison
│   │   └── ...
│   ├── clients/              # EDMS API HTTP clients
│   │   ├── base_client.py    # EdmsHttpClient base
│   │   ├── document_client.py
│   │   ├── employee_client.py
│   │   └── ...
│   ├── services/             # Business logic (Service Layer)
│   │   ├── nlp_service.py    # Semantic dispatcher, intent detection
│   │   ├── task_service.py   # Task creation with disambiguation
│   │   └── file_processor.py # Text extraction from files
│   ├── models/               # Internal domain models
│   ├── generated/            # Auto-generated OpenAPI models
│   └── utils/                # Shared utilities
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example              # Шаблон переменных окружения
├── pyproject.toml            # Project config + dependency groups
├── uv.lock                   # Locked dependency versions (commit this!)
├── .gitignore
└── README.md
```

---

## Troubleshooting

### `Access is denied` при `uv sync` (Windows)

Директория `.venv` заблокирована запущенным процессом (IDE, терминал, Python).

```powershell
# 1. Завершить все Python / uvicorn процессы
Get-Process python* | Stop-Process -Force
Get-Process uvicorn* | Stop-Process -Force

# 2. Удалить заблокированное окружение и артефакты сборки
Remove-Item -Recurse -Force .venv
Remove-Item -Recurse -Force edms_ai_assistant.egg-info

# 3. Пересоздать и синхронизировать
uv venv --python 3.13
uv sync --all-groups
```

> Если ошибка сохраняется — перезагрузите ПК и запустите `uv sync --all-groups` до открытия IDE.

### `Could not import module "edms_ai_assistant.main"`

Uvicorn запущен из неправильной директории или как скрипт вместо модуля.
Всегда запускайте из корня проекта:

```bash
uv run uvicorn edms_ai_assistant.main:app --reload
```

### Неверная версия Python

```bash
uv venv --python 3.13
uv sync --all-groups
```

### Ошибки setuptools после пересборки окружения

```bash
uv pip install --upgrade setuptools
uv sync
```

### Агент возвращает 503

Агент не инициализировался при старте. Проверьте логи:

```bash
# Смотреть логи uvicorn в реальном времени
uv run uvicorn edms_ai_assistant.main:app --reload --log-level debug
```

Частые причины: недоступен EDMS backend (`EDMS_BASE_URL`), неверный `OPENAI_API_KEY`, не запущен Redis или PostgreSQL.

### Файл не найден после загрузки

Проверьте что директория загрузок создана и доступна:

```bash
# Linux / macOS
ls -la /tmp/edms_ai_assistant_uploads/

# Windows PowerShell
Get-ChildItem $env:TEMP\edms_ai_assistant_uploads\
```

---

## License

MIT © Next