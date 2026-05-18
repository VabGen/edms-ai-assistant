# EDMS AI Assistant

> AI-powered assistant for electronic document management systems (EDMS/СЭД).  
> Automates document workflows, analysis, and task routing via LangGraph agents.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Contents

- [Overview](#overview)
- [Quick Start (Docker)](#quick-start-docker)
- [Deploying on a New Machine](#deploying-on-a-new-machine)
- [Configuration](#configuration)
- [Docker Commands](#docker-commands)
- [API](#api)
- [Ollama Setup](#ollama-setup)
- [Database Migrations](#database-migrations)
- [Development](#development)
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

**Stack:** FastAPI · LangGraph · PostgreSQL 18 · Redis 7 · Qdrant · Ollama · Docker

---

## Quick Start (Docker)

```bash
# 1. Clone
git clone https://github.com/your-org/edms-ai-assistant.git
cd edms-ai-assistant

# 2. Configure
cp .env.example .env
# edit .env with your values

# 3. Start
docker compose up -d --build

# 4. Check
curl http://localhost:8000/health
```

API docs: http://localhost:8000/docs

---

## Deploying on a New Machine

### Prerequisites

| Tool           | Version | Install                                                       |
|----------------|---------|---------------------------------------------------------------|
| Docker Desktop | latest  | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Ollama         | latest  | [ollama.com](https://ollama.com/download)                     |
| Git            | any     | [git-scm.com](https://git-scm.com/)                          |

### Step-by-step

**1. Install Ollama and pull the model**

```powershell
ollama pull gpt-oss:120b-cloud
```

**2. Configure Ollama to listen on all interfaces**

> Required so Docker containers can reach Ollama running on the host.

```powershell
# Run once, then restart Ollama
setx OLLAMA_HOST "0.0.0.0:11434"
```

Open a **new** terminal and start Ollama:
```powershell
ollama serve
```

Verify (must show `0.0.0.0:11434`):
```powershell
netstat -ano | findstr 11434
```

**3. Clone the repository**

```bash
git clone https://github.com/your-org/edms-ai-assistant.git
cd edms-ai-assistant
```

**4. Create `.env`**

Copy `.env.example` to `.env` and fill in the values (or copy a pre-filled `.env` from a teammate):

```bash
cp .env.example .env
```

**5. Start**

```bash
docker compose up -d --build
```

```bash
docker compose up -d --force-recreate app
```

**6. Verify**

```bash
docker compose ps
curl http://localhost:8000/health
docker exec edms-app curl -sf http://host.docker.internal:11434
```

---

## Configuration

All settings are loaded from `.env`. See `.env.example` for the full reference with comments.

Key variables:

| Variable               | Description                         | Default      |
|------------------------|-------------------------------------|--------------|
| `POSTGRES_PASSWORD`    | PostgreSQL password                 | *(required)* |
| `JWT_SECRET_KEY`       | JWT signing key (64-char hex)       | *(required)* |
| `LLM_GENERATIVE_URL`   | Ollama / OpenAI-compatible endpoint | *(required)* |
| `LLM_GENERATIVE_MODEL` | Model name                          | *(required)* |
| `EDMS_BASE_URL`        | EDMS backend URL                    | *(required)* |
| `API_PORT`             | Port to expose                      | `8000`       |
| `UVICORN_WORKERS`      | Number of uvicorn workers           | `1`          |
| `LOGGING_LEVEL`        | `DEBUG` / `INFO` / `WARNING`        | `INFO`       |

Generate a secure JWT key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Docker Commands

```bash
# First start / rebuild after code changes
docker compose up -d --build

# Update: pull latest code, then rebuild
git pull
docker compose up -d --build

# Follow application logs
docker compose logs -f app

# Check status of all services
docker compose ps

# Stop (data volumes preserved)
docker compose down

# Stop and wipe all data volumes
docker compose down -v

# Restart a single service
docker compose restart app

# Open a shell inside the app container
docker exec -it edms-app bash
```

### Dev mode (hot reload)

`docker-compose.override.yml` is auto-loaded — it mounts source code and enables `--reload`:

```bash
docker compose up -d --build
docker compose logs -f app
```

Infrastructure ports exposed locally in dev mode:
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- Qdrant: `localhost:6333`

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

## Ollama Setup

Ollama runs on the **host machine** (not in Docker). The app container reaches it via `host.docker.internal:11434`.

```powershell
# Set OLLAMA_HOST permanently (user-level, no admin required)
setx OLLAMA_HOST "0.0.0.0:11434"

# Kill current process if running
Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force

# Open a new terminal and start
ollama serve

# Verify — must show 0.0.0.0:11434
netstat -ano | findstr 11434

# Pull / list models
ollama pull gpt-oss:120b-cloud
ollama list
```

Test from inside container:
```bash
docker exec edms-app curl -sf http://host.docker.internal:11434
# Expected: Ollama is running
```

---

## Database Migrations

Migrations run **automatically** on container startup via `docker/entrypoint.sh`.

To run manually or create new migrations:

```bash
# Apply all pending migrations
docker exec edms-app alembic upgrade head

# Create a new migration (local dev)
uv run alembic revision --autogenerate -m "add_table_name"

# Other commands
docker exec edms-app alembic current     # current revision
docker exec edms-app alembic history     # migration history
docker exec edms-app alembic downgrade -1  # rollback one step
```

---

## Development

### Local setup

```bash
uv venv --python 3.13
uv sync --all-groups

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### Run locally (without Docker)

> Requires PostgreSQL 18, Redis, and Ollama running locally.

```bash
uv run uvicorn edms_ai_assistant.main:app --reload --reload-exclude ".venv"
```

### PyCharm Run Configuration

`Run → Edit Configurations → + → Python`

| Field                 | Value                                                        |
|-----------------------|--------------------------------------------------------------|
| **Module name**       | `uvicorn`                                                    |
| **Parameters**        | `edms_ai_assistant.main:app --reload --reload-exclude .venv` |
| **Working directory** | `<project root>`                                             |
| **Interpreter**       | `.venv\Scripts\python.exe`                                   |

> ⚠️ Use **Module name**, not Script path.

### Manage dependencies

```bash
uv add httpx                          # add runtime dependency
uv add --group dev pytest-mock        # add dev dependency
uv remove httpx                       # remove dependency
uv lock                               # update lock file
uv lock --upgrade-package langchain   # upgrade single package
```

> `uv.lock` must be committed — it ensures reproducible Docker builds.

### Tests

```bash
uv run pytest
uv run pytest --cov=edms_ai_assistant --cov-report=term-missing
uv run pytest -m "not integration"
```

### Code quality

```bash
uv run ruff check --fix .           # lint + autofix
uv run black edms_ai_assistant/     # format
uv run mypy .                       # type check
```

---

## Project Structure

```
edms-ai-assistant/
├── edms_ai_assistant/
│   ├── main.py               # FastAPI entry point
│   ├── config.py             # Settings (pydantic-settings)
│   ├── llm.py                # LLM factory
│   ├── agent/                # LangGraph orchestration
│   ├── api/routes/           # FastAPI routers
│   ├── clients/              # EDMS API HTTP clients
│   └── core/                 # Shared utilities, exceptions
├── migrations/               # Alembic migration scripts
├── docker/
│   └── entrypoint.sh         # Container startup script
├── Dockerfile                # Multi-stage production build
├── docker-compose.yml        # Production stack
├── docker-compose.override.yml  # Dev overrides (hot reload)
├── .env.example              # Environment variable template
├── pyproject.toml            # Project config + dependencies
├── uv.lock                   # Locked dependency versions (commit this!)
└── .dockerignore
```

---

## Troubleshooting

### Container `edms-postgres` is unhealthy

```bash
docker compose logs postgres
```

PostgreSQL 18 stores data in a version-specific subdirectory. If upgrading from an older image, wipe the old volume:
```bash
docker compose down
docker volume rm edms_pgdata
docker compose up -d
```

### App container exits immediately

```bash
docker compose logs app
```

Common causes: missing or incomplete `.env`, PostgreSQL not yet healthy, failed Alembic migration.

### Ollama not reachable from container

```bash
# Check Ollama is listening on 0.0.0.0 (not 127.0.0.1)
netstat -ano | findstr 11434

# If still 127.0.0.1 — set env var, then restart Ollama in a new terminal
setx OLLAMA_HOST "0.0.0.0:11434"
```

### Agent returns 503

```bash
docker compose logs app --tail=50
```

Common causes: `EDMS_BASE_URL` unreachable, wrong LLM endpoint, Ollama not running.

### Wipe everything and start fresh

```bash
docker compose down -v --remove-orphans
docker compose up -d --build
```

### `Access is denied` при `uv sync` (Windows)

```powershell
Get-Process python*, uvicorn* | Stop-Process -Force
Remove-Item -Recurse -Force .venv, edms_ai_assistant.egg-info
uv venv --python 3.13
uv sync --all-groups
```

---

## License

MIT © Next