# EDMS AI Assistant

> AI-powered assistant for electronic document management systems (EDMS/СЭД).  
> Automates document workflows, analysis, and task routing via LangGraph agents.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-0.5+-blueviolet)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [Development](#development)
- [Code Quality](#code-quality)
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

---

## Requirements

| Tool   | Version   | Install                                                                      |
|--------|-----------|------------------------------------------------------------------------------|
| Python | 3.13+     | [python.org](https://www.python.org/downloads/)                              |
| uv     | 0.5.0+    | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |

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
```

### Dependency groups

| Command                 | Installs             |
|-------------------------|----------------------|
| `uv sync --all-groups`  | Runtime + dev + lint |
| `uv sync --group dev`   | Runtime + tests      |
| `uv sync --group lint`  | Runtime + linters    |
| `uv sync --no-dev`      | Runtime only         |

### Activate virtual environment

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## Running

### Development

```bash
uv run uvicorn edms_ai_assistant.main:app --reload --reload-exclude ".venv"
```

### Production

```bash
uv run uvicorn edms_ai_assistant.main:app --host 0.0.0.0 --port 8000
```

### PyCharm Run Configuration

`Run → Edit Configurations → + → Python`

| Field                 | Value                                                          |
|-----------------------|----------------------------------------------------------------|
| **Module name**       | `uvicorn`                                                      |
| **Parameters**        | `edms_ai_assistant.main:app --reload --reload-exclude .venv`   |
| **Working directory** | `D:\project\edms-ai-assistant`                                 |
| **Interpreter**       | `.venv\Scripts\python.exe`                                     |

> ⚠️ Use **Module name**, not Script path — otherwise the package won't resolve correctly.

API docs: `http://localhost:8000/docs`

---

## Development

### Update lock file

```bash
# After editing pyproject.toml
uv lock

# Upgrade a specific package
uv lock --upgrade-package langchain
```

> `uv.lock` must be committed to the repository — it ensures reproducible builds
> across all environments and CI/CD pipelines (equivalent to `package-lock.json`).

### Run tests

```bash
uv run pytest

# With coverage report
uv run pytest --cov=edms_ai_assistant --cov-report=term-missing
```

---

## Code Quality

### Format

```bash
uv run black edms_ai_assistant/
```

### Lint

```bash
# Check only
uv run ruff check .

# Check and auto-fix
uv run ruff check --fix .
```

### Type check

```bash
uv run mypy .
```

### Run all checks

```bash
uv run black edms_ai_assistant/ && uv run ruff check --fix . && uv run mypy .
```

### AI code review

```bash
coderabbit review
```

---

## Project Structure

```
edms-ai-assistant/
├── edms_ai_assistant/
│   ├── main.py               # FastAPI entry point
│   ├── agent.py              # LangGraph orchestration
│   ├── config.py             # Settings (pydantic-settings)
│   ├── model.py              # Pydantic models
│   ├── llm.py                # LLM factory
│   ├── tools/                # LangChain tools
│   ├── clients/              # EDMS API clients
│   ├── services/             # Business logic
│   ├── generated/            # Auto-generated OpenAPI models
│   └── utils/                # Shared utilities
├── tests/
├── pyproject.toml            # Project config + dependency groups
├── uv.lock                   # Locked dependency versions (commit this)
├── .gitignore
└── README.md
```

---

## Troubleshooting

### `Access is denied` on `uv sync` (Windows)

The `.venv` directory is locked by a running process (IDE, terminal, Python).

```powershell
# 1. Kill all Python / uvicorn processes
Get-Process python* | Stop-Process -Force
Get-Process uvicorn* | Stop-Process -Force

# 2. Remove locked environment and build artifacts
Remove-Item -Recurse -Force .venv
Remove-Item -Recurse -Force edms_ai_assistant.egg-info

# 3. Recreate and sync
uv venv --python 3.13
uv sync --all-groups
```

> If the error persists — reboot and run `uv sync --all-groups` before opening any IDE.

### `Could not import module "edms_ai_assistant.main"`

Uvicorn is launched from the wrong directory or as a script instead of a module.
Always run from the project root:

```bash
uv run uvicorn edms_ai_assistant.main:app --reload
```

### Wrong Python version picked by `uv`

```bash
uv venv --python 3.13
uv sync --all-groups
```

### Setuptools errors after environment rebuild

```bash
uv pip install --upgrade setuptools
uv sync
```

---

## License

MIT © Next