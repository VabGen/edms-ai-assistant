ARG PYTHON_VERSION=3.13-slim

# ─────────────────────────────────────────────────────────────────────────────
# Build Browser Extension (WXT + React)
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS extension-builder

WORKDIR /extension

COPY edms-plugin/package.json edms-plugin/package-lock.json* ./

RUN npm install

COPY edms-plugin/ ./

RUN npm run zip

RUN zip_file=$(find . -name "*.zip" -type f | head -n 1) && \
    if [ -n "$zip_file" ]; then \
        echo "Found zip archive: $zip_file" && \
        mv "$zip_file" /tmp/extension.zip; \
    else \
        echo "ERROR: No .zip file found after npm run zip!" && exit 1; \
    fi


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Python Builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION} AS builder

ARG TORCH_VARIANT=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "${TORCH_VARIANT}" ]; then \
        uv pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache .

# -- Runtime ---------------------------------------------------------------

FROM python:${PYTHON_VERSION} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libreoffice-writer \
    libreoffice-common \
    libgl1 \
    libglib2.0-0 \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -- Verify Tesseract & set env vars ---------------------------------------

RUN tesseract --version && \
    tesseract --list-langs && \
    which soffice

ENV TESSERACT_CMD=/usr/bin/tesseract \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin -c "EDMS app user" appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Создаем папку и копируем собранный ZIP-архив плагина
RUN mkdir -p /app/static/plugin
COPY --from=extension-builder /tmp/extension.zip /app/static/plugin/extension.zip

COPY --chown=appuser:appuser edms_ai_assistant/ ./edms_ai_assistant/
COPY --chown=appuser:appuser migrations/ ./migrations/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser docker/entrypoint.sh ./entrypoint.sh

RUN mkdir -p /app/uploads && chown -R appuser:appuser /app/uploads /app/static
RUN chmod +x /app/entrypoint.sh

USER appuser

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]