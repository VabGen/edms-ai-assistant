ARG PYTHON_VERSION=3.13-slim

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

COPY --chown=appuser:appuser edms_ai_assistant/ ./edms_ai_assistant/
COPY --chown=appuser:appuser migrations/ ./migrations/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser docker/entrypoint.sh ./entrypoint.sh

RUN mkdir -p /app/uploads && chown -R appuser:appuser /app/uploads
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