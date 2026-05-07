# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# EDMS AI Assistant — Production Dockerfile
#
# Stages:
#   builder  — installs Python dependencies into an isolated venv
#   runtime  — minimal image with only runtime artifacts
#
# Build-time ARGs:
#   PYTHON_VERSION   — Python version tag (default: 3.13-slim)
#   TORCH_VARIANT    — "cpu" installs lighter CPU-only torch wheel
#                      set to "" to use the default wheel from pyproject.toml
#
# NOTE: For fully reproducible builds commit uv.lock to the repository
#       and uncomment the `COPY uv.lock` + `uv sync --frozen` lines below.
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.13-slim

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — builder
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:${PYTHON_VERSION} AS builder

ARG TORCH_VARIANT=cpu

# Build-time system deps (compilers, headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Pull uv binary directly from the official image (no pip overhead)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Create virtual environment
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the manifest first — layer is cached until deps change
COPY pyproject.toml README.md ./
# COPY uv.lock ./   # Uncomment after committing uv.lock

# Install PyTorch CPU-only wheel first (avoids pulling 2 GB CUDA wheel)
# To use CUDA, set TORCH_VARIANT="" and pin the correct index URL
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "${TORCH_VARIANT}" ]; then \
        uv pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install all remaining project dependencies
# RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev  # with lockfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache .

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — runtime
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:${PYTHON_VERSION} AS runtime

# Runtime system deps: OCR, PDF rendering, PostgreSQL client
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin -c "EDMS app user" appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source (owned by appuser)
COPY --chown=appuser:appuser edms_ai_assistant/ ./edms_ai_assistant/
COPY --chown=appuser:appuser migrations/ ./migrations/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser docker/entrypoint.sh ./entrypoint.sh

# Prepare upload directory
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
    CMD curl -sf http://localhost:${API_PORT:-8000}/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]