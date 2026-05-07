#!/bin/bash
# docker/entrypoint.sh — EDMS AI Assistant container entrypoint
# Runs Alembic migrations (idempotent), then starts uvicorn.
set -euo pipefail

log() {
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [entrypoint] $*"
}

# ── Wait for PostgreSQL ────────────────────────────────────────────────────────
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
MAX_RETRIES=30
RETRY_INTERVAL=2

log "Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}..."
for i in $(seq 1 $MAX_RETRIES); do
    if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -q 2>/dev/null; then
        log "PostgreSQL is ready."
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        log "ERROR: PostgreSQL did not become ready in time. Aborting."
        exit 1
    fi
    log "  attempt $i/$MAX_RETRIES — retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
done

# ── Run Alembic migrations ─────────────────────────────────────────────────────
log "Running Alembic migrations..."
alembic upgrade head
log "Migrations complete."

# ── Start application ──────────────────────────────────────────────────────────
WORKERS="${UVICORN_WORKERS:-1}"
PORT="${API_PORT:-8000}"
LOG_LEVEL=$(echo "${LOGGING_LEVEL:-INFO}" | tr '[:upper:]' '[:lower:]')

log "Starting uvicorn: workers=${WORKERS} port=${PORT} log_level=${LOG_LEVEL}"
exec uvicorn edms_ai_assistant.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}" \
    --proxy-headers \
    --forwarded-allow-ips '*'
