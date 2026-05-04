"""Create summarization_cache table.

This migration creates the new v2 cache table for summarization results.
It replaces the old `summarization_cache` table with a richer schema that:
  - Uses content-addressed cache keys (SHA-256 based)
  - Stores structured metadata (file_hash, mode, language, prompt_version)
  - Has proper expiry via expires_at column
  - Tracks token costs and model information
  - Includes updated_at for cache refresh tracking

Revision: 001
Created: 2025-06-01

alembic upgrade head
DELETE FROM alembic_version;
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Revision identifiers
revision: str = "001_summarization_cache"
down_revision: str | None = None  # First migration
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ── Create schema if not exists ────────────────────────────────────────
    op.execute("CREATE SCHEMA IF NOT EXISTS edms")

    # ── summarization_cache ─────────────────────────────────────────────
    op.create_table(
        "summarization_cache",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "cache_key",
            sa.String(64),
            nullable=False,
            comment="SHA-256 derived cache key: SHA256(file_hash::mode::language::prompt_version)[:32] with 'smz:' prefix",
        ),
        sa.Column(
            "file_hash",
            sa.String(64),
            nullable=False,
            comment="SHA-256 of raw file bytes — content-addressed, mode-independent",
        ),
        sa.Column(
            "mode",
            sa.String(32),
            nullable=False,
            comment="SummaryMode value: executive, abstractive, etc.",
        ),
        sa.Column(
            "language",
            sa.String(16),
            nullable=False,
            comment="BCP-47 output language tag",
        ),
        sa.Column(
            "prompt_version",
            sa.String(64),
            nullable=False,
            comment="Prompt registry version — bump to invalidate all entries",
        ),
        sa.Column(
            "cache_entry_json",
            sa.Text(),
            nullable=False,
            comment="JSON-serialized CacheEntry (contains result_json + metadata)",
        ),
        sa.Column(
            "input_tokens",
            sa.Integer(),
            nullable=False,
            default=0,
            comment="Total LLM input tokens consumed",
        ),
        sa.Column(
            "output_tokens",
            sa.Integer(),
            nullable=False,
            default=0,
            comment="Total LLM output tokens generated",
        ),
        sa.Column(
            "cost_usd",
            sa.Numeric(precision=10, scale=6),
            nullable=False,
            default=0.0,
            comment="Estimated cost in USD",
        ),
        sa.Column(
            "model_name",
            sa.String(128),
            nullable=False,
            comment="LLM model identifier used for generation",
        ),
        sa.Column(
            "expires_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            comment="Entry expiry time — filter in SELECT queries",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=True,
            comment="Set on cache refresh (ON CONFLICT DO UPDATE)",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="edms",
        comment="Summarization results cache — content-addressed, multi-level",
    )

    # ── Indexes ────────────────────────────────────────────────────────────

    # Primary lookup: exact cache key match
    op.create_index(
        "ix_summ_cache_cache_key",
        "summarization_cache",
        ["cache_key"],
        unique=True,
        schema="edms",
    )

    # File-level invalidation: delete all modes/languages for a file
    op.create_index(
        "ix_summ_cache_file_hash",
        "summarization_cache",
        ["file_hash"],
        schema="edms",
    )

    # Expiry cleanup: SELECT/DELETE expired entries efficiently
    op.create_index(
        "ix_summ_cache_expires_at",
        "summarization_cache",
        ["expires_at"],
        schema="edms",
    )

    # Compound: mode + language for analytics queries
    op.create_index(
        "ix_summ_cache_mode_lang",
        "summarization_cache",
        ["mode", "language"],
        schema="edms",
    )

    # ── PostgreSQL: auto-update updated_at trigger ─────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION edms.update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER trg_summ_cache_updated_at
        BEFORE UPDATE ON edms.summarization_cache
        FOR EACH ROW
        EXECUTE FUNCTION edms.update_updated_at_column();
    """)

    # ── Cleanup job: schedule via pg_cron or application-level ────────────
    # Comment explains intent — actual scheduling done at app level
    op.execute("""
        COMMENT ON TABLE edms.summarization_cache IS
        'Expired rows are cleaned up by SummarizationCacheCleanupJob (runs daily).
         Query: DELETE FROM edms.summarization_cache WHERE expires_at < NOW();';
    """)


def downgrade() -> None:
    # Drop trigger
    op.execute(
        "DROP TRIGGER IF EXISTS trg_summ_cache_updated_at "
        "ON edms.summarization_cache;"
    )
    # Drop function
    op.execute(
        "DROP FUNCTION IF EXISTS edms.update_updated_at_column();"
    )
    # Drop indexes
    for idx in [
        "ix_summ_cache_mode_lang",
        "ix_summ_cache_expires_at",
        "ix_summ_cache_file_hash",
        "ix_summ_cache_cache_key",
    ]:
        op.drop_index(idx, table_name="summarization_cache", schema="edms")
    # Drop table
    op.drop_table("summarization_cache", schema="edms")

