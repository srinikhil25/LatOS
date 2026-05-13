"""add analysis_results table

Stage 3A: Analysis Layer foundation. Stores derived results from
analyzers (Tauc plot, peak fit, ZT calculation, ...). One row per
analyzer run; multiple results allowed per measurement.

`params` and `outputs` are JSON columns: SQLite stores them as TEXT
and SQLAlchemy round-trips through Python dicts. `issues_json` keeps
the analyzer's ValidationIssues in one column rather than a separate
table — they're a short, append-only list per result.

Revision ID: 0003_add_analysis_results
Revises: 0002_drop_files_sha256_unique
Create Date: 2026-05-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_add_analysis_results"
down_revision: str | Sequence[str] | None = "0002_drop_files_sha256_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the `analysis_results` table."""
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("measurement_id", sa.String(length=32), nullable=False),
        sa.Column("analyzer_name", sa.String(), nullable=False),
        sa.Column("analyzer_version", sa.String(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("outputs", sa.JSON(), nullable=False),
        sa.Column("derived_arrays_path", sa.String(), nullable=True),
        sa.Column("issues_json", sa.JSON(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["measurement_id"], ["measurements.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("analysis_results", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_analysis_results_measurement_id"),
            ["measurement_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_analysis_results_analyzer_name"),
            ["analyzer_name"],
            unique=False,
        )


def downgrade() -> None:
    """Drop the `analysis_results` table."""
    with op.batch_alter_table("analysis_results", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_analysis_results_analyzer_name"))
        batch_op.drop_index(batch_op.f("ix_analysis_results_measurement_id"))
    op.drop_table("analysis_results")
