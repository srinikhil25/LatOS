"""add project review_status + confirmed_at

RB8: human-verification gate. After ingestion a project sits in
`needs_review`; the downstream pipeline (analysis, correlation,
optimization) requires `confirmed`. Existing rows default to
`needs_review` via the column server_default.

Revision ID: 0004_add_project_review_status
Revises: 0003_add_analysis_results
Create Date: 2026-06-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_add_project_review_status"
down_revision: str | Sequence[str] | None = "0003_add_analysis_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add `review_status` and `confirmed_at` to `projects`."""
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "review_status",
                sa.String(),
                nullable=False,
                server_default="needs_review",
            ),
        )
        batch_op.add_column(
            sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    """Drop the review columns."""
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_column("confirmed_at")
        batch_op.drop_column("review_status")
