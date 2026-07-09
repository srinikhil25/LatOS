"""add measurement features JSON column

Stage 4B: feature-extraction layer. Parsers compute scalar features
(Hall carrier concentration + mobility, EDS beam kV, …) that were
previously dropped after ingestion. This column persists them per
measurement so the UI and optimizer can use them. Existing rows default
to an empty object.

Revision ID: 0005_add_measurement_features
Revises: 0004_add_project_review_status
Create Date: 2026-06-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_add_measurement_features"
down_revision: str | Sequence[str] | None = "0004_add_project_review_status"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the `features` JSON column to `measurements`."""
    with op.batch_alter_table("measurements", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "features",
                sa.JSON(),
                nullable=False,
                server_default="{}",
            ),
        )


def downgrade() -> None:
    """Drop the `features` column."""
    with op.batch_alter_table("measurements", schema=None) as batch_op:
        batch_op.drop_column("features")
