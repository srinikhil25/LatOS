"""drop UNIQUE on files.sha256

Stage 1F: multi-sheet workbook support. A single file produces one
measurement per non-empty sheet, all sharing the same sha256. The
old UNIQUE constraint blocked persistence after the first sibling
measurement.

The non-unique INDEX on sha256 is preserved (it powers the parse
cache lookup), only the UNIQUE constraint is dropped.

Revision ID: 0002_drop_files_sha256_unique
Revises: 0001_initial
Create Date: 2026-05-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_drop_files_sha256_unique"
down_revision: str | Sequence[str] | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the UNIQUE(sha256) constraint on `files`."""
    # SQLite doesn't support `ALTER TABLE DROP CONSTRAINT`. Use
    # `batch_alter_table` which transparently rewrites the table.
    with op.batch_alter_table("files", schema=None) as batch_op:
        batch_op.drop_constraint("uq_file_sha256", type_="unique")


def downgrade() -> None:
    """Re-add the UNIQUE(sha256) constraint."""
    with op.batch_alter_table("files", schema=None) as batch_op:
        batch_op.create_unique_constraint("uq_file_sha256", ["sha256"])
