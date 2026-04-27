"""SQLAlchemy ORM schema — mirrors domain models from `latos.core.models`.

This module defines table structure ONLY. It does not contain conversion
logic between domain dataclasses and ORM rows — that lives in `mappers.py`.

Schema versioning: when this file changes in a way that affects on-disk
representation, an Alembic migration must be added under
`migrations/versions/` and `LATEST_SCHEMA_VERSION` bumped.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Dialect,
    ForeignKey,
    Integer,
    String,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Bumped on any schema change. Stored on every Project row so migrations
# can detect mismatches between code and on-disk data.
LATEST_SCHEMA_VERSION = 1


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware DateTime that always round-trips as UTC.

    SQLite has no native timezone storage. SQLAlchemy's `UtcDateTime()`
    stores the value but returns a naive `datetime` on load — silently dropping
    the tzinfo. That breaks our domain invariant that all timestamps are
    timezone-aware.

    This type:
    - On bind: converts any tz-aware input to UTC before storage.
      Rejects naive datetimes (loud failure beats silent data corruption).
    - On result: re-attaches `UTC` tzinfo to whatever SQLite returns.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        """Normalize Python → SQL: tz-aware → UTC; naive raises."""
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("UtcDateTime requires a timezone-aware datetime; got naive.")
        return value.astimezone(UTC)

    def process_result_value(
        self,
        value: Any,
        dialect: Dialect,
    ) -> datetime | None:
        """Normalize SQL → Python: re-attach UTC tzinfo, never naive."""
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"UtcDateTime got non-datetime from DB: {type(value)}")
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    """Base class for all Latos ORM tables."""


class ProjectRow(Base):
    """One row per project. Lives at `<project_root>/.latos/data.db`."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    root_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)

    samples: Mapped[list[SampleRow]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    unassigned_files: Mapped[list[FileRow]] = relationship(
        primaryjoin="and_(FileRow.project_id == ProjectRow.id, FileRow.measurement_id.is_(None))",
        cascade="all, delete-orphan",
        lazy="selectin",
        viewonly=False,
        overlaps="files,measurement",
    )


class SampleRow(Base):
    """A physical sample within a project."""

    __tablename__ = "samples"
    __table_args__ = (
        UniqueConstraint("project_id", "canonical_name", name="uq_sample_canonical_name"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    canonical_name: Mapped[str] = mapped_column(String, nullable=False)
    aliases: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    project: Mapped[ProjectRow] = relationship(back_populates="samples")
    measurements: Mapped[list[MeasurementRow]] = relationship(
        back_populates="sample",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class MeasurementRow(Base):
    """One measurement (one technique on one sample)."""

    __tablename__ = "measurements"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    sample_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("samples.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    technique: Mapped[str] = mapped_column(String, nullable=False, index=True)
    instrument: Mapped[str | None] = mapped_column(String, nullable=True)
    measured_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    parsed_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    parser_version: Mapped[str] = mapped_column(String, nullable=False)
    parsed_data_path: Mapped[str | None] = mapped_column(String, nullable=True)

    sample: Mapped[SampleRow] = relationship(back_populates="measurements")
    files: Mapped[list[FileRow]] = relationship(
        back_populates="measurement",
        cascade="all, delete-orphan",
        lazy="selectin",
        primaryjoin="MeasurementRow.id == FileRow.measurement_id",
        overlaps="unassigned_files",
    )
    issues: Mapped[list[ValidationIssueRow]] = relationship(
        back_populates="measurement",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class FileRow(Base):
    """A file on disk that contributed to a measurement (or is unassigned).

    `sha256` is unique across the whole project — same content = same file row.
    Either `measurement_id` is set (file belongs to a measurement) or
    `project_id` is set with `measurement_id` NULL (unassigned).
    """

    __tablename__ = "files"
    __table_args__ = (UniqueConstraint("sha256", name="uq_file_sha256"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    measurement_id: Mapped[str | None] = mapped_column(
        String(32),
        ForeignKey("measurements.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    path: Mapped[str] = mapped_column(String, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    scanned_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)

    measurement: Mapped[MeasurementRow | None] = relationship(
        back_populates="files",
        primaryjoin="MeasurementRow.id == FileRow.measurement_id",
    )


class ValidationIssueRow(Base):
    """A validation problem flagged on a measurement."""

    __tablename__ = "validation_issues"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    measurement_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("measurements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    field: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False, index=True)
    message: Mapped[str] = mapped_column(String, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    measurement: Mapped[MeasurementRow] = relationship(back_populates="issues")
