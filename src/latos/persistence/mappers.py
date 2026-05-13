"""Conversion between domain dataclasses and SQLAlchemy ORM rows.

Domain code (UI, analysis, ingestion) only ever sees `latos.core.models`
types. Persistence code only ever sees `latos.persistence.schema` rows.
This module bridges the two — and is the ONLY module allowed to bridge them.
"""

from __future__ import annotations

from pathlib import Path

from latos.core.enums import FileRole, Severity, Technique
from latos.core.models import (
    FileRef,
    Measurement,
    Project,
    Sample,
    ValidationIssue,
)
from latos.persistence.schema import (
    FileRow,
    MeasurementRow,
    ProjectRow,
    SampleRow,
    ValidationIssueRow,
)


# ─── Domain → ORM ───────────────────────────────────────────────────
def project_to_row(project: Project) -> ProjectRow:
    """Convert a Project dataclass into a fresh ProjectRow.

    Does NOT recursively materialize samples/measurements — call the
    sample/measurement converters separately and attach.
    """
    return ProjectRow(
        id=project.id,
        name=project.name,
        root_path=str(project.root_path),
        created_at=project.created_at,
        schema_version=project.schema_version,
    )


def sample_to_row(sample: Sample) -> SampleRow:
    """Convert a Sample dataclass into a fresh SampleRow."""
    return SampleRow(
        id=sample.id,
        project_id=sample.project_id,
        canonical_name=sample.canonical_name,
        aliases=list(sample.aliases),
    )


def measurement_to_row(measurement: Measurement) -> MeasurementRow:
    """Convert a Measurement dataclass into a fresh MeasurementRow."""
    return MeasurementRow(
        id=measurement.id,
        sample_id=measurement.sample_id,
        technique=measurement.technique.value,
        instrument=measurement.instrument,
        measured_at=measurement.measured_at,
        parsed_at=measurement.parsed_at,
        parser_version=measurement.parser_version,
        parsed_data_path=(
            str(measurement.parsed_data_path) if measurement.parsed_data_path else None
        ),
    )


def file_to_row(file_ref: FileRef, project_id: str, measurement_id: str | None) -> FileRow:
    """Convert a FileRef dataclass into a fresh FileRow.

    Args:
        file_ref: The domain object.
        project_id: Required — files always belong to a project.
        measurement_id: None for unassigned files.

    The row ID is derived from (measurement_id, sha256) - or from
    (project_id, sha256) for unassigned files. Stage 1 used
    `sha256[:32]` alone, assuming one file = one row, but Stage 1F's
    multi-sheet support means several measurements can reference one
    file (one row per (measurement, file) pair). Mixing the
    measurement_id into the derivation keeps the row ID stable across
    re-saves of the same (measurement, file) pair while avoiding
    collisions between siblings.
    """
    import hashlib  # noqa: PLC0415 — only needed inside this hot path

    discriminator = (measurement_id or project_id).encode("utf-8")
    seed = discriminator + b":" + file_ref.sha256.encode("utf-8")
    row_id = hashlib.sha256(seed).hexdigest()[:32]
    return FileRow(
        id=row_id,
        measurement_id=measurement_id,
        project_id=project_id,
        path=str(file_ref.path),
        sha256=file_ref.sha256,
        size_bytes=file_ref.size_bytes,
        role=file_ref.role.value,
        scanned_at=file_ref.scanned_at,
    )


def issue_to_row(
    issue: ValidationIssue, *, issue_id: str, measurement_id: str
) -> ValidationIssueRow:
    """Convert a ValidationIssue dataclass into a fresh ValidationIssueRow."""
    return ValidationIssueRow(
        id=issue_id,
        measurement_id=measurement_id,
        field=issue.field,
        severity=issue.severity.value,
        message=issue.message,
        detected_at=issue.detected_at,
        acknowledged=issue.acknowledged,
    )


# ─── ORM → Domain ───────────────────────────────────────────────────
def row_to_file_ref(row: FileRow) -> FileRef:
    """Convert a FileRow into a FileRef domain dataclass."""
    return FileRef(
        path=Path(row.path),
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        role=FileRole(row.role),
        scanned_at=row.scanned_at,
    )


def row_to_issue(row: ValidationIssueRow) -> ValidationIssue:
    """Convert a ValidationIssueRow into a ValidationIssue domain dataclass."""
    return ValidationIssue(
        field=row.field,
        severity=Severity(row.severity),
        message=row.message,
        detected_at=row.detected_at,
        acknowledged=row.acknowledged,
    )


def row_to_measurement(row: MeasurementRow) -> Measurement:
    """Convert a MeasurementRow (with files + issues loaded) into a Measurement."""
    return Measurement(
        id=row.id,
        sample_id=row.sample_id,
        technique=Technique(row.technique),
        instrument=row.instrument,
        measured_at=row.measured_at,
        parsed_at=row.parsed_at,
        parser_version=row.parser_version,
        files=tuple(row_to_file_ref(f) for f in row.files),
        issues=tuple(row_to_issue(i) for i in row.issues),
        parsed_data_path=Path(row.parsed_data_path) if row.parsed_data_path else None,
    )


def row_to_sample(row: SampleRow) -> Sample:
    """Convert a SampleRow (with measurements loaded) into a Sample."""
    return Sample(
        id=row.id,
        project_id=row.project_id,
        canonical_name=row.canonical_name,
        aliases=tuple(row.aliases),
        measurements=tuple(row_to_measurement(m) for m in row.measurements),
    )


def row_to_project(row: ProjectRow) -> Project:
    """Convert a ProjectRow (with samples + unassigned files loaded) into a Project."""
    return Project(
        id=row.id,
        name=row.name,
        root_path=Path(row.root_path),
        created_at=row.created_at,
        schema_version=row.schema_version,
        samples=tuple(row_to_sample(s) for s in row.samples),
        unassigned_files=tuple(row_to_file_ref(f) for f in row.unassigned_files),
    )
