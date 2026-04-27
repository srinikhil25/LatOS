"""Domain models — the shapes that flow through the entire Latos pipeline.

All models are frozen dataclasses (immutable). Mutations create new instances.
This makes it safe to share instances across UI threads, cache them freely,
and reason about state transitions explicitly.

No I/O, no DB, no UI dependencies live here. Anything imported from this
module must work in a Jupyter notebook with no Qt or SQLAlchemy installed.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from latos.core.enums import FileRole, Severity, Technique
from latos.core.exceptions import ValidationError

__all__ = [
    "FileRef",
    "Measurement",
    "Project",
    "Sample",
    "ValidationIssue",
    "new_id",
    "utc_now",
]


# ─── ID & timestamp helpers ─────────────────────────────────────────
def new_id() -> str:
    """Generate a fresh UUID4 hex string for use as a primary key.

    Returns:
        32-character lowercase hex string.
    """
    return uuid.uuid4().hex


def utc_now() -> datetime:
    """Timezone-aware UTC `now()`. Always use this instead of `datetime.now()`."""
    return datetime.now(UTC)


# UUID hex format used as primary key in our DB.
_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# SHA-256 produces 256 bits = 32 bytes = 64 hex characters.
_SHA256_HEX_LEN = 64


def _check_id(value: str, field_name: str) -> None:
    """Raise ValidationError if `value` is not a valid 32-char hex UUID."""
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValidationError(f"{field_name} must be a 32-character hex UUID, got: {value!r}")


# ─── FileRef ────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FileRef:
    """A reference to a file on disk that contributed to a measurement.

    Files are tracked by content hash so duplicates and modifications can be
    detected without re-parsing.

    Attributes:
        path: Absolute path on the user's filesystem.
        sha256: 64-character hex digest of file contents.
        size_bytes: File size in bytes (non-negative).
        role: How this file participates in the measurement.
        scanned_at: When the crawler last verified this file existed.
    """

    path: Path
    sha256: str
    size_bytes: int
    role: FileRole
    scanned_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ValidationError(
                f"FileRef.path must be a pathlib.Path, got {type(self.path).__name__}"
            )
        if not (
            isinstance(self.sha256, str)
            and len(self.sha256) == _SHA256_HEX_LEN
            and all(c in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValidationError(
                f"FileRef.sha256 must be a 64-char lowercase hex digest, got {self.sha256!r}"
            )
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValidationError(
                f"FileRef.size_bytes must be a non-negative int, got {self.size_bytes!r}"
            )
        if self.scanned_at.tzinfo is None:
            raise ValidationError("FileRef.scanned_at must be timezone-aware")


# ─── ValidationIssue ────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A problem detected during ingestion or analysis.

    Issues are surfaced to the user but do not abort processing. The user can
    acknowledge them ("I know my zT looks weird, accept anyway") to suppress
    repeat warnings.

    Attributes:
        field: Which field of the parsed data has the issue (e.g. "zT").
        severity: How serious the issue is.
        message: Human-readable explanation.
        detected_at: When the issue was first flagged.
        acknowledged: Whether the user has explicitly accepted it.
    """

    field: str
    severity: Severity
    message: str
    detected_at: datetime
    acknowledged: bool = False

    def __post_init__(self) -> None:
        if not self.field:
            raise ValidationError("ValidationIssue.field cannot be empty")
        if not self.message:
            raise ValidationError("ValidationIssue.message cannot be empty")
        if self.detected_at.tzinfo is None:
            raise ValidationError("ValidationIssue.detected_at must be timezone-aware")


# ─── Measurement ────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Measurement:
    """A single measurement performed on a sample.

    A "measurement" is the unit at which we cache parsed data. One physical
    measurement (one XRD scan, one XPS region, one UV-DRS sweep) produces one
    Measurement. A sample typically has multiple measurements (one per
    technique, sometimes more).

    Attributes:
        id: Primary key (32-char hex UUID).
        sample_id: Foreign key to the owning Sample.
        technique: Which technique this measurement was performed with.
        instrument: Free-text instrument identifier (e.g. "JEOL JEM-2100F").
        measured_at: When the experiment was performed (from file metadata).
        parsed_at: When Latos most recently parsed this measurement.
        parser_version: Version string of the parser used. Drives re-parsing.
        files: Files that contributed to this measurement.
        issues: Validation problems detected so far.
        parsed_data_path: Optional path to a Parquet file holding bulk arrays.
    """

    id: str
    sample_id: str
    technique: Technique
    instrument: str | None
    measured_at: datetime | None
    parsed_at: datetime
    parser_version: str
    files: tuple[FileRef, ...]
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    parsed_data_path: Path | None = None

    def __post_init__(self) -> None:
        _check_id(self.id, "Measurement.id")
        _check_id(self.sample_id, "Measurement.sample_id")
        if self.parsed_at.tzinfo is None:
            raise ValidationError("Measurement.parsed_at must be timezone-aware")
        if self.measured_at is not None and self.measured_at.tzinfo is None:
            raise ValidationError("Measurement.measured_at must be timezone-aware")
        if not self.parser_version:
            raise ValidationError("Measurement.parser_version cannot be empty")
        if not isinstance(self.files, tuple):
            raise ValidationError("Measurement.files must be a tuple (immutable)")
        if not isinstance(self.issues, tuple):
            raise ValidationError("Measurement.issues must be a tuple (immutable)")

    @property
    def has_errors(self) -> bool:
        """True if any issue has Severity.ERROR."""
        return any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """True if any issue has Severity.WARNING or higher."""
        return any(i.severity.order >= Severity.WARNING.order for i in self.issues)


# ─── Sample ─────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Sample:
    """A physical sample with one or more measurements.

    Attributes:
        id: Primary key (32-char hex UUID).
        project_id: Foreign key to the owning Project.
        canonical_name: The "real" name decided by the user / labeling agent.
        aliases: All name variants observed in the source data.
        measurements: Measurements performed on this sample.
    """

    id: str
    project_id: str
    canonical_name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    measurements: tuple[Measurement, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_id(self.id, "Sample.id")
        _check_id(self.project_id, "Sample.project_id")
        if not self.canonical_name or not self.canonical_name.strip():
            raise ValidationError("Sample.canonical_name cannot be empty or whitespace")
        if not isinstance(self.aliases, tuple):
            raise ValidationError("Sample.aliases must be a tuple (immutable)")
        if not isinstance(self.measurements, tuple):
            raise ValidationError("Sample.measurements must be a tuple (immutable)")
        # Aliases must be unique and non-empty
        seen: set[str] = set()
        for a in self.aliases:
            if not isinstance(a, str) or not a.strip():
                raise ValidationError(f"Sample.aliases contains empty/non-string: {a!r}")
            if a in seen:
                raise ValidationError(f"Sample.aliases contains duplicate: {a!r}")
            seen.add(a)
        # All measurements must point to this sample
        for m in self.measurements:
            if m.sample_id != self.id:
                raise ValidationError(
                    f"Measurement {m.id} has sample_id={m.sample_id} "
                    f"but is owned by Sample {self.id}"
                )

    @property
    def techniques(self) -> frozenset[Technique]:
        """Unique set of techniques present across all measurements."""
        return frozenset(m.technique for m in self.measurements)

    def has_technique(self, technique: Technique) -> bool:
        """Whether this sample has at least one measurement of the given technique."""
        return any(m.technique is technique for m in self.measurements)

    def measurements_for(self, technique: Technique) -> tuple[Measurement, ...]:
        """All measurements of the given technique on this sample."""
        return tuple(m for m in self.measurements if m.technique is technique)


# ─── Project ────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Project:
    """A collection of samples + measurements rooted at a folder on disk.

    A project is the top-level organizational unit. Each project corresponds
    to one folder of raw data and one SQLite database stored under
    `<root_path>/.latos/data.db`.

    Attributes:
        id: Primary key (32-char hex UUID).
        name: User-assigned display name (often defaults to folder basename).
        root_path: Absolute path to the folder of raw data.
        created_at: When the project was first created in Latos.
        schema_version: DB schema version used when this project was last opened.
        samples: Samples found in this project.
        unassigned_files: Files the crawler found but couldn't attribute to a sample.
    """

    id: str
    name: str
    root_path: Path
    created_at: datetime
    schema_version: int
    samples: tuple[Sample, ...] = field(default_factory=tuple)
    unassigned_files: tuple[FileRef, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_id(self.id, "Project.id")
        if not self.name or not self.name.strip():
            raise ValidationError("Project.name cannot be empty or whitespace")
        if not isinstance(self.root_path, Path):
            raise ValidationError(
                f"Project.root_path must be Path, got {type(self.root_path).__name__}"
            )
        if self.created_at.tzinfo is None:
            raise ValidationError("Project.created_at must be timezone-aware")
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValidationError(
                f"Project.schema_version must be a positive int, got {self.schema_version!r}"
            )
        if not isinstance(self.samples, tuple):
            raise ValidationError("Project.samples must be a tuple (immutable)")
        if not isinstance(self.unassigned_files, tuple):
            raise ValidationError("Project.unassigned_files must be a tuple (immutable)")
        # Every sample must point to this project
        for s in self.samples:
            if s.project_id != self.id:
                raise ValidationError(
                    f"Sample {s.id} has project_id={s.project_id} but is owned by Project {self.id}"
                )

    @property
    def total_files(self) -> int:
        """Files attributed to measurements + unassigned files."""
        attributed = sum(len(m.files) for s in self.samples for m in s.measurements)
        return attributed + len(self.unassigned_files)

    @property
    def total_measurements(self) -> int:
        """Number of measurements across all samples."""
        return sum(len(s.measurements) for s in self.samples)

    @property
    def all_techniques(self) -> frozenset[Technique]:
        """Unique set of techniques present in any measurement of any sample."""
        return frozenset(t for s in self.samples for t in s.techniques)

    def sample_by_id(self, sample_id: str) -> Sample | None:
        """Look up a sample by ID, or None if not found."""
        return next((s for s in self.samples if s.id == sample_id), None)

    def sample_by_name(self, canonical_name: str) -> Sample | None:
        """Look up a sample by canonical name, or None if not found."""
        return next((s for s in self.samples if s.canonical_name == canonical_name), None)

    def summary(self) -> dict[str, Any]:
        """Lightweight summary suitable for UI dashboards / JSON export."""
        return {
            "id": self.id,
            "name": self.name,
            "root_path": str(self.root_path),
            "created_at": self.created_at.isoformat(),
            "schema_version": self.schema_version,
            "n_samples": len(self.samples),
            "n_measurements": self.total_measurements,
            "n_files": self.total_files,
            "n_unassigned": len(self.unassigned_files),
            "techniques": sorted(t.value for t in self.all_techniques),
        }
