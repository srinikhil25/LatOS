"""Latos core domain layer — models, enums, exceptions.

No I/O, no DB, no UI dependencies. Imports from this package must work in any
Python environment with `latos` installed.
"""

from __future__ import annotations

from latos.core.enums import FileRole, Severity, Technique
from latos.core.exceptions import (
    AnalysisError,
    ConfigurationError,
    CorruptedFileError,
    FitConvergenceError,
    IngestionError,
    InsufficientDataError,
    LatosError,
    ParserError,
    PersistenceError,
    ProjectNotFoundError,
    SampleResolutionError,
    SchemaVersionError,
    UnsupportedFileError,
    ValidationError,
)
from latos.core.models import (
    FileRef,
    Measurement,
    Project,
    Sample,
    ValidationIssue,
    new_id,
    utc_now,
)

__all__ = [
    # Enums
    "FileRole",
    "Severity",
    "Technique",
    # Exceptions
    "AnalysisError",
    "ConfigurationError",
    "CorruptedFileError",
    "FitConvergenceError",
    "IngestionError",
    "InsufficientDataError",
    "LatosError",
    "ParserError",
    "PersistenceError",
    "ProjectNotFoundError",
    "SampleResolutionError",
    "SchemaVersionError",
    "UnsupportedFileError",
    "ValidationError",
    # Models
    "FileRef",
    "Measurement",
    "Project",
    "Sample",
    "ValidationIssue",
    "new_id",
    "utc_now",
]
