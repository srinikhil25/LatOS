"""Latos exception hierarchy.

All Latos-raised exceptions inherit from `LatosError`, so callers can use one
`except` clause to catch anything from the platform without swallowing other
errors.
"""

from __future__ import annotations


class LatosError(Exception):
    """Base class for every error raised by Latos.

    Use specific subclasses where possible; only raise this directly when no
    more specific class fits.
    """


# ─── Persistence ────────────────────────────────────────────────────
class PersistenceError(LatosError):
    """Failure in the persistence layer (SQLite, Parquet, file I/O)."""


class ProjectNotFoundError(PersistenceError):
    """A project was requested that does not exist."""


class SchemaVersionError(PersistenceError):
    """Project DB schema version is incompatible with this Latos version."""


# ─── Ingestion ──────────────────────────────────────────────────────
class IngestionError(LatosError):
    """Failure during file ingestion (crawling, parsing, labeling)."""


class ParserError(IngestionError):
    """Generic parser failure. Subclass for parser-specific errors."""


class UnsupportedFileError(ParserError):
    """No parser claims to handle the given file."""


class CorruptedFileError(ParserError):
    """File is recognized as a known type but cannot be parsed."""


class SampleResolutionError(IngestionError):
    """Failed to assign a file to a sample (Stage 2)."""


# ─── Analysis ───────────────────────────────────────────────────────
class AnalysisError(LatosError):
    """Failure in an analysis function."""


class FitConvergenceError(AnalysisError):
    """A fit (peak deconvolution, GP, etc.) did not converge."""


class InsufficientDataError(AnalysisError):
    """Analysis was requested but the input data is too sparse."""


# ─── Validation ─────────────────────────────────────────────────────
class ValidationError(LatosError):
    """A piece of data violated a domain invariant."""


# ─── Configuration ──────────────────────────────────────────────────
class ConfigurationError(LatosError):
    """The user's configuration or environment is invalid."""
