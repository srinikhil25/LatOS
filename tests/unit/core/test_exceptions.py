"""Tests for `latos.core.exceptions`.

Verify the hierarchy: every Latos exception inherits from `LatosError`, and
specialized exceptions have the right parent chain.
"""

from __future__ import annotations

import pytest

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

ALL_EXCEPTIONS: list[type[LatosError]] = [
    LatosError,
    PersistenceError,
    ProjectNotFoundError,
    SchemaVersionError,
    IngestionError,
    ParserError,
    UnsupportedFileError,
    CorruptedFileError,
    SampleResolutionError,
    AnalysisError,
    FitConvergenceError,
    InsufficientDataError,
    ValidationError,
    ConfigurationError,
]


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
def test_all_inherit_from_latos_error(exc_cls: type[Exception]) -> None:
    assert issubclass(exc_cls, LatosError)


@pytest.mark.parametrize(
    ("exc_cls", "parent"),
    [
        (ProjectNotFoundError, PersistenceError),
        (SchemaVersionError, PersistenceError),
        (ParserError, IngestionError),
        (UnsupportedFileError, ParserError),
        (CorruptedFileError, ParserError),
        (SampleResolutionError, IngestionError),
        (FitConvergenceError, AnalysisError),
        (InsufficientDataError, AnalysisError),
    ],
)
def test_subclass_relationships(exc_cls: type[Exception], parent: type[Exception]) -> None:
    assert issubclass(exc_cls, parent)


def test_can_be_caught_as_latos_error() -> None:
    """One except clause catches everything from Latos."""
    with pytest.raises(LatosError):
        raise UnsupportedFileError("nope.foo")


def test_can_be_caught_as_specific_type() -> None:
    """Specific catches still work."""
    with pytest.raises(UnsupportedFileError):
        raise UnsupportedFileError("nope.foo")


def test_message_passes_through() -> None:
    err = ValidationError("zT out of range")
    assert str(err) == "zT out of range"
