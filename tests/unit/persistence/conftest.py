"""Shared fixtures for persistence tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from latos.core.enums import FileRole, Severity, Technique
from latos.core.models import (
    FileRef,
    Measurement,
    Project,
    Sample,
    ValidationIssue,
    new_id,
    utc_now,
)
from latos.persistence.db import (
    create_memory_engine,
    init_schema,
    make_session_factory,
)
from latos.persistence.repository import ProjectRepository


@pytest.fixture
def engine() -> Iterator[Engine]:
    """Fresh in-memory SQLite engine with schema applied."""
    eng = create_memory_engine()
    init_schema(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory bound to the in-memory engine."""
    return make_session_factory(engine)


@pytest.fixture
def repo(session_factory: sessionmaker[Session]) -> ProjectRepository:
    """ProjectRepository bound to a fresh in-memory DB."""
    return ProjectRepository(session_factory)


# ─── Builder helpers used by multiple test files ────────────────────
def make_file_ref(*, sha256: str | None = None, role: FileRole = FileRole.RAW) -> FileRef:
    return FileRef(
        path=Path("/data/sample.xy"),
        sha256=sha256 if sha256 is not None else "a" * 64,
        size_bytes=1024,
        role=role,
        scanned_at=utc_now(),
    )


def make_issue(*, severity: Severity = Severity.WARNING) -> ValidationIssue:
    return ValidationIssue(
        field="zT",
        severity=severity,
        message="zT out of range",
        detected_at=utc_now(),
    )


def make_measurement(
    sample_id: str,
    *,
    technique: Technique = Technique.XRD,
    instrument: str | None = "JEOL",
    file_sha: str | None = None,
    issues: tuple[ValidationIssue, ...] = (),
) -> Measurement:
    # Slightly stagger times so equality checks don't collide.
    now = utc_now()
    return Measurement(
        id=new_id(),
        sample_id=sample_id,
        technique=technique,
        instrument=instrument,
        measured_at=now - timedelta(days=1),
        parsed_at=now,
        parser_version="1.0.0",
        files=(make_file_ref(sha256=file_sha),),
        issues=issues,
    )


def make_sample(
    project_id: str,
    *,
    name: str = "CS",
    aliases: tuple[str, ...] = (),
    measurements: tuple[Measurement, ...] = (),
) -> Sample:
    return Sample(
        id=new_id(),
        project_id=project_id,
        canonical_name=name,
        aliases=aliases,
        measurements=measurements,
    )


def make_project(
    *,
    samples: tuple[Sample, ...] = (),
    unassigned: tuple[FileRef, ...] = (),
) -> Project:
    return Project(
        id=new_id(),
        name="Demo Project",
        root_path=Path("/data/demo"),
        created_at=utc_now(),
        schema_version=1,
        samples=samples,
        unassigned_files=unassigned,
    )
