"""Repository pattern for Latos persistence.

`ProjectRepository` is the only public API for reading and writing projects.
Domain code never touches SQLAlchemy directly.

Design choices:
- All repository methods take/return domain dataclasses, not ORM rows.
- `save()` is a full upsert: deletes the old project tree and writes fresh.
  We trade write performance for correctness simplicity. Stage 1B can
  afford this; we'll optimize in Stage 8 if needed.
- `load()` eagerly loads samples + measurements + files + issues
  (relationships use `lazy="selectin"`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from latos.core.exceptions import ProjectNotFoundError
from latos.core.models import Project, new_id
from latos.persistence.db import session_scope
from latos.persistence.mappers import (
    analysis_result_to_row,
    file_to_row,
    issue_to_row,
    measurement_to_row,
    project_to_row,
    row_to_project,
    sample_to_row,
)
from latos.persistence.schema import ProjectRow


@dataclass(frozen=True, slots=True)
class ProjectSummary:
    """Lightweight project listing entry — no samples loaded.

    Used by the "Recent Projects" UI.
    """

    id: str
    name: str
    root_path: Path
    created_at: datetime
    schema_version: int


class ProjectRepository:
    """Read/write `Project` aggregates.

    A repository is bound to one engine (one project DB). Multiple
    repositories on different engines coexist — each represents a different
    project on disk.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    # ─── Read ────────────────────────────────────────────────────
    def list_projects(self) -> list[ProjectSummary]:
        """Return summaries of every project in this DB.

        For per-project DBs (the standard case), this returns 0 or 1.
        Useful for the user-level recent-projects index.
        """
        with session_scope(self._sessions) as session:
            rows = session.execute(select(ProjectRow)).scalars().all()
            return [
                ProjectSummary(
                    id=r.id,
                    name=r.name,
                    root_path=Path(r.root_path),
                    created_at=r.created_at,
                    schema_version=r.schema_version,
                )
                for r in rows
            ]

    def load(self, project_id: str) -> Project:
        """Load a complete project aggregate by ID.

        Raises:
            ProjectNotFoundError: If no project with that ID exists.
        """
        with session_scope(self._sessions) as session:
            row = session.get(ProjectRow, project_id)
            if row is None:
                raise ProjectNotFoundError(f"No project with id={project_id!r}")
            return row_to_project(row)

    def load_first(self) -> Project | None:
        """Load the only project in this DB (typical for per-project DBs).

        Returns None if there are no projects.
        """
        with session_scope(self._sessions) as session:
            row = session.execute(select(ProjectRow).limit(1)).scalar_one_or_none()
            return row_to_project(row) if row else None

    def exists(self, project_id: str) -> bool:
        """True if a project with the given ID exists."""
        with session_scope(self._sessions) as session:
            return session.get(ProjectRow, project_id) is not None

    # ─── Write ───────────────────────────────────────────────────
    def save(self, project: Project) -> None:
        """Persist a complete project aggregate.

        If a project with this ID already exists, it is fully replaced
        (all samples, measurements, files, issues are dropped and rewritten).
        Cascading FK deletes handle the cleanup.
        """
        with session_scope(self._sessions) as session:
            # Drop any existing aggregate. Relies on cascade ON DELETE.
            existing = session.get(ProjectRow, project.id)
            if existing is not None:
                session.delete(existing)
                session.flush()  # ensure deletes commit before re-insert

            project_row = project_to_row(project)
            session.add(project_row)
            session.flush()

            # Add samples, measurements, files (assigned), issues
            for sample in project.samples:
                sample_row = sample_to_row(sample)
                session.add(sample_row)
                session.flush()

                for measurement in sample.measurements:
                    measurement_row = measurement_to_row(measurement)
                    session.add(measurement_row)
                    session.flush()

                    for file_ref in measurement.files:
                        file_row = file_to_row(
                            file_ref,
                            project_id=project.id,
                            measurement_id=measurement.id,
                        )
                        session.add(file_row)

                    for issue in measurement.issues:
                        issue_row = issue_to_row(
                            issue,
                            issue_id=new_id(),
                            measurement_id=measurement.id,
                        )
                        session.add(issue_row)

                    for analysis_result in measurement.analysis_results:
                        analysis_row = analysis_result_to_row(analysis_result)
                        session.add(analysis_row)

            # Add unassigned files
            for file_ref in project.unassigned_files:
                file_row = file_to_row(
                    file_ref,
                    project_id=project.id,
                    measurement_id=None,
                )
                session.add(file_row)

    def delete(self, project_id: str) -> None:
        """Delete a project and all its descendants (cascading).

        No-op if the project doesn't exist.
        """
        with session_scope(self._sessions) as session:
            row = session.get(ProjectRow, project_id)
            if row is not None:
                session.delete(row)
