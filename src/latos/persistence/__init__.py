"""Persistence layer — SQLite + Parquet, repository pattern.

Public API:
- `ProjectRepository` — read/write Project aggregates
- `ProjectSummary` — lightweight project listing entry
- `create_project_engine` — build an engine bound to a project's DB file
- `create_memory_engine` — in-memory engine (for tests)
- `init_schema` — create tables on a fresh engine
- `make_session_factory` — wrap an engine in a sessionmaker
- `project_db_path`, `project_arrays_dir`, `project_exports_dir` — path helpers

SQL **never** leaves this package. All other Latos code uses Repositories.
"""

from __future__ import annotations

from latos.persistence.db import (
    ARRAYS_DIR_NAME,
    DB_FILE_NAME,
    EXPORTS_DIR_NAME,
    LATOS_DIR_NAME,
    create_memory_engine,
    create_project_engine,
    ensure_project_dirs,
    init_schema,
    make_session_factory,
    project_arrays_dir,
    project_db_path,
    project_exports_dir,
    session_scope,
)
from latos.persistence.repository import ProjectRepository, ProjectSummary
from latos.persistence.schema import LATEST_SCHEMA_VERSION

__all__ = [
    # Constants
    "ARRAYS_DIR_NAME",
    "DB_FILE_NAME",
    "EXPORTS_DIR_NAME",
    "LATEST_SCHEMA_VERSION",
    "LATOS_DIR_NAME",
    # Engine / session
    "create_memory_engine",
    "create_project_engine",
    "ensure_project_dirs",
    "init_schema",
    "make_session_factory",
    "session_scope",
    # Path helpers
    "project_arrays_dir",
    "project_db_path",
    "project_exports_dir",
    # Repository
    "ProjectRepository",
    "ProjectSummary",
]
