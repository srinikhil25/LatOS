"""SQLite engine, session factory, and project DB path resolution.

Latos stores **one SQLite file per project** at
`<project_root>/.latos/data.db`. This module is the only place that knows
how to construct an engine bound to that file.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from latos.persistence.schema import Base

# Subdirectory created inside every project root.
LATOS_DIR_NAME = ".latos"
DB_FILE_NAME = "data.db"
ARRAYS_DIR_NAME = "arrays"
EXPORTS_DIR_NAME = "exports"


def project_db_path(project_root: Path) -> Path:
    """Resolve the SQLite path for a given project root folder.

    Args:
        project_root: The folder of raw data the project is rooted at.

    Returns:
        Absolute path to `<project_root>/.latos/data.db`. The file may not
        exist yet — callers are responsible for ensuring it does.
    """
    return project_root / LATOS_DIR_NAME / DB_FILE_NAME


def project_arrays_dir(project_root: Path) -> Path:
    """Resolve the Parquet arrays directory for a given project."""
    return project_root / LATOS_DIR_NAME / ARRAYS_DIR_NAME


def project_exports_dir(project_root: Path) -> Path:
    """Resolve the exports directory for a given project."""
    return project_root / LATOS_DIR_NAME / EXPORTS_DIR_NAME


def ensure_project_dirs(project_root: Path) -> None:
    """Create `.latos/`, `.latos/arrays/`, `.latos/exports/` if missing.

    Idempotent — safe to call on every project open.
    """
    (project_root / LATOS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    project_arrays_dir(project_root).mkdir(parents=True, exist_ok=True)
    project_exports_dir(project_root).mkdir(parents=True, exist_ok=True)


# ─── Engine factory ─────────────────────────────────────────────────
def _apply_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
    """Apply per-connection PRAGMAs that SQLite needs for correctness + speed.

    - `journal_mode=WAL`: enables concurrent readers + single writer.
    - `foreign_keys=ON`: SQLite default is OFF (cascades fail silently!).
    - `busy_timeout=5000`: wait up to 5s on lock contention before erroring.
    - `synchronous=NORMAL`: safe with WAL, faster than FULL.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def create_project_engine(project_root: Path, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine for the project DB at `project_root`.

    Ensures `.latos/` exists. Does NOT create tables — use `init_schema()`
    or run an Alembic migration first.

    Args:
        project_root: Folder of raw data this project is rooted at.
        echo: If True, log all SQL to stderr (debugging only).

    Returns:
        SQLAlchemy engine. Caller owns the engine and should `dispose()` it
        when the project is closed.
    """
    ensure_project_dirs(project_root)
    db_path = project_db_path(project_root)
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, echo=echo, future=True)
    event.listen(engine, "connect", _apply_sqlite_pragmas)
    return engine


def create_memory_engine(*, echo: bool = False) -> Engine:
    """Create an in-memory SQLite engine for testing.

    The engine uses a shared cache so the database persists across multiple
    connections within the same process.
    """
    engine = create_engine("sqlite://", echo=echo, future=True)
    event.listen(engine, "connect", _apply_sqlite_pragmas)
    return engine


def init_schema(engine: Engine) -> None:
    """Create all Latos tables on the given engine.

    Used in tests and as a fallback when Alembic isn't available. In normal
    operation, schema is created and migrated by Alembic.
    """
    Base.metadata.create_all(engine)


# ─── Session helpers ────────────────────────────────────────────────
def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory bound to the given engine."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Context manager that yields a session and commits/rolls back on exit.

    Example:
        ```python
        factory = make_session_factory(engine)
        with session_scope(factory) as session:
            session.add(row)
        ```
    """
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
