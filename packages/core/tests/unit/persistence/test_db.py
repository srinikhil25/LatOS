"""Tests for `latos.persistence.db` — paths, engine factory, sessions."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

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


# ─── Path helpers ───────────────────────────────────────────────────
class TestPathHelpers:
    def test_project_db_path(self, tmp_path: Path) -> None:
        result = project_db_path(tmp_path)
        assert result == tmp_path / LATOS_DIR_NAME / DB_FILE_NAME

    def test_project_arrays_dir(self, tmp_path: Path) -> None:
        result = project_arrays_dir(tmp_path)
        assert result == tmp_path / LATOS_DIR_NAME / ARRAYS_DIR_NAME

    def test_project_exports_dir(self, tmp_path: Path) -> None:
        result = project_exports_dir(tmp_path)
        assert result == tmp_path / LATOS_DIR_NAME / EXPORTS_DIR_NAME

    def test_ensure_project_dirs_creates_all(self, tmp_path: Path) -> None:
        ensure_project_dirs(tmp_path)
        assert (tmp_path / LATOS_DIR_NAME).is_dir()
        assert project_arrays_dir(tmp_path).is_dir()
        assert project_exports_dir(tmp_path).is_dir()

    def test_ensure_project_dirs_idempotent(self, tmp_path: Path) -> None:
        ensure_project_dirs(tmp_path)
        ensure_project_dirs(tmp_path)  # second call must not error
        assert (tmp_path / LATOS_DIR_NAME).is_dir()


# ─── Engine factory ─────────────────────────────────────────────────
class TestProjectEngine:
    def test_creates_db_file(self, tmp_path: Path) -> None:
        engine = create_project_engine(tmp_path)
        try:
            init_schema(engine)
            assert project_db_path(tmp_path).is_file()
        finally:
            engine.dispose()

    def test_pragmas_applied(self, tmp_path: Path) -> None:
        engine = create_project_engine(tmp_path)
        try:
            init_schema(engine)
            with engine.connect() as conn:
                fk = conn.execute(text("PRAGMA foreign_keys")).scalar()
                journal = conn.execute(text("PRAGMA journal_mode")).scalar()
                busy = conn.execute(text("PRAGMA busy_timeout")).scalar()
            assert fk == 1
            assert journal == "wal"
            assert busy == 5000
        finally:
            engine.dispose()

    def test_memory_engine_pragmas(self) -> None:
        engine = create_memory_engine()
        try:
            with engine.connect() as conn:
                fk = conn.execute(text("PRAGMA foreign_keys")).scalar()
            assert fk == 1
        finally:
            engine.dispose()


# ─── Session helpers ────────────────────────────────────────────────
class TestSessionScope:
    def test_commits_on_success(self, engine: Engine) -> None:
        factory = make_session_factory(engine)
        with session_scope(factory) as s:
            s.execute(
                text(
                    "INSERT INTO projects (id, name, root_path, created_at, schema_version) "
                    "VALUES (:id, :name, :root, :created, :sv)"
                ),
                {
                    "id": "a" * 32,
                    "name": "X",
                    "root": "/x",
                    "created": "2026-01-01 00:00:00",
                    "sv": 1,
                },
            )
        # Verify in a new session
        with session_scope(factory) as s:
            count = s.execute(text("SELECT COUNT(*) FROM projects")).scalar()
        assert count == 1

    def test_rolls_back_on_exception(self, engine: Engine) -> None:
        factory = make_session_factory(engine)

        class _Boom(Exception):
            pass

        with pytest.raises(_Boom), session_scope(factory) as s:
            s.execute(
                text(
                    "INSERT INTO projects (id, name, root_path, created_at, schema_version) "
                    "VALUES (:id, :name, :root, :created, :sv)"
                ),
                {
                    "id": "b" * 32,
                    "name": "X",
                    "root": "/x",
                    "created": "2026-01-01 00:00:00",
                    "sv": 1,
                },
            )
            raise _Boom

        with session_scope(factory) as s:
            count = s.execute(text("SELECT COUNT(*) FROM projects")).scalar()
        assert count == 0


def test_engine_fixture_yields_working_engine(engine: Engine) -> None:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_session_factory_fixture(session_factory: sessionmaker[Session]) -> None:
    assert callable(session_factory)
