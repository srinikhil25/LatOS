"""Tests for Alembic migrations.

These run the actual migrations against a temp DB to verify forward and
backward compatibility. Critical guard against schema drift.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_alembic(args: list[str], db_url: str) -> subprocess.CompletedProcess[str]:
    """Invoke alembic via the same Python interpreter pytest is using."""
    import os

    env = os.environ.copy()
    env["LATOS_DB_URL"] = db_url
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.slow
def test_upgrade_from_empty_creates_all_tables(tmp_path: Path) -> None:
    """`alembic upgrade head` on a fresh DB must create the full schema."""
    db_path = tmp_path / "fresh.db"
    db_url = f"sqlite:///{db_path}"

    result = _run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed: {result.stderr}"
    assert db_path.is_file()

    # Verify tables exist
    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    finally:
        con.close()

    table_names = {r[0] for r in rows}
    assert "alembic_version" in table_names
    assert "projects" in table_names
    assert "samples" in table_names
    assert "measurements" in table_names
    assert "files" in table_names
    assert "validation_issues" in table_names
    assert "analysis_results" in table_names


@pytest.mark.slow
def test_downgrade_to_base_drops_all_tables(tmp_path: Path) -> None:
    """`alembic downgrade base` must remove every Latos table."""
    db_path = tmp_path / "down.db"
    db_url = f"sqlite:///{db_path}"

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade failed: {up.stderr}"

    down = _run_alembic(["downgrade", "base"], db_url)
    assert down.returncode == 0, f"downgrade failed: {down.stderr}"

    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    finally:
        con.close()

    table_names = {r[0] for r in rows}
    # alembic_version may stick around; but Latos tables must be gone
    for t in (
        "projects",
        "samples",
        "measurements",
        "files",
        "validation_issues",
        "analysis_results",
    ):
        assert t not in table_names, f"{t} survived downgrade"


@pytest.mark.slow
def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Running `upgrade head` twice should be a no-op the second time."""
    db_path = tmp_path / "twice.db"
    db_url = f"sqlite:///{db_path}"

    first = _run_alembic(["upgrade", "head"], db_url)
    second = _run_alembic(["upgrade", "head"], db_url)
    assert first.returncode == 0
    assert second.returncode == 0
