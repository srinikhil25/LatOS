"""Tests for `latos.ui.services.recent_projects`.

This module is pure Python (no Qt), so the tests don't need `qtbot` and
they run in the default `not ui` slice of pytest. That keeps them on the
critical path even on headless CI workers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from latos.ui.services.recent_projects import (
    DEFAULT_MAX_ENTRIES,
    RecentProject,
    RecentProjectsService,
    default_state_path,
)


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    """A fresh JSON file path under tmp; the service creates it on save."""
    return tmp_path / "recent.json"


@pytest.fixture
def service(state_path: Path) -> RecentProjectsService:
    return RecentProjectsService(state_path)


def _make_project_dir(parent: Path, name: str) -> Path:
    """Create an actual directory so `add()` doesn't filter it out."""
    p = parent / name
    p.mkdir()
    return p


# ---------------------------------------------------------------------------
# default_state_path
# ---------------------------------------------------------------------------


class TestDefaultStatePath:
    def test_falls_back_to_home(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LATOS_HOME", raising=False)
        path = default_state_path()
        # We don't care what HOME resolves to in CI — just that the file
        # name + the parent ".latos" directory are correct.
        assert path.name == "recent.json"
        assert path.parent.name == ".latos"

    def test_honors_latos_home_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("LATOS_HOME", str(tmp_path))
        path = default_state_path()
        assert path == tmp_path / "recent.json"


# ---------------------------------------------------------------------------
# Empty / fresh state
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_list_when_file_missing_returns_empty(self, service: RecentProjectsService):
        assert service.entries() == []

    def test_remove_missing_path_is_noop(self, service: RecentProjectsService, tmp_path: Path):
        # Should not raise even though the file doesn't exist yet.
        service.remove(tmp_path / "nope")
        assert service.entries() == []

    def test_clear_creates_empty_file(self, service: RecentProjectsService, state_path: Path):
        service.clear()
        assert state_path.exists()
        data = json.loads(state_path.read_text(encoding="utf-8"))
        assert data["entries"] == []


# ---------------------------------------------------------------------------
# add() / list() basics
# ---------------------------------------------------------------------------


class TestAdd:
    def test_adds_entry_and_persists(
        self,
        service: RecentProjectsService,
        state_path: Path,
        tmp_path: Path,
    ):
        proj = _make_project_dir(tmp_path, "Project_A")
        entry = service.add(proj)
        assert entry.path == proj.resolve()
        assert entry.name == "Project_A"

        # Persisted to disk.
        assert state_path.exists()
        # Round-trips through a fresh service instance.
        fresh = RecentProjectsService(state_path).entries()
        assert len(fresh) == 1
        assert fresh[0].path == proj.resolve()

    def test_default_name_is_folder_name(self, service: RecentProjectsService, tmp_path: Path):
        proj = _make_project_dir(tmp_path, "MyData_2025")
        entry = service.add(proj)
        assert entry.name == "MyData_2025"

    def test_explicit_name_overrides_default(self, service: RecentProjectsService, tmp_path: Path):
        proj = _make_project_dir(tmp_path, "raw")
        entry = service.add(proj, name="Dhivya thesis run")
        assert entry.name == "Dhivya thesis run"

    def test_resolves_relative_paths(self, service: RecentProjectsService, tmp_path: Path):
        proj = _make_project_dir(tmp_path, "rel")
        # An equivalent path with a `..` segment should normalize to the
        # same resolved path on read.
        wonky = proj.parent / ".." / proj.parent.name / proj.name
        service.add(wonky)
        entries = service.entries()
        assert len(entries) == 1
        assert entries[0].path == proj.resolve()


class TestMru:
    def test_re_add_promotes_to_top_without_duplicating(
        self, service: RecentProjectsService, tmp_path: Path
    ):
        a = _make_project_dir(tmp_path, "A")
        b = _make_project_dir(tmp_path, "B")
        service.add(a)
        service.add(b)
        # Re-add A — it should jump back to position 0 and we should still
        # have only two entries.
        service.add(a)
        entries = service.entries()
        assert [e.path for e in entries] == [a.resolve(), b.resolve()]

    def test_max_entries_trims_oldest(self, state_path: Path, tmp_path: Path):
        svc = RecentProjectsService(state_path, max_entries=3)
        dirs = [_make_project_dir(tmp_path, f"P{i}") for i in range(5)]
        for d in dirs:
            svc.add(d)
        entries = svc.entries()
        # Only the three most recently added survive, MRU first.
        assert len(entries) == 3
        assert [e.path for e in entries] == [d.resolve() for d in reversed(dirs[-3:])]

    def test_default_max_entries_is_twenty(self):
        assert DEFAULT_MAX_ENTRIES == 20


# ---------------------------------------------------------------------------
# remove() / clear()
# ---------------------------------------------------------------------------


class TestRemove:
    def test_removes_present_entry(self, service: RecentProjectsService, tmp_path: Path):
        a = _make_project_dir(tmp_path, "A")
        b = _make_project_dir(tmp_path, "B")
        service.add(a)
        service.add(b)
        service.remove(a)
        entries = service.entries()
        assert [e.path for e in entries] == [b.resolve()]


class TestClear:
    def test_clear_removes_all(self, service: RecentProjectsService, tmp_path: Path):
        a = _make_project_dir(tmp_path, "A")
        service.add(a)
        service.clear()
        assert service.entries() == []


# ---------------------------------------------------------------------------
# Filter-on-read
# ---------------------------------------------------------------------------


class TestFilterMissingPaths:
    def test_path_deleted_after_add_is_dropped_from_list(
        self, service: RecentProjectsService, tmp_path: Path
    ):
        a = _make_project_dir(tmp_path, "A")
        b = _make_project_dir(tmp_path, "B")
        service.add(a)
        service.add(b)
        a.rmdir()  # path no longer exists
        entries = service.entries()
        assert [e.path for e in entries] == [b.resolve()]

    def test_file_path_treated_as_missing(self, service: RecentProjectsService, tmp_path: Path):
        # We only accept directories — a regular file at the path counts
        # as "no longer a project folder" and should be filtered out.
        d = _make_project_dir(tmp_path, "D")
        service.add(d)
        # Replace the directory with a file at the same path.
        import shutil

        shutil.rmtree(d)
        d.write_text("not a project")
        assert service.entries() == []


# ---------------------------------------------------------------------------
# Tolerant load
# ---------------------------------------------------------------------------


class TestTolerantLoad:
    def test_corrupt_json_treated_as_empty(self, state_path: Path):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{not valid json", encoding="utf-8")
        svc = RecentProjectsService(state_path)
        assert svc.entries() == []

    def test_wrong_schema_version_treated_as_empty(self, state_path: Path):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"version": 999, "entries": []}), encoding="utf-8")
        assert RecentProjectsService(state_path).entries() == []

    def test_entries_not_list_treated_as_empty(self, state_path: Path):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"version": 1, "entries": "oops"}), encoding="utf-8")
        assert RecentProjectsService(state_path).entries() == []

    def test_malformed_entry_skipped(self, state_path: Path, tmp_path: Path):
        good = _make_project_dir(tmp_path, "Good")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {"path": "missing keys"},  # bad shape
                        {
                            "path": str(good),
                            "name": "Good",
                            "last_opened_at": datetime.now(UTC).isoformat(),
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        entries = RecentProjectsService(state_path).entries()
        assert len(entries) == 1
        assert entries[0].path == good.resolve()

    def test_overwrite_after_corrupt_load(
        self,
        state_path: Path,
        tmp_path: Path,
    ):
        # Corrupt file → next add() should still succeed and produce a
        # valid file, not propagate the corruption.
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{garbage", encoding="utf-8")
        svc = RecentProjectsService(state_path)
        a = _make_project_dir(tmp_path, "A")
        svc.add(a)
        # Reloading from disk now yields a valid single-entry list.
        assert [e.path for e in RecentProjectsService(state_path).entries()] == [a.resolve()]


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_no_tmp_file_left_after_save(
        self, service: RecentProjectsService, state_path: Path, tmp_path: Path
    ):
        a = _make_project_dir(tmp_path, "A")
        service.add(a)
        siblings = list(state_path.parent.iterdir())
        # We expect exactly the project dir + recent.json in the tmp_path —
        # no `.tmp` carcass.
        assert not any(s.name.endswith(".tmp") for s in siblings)


# ---------------------------------------------------------------------------
# Dataclass smoke
# ---------------------------------------------------------------------------


class TestRecentProjectDataclass:
    def test_is_frozen(self, tmp_path: Path):
        entry = RecentProject(
            path=tmp_path,
            name="x",
            last_opened_at=datetime.now(UTC),
        )
        with pytest.raises((AttributeError, TypeError)):
            entry.name = "y"  # type: ignore[misc]

    def test_ordering_is_caller_concern(self, tmp_path: Path):
        # The dataclass itself is unordered; MRU ordering is enforced by
        # the service. A regression here would mean someone added
        # `order=True` and accidentally let datetime control sort order.
        now = datetime.now(UTC)
        a = RecentProject(tmp_path / "a", "a", now)
        b = RecentProject(tmp_path / "b", "b", now + timedelta(seconds=1))
        with pytest.raises(TypeError):
            _ = a < b
