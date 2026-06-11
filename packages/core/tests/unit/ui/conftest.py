"""Shared fixtures for UI tests.

`pytest-qt` provides the `qtbot` fixture which manages a `QApplication`
instance per test session. We add a `latos_window` fixture that builds
a fresh `LatosMainWindow`, registers it for cleanup, and returns it.

UI tests are marked with `@pytest.mark.ui` (or via the module-level
`pytestmark`) so CI's "not ui" exclusion can opt them out on headless
Linux until X11/Wayland setup is sorted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from latos.core.models import Project
from latos.ingestion.orchestrator import IngestionResult, Orchestrator
from latos.ui.main_window import LatosMainWindow
from latos.ui.services.recent_projects import RecentProjectsService

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def recent_service(tmp_path: Path) -> RecentProjectsService:
    """Build a `RecentProjectsService` rooted at a tmp dir.

    Tests must never touch the user's real `~/.latos/recent.json`. This
    fixture isolates state per-test and is shared by the main-window and
    picker-page UI fixtures below.
    """
    return RecentProjectsService(tmp_path / "recent.json")


def _stub_ingestion_result(root: Path) -> IngestionResult:
    """A minimal `IngestionResult` for tests — no samples, no outcomes."""
    project = Project(
        id="0" * 32,
        name=root.name or "stub",
        root_path=root,
        created_at=datetime.now(UTC),
        schema_version=1,
    )
    return IngestionResult(project=project, outcomes=())


def _stub_orchestrator_factory(*, root_holder: list[Path]):
    """Build a factory whose Orchestrator returns immediately.

    `root_holder` is a one-slot list the fixture mutates so the stub
    can hand the *actual* invoked path back into `_stub_ingestion_result`.
    """

    def factory() -> Orchestrator:
        orch = MagicMock(spec=Orchestrator)
        orch.ingest.side_effect = lambda root, **_kw: _stub_ingestion_result(Path(root))
        return orch

    _ = root_holder  # kept for symmetry; not currently used
    return factory


@pytest.fixture
def stub_orchestrator_factory():
    """A no-op `OrchestratorFactory` returning an empty `IngestionResult`.

    Tests that need to drive the project picker through the main window
    use this so `_on_project_opened` doesn't fire up real ingestion. The
    spawned QThread completes within milliseconds.
    """
    return _stub_orchestrator_factory(root_holder=[])


@pytest.fixture
def latos_window(
    qtbot: QtBot,
    recent_service: RecentProjectsService,
    stub_orchestrator_factory,
) -> LatosMainWindow:
    """Build a fresh LatosMainWindow registered with qtbot for cleanup."""
    window = LatosMainWindow(
        recent_service=recent_service,
        orchestrator_factory=stub_orchestrator_factory,
    )
    qtbot.addWidget(window)  # ensures Qt cleans up at test end
    return window
