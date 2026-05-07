"""Shared fixtures for UI tests.

`pytest-qt` provides the `qtbot` fixture which manages a `QApplication`
instance per test session. We add a `latos_window` fixture that builds
a fresh `LatosMainWindow`, registers it for cleanup, and returns it.

UI tests are marked with `@pytest.mark.ui` (or via the module-level
`pytestmark`) so CI's "not ui" exclusion can opt them out on headless
Linux until X11/Wayland setup is sorted.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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


@pytest.fixture
def latos_window(qtbot: QtBot, recent_service: RecentProjectsService) -> LatosMainWindow:
    """Build a fresh LatosMainWindow registered with qtbot for cleanup."""
    window = LatosMainWindow(recent_service=recent_service)
    qtbot.addWidget(window)  # ensures Qt cleans up at test end
    return window
