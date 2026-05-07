"""Shared fixtures for UI tests.

`pytest-qt` provides the `qtbot` fixture which manages a `QApplication`
instance per test session. We add a `latos_window` fixture that builds
a fresh `LatosMainWindow`, registers it for cleanup, and returns it.

UI tests are marked with `@pytest.mark.ui` (or via the module-level
`pytestmark`) so CI's "not ui" exclusion can opt them out on headless
Linux until X11/Wayland setup is sorted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from latos.ui.main_window import LatosMainWindow

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def latos_window(qtbot: QtBot) -> LatosMainWindow:
    """Build a fresh LatosMainWindow registered with qtbot for cleanup."""
    window = LatosMainWindow()
    qtbot.addWidget(window)  # ensures Qt cleans up at test end
    return window
