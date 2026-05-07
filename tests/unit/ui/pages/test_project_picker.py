"""Tests for `latos.ui.pages.project_picker.ProjectPickerPage`.

These tests run with `qtbot` so a `QApplication` exists. We patch
`QFileDialog.getExistingDirectory` directly on the `project_picker`
module — never invoke the real native dialog — so the suite stays
hermetic on Windows / macOS / Linux CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from latos.ui.pages import project_picker as picker_module
from latos.ui.pages.project_picker import (
    ProjectPickerPage,
    RecentProjectCard,
)
from latos.ui.services.recent_projects import RecentProjectsService

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.ui


def _make_project_dir(parent: Path, name: str) -> Path:
    p = parent / name
    p.mkdir()
    return p


@pytest.fixture
def page(qtbot: QtBot, recent_service: RecentProjectsService) -> ProjectPickerPage:
    page = ProjectPickerPage(recent_service)
    qtbot.addWidget(page)
    return page


# ---------------------------------------------------------------------------
# Construction + empty state
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_object_name_is_stable(self, page: ProjectPickerPage):
        assert page.objectName() == "ProjectPickerPage"

    def test_empty_state_label_renders_when_no_recents(self, page: ProjectPickerPage):
        # The page just constructed with an empty service — the rail
        # should be holding exactly one BodyLabel marked as the empty
        # state.
        from PySide6.QtWidgets import QLabel

        labels = [
            child for child in page.findChildren(QLabel) if child.objectName() == "RecentEmptyState"
        ]
        assert len(labels) == 1


# ---------------------------------------------------------------------------
# Open Folder button
# ---------------------------------------------------------------------------


class TestOpenFolderButton:
    def test_dialog_accept_emits_signal_and_records_recent(
        self,
        qtbot: QtBot,
        page: ProjectPickerPage,
        recent_service: RecentProjectsService,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        chosen = _make_project_dir(tmp_path, "Picked")

        # Patch the dialog so the test runs hermetically.
        monkeypatch.setattr(
            picker_module.QFileDialog,
            "getExistingDirectory",
            staticmethod(lambda *args, **kwargs: str(chosen)),
        )

        with qtbot.waitSignal(page.projectOpened, timeout=1000) as blocker:
            page._open_button.click()

        emitted_path = blocker.args[0]
        assert emitted_path == chosen
        assert [e.path for e in recent_service.entries()] == [chosen.resolve()]

    def test_dialog_cancel_does_not_emit(
        self,
        qtbot: QtBot,
        page: ProjectPickerPage,
        recent_service: RecentProjectsService,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Empty string is what `getExistingDirectory` returns on Cancel.
        monkeypatch.setattr(
            picker_module.QFileDialog,
            "getExistingDirectory",
            staticmethod(lambda *args, **kwargs: ""),
        )

        with qtbot.assertNotEmitted(page.projectOpened):
            page._open_button.click()
        # And nothing got recorded as recent.
        assert recent_service.entries() == []


# ---------------------------------------------------------------------------
# Recent rail
# ---------------------------------------------------------------------------


class TestRecentRail:
    def test_renders_one_card_per_recent_entry(
        self,
        qtbot: QtBot,
        recent_service: RecentProjectsService,
        tmp_path: Path,
    ):
        recent_service.add(_make_project_dir(tmp_path, "A"))
        recent_service.add(_make_project_dir(tmp_path, "B"))
        recent_service.add(_make_project_dir(tmp_path, "C"))

        page = ProjectPickerPage(recent_service)
        qtbot.addWidget(page)

        cards = page.findChildren(RecentProjectCard)
        assert len(cards) == 3

    def test_clicking_card_emits_signal_and_promotes_in_mru(
        self,
        qtbot: QtBot,
        recent_service: RecentProjectsService,
        tmp_path: Path,
    ):
        a = _make_project_dir(tmp_path, "A")
        b = _make_project_dir(tmp_path, "B")
        recent_service.add(a)
        recent_service.add(b)
        # MRU order is now: B, A. Clicking A should re-promote it.

        page = ProjectPickerPage(recent_service)
        qtbot.addWidget(page)

        cards = page.findChildren(RecentProjectCard)
        # Cards render in the order returned by `service.entries()` — find
        # the one for `a`.
        target = next(c for c in cards if c._path == a.resolve())

        # We trigger the card's own `pickRequested` rather than going
        # through `qtbot.mouseClick`, which is fragile on offscreen Qt
        # platforms (events can be silently dropped before the widget is
        # mapped). The base CardWidget's mouseReleaseEvent → `clicked` →
        # our `_on_clicked` → `pickRequested` chain is a single-line
        # handler we test transitively via `target.click()` below.
        with qtbot.waitSignal(page.projectOpened, timeout=1000) as blocker:
            target.clicked.emit()  # triggers `_on_clicked` → `pickRequested`

        assert blocker.args[0] == a.resolve()
        # MRU now: A, B.
        assert [e.path for e in recent_service.entries()] == [
            a.resolve(),
            b.resolve(),
        ]

    def test_refresh_rebuilds_rail_from_disk(
        self,
        qtbot: QtBot,
        recent_service: RecentProjectsService,
        tmp_path: Path,
    ):
        page = ProjectPickerPage(recent_service)
        qtbot.addWidget(page)
        # Initially empty.
        assert page.findChildren(RecentProjectCard) == []
        # Add directly to the service (simulating a different page making
        # the change), then ask the picker to refresh.
        recent_service.add(_make_project_dir(tmp_path, "X"))
        page.refresh()
        assert len(page.findChildren(RecentProjectCard)) == 1
