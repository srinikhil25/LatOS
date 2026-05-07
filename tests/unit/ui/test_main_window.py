"""Tests for `latos.ui.main_window.LatosMainWindow`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from latos.ui.main_window import LatosMainWindow
from latos.ui.pages.overview import OverviewPage
from latos.ui.pages.project_picker import ProjectPickerPage
from latos.ui.pages.sample_review import SampleReviewPage
from latos.ui.pages.welcome import WelcomePage

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    from latos.ui.services.recent_projects import RecentProjectsService

pytestmark = pytest.mark.ui


class TestWindowConstruction:
    def test_window_constructs(self, latos_window: LatosMainWindow):
        # If the constructor raises (Qt theme miswire, missing widget,
        # etc.), the test fails here before any assertion runs.
        assert latos_window is not None

    def test_window_title(self, latos_window: LatosMainWindow):
        assert latos_window.windowTitle() == "Latos"

    def test_default_window_size(self, latos_window: LatosMainWindow):
        # Matches `_DEFAULT_WINDOW_SIZE`. Fixed values catch accidental
        # regressions from someone "tidying" the constants.
        size = latos_window.size()
        assert size.width() == 1280
        assert size.height() == 800

    def test_minimum_window_size(self, latos_window: LatosMainWindow):
        min_size = latos_window.minimumSize()
        assert min_size.width() == 960
        assert min_size.height() == 600


class TestPagesRegistered:
    def test_welcome_page_present_in_widget_tree(self, latos_window: LatosMainWindow):
        # `addSubInterface` parents the page to the FluentWindow's
        # stacked widget. A simple findChild verifies the registration
        # succeeded.
        welcome = latos_window.findChild(WelcomePage, "WelcomePage")
        assert welcome is not None

    def test_project_picker_page_present_in_widget_tree(self, latos_window: LatosMainWindow):
        picker = latos_window.findChild(ProjectPickerPage, "ProjectPickerPage")
        assert picker is not None

    def test_overview_page_present_in_widget_tree(self, latos_window: LatosMainWindow):
        overview = latos_window.findChild(OverviewPage, "OverviewPage")
        assert overview is not None
        # Empty state until a project is opened.
        assert overview.project is None

    def test_sample_review_page_present_in_widget_tree(self, latos_window: LatosMainWindow):
        review = latos_window.findChild(SampleReviewPage, "SampleReviewPage")
        assert review is not None
        assert review.project is None


class TestProjectOpenedSlot:
    def test_initial_current_project_is_none(self, latos_window: LatosMainWindow):
        assert latos_window.current_project_root is None

    def test_picker_signal_updates_current_project_root(
        self,
        qtbot: QtBot,
        recent_service: RecentProjectsService,
        latos_window: LatosMainWindow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Drive the picker exactly the way a real user would: patch the
        # dialog to return a chosen folder and click the open button.
        # The `latos_window` fixture wires a stub orchestrator into the
        # ingestion dialog so this completes synchronously without
        # touching real SQLite / Parquet.
        chosen = tmp_path / "ChosenProject"
        chosen.mkdir()

        from latos.ui.pages import project_picker as picker_module

        monkeypatch.setattr(
            picker_module.QFileDialog,
            "getExistingDirectory",
            staticmethod(lambda *args, **kwargs: str(chosen)),
        )

        picker = latos_window.findChild(ProjectPickerPage, "ProjectPickerPage")
        assert picker is not None

        with qtbot.waitSignal(picker.projectOpened, timeout=2000):
            picker._open_button.click()

        assert latos_window.current_project_root == chosen
        # The stub orchestrator returns an empty IngestionResult, which
        # the main window stores after the dialog accepts.
        assert latos_window.last_ingestion_result is not None
        assert latos_window.last_ingestion_result.project.root_path == chosen
        # And the folder shows up as a recent in the injected service.
        assert [e.path for e in recent_service.entries()] == [chosen.resolve()]
        # Overview page got populated with the (empty) project.
        overview = latos_window.findChild(OverviewPage, "OverviewPage")
        assert overview is not None
        assert overview.project is not None
        assert overview.project.root_path == chosen
        # Same for the sample review page — both are populated in
        # `_on_project_opened`.
        review = latos_window.findChild(SampleReviewPage, "SampleReviewPage")
        assert review is not None
        assert review.project is not None
        assert review.project.root_path == chosen
