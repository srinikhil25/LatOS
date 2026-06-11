"""Tests for `latos.ui.main_window.LatosMainWindow`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PySide6.QtWidgets import QWidget
from qfluentwidgets import PrimaryPushButton

from latos.ui.main_window import LatosMainWindow
from latos.ui.pages.cluster_review import ClusterReviewPage
from latos.ui.pages.project_hub import ProjectHubPage
from latos.ui.pages.sample_review import SampleReviewPage

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
    def test_hub_page_present_in_widget_tree(self, latos_window: LatosMainWindow):
        # The Project Hub is the home page (it replaced the old Welcome +
        # Open pages). `addSubInterface` parents it to the FluentWindow's
        # stacked widget; a findChild verifies the registration succeeded.
        hub = latos_window.findChild(ProjectHubPage, "ProjectHubPage")
        assert hub is not None
        # Start view until a project is opened.
        assert hub.project is None

    def test_overview_page_is_retired(self, latos_window: LatosMainWindow):
        # The Overview page merged into the hub (UR6): its stats and
        # preview plot live on ProjectHubPage now. Guard against an
        # accidental re-registration.
        assert latos_window.findChild(QWidget, "OverviewPage") is None

    def test_sample_review_page_present_in_widget_tree(self, latos_window: LatosMainWindow):
        review = latos_window.findChild(SampleReviewPage, "SampleReviewPage")
        assert review is not None
        assert review.project is None

    def test_cluster_review_page_present_in_widget_tree(self, latos_window: LatosMainWindow):
        cluster = latos_window.findChild(ClusterReviewPage, "ClusterReviewPage")
        assert cluster is not None
        # Empty until a project is opened — no clusters, no project root.
        assert cluster.clusters == ()
        assert cluster.project_root is None


class TestProjectOpenedSlot:
    def test_initial_current_project_is_none(self, latos_window: LatosMainWindow):
        assert latos_window.current_project_root is None

    def test_hub_open_updates_current_project_root(
        self,
        qtbot: QtBot,
        recent_service: RecentProjectsService,
        latos_window: LatosMainWindow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Drive the hub exactly the way a real user would: patch the
        # dialog to return a chosen folder and click the "Open Folder"
        # button on the start view. The `latos_window` fixture wires a
        # stub orchestrator into the ingestion dialog so this completes
        # synchronously without touching real SQLite / Parquet.
        chosen = tmp_path / "ChosenProject"
        chosen.mkdir()

        from latos.ui.pages import project_hub as hub_module

        monkeypatch.setattr(
            hub_module.QFileDialog,
            "getExistingDirectory",
            staticmethod(lambda *args, **kwargs: str(chosen)),
        )

        hub = latos_window.findChild(ProjectHubPage, "ProjectHubPage")
        assert hub is not None
        open_button = hub.findChild(PrimaryPushButton, "OpenFolderButton")
        assert open_button is not None

        with qtbot.waitSignal(hub.projectOpened, timeout=2000):
            open_button.click()

        assert latos_window.current_project_root == chosen
        # The stub orchestrator returns an empty IngestionResult, which
        # the main window stores after the dialog accepts.
        assert latos_window.last_ingestion_result is not None
        assert latos_window.last_ingestion_result.project.root_path == chosen
        # And the folder shows up as a recent in the injected service.
        assert [e.path for e in recent_service.entries()] == [chosen.resolve()]
        # The sample review page is populated in `_on_project_opened`.
        review = latos_window.findChild(SampleReviewPage, "SampleReviewPage")
        assert review is not None
        assert review.project is not None
        assert review.project.root_path == chosen
        # Cluster review picks up the project root so Apply can persist
        # decisions to `<root>/.latos/cluster_decisions.json`. The stub
        # ingestion has zero samples, so the cluster list is empty.
        cluster = latos_window.findChild(ClusterReviewPage, "ClusterReviewPage")
        assert cluster is not None
        assert cluster.project_root == chosen
        # The hub itself switches to its populated state after open.
        assert hub.project is not None
        assert hub.project.root_path == chosen
