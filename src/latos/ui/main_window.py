"""`LatosMainWindow` — the single window containing every page.

Built on QFluentWidgets' `FluentWindow`, which provides a sidebar
navigation + stacked content area for free. Pages are added via
`addSubInterface(widget, icon, label)`; switching the sidebar swaps
the visible page.

Why a single window
-------------------
- Modern desktop UX expects one app icon in the taskbar, one window.
- `FluentWindow.navigationInterface` lets us add/remove sidebar items
  dynamically as state changes (e.g. show "Overview" only when a
  project is open).
- Single window = single ownership of the recent-projects state, the
  current `Project`, and the running `IngestionWorker` (Stage 1E.3).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QDialog
from qfluentwidgets import FluentIcon, FluentWindow

from latos.analysis import AnalysisService
from latos.analysis import default_registry as analysis_default_registry
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.labeling.pipeline import cluster_project
from latos.ingestion.orchestrator import IngestionResult
from latos.persistence.db import (
    create_project_engine,
    init_schema,
    make_session_factory,
    project_arrays_dir,
)
from latos.persistence.repository import ProjectRepository
from latos.ui.dialogs.ingestion_progress import IngestionProgressDialog
from latos.ui.pages.analysis import AnalysisPage
from latos.ui.pages.cluster_review import ClusterReviewPage
from latos.ui.pages.overview import OverviewPage
from latos.ui.pages.project_picker import ProjectPickerPage
from latos.ui.pages.sample_review import SampleReviewPage
from latos.ui.pages.welcome import WelcomePage
from latos.ui.services.ingestion_worker import OrchestratorFactory
from latos.ui.services.recent_projects import RecentProjectsService

__all__ = ["LatosMainWindow"]


# Default window size. Big enough to fit the four-pane Overview layout
# we'll ship in 1E.4 without scrollbars on a typical 1080p display, but
# small enough that a 1366x768 laptop can show it without maximizing.
_DEFAULT_WINDOW_SIZE = QSize(1280, 800)
_MINIMUM_WINDOW_SIZE = QSize(960, 600)


class LatosMainWindow(FluentWindow):  # type: ignore[misc]
    """The single main window. All pages live inside its stacked content area."""

    def __init__(
        self,
        recent_service: RecentProjectsService | None = None,
        *,
        orchestrator_factory: OrchestratorFactory | None = None,
    ) -> None:
        """Build the window and register every page.

        Args:
            recent_service: The recent-projects state owner. Tests inject
                a service rooted at a `tmp_path`; the packaged app passes
                `None`, which falls back to `~/.latos/recent.json`.
            orchestrator_factory: Hook that returns the `Orchestrator`
                used during ingestion. `None` → real orchestrator with
                the auto-discovered parser registry. Tests pass a stub
                so the suite never touches the real ingestion stack.
        """
        super().__init__()
        self.setWindowTitle("Latos")
        self.resize(_DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(_MINIMUM_WINDOW_SIZE)

        self._recent_service = recent_service or RecentProjectsService()
        self._orchestrator_factory = orchestrator_factory
        # Set when the user picks a project; consumed by Stage 1E.4+.
        self._current_project_root: Path | None = None
        # Latest ingestion result, available to the Overview page in 1E.4.
        self._last_ingestion_result: IngestionResult | None = None

        self._init_pages()

    @property
    def current_project_root(self) -> Path | None:
        """The currently open project folder, or `None` if none is open."""
        return self._current_project_root

    @property
    def last_ingestion_result(self) -> IngestionResult | None:
        """The most recent successful `IngestionResult`, if any."""
        return self._last_ingestion_result

    def _init_pages(self) -> None:
        """Construct every page and register it with the sidebar.

        Order matters: the first page registered is the one shown on
        startup. Stage 1E.5 will add Sample / Measurement detail pages
        here.
        """
        self._welcome = WelcomePage()
        self.addSubInterface(self._welcome, FluentIcon.HOME, "Welcome")

        self._project_picker = ProjectPickerPage(self._recent_service)
        self._project_picker.projectOpened.connect(self._on_project_opened)
        self.addSubInterface(self._project_picker, FluentIcon.FOLDER_ADD, "Open")

        # Overview is registered up-front (showing the empty state) so
        # the sidebar layout is stable across "no project" / "project
        # open" states. After a successful ingestion we populate it and
        # navigate to it.
        self._overview = OverviewPage()
        self.addSubInterface(self._overview, FluentIcon.PIE_SINGLE, "Overview")

        # Cluster review page. Lets the user merge/rename auto-clustered
        # samples produced by Stage 2C (or, for now, derived from the
        # ingested project's `Sample` rows). Sits before "Review" in the
        # rail because clustering decisions are upstream of the
        # measurement-level drill-down.
        self._cluster_review = ClusterReviewPage()
        self.addSubInterface(self._cluster_review, FluentIcon.TILES, "Clustering")

        # Same pattern as Overview — register early, populate on
        # ingestion-complete. The Review page lets the user drill into
        # individual samples / measurements.
        self._sample_review = SampleReviewPage()
        self.addSubInterface(self._sample_review, FluentIcon.SEARCH, "Review")

        # Analysis page (Stage 3C) — register empty; bind a runtime
        # AnalysisService + AnalyzerRegistry + ArrayStore after the
        # project loads in `_on_project_opened`. Sits after Review
        # because it operates on already-loaded measurements.
        self._analysis = AnalysisPage()
        self.addSubInterface(self._analysis, FluentIcon.IOT, "Analysis")

    def _on_project_opened(self, path: Path) -> None:
        """Slot fired when the user picks a folder.

        Records the path on `current_project_root`, runs the
        `IngestionProgressDialog`, and on success populates the
        Overview page and switches the sidebar to it.
        """
        self._current_project_root = path
        dialog = self._make_ingestion_dialog(path)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.ingestion_result()
            self._last_ingestion_result = result
            if result is not None:
                self._overview.set_project(result.project)
                self._sample_review.set_project(result.project)
                self._cluster_review.set_clusters(
                    cluster_project(result.project),
                    project_root=result.project.root_path,
                )
                self._bind_analysis_runtime(result.project.root_path)
                self._analysis.set_project(result.project)
                self.switchTo(self._overview)
        # Cancel / failure paths leave `_last_ingestion_result` untouched
        # — the user can re-pick the folder to retry. The project picker
        # remains the active page so the user can correct course.

    # ------------------------------------------------------------------
    # Hook so tests can swap in a stub dialog.
    # ------------------------------------------------------------------

    def _make_ingestion_dialog(self, path: Path) -> IngestionProgressDialog:
        """Build the ingestion dialog. Tests override this to inject a stub."""
        return IngestionProgressDialog(
            path,
            orchestrator_factory=self._orchestrator_factory,
            parent=self,
        )

    def _bind_analysis_runtime(self, project_root: Path) -> None:
        """Construct the Stage 3 runtime and hand it to the Analysis page.

        Builds a fresh SQLAlchemy engine + session factory pointed at
        the project's data.db, a ProjectRepository on top, an
        AnalysisService, the default AnalyzerRegistry, and an
        ArrayStore over `.latos/arrays/`. Tests can monkey-patch this
        method to inject stubs.

        The engine is created per project-open: opening a different
        project rebinds with a new engine. We don't currently dispose
        the previous engine here — Stage 1's repository factory is the
        owner of engine lifetime, and rebinding `_analysis_service`
        drops the only reference. A future "Close project" action
        will handle explicit teardown.
        """
        engine = create_project_engine(project_root)
        init_schema(engine)
        session_factory = make_session_factory(engine)
        repository = ProjectRepository(session_factory)
        array_store = ArrayStore(project_arrays_dir(project_root))
        service = AnalysisService(repository=repository, array_store=array_store)
        registry = analysis_default_registry()
        self._analysis.bind_runtime(
            service=service,
            registry=registry,
            array_store=array_store,
        )
