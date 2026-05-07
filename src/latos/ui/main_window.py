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
from qfluentwidgets import FluentIcon, FluentWindow

from latos.ui.pages.project_picker import ProjectPickerPage
from latos.ui.pages.welcome import WelcomePage
from latos.ui.services.recent_projects import RecentProjectsService

__all__ = ["LatosMainWindow"]


# Default window size. Big enough to fit the four-pane Overview layout
# we'll ship in 1E.4 without scrollbars on a typical 1080p display, but
# small enough that a 1366x768 laptop can show it without maximizing.
_DEFAULT_WINDOW_SIZE = QSize(1280, 800)
_MINIMUM_WINDOW_SIZE = QSize(960, 600)


class LatosMainWindow(FluentWindow):  # type: ignore[misc]
    """The single main window. All pages live inside its stacked content area."""

    def __init__(self, recent_service: RecentProjectsService | None = None) -> None:
        """Build the window and register every page.

        Args:
            recent_service: The recent-projects state owner. Tests inject
                a service rooted at a `tmp_path`; the packaged app passes
                `None`, which falls back to `~/.latos/recent.json`.
        """
        super().__init__()
        self.setWindowTitle("Latos")
        self.resize(_DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(_MINIMUM_WINDOW_SIZE)

        self._recent_service = recent_service or RecentProjectsService()
        # Set when the user picks a project; consumed by Stage 1E.3+.
        self._current_project_root: Path | None = None

        self._init_pages()

    @property
    def current_project_root(self) -> Path | None:
        """The currently open project folder, or `None` if none is open."""
        return self._current_project_root

    def _init_pages(self) -> None:
        """Construct every page and register it with the sidebar.

        Order matters: the first page registered is the one shown on
        startup. Subsequent stages will add Overview and Review pages
        here.
        """
        self._welcome = WelcomePage()
        self.addSubInterface(self._welcome, FluentIcon.HOME, "Welcome")

        self._project_picker = ProjectPickerPage(self._recent_service)
        self._project_picker.projectOpened.connect(self._on_project_opened)
        self.addSubInterface(self._project_picker, FluentIcon.FOLDER_ADD, "Open")

    def _on_project_opened(self, path: Path) -> None:
        """Slot fired when the user picks a folder.

        Stage 1E.2 just records the chosen path so later stages
        (ingestion, overview) can read it. Stage 1E.3 will replace this
        with a kickoff into the ingestion worker.
        """
        self._current_project_root = path
