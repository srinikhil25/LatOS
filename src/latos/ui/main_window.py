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

from PySide6.QtCore import QSize
from qfluentwidgets import FluentIcon, FluentWindow

from latos.ui.pages.welcome import WelcomePage

__all__ = ["LatosMainWindow"]


# Default window size. Big enough to fit the four-pane Overview layout
# we'll ship in 1E.4 without scrollbars on a typical 1080p display, but
# small enough that a 1366x768 laptop can show it without maximizing.
_DEFAULT_WINDOW_SIZE = QSize(1280, 800)
_MINIMUM_WINDOW_SIZE = QSize(960, 600)


class LatosMainWindow(FluentWindow):  # type: ignore[misc]
    """The single main window. All pages live inside its stacked content area."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Latos")
        self.resize(_DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(_MINIMUM_WINDOW_SIZE)

        self._init_pages()

    def _init_pages(self) -> None:
        """Construct every page and register it with the sidebar.

        Order matters: the first page registered is the one shown on
        startup. Subsequent stages will add Project Picker, Overview,
        and Review pages here.
        """
        self._welcome = WelcomePage()
        self.addSubInterface(self._welcome, FluentIcon.HOME, "Welcome")
