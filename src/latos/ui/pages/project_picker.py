"""`ProjectPickerPage` — second sidebar page; the user's first action.

Layout
------
- Hero block at top: bold title, one-line subtitle, big primary "Open
  Folder…" button.
- Below it: "Recent" header + a vertical stack of clickable recent-project
  cards (or an empty-state hint if there are none yet).

Behavior
--------
The page never opens a project itself. It just emits `projectOpened(Path)`
when the user picks a folder (via dialog or the Recent rail). The
`LatosMainWindow` listens to that signal and decides what to do next —
in Stage 1E.2 it just stashes the path; Stage 1E.3 will trigger ingestion.

Why a separate page (vs. a dialog)
----------------------------------
A page slot in the sidebar matches the modern "hub" pattern (VS Code's
Welcome, Fusion 360's Data Panel) and lets the user return to the picker
without dismissing other state. A modal dialog would block the entire
window every time the user wanted to switch projects.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    IconWidget,
    PrimaryPushButton,
    StrongBodyLabel,
    SubtitleLabel,
    TitleLabel,
)

from latos.ui.services.recent_projects import (
    RecentProject,
    RecentProjectsService,
)

__all__ = ["ProjectPickerPage", "RecentProjectCard"]


class RecentProjectCard(CardWidget):  # type: ignore[misc]
    """One row in the Recent rail.

    Emits `pickRequested(Path)` on left-click. We deliberately don't
    reuse `QPushButton` because we want a card-shaped affordance with
    name + path subtitle on two lines, which fits CardWidget's content
    area cleanly. Note: we avoid the name `clicked` because the base
    `CardWidget` already defines its own zero-arg `clicked` signal that
    it fires from `mouseReleaseEvent`; shadowing it with a different
    signature breaks the base implementation.
    """

    # Qt signals use mixedCase by convention; ruff's N815 doesn't know
    # about that. Suppress per-attribute rather than per-file so any
    # accidental non-signal mixedCase still gets caught.
    pickRequested = Signal(Path)  # noqa: N815

    def __init__(self, entry: RecentProject, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = entry.path
        self.setObjectName("RecentProjectCard")
        # Keep the card's height stable as the rail grows; the layout
        # below stretches the name label horizontally.
        self.setFixedHeight(72)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(12)

        icon = IconWidget(FluentIcon.FOLDER, self)
        icon.setFixedSize(24, 24)
        outer.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        name_label = StrongBodyLabel(entry.name, self)
        path_label = CaptionLabel(str(entry.path), self)
        # Caption color is already muted; we just want the path to wrap if
        # the project lives at a long path so the card stays readable.
        path_label.setWordWrap(False)
        path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        text_col.addWidget(name_label)
        text_col.addWidget(path_label)
        outer.addLayout(text_col, 1)

        # `CardWidget.clicked` is a zero-arg signal the base class fires
        # on mouseRelease; we re-emit with the bound path so consumers
        # don't have to track which card sent what.
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self) -> None:
        """Re-emit the base CardWidget click as `pickRequested(path)`."""
        self.pickRequested.emit(self._path)


class ProjectPickerPage(QWidget):
    """Project picker page registered as the "Open" sidebar entry."""

    projectOpened = Signal(Path)  # noqa: N815  (Qt signals use mixedCase)

    def __init__(
        self,
        recent_service: RecentProjectsService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Object name is what `FluentWindow.addSubInterface` keys off for
        # the navigation rail; tests find the page by this name.
        self.setObjectName("ProjectPickerPage")
        self._service = recent_service
        # Stable references so tests can introspect / drive them, and so
        # `refresh()` knows what to clear and re-build.
        self._recent_container: QWidget
        self._recent_layout: QVBoxLayout
        self._open_button: PrimaryPushButton

        self._build_layout()
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read the recent list and rebuild the rail.

        Called automatically after the user picks a folder. The main
        window also calls this when the page becomes visible, so a
        recent-projects edit made elsewhere shows up immediately.
        """
        # Clear existing children. `deleteLater()` keeps the destruction
        # off the immediate stack, which is the safe pattern for widgets
        # whose parent layout we're still iterating.
        while self._recent_layout.count():
            item = self._recent_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

        entries = self._service.entries()
        if not entries:
            empty = BodyLabel("No recent projects yet.")
            empty.setObjectName("RecentEmptyState")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._recent_layout.addWidget(empty)
            return

        for entry in entries:
            card = RecentProjectCard(entry, self._recent_container)
            card.pickRequested.connect(self._handle_recent_clicked)
            self._recent_layout.addWidget(card)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _handle_open_folder_clicked(self) -> None:
        """Run the native folder picker; if accepted, open the chosen path."""
        chosen = QFileDialog.getExistingDirectory(self, "Choose a project folder")
        if not chosen:
            return  # user dismissed the dialog
        self._open_path(Path(chosen))

    def _handle_recent_clicked(self, path: Path) -> None:
        """User clicked a card in the Recent rail."""
        self._open_path(path)

    def _open_path(self, path: Path) -> None:
        """Common path: record in MRU + emit projectOpened + refresh rail."""
        self._service.add(path)
        self.refresh()
        self.projectOpened.emit(path)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 32, 40, 32)
        outer.setSpacing(24)

        outer.addLayout(self._build_hero())
        outer.addLayout(self._build_recent_section())
        outer.addStretch(1)

    def _build_hero(self) -> QVBoxLayout:
        hero = QVBoxLayout()
        hero.setSpacing(8)

        title = TitleLabel("Open a project", self)
        subtitle = SubtitleLabel(
            "Pick a folder of measurement data to ingest.",
            self,
        )

        # PrimaryPushButton has no `(icon, text, parent)` constructor in
        # all qfluentwidgets versions, so build it bare and configure
        # afterwards — this stays version-tolerant.
        self._open_button = PrimaryPushButton("Open Folder", self)
        self._open_button.setIcon(FluentIcon.FOLDER_ADD)
        self._open_button.setObjectName("OpenFolderButton")
        self._open_button.setMinimumHeight(40)
        self._open_button.clicked.connect(self._handle_open_folder_clicked)

        hero.addWidget(title)
        hero.addWidget(subtitle)
        hero.addSpacing(8)
        # Keep the button to its natural width — full-bleed primary
        # buttons feel pushy on a hub-style page.
        button_row = QHBoxLayout()
        button_row.addWidget(self._open_button)
        button_row.addStretch(1)
        hero.addLayout(button_row)

        return hero

    def _build_recent_section(self) -> QVBoxLayout:
        section = QVBoxLayout()
        section.setSpacing(8)

        header = StrongBodyLabel("Recent", self)
        header.setObjectName("RecentHeader")
        section.addWidget(header)

        self._recent_container = QWidget(self)
        self._recent_layout = QVBoxLayout(self._recent_container)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(8)
        section.addWidget(self._recent_container)

        return section
