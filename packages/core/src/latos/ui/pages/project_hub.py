"""`ProjectHubPage` — the single home screen for Latos.

Replaces the old `WelcomePage` + `ProjectPickerPage` split *and* the
old `OverviewPage` dashboard. One page, two states:

- **Start** (no project open): brand hero, a big "Open Folder" button,
  and the recent-projects rail. The user's first action lives here.
- **Hub** (project open): the project name, ingestion outcome line,
  summary stat tiles, a themed preview plot, and a grid of large
  `ActivityCard`s that launch each workspace section (Samples,
  Analysis).

The page owns no project state beyond what it displays. It emits:
- `projectOpened(Path)` when the user picks a folder (dialog or recent),
- `activitySelected(str)` when an activity card is clicked (the key maps
  to a navigation target in `LatosMainWindow`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
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

from latos.ingestion.array_store import ArrayStore
from latos.ui.components import (
    ActivityCard,
    Card,
    PageContainer,
    SectionHeader,
    StatTile,
)
from latos.ui.design import plot_theme, tokens
from latos.ui.services.recent_projects import RecentProject, RecentProjectsService

if TYPE_CHECKING:
    import numpy as np

    from latos.core.models import Measurement, Project
    from latos.ingestion.orchestrator import IngestionResult

__all__ = ["ProjectHubPage", "RecentProjectCard"]


# The launcher cards on the hub, in display order:
# (route key, icon, title, one-line description). The key is matched in
# `LatosMainWindow` to the corresponding sub-interface. The old
# "Overview" card is gone — the hub itself now carries the summary +
# preview, and "Clustering" is folding into Samples (UR7).
_ACTIVITIES: tuple[tuple[str, object, str, str], ...] = (
    (
        "review",
        FluentIcon.SEARCH,
        "Samples",
        "Browse samples, measurements, and images",
    ),
    (
        "analysis",
        FluentIcon.IOT,
        "Analysis",
        "Run analyzers — band gap, peak fit, and more",
    ),
)

# Summary tiles on the hub: (key, label). Values filled from the project.
_STAT_TILES: tuple[tuple[str, str], ...] = (
    ("samples", "Samples"),
    ("measurements", "Measurements"),
    ("techniques", "Techniques"),
)

# Activity grid column count.
_GRID_COLUMNS = 2

# How many distinct 1-D arrays a measurement needs before we plot one
# against another (x vs y). Below that, the lone array plots vs index.
_MIN_ARRAYS_FOR_XY_PLOT = 2


class RecentProjectCard(CardWidget):  # type: ignore[misc]
    """One row in the start view's Recent rail.

    Moved here from the retired `project_picker` module — the hub is its
    only consumer. Emits `pickRequested(Path)` on click; we avoid the
    name `clicked` because the base `CardWidget` already defines a
    zero-arg `clicked` signal it fires from `mouseReleaseEvent`.
    """

    pickRequested = Signal(Path)  # noqa: N815  (Qt signals use mixedCase)

    def __init__(self, entry: RecentProject, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = entry.path
        self.setObjectName("RecentProjectCard")
        self.setFixedHeight(72)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(
            tokens.SPACE_LG,
            tokens.SPACE_MD,
            tokens.SPACE_LG,
            tokens.SPACE_MD,
        )
        outer.setSpacing(tokens.SPACE_MD)

        icon = IconWidget(FluentIcon.FOLDER, self)
        icon.setFixedSize(24, 24)
        outer.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(tokens.SPACE_2XS)

        name_label = StrongBodyLabel(entry.name, self)
        path_label = CaptionLabel(str(entry.path), self)
        path_label.setWordWrap(False)
        path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        text_col.addWidget(name_label)
        text_col.addWidget(path_label)
        outer.addLayout(text_col, 1)

        self.clicked.connect(self._on_clicked)

    def _on_clicked(self) -> None:
        """Re-emit the base CardWidget click as `pickRequested(path)`."""
        self.pickRequested.emit(self._path)


class ProjectHubPage(QWidget):
    """Home page: start screen before a project, activity hub after."""

    projectOpened = Signal(Path)  # noqa: N815  (Qt signals use mixedCase)
    activitySelected = Signal(str)  # noqa: N815

    def __init__(
        self,
        recent_service: RecentProjectsService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ProjectHubPage")
        self._service = recent_service

        self._project: Project | None = None
        self._stat_tiles: dict[str, StatTile] = {}
        self._hub_title: TitleLabel
        self._hub_subtitle: BodyLabel
        self._ingestion_caption: CaptionLabel
        self._plot_widget: pg.PlotWidget
        self._plot_caption: CaptionLabel
        self._recent_container: QWidget
        self._recent_layout: QVBoxLayout

        self._stack = QStackedWidget(self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._stack)

        self._start_view = self._build_start_view()
        self._hub_view = self._build_hub_view()
        self._stack.addWidget(self._start_view)
        self._stack.addWidget(self._hub_view)

        self.refresh()
        self._stack.setCurrentWidget(self._start_view)

    # ─── Public API ──────────────────────────────────────────────────
    @property
    def project(self) -> Project | None:
        """The project currently shown on the hub, or `None` (start view)."""
        return self._project

    def refresh(self) -> None:
        """Re-read recent projects and rebuild the start-view rail."""
        while self._recent_layout.count():
            item = self._recent_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

        entries = self._service.entries()
        if not entries:
            empty = BodyLabel("No recent projects yet.")
            empty.setObjectName("RecentEmptyState")
            empty.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self._recent_layout.addWidget(empty)
            return

        for entry in entries:
            card = RecentProjectCard(entry, self._recent_container)
            card.pickRequested.connect(self._open_path)
            self._recent_layout.addWidget(card)

    def set_project(
        self,
        project: Project,
        *,
        array_store: ArrayStore | None = None,
        ingestion: IngestionResult | None = None,
    ) -> None:
        """Populate and show the hub view for `project`.

        Args:
            project: The ingested project to summarize.
            array_store: Where measurement arrays live, for the preview
                plot. `None` → open `<root>/.latos/arrays` (production
                default); tests inject a tmp-backed store.
            ingestion: The per-file outcome ledger from the run that
                produced `project`. When present, the hub shows real
                parsed / failed / unclassified counts; when absent
                (e.g. reopening an existing project), the line is
                hidden rather than showing made-up zeros.
        """
        self._project = project
        self._hub_title.setText(project.name)

        stats = _summary_stats(project)
        self._hub_subtitle.setText(
            f"{stats['samples']} samples · {stats['measurements']} measurements",
        )
        for key, tile in self._stat_tiles.items():
            tile.set_value(stats.get(key, 0))

        if ingestion is not None:
            self._ingestion_caption.setText(
                f"{ingestion.parsed_count} files parsed · "
                f"{ingestion.failed_count} failed · "
                f"{ingestion.unclassified_count} unclassified",
            )
            self._ingestion_caption.setVisible(True)
        else:
            self._ingestion_caption.setVisible(False)

        self._render_preview(project, array_store)
        self._stack.setCurrentWidget(self._hub_view)

    def clear(self) -> None:
        """Return to the start view (no project)."""
        self._project = None
        self.refresh()
        self._stack.setCurrentWidget(self._start_view)

    # ─── Start view ──────────────────────────────────────────────────
    def _build_start_view(self) -> QWidget:
        container = PageContainer()

        hero = QVBoxLayout()
        hero.setSpacing(tokens.SPACE_SM)
        title = TitleLabel("Latos")
        subtitle = SubtitleLabel("Multi-modal materials characterization")
        hero.addWidget(title)
        hero.addWidget(subtitle)

        open_button = PrimaryPushButton("Open Folder")
        open_button.setIcon(FluentIcon.FOLDER_ADD)
        open_button.setObjectName("OpenFolderButton")
        open_button.setMinimumHeight(40)
        open_button.clicked.connect(self._handle_open_folder_clicked)
        button_row = QHBoxLayout()
        button_row.addWidget(open_button)
        button_row.addStretch(1)
        hero.addSpacing(tokens.SPACE_SM)
        hero.addLayout(button_row)
        container.add_layout(hero)

        recent_header = SectionHeader(
            "Recent",
            caption="Pick up where you left off",
        )
        container.add_widget(recent_header)

        self._recent_container = QWidget()
        self._recent_layout = QVBoxLayout(self._recent_container)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(tokens.SPACE_SM)
        container.add_widget(self._recent_container)

        container.add_stretch(1)
        return container

    # ─── Hub view ────────────────────────────────────────────────────
    def _build_hub_view(self) -> QWidget:
        container = PageContainer()

        header = QVBoxLayout()
        header.setSpacing(tokens.SPACE_2XS)
        self._hub_title = TitleLabel("Project")
        self._hub_subtitle = BodyLabel("")
        self._ingestion_caption = CaptionLabel("")
        self._ingestion_caption.setObjectName("IngestionCaption")
        self._ingestion_caption.setVisible(False)
        header.addWidget(self._hub_title)
        header.addWidget(self._hub_subtitle)
        header.addWidget(self._ingestion_caption)
        container.add_layout(header)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(tokens.SPACE_MD)
        for key, label in _STAT_TILES:
            tile = StatTile(label, 0)
            self._stat_tiles[key] = tile
            stats_row.addWidget(tile)
        container.add_layout(stats_row)

        container.add_widget(SectionHeader("Workspace", caption="Choose where to go next"))

        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(tokens.SPACE_MD)
        grid.setVerticalSpacing(tokens.SPACE_MD)
        for index, (key, icon, title, description) in enumerate(_ACTIVITIES):
            card = ActivityCard(
                key=key,
                icon=icon,
                title=title,
                description=description,
            )
            card.activated.connect(self.activitySelected)
            row, col = divmod(index, _GRID_COLUMNS)
            grid.addWidget(card, row, col)
        for col in range(_GRID_COLUMNS):
            grid.setColumnStretch(col, 1)
        container.add_widget(grid_holder)

        # Preview plot — carried over from the retired Overview page,
        # now themed instead of forced-white. Lives inside a Card so it
        # sits on the same elevation language as the stat tiles.
        preview_card = Card(padding=tokens.SPACE_LG, spacing=tokens.SPACE_SM)
        preview_header = StrongBodyLabel("Preview")
        preview_header.setObjectName("PreviewHeader")
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setObjectName("PreviewPlot")
        self._plot_widget.setMinimumHeight(220)
        plot_theme.style_plot_widget(self._plot_widget)
        self._plot_caption = CaptionLabel("")
        self._plot_caption.setObjectName("PreviewCaption")
        self._plot_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_card.add(preview_header)
        preview_card.add(self._plot_widget)
        preview_card.add(self._plot_caption)
        container.add_widget(preview_card)

        container.add_stretch(1)
        return container

    def _render_preview(self, project: Project, array_store: ArrayStore | None) -> None:
        """Plot the first measurement with loadable arrays (themed)."""
        self._plot_widget.clear()
        plot_theme.style_plot_widget(self._plot_widget)

        store = array_store or ArrayStore(project.root_path / ".latos" / "arrays")
        match = _find_first_plottable(project, store)
        if match is None:
            self._plot_caption.setText("No plottable arrays in this project yet.")
            return

        measurement, arrays = match
        # X/Y heuristic: ≥2 arrays → first vs second (two_theta vs
        # intensity, wavelength vs absorbance, …); 1 array → vs index.
        names = list(arrays.keys())
        if len(names) >= _MIN_ARRAYS_FOR_XY_PLOT:
            x_name, y_name = names[0], names[1]
            x = arrays[x_name]
            y = arrays[y_name]
        else:
            x_name, y_name = "index", names[0]
            y = arrays[y_name]
            x = _arange_like(y)
        self._plot_widget.setLabel("bottom", x_name)
        self._plot_widget.setLabel("left", y_name)
        self._plot_widget.plot(x, y, pen=plot_theme.accent_pen())
        self._plot_caption.setText(f"{measurement.technique.value} · {y_name} vs {x_name}")

    # ─── Slots ───────────────────────────────────────────────────────
    def _handle_open_folder_clicked(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose a project folder")
        if not chosen:
            return
        self._open_path(Path(chosen))

    def _open_path(self, path: Path) -> None:
        """Record in the recent list, refresh, and announce the open."""
        self._service.add(path)
        self.refresh()
        self.projectOpened.emit(path)


def _summary_stats(project: Project) -> dict[str, int]:
    """Counts shown on the hub: samples, measurements, distinct techniques."""
    measurements = [m for s in project.samples for m in s.measurements]
    techniques = {m.technique for m in measurements}
    return {
        "samples": len(project.samples),
        "measurements": len(measurements),
        "techniques": len(techniques),
    }


def _find_first_plottable(
    project: Project,
    store: ArrayStore,
) -> tuple[Measurement, dict[str, np.ndarray]] | None:
    """First (measurement, arrays) pair where the store has data on disk.

    `ArrayStore.load` returns `{}` for measurements that didn't write a
    Parquet file (metadata-only TIFs, etc.), so the empty-dict check is
    enough — no exception handling needed.
    """
    for sample in project.samples:
        for measurement in sample.measurements:
            arrays = store.load(measurement.id)
            if arrays:
                return measurement, arrays
    return None


def _arange_like(arr: np.ndarray) -> np.ndarray:
    """`np.arange(len(arr))` — wrapper for type-checker friendliness."""
    import numpy as np  # noqa: PLC0415  (cheap on this cold path)

    return np.arange(len(arr))
