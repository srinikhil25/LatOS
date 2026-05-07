"""`OverviewPage` — the dashboard shown after a successful ingestion.

Layout
------
- Header row: project name + a row of summary "stat cards" (samples,
  measurements, parsed / cached / failed counts).
- Sample list: one row per sample, with its measurement count.
- Preview plot: pyqtgraph `PlotWidget` showing the first measurement
  that has 1-D arrays attached. Stage 1E.5 will wire individual sample
  → measurement navigation; for 1E.4 we just show "the first thing we
  could plot" so the user gets visual confirmation that ingestion + the
  array store work end-to-end.

The page is constructed once and reused across ingestions via
`set_project(project, array_store=...)`. An empty state ("No project
open yet") is shown until the first call.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    StrongBodyLabel,
    SubtitleLabel,
    TitleLabel,
)

from latos.ingestion.array_store import ArrayStore

if TYPE_CHECKING:
    import numpy as np

    from latos.core.models import Measurement, Project

__all__ = ["OverviewPage", "StatCard"]


# Stat cards we show in the header row, in display order. The display
# label and the `IngestionResult`/`Project`-derived value live together
# so adding a new card is one tuple instead of three scattered edits.
_STAT_CARD_KEYS: tuple[str, ...] = (
    "samples",
    "measurements",
    "parsed",
    "cached",
    "failed",
)

# How many distinct 1-D arrays a measurement needs before we plot one
# against another. Below that, we plot the lone array vs. its index.
_MIN_ARRAYS_FOR_XY_PLOT = 2


class StatCard(CardWidget):  # type: ignore[misc]
    """One summary card in the Overview header.

    Displays a big number on top and a caption below. Pure presentation
    — the value and label are passed in.
    """

    def __init__(self, label: str, value: str | int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(f"StatCard_{label}")
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)

        self._value_label = SubtitleLabel(str(value), self)
        self._value_label.setObjectName("StatCardValue")
        self._caption_label = CaptionLabel(label, self)

        layout.addWidget(self._value_label)
        layout.addWidget(self._caption_label)

    def set_value(self, value: str | int) -> None:
        """Update the big number on this card."""
        self._value_label.setText(str(value))


class OverviewPage(QWidget):
    """Dashboard for the currently open project."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Object name is what `FluentWindow.addSubInterface` keys off for
        # the navigation rail; tests find the page by this name.
        self.setObjectName("OverviewPage")

        # State holders so tests / Stage 1E.5 can read what's currently
        # displayed without scraping widget text.
        self._project: Project | None = None

        # References we mutate in `set_project`.
        self._stat_cards: dict[str, StatCard] = {}
        self._title_label: TitleLabel
        self._samples_container: QWidget
        self._samples_layout: QVBoxLayout
        self._plot_widget: pg.PlotWidget
        self._plot_caption: CaptionLabel

        self._build_layout()
        self._render_empty_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def project(self) -> Project | None:
        """The currently displayed project, or `None` if empty."""
        return self._project

    def set_project(self, project: Project, *, array_store: ArrayStore | None = None) -> None:
        """Re-render the page with `project`'s data.

        Args:
            project: The newly ingested project to display.
            array_store: Optional `ArrayStore` to load measurement arrays
                from for the preview plot. `None` means "build one from
                `project.root_path/.latos/arrays`" (production default).
                Tests inject an in-memory or `tmp_path`-backed store.
        """
        self._project = project

        self._title_label.setText(project.name)

        for key, value in _stats_for_project(project).items():
            card = self._stat_cards.get(key)
            if card is not None:
                card.set_value(value)

        self._render_samples(project)
        self._render_plot(project, array_store)

    def clear(self) -> None:
        """Reset to the empty-state placeholder."""
        self._project = None
        self._render_empty_state()

    # ------------------------------------------------------------------
    # Internals — layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 32, 40, 32)
        outer.setSpacing(20)

        outer.addLayout(self._build_header())
        outer.addLayout(self._build_stats_row())
        outer.addLayout(self._build_samples_section())
        outer.addLayout(self._build_plot_section())
        outer.addStretch(1)

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)

        self._title_label = TitleLabel("Overview", self)
        header.addWidget(self._title_label)

        return header

    def _build_stats_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        for key in _STAT_CARD_KEYS:
            card = StatCard(label=key, value=0, parent=self)
            self._stat_cards[key] = card
            row.addWidget(card)

        return row

    def _build_samples_section(self) -> QVBoxLayout:
        section = QVBoxLayout()
        section.setSpacing(8)

        header = StrongBodyLabel("Samples", self)
        header.setObjectName("SamplesHeader")
        section.addWidget(header)

        self._samples_container = QWidget(self)
        self._samples_layout = QVBoxLayout(self._samples_container)
        self._samples_layout.setContentsMargins(0, 0, 0, 0)
        self._samples_layout.setSpacing(4)
        section.addWidget(self._samples_container)

        return section

    def _build_plot_section(self) -> QVBoxLayout:
        section = QVBoxLayout()
        section.setSpacing(8)

        header = StrongBodyLabel("Preview", self)
        header.setObjectName("PreviewHeader")
        section.addWidget(header)

        # `setBackground('w')` forces a white background regardless of
        # the QFluentWidgets theme so the default dark plot lines remain
        # visible. We can theme-match later (Stage 2+).
        self._plot_widget = pg.PlotWidget(self)
        self._plot_widget.setObjectName("PreviewPlot")
        self._plot_widget.setBackground("w")
        self._plot_widget.setMinimumHeight(220)
        section.addWidget(self._plot_widget)

        self._plot_caption = CaptionLabel("", self)
        self._plot_caption.setObjectName("PreviewCaption")
        self._plot_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        section.addWidget(self._plot_caption)

        return section

    # ------------------------------------------------------------------
    # Internals — render passes
    # ------------------------------------------------------------------

    def _render_empty_state(self) -> None:
        """Show the placeholder when no project is loaded."""
        self._title_label.setText("No project open yet")
        for card in self._stat_cards.values():
            card.set_value(0)
        self._clear_samples_layout()
        empty = BodyLabel("Open a project to see its overview.", self)
        empty.setObjectName("SamplesEmptyState")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._samples_layout.addWidget(empty)

        self._plot_widget.clear()
        self._plot_caption.setText("")

    def _render_samples(self, project: Project) -> None:
        self._clear_samples_layout()
        if not project.samples:
            empty = BodyLabel("No samples were detected.", self)
            empty.setObjectName("SamplesEmptyState")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._samples_layout.addWidget(empty)
            return
        for sample in project.samples:
            row = StrongBodyLabel(
                f"{sample.canonical_name}  —  {len(sample.measurements)} measurement(s)",
                self,
            )
            row.setObjectName("SampleRow")
            self._samples_layout.addWidget(row)

    def _render_plot(self, project: Project, array_store: ArrayStore | None) -> None:
        """Plot the first measurement we can find arrays for."""
        self._plot_widget.clear()

        store = array_store or _default_store_for(project.root_path)
        match = _find_first_plottable(project, store)
        if match is None:
            self._plot_caption.setText("No plottable arrays in this project yet.")
            return

        measurement, arrays = match
        # Heuristic for which columns to use as X / Y:
        # - If the measurement has at least 2 arrays, take the first as X
        #   and the second as Y. This matches XRD (two_theta vs intensity),
        #   UV-DRS (wavelength vs absorbance), etc.
        # - If only one array, plot it against its index.
        names = list(arrays.keys())
        if len(names) >= _MIN_ARRAYS_FOR_XY_PLOT:
            x_name, y_name = names[0], names[1]
            x = arrays[x_name]
            y = arrays[y_name]
            self._plot_widget.setLabel("bottom", x_name)
            self._plot_widget.setLabel("left", y_name)
        else:
            x_name = "index"
            y_name = names[0]
            y = arrays[y_name]
            x = _arange_like(y)
            self._plot_widget.setLabel("bottom", x_name)
            self._plot_widget.setLabel("left", y_name)

        # `pen='b'` keeps the trace visible on the white background. We'll
        # do proper themed colors when we tie pyqtgraph into Stage 2's
        # design tokens.
        self._plot_widget.plot(x, y, pen="b")
        self._plot_caption.setText(f"{measurement.technique.value} · {y_name} vs {x_name}")

    def _clear_samples_layout(self) -> None:
        while self._samples_layout.count():
            item = self._samples_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()


# ---------------------------------------------------------------------------
# Helpers (module-level so tests can import them directly)
# ---------------------------------------------------------------------------


def _stats_for_project(project: Project) -> dict[str, int]:
    """Counts that drive the stat-card row."""
    measurements = sum(len(s.measurements) for s in project.samples)
    parsed = measurements  # everything in `project.samples` was parsed
    cached = 0  # cached files don't add new measurements; left at 0 for now
    failed = 0  # failed files don't appear on the project; left at 0 for now
    return {
        "samples": len(project.samples),
        "measurements": measurements,
        "parsed": parsed,
        "cached": cached,
        "failed": failed,
    }


def _default_store_for(root: Path) -> ArrayStore:
    """Production default: open the array store under `<root>/.latos/arrays`."""
    return ArrayStore(root / ".latos" / "arrays")


def _find_first_plottable(
    project: Project, store: ArrayStore
) -> tuple[Measurement, dict[str, np.ndarray]] | None:
    """First (measurement, arrays) pair where the store has data on disk.

    `ArrayStore.load` returns `{}` for measurements that didn't write a
    Parquet file (metadata-only TIFs, etc.), so the empty-dict check is
    enough — we don't need to catch any exception here.
    """
    for sample in project.samples:
        for measurement in sample.measurements:
            arrays = store.load(measurement.id)
            if arrays:
                return measurement, arrays
    return None


def _arange_like(arr: np.ndarray) -> np.ndarray:
    """`np.arange(len(arr))` — kept as a wrapper for type-checker friendliness."""
    import numpy as np  # noqa: PLC0415  (cheap on this hot path)

    return np.arange(len(arr))
