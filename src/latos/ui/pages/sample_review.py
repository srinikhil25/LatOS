"""`SampleReviewPage` — sidebar entry for drilling into samples and measurements.

Layout
------
Two panes inside a `QSplitter`:

- **Left pane** (`TreeWidget`): one top-level item per sample, its
  measurements as children. Sample nodes are headers (not selectable
  for the detail panel); measurement leaves are. Each measurement node
  shows the technique + instrument so the user can pick by sight.
- **Right pane** (detail): title (sample name + technique), a metadata
  grid (instrument, measured_at, parser version, parsed_at), the list
  of contributing files, validation issues with severity dots, and a
  `pyqtgraph.PlotWidget` showing the measurement's arrays. If the
  measurement has no arrays attached, the plot collapses to a caption.

State
-----
The page owns:
- `_project`: the currently displayed `Project`.
- `_array_store`: the `ArrayStore` to load measurements from. Production
  default is `<project.root_path>/.latos/arrays`; tests inject a
  `tmp_path`-backed store.
- `_selected_measurement`: the measurement currently shown on the
  right (or `None` for the empty placeholder).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSplitter,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    StrongBodyLabel,
    SubtitleLabel,
    TreeWidget,
)

from latos.core.enums import Severity
from latos.ingestion.array_store import ArrayStore

if TYPE_CHECKING:
    from latos.core.models import Measurement, Project, Sample

__all__ = ["SampleReviewPage"]


# Tree-item user-role storage. Qt user data is keyed by an int role; we
# pick a value above `Qt.ItemDataRole.UserRole` so it doesn't clash with
# anything Qt itself sets.
_MEASUREMENT_ID_ROLE = Qt.ItemDataRole.UserRole + 1

# How many distinct 1-D arrays we need before plotting one against
# another. Mirrors the heuristic in `OverviewPage._render_plot`.
_MIN_ARRAYS_FOR_XY_PLOT = 2


# A small dot character we colorize per-severity to draw the eye to
# error/warning issues without rendering custom icons.
_SEVERITY_DOT = "●"
_SEVERITY_COLOR = {
    Severity.ERROR: "#D13438",
    Severity.WARNING: "#CA5010",
    Severity.INFO: "#0F6CBD",
}


class SampleReviewPage(QWidget):
    """Drill-down review page: tree of samples → measurements + detail panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # `findChild(SampleReviewPage, "SampleReviewPage")` is how the
        # main window and tests locate the page.
        self.setObjectName("SampleReviewPage")

        self._project: Project | None = None
        self._array_store: ArrayStore | None = None
        self._selected_measurement: Measurement | None = None

        # Forward declarations so attributes are visible to type checkers.
        self._tree: TreeWidget
        self._title_label: SubtitleLabel
        self._meta_label: BodyLabel
        self._files_label: BodyLabel
        self._issues_container: QWidget
        self._issues_layout: QVBoxLayout
        self._plot_widget: pg.PlotWidget
        self._plot_caption: CaptionLabel

        self._build_layout()
        self._render_empty_detail()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def project(self) -> Project | None:
        """The currently displayed project, or `None`."""
        return self._project

    @property
    def selected_measurement(self) -> Measurement | None:
        """The measurement currently shown on the right pane, or `None`."""
        return self._selected_measurement

    def set_project(self, project: Project, *, array_store: ArrayStore | None = None) -> None:
        """Re-populate the tree from `project` and reset the detail pane."""
        self._project = project
        self._array_store = array_store or ArrayStore(project.root_path / ".latos" / "arrays")
        self._populate_tree(project)
        self._selected_measurement = None
        self._render_empty_detail()

    def clear(self) -> None:
        """Reset to the empty placeholder."""
        self._project = None
        self._array_store = None
        self._selected_measurement = None
        self._tree.clear()
        self._render_empty_detail()

    # ------------------------------------------------------------------
    # Internals — layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("SampleReviewSplitter")
        splitter.addWidget(self._build_tree_pane())
        splitter.addWidget(self._build_detail_pane())
        # Roughly 1:2 split — detail pane gets more room.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter)

    def _build_tree_pane(self) -> QWidget:
        wrapper = QWidget(self)
        col = QVBoxLayout(wrapper)
        col.setContentsMargins(0, 0, 12, 0)
        col.setSpacing(8)

        header = StrongBodyLabel("Samples", wrapper)
        header.setObjectName("SampleReviewTreeHeader")
        col.addWidget(header)

        self._tree = TreeWidget(wrapper)
        self._tree.setObjectName("SampleReviewTree")
        self._tree.setHeaderHidden(True)
        # Selection drives the right-pane render. We listen on
        # `currentItemChanged` rather than `itemClicked` so keyboard
        # navigation also works.
        self._tree.currentItemChanged.connect(self._on_tree_selection_changed)
        col.addWidget(self._tree, 1)

        return wrapper

    def _build_detail_pane(self) -> QWidget:
        wrapper = QWidget(self)
        col = QVBoxLayout(wrapper)
        col.setContentsMargins(12, 0, 0, 0)
        col.setSpacing(12)

        self._title_label = SubtitleLabel("", wrapper)
        self._title_label.setObjectName("SampleReviewTitle")
        col.addWidget(self._title_label)

        self._meta_label = BodyLabel("", wrapper)
        self._meta_label.setObjectName("SampleReviewMeta")
        self._meta_label.setWordWrap(True)
        col.addWidget(self._meta_label)

        files_header = StrongBodyLabel("Files", wrapper)
        files_header.setObjectName("SampleReviewFilesHeader")
        col.addWidget(files_header)
        self._files_label = BodyLabel("", wrapper)
        self._files_label.setObjectName("SampleReviewFiles")
        self._files_label.setWordWrap(True)
        col.addWidget(self._files_label)

        issues_header = StrongBodyLabel("Issues", wrapper)
        issues_header.setObjectName("SampleReviewIssuesHeader")
        col.addWidget(issues_header)
        self._issues_container = QWidget(wrapper)
        self._issues_layout = QVBoxLayout(self._issues_container)
        self._issues_layout.setContentsMargins(0, 0, 0, 0)
        self._issues_layout.setSpacing(2)
        col.addWidget(self._issues_container)

        plot_header = StrongBodyLabel("Arrays", wrapper)
        plot_header.setObjectName("SampleReviewPlotHeader")
        col.addWidget(plot_header)
        self._plot_widget = pg.PlotWidget(wrapper)
        self._plot_widget.setObjectName("SampleReviewPlot")
        self._plot_widget.setBackground("w")
        self._plot_widget.setMinimumHeight(220)
        col.addWidget(self._plot_widget, 1)
        self._plot_caption = CaptionLabel("", wrapper)
        self._plot_caption.setObjectName("SampleReviewPlotCaption")
        self._plot_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self._plot_caption)

        return wrapper

    # ------------------------------------------------------------------
    # Internals — tree population / selection
    # ------------------------------------------------------------------

    def _populate_tree(self, project: Project) -> None:
        self._tree.clear()
        for sample in project.samples:
            sample_node = QTreeWidgetItem([sample.canonical_name])
            # Sample nodes carry no measurement id — selecting one shows
            # the empty detail placeholder.
            sample_node.setData(0, _MEASUREMENT_ID_ROLE, None)
            for measurement in sample.measurements:
                label = (
                    f"{measurement.technique.value} · "
                    f"{measurement.instrument or 'unknown instrument'}"
                )
                child = QTreeWidgetItem([label])
                child.setData(0, _MEASUREMENT_ID_ROLE, measurement.id)
                sample_node.addChild(child)
            self._tree.addTopLevelItem(sample_node)
            sample_node.setExpanded(True)

    def _on_tree_selection_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            self._selected_measurement = None
            self._render_empty_detail()
            return

        measurement_id = current.data(0, _MEASUREMENT_ID_ROLE)
        if not measurement_id:
            # Sample node — empty detail until the user expands and picks.
            self._selected_measurement = None
            self._render_empty_detail()
            return

        measurement = self._lookup_measurement(measurement_id)
        if measurement is None:
            # The tree got out of sync with the project — bail to the
            # empty state rather than crashing.
            self._selected_measurement = None
            self._render_empty_detail()
            return
        self._selected_measurement = measurement
        self._render_detail(measurement)

    def _lookup_measurement(self, measurement_id: str) -> Measurement | None:
        if self._project is None:
            return None
        for sample in self._project.samples:
            for m in sample.measurements:
                if m.id == measurement_id:
                    return m
        return None

    # ------------------------------------------------------------------
    # Internals — detail render passes
    # ------------------------------------------------------------------

    def _render_empty_detail(self) -> None:
        self._title_label.setText("Select a measurement")
        self._meta_label.setText("Pick a measurement on the left to see its details.")
        self._files_label.setText("")
        self._clear_issues()
        self._plot_widget.clear()
        self._plot_caption.setText("")

    def _render_detail(self, measurement: Measurement) -> None:
        sample = self._sample_for(measurement)
        sample_name = sample.canonical_name if sample is not None else "?"
        self._title_label.setText(f"{sample_name} · {measurement.technique.value}")

        # Metadata block — multi-line plain text. We could promote this
        # to a grid widget later; one BodyLabel is enough for 1E.5 and
        # keeps the test surface small.
        meta_lines = [
            f"Instrument: {measurement.instrument or 'unknown'}",
            f"Measured at: {measurement.measured_at or 'unknown'}",
            f"Parser version: {measurement.parser_version}",
            f"Parsed at: {measurement.parsed_at}",
        ]
        self._meta_label.setText("\n".join(meta_lines))

        files_lines = [
            f"• {f.path.name} ({f.role.value}, {f.size_bytes} bytes)" for f in measurement.files
        ] or ["(no files)"]
        self._files_label.setText("\n".join(files_lines))

        self._render_issues(measurement)
        self._render_plot(measurement)

    def _render_issues(self, measurement: Measurement) -> None:
        self._clear_issues()
        if not measurement.issues:
            empty = BodyLabel("No issues recorded.", self._issues_container)
            empty.setObjectName("SampleReviewIssuesEmpty")
            self._issues_layout.addWidget(empty)
            return
        for issue in measurement.issues:
            color = _SEVERITY_COLOR.get(issue.severity, "#888888")
            label = BodyLabel(
                f"{_SEVERITY_DOT}  {issue.severity.value.upper()} · {issue.field}: {issue.message}",
                self._issues_container,
            )
            label.setObjectName(f"SampleReviewIssue_{issue.severity.value}")
            label.setStyleSheet(f"color: {color};")
            label.setWordWrap(True)
            self._issues_layout.addWidget(label)

    def _render_plot(self, measurement: Measurement) -> None:
        self._plot_widget.clear()
        if self._array_store is None:
            self._plot_caption.setText("No array store available.")
            return

        arrays = self._array_store.load(measurement.id)
        if not arrays:
            self._plot_caption.setText("This measurement has no plottable arrays.")
            return

        names = list(arrays.keys())
        if len(names) >= _MIN_ARRAYS_FOR_XY_PLOT:
            x_name, y_name = names[0], names[1]
            x = arrays[x_name]
            y = arrays[y_name]
            self._plot_widget.setLabel("bottom", x_name)
            self._plot_widget.setLabel("left", y_name)
            self._plot_widget.plot(x, y, pen="b")
            self._plot_caption.setText(f"{y_name} vs {x_name}")
        else:
            import numpy as np  # noqa: PLC0415  (cheap on this hot path)

            y_name = names[0]
            y = arrays[y_name]
            x = np.arange(len(y))
            self._plot_widget.setLabel("bottom", "index")
            self._plot_widget.setLabel("left", y_name)
            self._plot_widget.plot(x, y, pen="b")
            self._plot_caption.setText(f"{y_name} vs index")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_for(self, measurement: Measurement) -> Sample | None:
        if self._project is None:
            return None
        for sample in self._project.samples:
            if any(m.id == measurement.id for m in sample.measurements):
                return sample
        return None

    def _clear_issues(self) -> None:
        while self._issues_layout.count():
            item = self._issues_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()


def _default_store_for(root: Path) -> ArrayStore:
    """Production default array store for a project root."""
    return ArrayStore(root / ".latos" / "arrays")
