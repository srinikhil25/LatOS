"""`AnalysisPage` — Stage 3C UI surface for running analyzers on measurements.

Layout (mirrors `SampleReviewPage`):

- **Left pane** (`TreeWidget`): samples → measurements, but filtered to
  measurements that at least one registered analyzer accepts. A
  measurement with no applicable analyzer is hidden — there's nothing
  the user could do with it on this page.
- **Right pane**:
  1. Measurement header (sample · technique · instrument).
  2. **Analyzer selector** — combo of analyzers that accept this
     measurement.
  3. **Parameter form** — auto-generated widgets from
     `analyzer.default_params` (bool → CheckBox, int → SpinBox, float
     → DoubleSpinBox, str → LineEdit). The user can override any
     default before running.
  4. **Run / Re-run buttons** — Run uses the cache (no-op on hit),
     Re-run forces a fresh compute (`force=True`).
  5. **Results list** — every `AnalysisResult` already on this
     measurement (most recent first).
  6. **Outputs panel + plot** — when the user picks a result, show the
     scalar outputs as key/value rows and plot the derived arrays.
     For UV-DRS Tauc, that's (photon_energy_ev, tauc_y) with the
     linear fit overlaid and a vertical line at the extracted band
     gap.
  7. **Issues** — analyzer-emitted `ValidationIssue`s, severity-colored.

State
-----
- `_project`: the current `Project`.
- `_array_store`: where parsed + derived arrays live (one per project).
- `_service`: the `AnalysisService` instance.
- `_registry`: the `AnalyzerRegistry` discovered analyzers.
- `_selected_measurement`: the measurement on the right.
- `_selected_result`: the AnalysisResult currently shown in the plot.

The page does NOT own the service or registry — they're injected via
`bind_runtime()` after construction so tests can inject stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSpinBox,
    QSplitter,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
    SubtitleLabel,
    TreeWidget,
)

from latos.core.enums import Severity
from latos.core.exceptions import AnalysisError

if TYPE_CHECKING:
    import numpy as np

    from latos.analysis import AnalysisService, AnalyzerRegistry, BaseAnalyzer
    from latos.core.models import AnalysisResult, Measurement, Project, Sample
    from latos.ingestion.array_store import ArrayStore

__all__ = ["AnalysisPage"]


# Tree-item user-role storage for the measurement id behind each leaf.
_MEASUREMENT_ID_ROLE = Qt.ItemDataRole.UserRole + 1

# Severity coloring — same palette as SampleReviewPage so the app feels
# consistent across pages.
_SEVERITY_DOT = "●"
_SEVERITY_COLOR = {
    Severity.ERROR: "#D13438",
    Severity.WARNING: "#CA5010",
    Severity.INFO: "#0F6CBD",
}

# Plot styling kept in sync with SampleReviewPage.
_PLOT_PEN = {"color": "#0F6CBD", "width": 2}
_FIT_PEN = {"color": "#CA5010", "width": 2, "style": Qt.PenStyle.DashLine}
_BAND_GAP_PEN = {"color": "#107C10", "width": 1, "style": Qt.PenStyle.DotLine}

# Bounds for spin-box widgets generated from `default_params`. Generous
# enough to fit any sensible analyzer parameter; we don't try to infer
# per-parameter ranges (the analyzer's own validation does that).
_FLOAT_MIN = -1.0e9
_FLOAT_MAX = 1.0e9
_FLOAT_DECIMALS = 6
_INT_MIN = -1_000_000_000
_INT_MAX = 1_000_000_000

# Minimum number of derived arrays required before we'll plot one
# against another (x vs y). Below this we fall back to plotting the
# single array against its index.
_MIN_DERIVED_ARRAYS_FOR_XY_PLOT = 2


class AnalysisPage(QWidget):
    """Run analyzers on measurements and visualize results."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AnalysisPage")

        self._project: Project | None = None
        self._array_store: ArrayStore | None = None
        self._service: AnalysisService | None = None
        self._registry: AnalyzerRegistry | None = None

        self._selected_measurement: Measurement | None = None
        self._selected_result: AnalysisResult | None = None
        # Widgets built by `_build_param_form` — flushed each time the
        # analyzer combo selection changes.
        self._param_widgets: dict[str, QWidget] = {}

        # Forward declarations for type checkers.
        self._tree: TreeWidget
        self._title_label: SubtitleLabel
        self._meta_label: BodyLabel
        self._analyzer_combo: ComboBox
        self._param_form: QFormLayout
        self._param_container: QWidget
        self._run_button: PrimaryPushButton
        self._rerun_button: PushButton
        self._results_list: QListWidget
        self._outputs_label: BodyLabel
        self._plot_widget: pg.PlotWidget
        self._plot_caption: CaptionLabel
        self._issues_container: QWidget
        self._issues_layout: QVBoxLayout

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

    @property
    def selected_result(self) -> AnalysisResult | None:
        """The analysis result currently plotted, or `None`."""
        return self._selected_result

    def bind_runtime(
        self,
        *,
        service: AnalysisService,
        registry: AnalyzerRegistry,
        array_store: ArrayStore,
    ) -> None:
        """Inject the runtime dependencies. Call once per opened project.

        The page can be constructed before a project is open; the
        runtime is bound when the project actually loads. Tests inject
        stubs here.
        """
        self._service = service
        self._registry = registry
        self._array_store = array_store

    def set_project(self, project: Project) -> None:
        """Populate the tree from `project` and reset the detail pane."""
        self._project = project
        self._populate_tree(project)
        self._selected_measurement = None
        self._selected_result = None
        self._render_empty_detail()

    def clear(self) -> None:
        """Reset to the empty state."""
        self._project = None
        self._selected_measurement = None
        self._selected_result = None
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
        splitter.setObjectName("AnalysisSplitter")
        splitter.addWidget(self._build_tree_pane())
        splitter.addWidget(self._build_detail_pane())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter)

    def _build_tree_pane(self) -> QWidget:
        wrapper = QWidget(self)
        col = QVBoxLayout(wrapper)
        col.setContentsMargins(0, 0, 12, 0)
        col.setSpacing(8)

        header = StrongBodyLabel("Measurements", wrapper)
        header.setObjectName("AnalysisTreeHeader")
        col.addWidget(header)

        self._tree = TreeWidget(wrapper)
        self._tree.setObjectName("AnalysisTree")
        self._tree.setHeaderHidden(True)
        self._tree.currentItemChanged.connect(self._on_tree_selection_changed)
        col.addWidget(self._tree, 1)

        return wrapper

    def _build_detail_pane(self) -> QWidget:  # noqa: PLR0915
        # Long by design: the detail pane stacks seven labelled sections
        # (header, analyzer combo, param form, buttons, results list,
        # outputs, plot, issues). Splitting into helpers fragments the
        # widget-attribute bindings each section needs to register on
        # self for later `_render_*` passes.
        wrapper = QWidget(self)
        col = QVBoxLayout(wrapper)
        col.setContentsMargins(12, 0, 0, 0)
        col.setSpacing(10)

        self._title_label = SubtitleLabel("", wrapper)
        self._title_label.setObjectName("AnalysisTitle")
        col.addWidget(self._title_label)

        self._meta_label = BodyLabel("", wrapper)
        self._meta_label.setObjectName("AnalysisMeta")
        self._meta_label.setWordWrap(True)
        col.addWidget(self._meta_label)

        # ─── Analyzer + params ──────────────────────────────────────
        col.addWidget(StrongBodyLabel("Analyzer", wrapper))
        self._analyzer_combo = ComboBox(wrapper)
        self._analyzer_combo.setObjectName("AnalysisAnalyzerCombo")
        self._analyzer_combo.currentIndexChanged.connect(self._on_analyzer_changed)
        col.addWidget(self._analyzer_combo)

        self._param_container = QWidget(wrapper)
        self._param_container.setObjectName("AnalysisParamContainer")
        self._param_form = QFormLayout(self._param_container)
        self._param_form.setContentsMargins(0, 0, 0, 0)
        self._param_form.setSpacing(6)
        col.addWidget(self._param_container)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self._run_button = PrimaryPushButton("Run analysis", wrapper)
        self._run_button.setObjectName("AnalysisRunButton")
        self._run_button.clicked.connect(self._on_run_clicked)
        button_row.addWidget(self._run_button)
        self._rerun_button = PushButton("Re-run (force)", wrapper)
        self._rerun_button.setObjectName("AnalysisRerunButton")
        self._rerun_button.clicked.connect(self._on_rerun_clicked)
        button_row.addWidget(self._rerun_button)
        button_row.addStretch(1)
        col.addLayout(button_row)

        # ─── Results list ───────────────────────────────────────────
        col.addWidget(StrongBodyLabel("Results", wrapper))
        self._results_list = QListWidget(wrapper)
        self._results_list.setObjectName("AnalysisResultsList")
        self._results_list.currentItemChanged.connect(self._on_result_selection_changed)
        self._results_list.setMaximumHeight(120)
        col.addWidget(self._results_list)

        # ─── Outputs + plot ─────────────────────────────────────────
        col.addWidget(StrongBodyLabel("Outputs", wrapper))
        self._outputs_label = BodyLabel("", wrapper)
        self._outputs_label.setObjectName("AnalysisOutputs")
        self._outputs_label.setWordWrap(True)
        col.addWidget(self._outputs_label)

        col.addWidget(StrongBodyLabel("Plot", wrapper))
        self._plot_widget = pg.PlotWidget(wrapper)
        self._plot_widget.setObjectName("AnalysisPlot")
        self._plot_widget.setBackground("w")
        self._plot_widget.setMinimumHeight(220)
        col.addWidget(self._plot_widget, 1)
        self._plot_caption = CaptionLabel("", wrapper)
        self._plot_caption.setObjectName("AnalysisPlotCaption")
        self._plot_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self._plot_caption)

        # ─── Issues ─────────────────────────────────────────────────
        col.addWidget(StrongBodyLabel("Issues", wrapper))
        self._issues_container = QWidget(wrapper)
        self._issues_layout = QVBoxLayout(self._issues_container)
        self._issues_layout.setContentsMargins(0, 0, 0, 0)
        self._issues_layout.setSpacing(2)
        col.addWidget(self._issues_container)

        return wrapper

    # ------------------------------------------------------------------
    # Tree population / selection
    # ------------------------------------------------------------------

    def _populate_tree(self, project: Project) -> None:
        """Fill the tree with samples → analyzable measurements only.

        Filtering at populate-time (rather than greying-out non-
        analyzable rows) keeps the tree visually tidy. The user can
        cross-reference SampleReviewPage for the full measurement set.
        """
        self._tree.clear()
        for sample in project.samples:
            analyzable = [m for m in sample.measurements if self._is_analyzable(m)]
            if not analyzable:
                continue
            sample_node = QTreeWidgetItem([sample.canonical_name])
            sample_node.setData(0, _MEASUREMENT_ID_ROLE, None)
            for measurement in analyzable:
                label = (
                    f"{measurement.technique.value} · "
                    f"{measurement.instrument or 'unknown instrument'}"
                )
                child = QTreeWidgetItem([label])
                child.setData(0, _MEASUREMENT_ID_ROLE, measurement.id)
                sample_node.addChild(child)
            self._tree.addTopLevelItem(sample_node)
            sample_node.setExpanded(True)

    def _is_analyzable(self, measurement: Measurement) -> bool:
        if self._registry is None:
            return False
        return bool(self._registry.find_for(measurement))

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
            self._selected_measurement = None
            self._render_empty_detail()
            return
        measurement = self._lookup_measurement(measurement_id)
        if measurement is None:
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
    # Detail render — top half (analyzer combo + param form)
    # ------------------------------------------------------------------

    def _render_empty_detail(self) -> None:
        self._title_label.setText("Select a measurement")
        self._meta_label.setText("Pick a measurement on the left to choose an analyzer and run it.")
        # Block the analyzer-combo signal while clearing so we don't
        # fire `_on_analyzer_changed` with index=-1 in the middle of a
        # render and re-enter the empty path.
        self._analyzer_combo.blockSignals(True)
        self._analyzer_combo.clear()
        self._analyzer_combo.blockSignals(False)
        self._clear_param_form()
        self._results_list.clear()
        self._outputs_label.setText("")
        self._plot_widget.clear()
        self._plot_caption.setText("")
        self._clear_issues()
        self._run_button.setEnabled(False)
        self._rerun_button.setEnabled(False)

    def _render_detail(self, measurement: Measurement) -> None:
        sample = self._sample_for(measurement)
        sample_name = sample.canonical_name if sample is not None else "?"
        self._title_label.setText(f"{sample_name} · {measurement.technique.value}")
        meta_lines = [
            f"Instrument: {measurement.instrument or 'unknown'}",
            f"Files: {len(measurement.files)}",
            f"Parser: {measurement.parser_version}",
        ]
        self._meta_label.setText(" · ".join(meta_lines))

        # Populate the analyzer combo with the registry's matches.
        # `blockSignals` prevents an interim selection change from
        # triggering a half-built render.
        self._analyzer_combo.blockSignals(True)
        self._analyzer_combo.clear()
        analyzers = self._registry.find_for(measurement) if self._registry else ()
        for analyzer in analyzers:
            self._analyzer_combo.addItem(f"{analyzer.name} ({analyzer.version})")
        self._analyzer_combo.blockSignals(False)
        if analyzers:
            self._analyzer_combo.setCurrentIndex(0)
            # Force a re-render of the param form (setCurrentIndex(0)
            # is a no-op if the combo was already at 0).
            self._render_param_form(analyzers[0])
        else:
            self._clear_param_form()

        self._populate_results_list(measurement)
        # No selected result yet — empty the plot / outputs / issues
        # blocks until the user picks one (or runs).
        self._selected_result = None
        self._outputs_label.setText("")
        self._plot_widget.clear()
        self._plot_caption.setText("")
        self._clear_issues()

        self._run_button.setEnabled(bool(analyzers))
        self._rerun_button.setEnabled(bool(analyzers))

    def _on_analyzer_changed(self, index: int) -> None:
        if index < 0 or self._selected_measurement is None or self._registry is None:
            return
        analyzers = self._registry.find_for(self._selected_measurement)
        if index >= len(analyzers):
            return
        self._render_param_form(analyzers[index])

    # ------------------------------------------------------------------
    # Parameter form (auto-generated from default_params)
    # ------------------------------------------------------------------

    def _render_param_form(self, analyzer: BaseAnalyzer) -> None:
        """Build the parameter form for `analyzer`.

        Type dispatch: bool → CheckBox, int → SpinBox, float →
        DoubleSpinBox, anything else → LineEdit. The same defaults the
        analyzer declares are pre-filled; the user can override any
        of them before clicking Run.
        """
        self._clear_param_form()
        for key, default in analyzer.default_params.items():
            widget = self._make_param_widget(default)
            self._param_widgets[key] = widget
            self._param_form.addRow(BodyLabel(key, self._param_container), widget)

    def _make_param_widget(self, default: Any) -> QWidget:
        # bool MUST be checked before int — bool is a subclass of int
        # in Python, so the isinstance(int) branch would catch True/False
        # first and you'd get a SpinBox instead of a CheckBox.
        if isinstance(default, bool):
            w = QCheckBox(self._param_container)
            w.setChecked(default)
            return w
        if isinstance(default, int):
            sb = QSpinBox(self._param_container)
            sb.setRange(_INT_MIN, _INT_MAX)
            sb.setValue(default)
            return sb
        if isinstance(default, float):
            dsb = QDoubleSpinBox(self._param_container)
            dsb.setDecimals(_FLOAT_DECIMALS)
            dsb.setRange(_FLOAT_MIN, _FLOAT_MAX)
            dsb.setValue(default)
            return dsb
        # Fallback: LineEdit holds the str() of whatever the default is.
        # Suits the Tauc analyzer's "direct"/"indirect" string param and
        # any future enum-like text params. Lists/dicts get a JSON-ish
        # rendering the user can edit at their own risk.
        le = QLineEdit(self._param_container)
        le.setText(str(default))
        return le

    def _collect_params(self) -> dict[str, Any]:
        """Read the param-form widgets back into a JSON-safe dict."""
        out: dict[str, Any] = {}
        for key, widget in self._param_widgets.items():
            if isinstance(widget, QCheckBox):
                out[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                out[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                out[key] = float(widget.value())
            elif isinstance(widget, QLineEdit):
                out[key] = widget.text()
            else:
                # Defensive: a future widget type slipped through.
                # Mypy will scream if the type isn't covered, but
                # runtime gets a string fallback.
                out[key] = str(widget)
        return out

    def _clear_param_form(self) -> None:
        # Remove every existing row. `removeRow(0)` doesn't exist on
        # QFormLayout in older Qt; we manually pop fields instead.
        while self._param_form.rowCount() > 0:
            self._param_form.removeRow(0)
        self._param_widgets.clear()

    # ------------------------------------------------------------------
    # Run / Re-run
    # ------------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        self._run_analyzer(force=False)

    def _on_rerun_clicked(self) -> None:
        self._run_analyzer(force=True)

    def _run_analyzer(self, *, force: bool) -> None:
        if self._service is None or self._registry is None or self._selected_measurement is None:
            return
        analyzers = self._registry.find_for(self._selected_measurement)
        idx = self._analyzer_combo.currentIndex()
        if idx < 0 or idx >= len(analyzers):
            return
        analyzer = analyzers[idx]
        params = self._collect_params()

        try:
            outcome = self._service.run(
                analyzer,
                self._selected_measurement,
                params=params,
                force=force,
            )
        except AnalysisError as exc:
            # Surface as a modal dialog rather than a silent log. An
            # AnalysisError here means the analyzer rejected the
            # measurement (wrong technique / accepts() returned False) —
            # always a user-actionable surprise.
            QMessageBox.warning(self, "Analysis failed", str(exc))
            return

        # Re-fetch the project so its in-memory copy reflects the
        # freshly-persisted AnalysisResult. The service writes via
        # ProjectRepository.save which replaces the whole aggregate;
        # holding our old `_project` would mean the results list is
        # one step behind.
        self._refresh_project_from_service()
        # Find the new measurement object inside the reloaded project
        # and re-render the right pane around it.
        if self._selected_measurement is not None:
            reloaded = self._lookup_measurement(self._selected_measurement.id)
            if reloaded is not None:
                self._selected_measurement = reloaded
                self._populate_results_list(reloaded)
                # Auto-select the result we just produced so the user
                # sees the plot without an extra click.
                self._select_result_in_list(outcome.result.id)

    def _refresh_project_from_service(self) -> None:
        """Reload the project from the repository to pick up new results.

        We don't have a direct handle to the repository, but the
        service does — and we use its `_repo.load_first()` indirectly.
        Tests can monkey-patch this if they want to keep things
        purely in-memory.
        """
        if self._service is None or self._project is None:
            return
        # Reach into the service for the repository. The single-project
        # invariant (one DB per project) means `load_first()` is the
        # right call.
        loaded = self._service._repo.load_first()
        if loaded is not None:
            self._project = loaded

    # ------------------------------------------------------------------
    # Results list + selected result render
    # ------------------------------------------------------------------

    def _populate_results_list(self, measurement: Measurement) -> None:
        self._results_list.clear()
        # Most-recent first.
        sorted_results = sorted(
            measurement.analysis_results,
            key=lambda r: r.computed_at,
            reverse=True,
        )
        for result in sorted_results:
            stamp = result.computed_at.strftime("%Y-%m-%d %H:%M:%S")
            n_issues = len(result.issues)
            if n_issues == 0:
                issue_tag = ""
            else:
                plural = "s" if n_issues > 1 else ""
                issue_tag = f"  ({n_issues} issue{plural})"
            label = f"{result.analyzer_name} v{result.analyzer_version}  ·  {stamp}{issue_tag}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole + 1, result.id)
            self._results_list.addItem(item)

    def _select_result_in_list(self, result_id: str) -> None:
        for i in range(self._results_list.count()):
            item = self._results_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole + 1) == result_id:
                self._results_list.setCurrentRow(i)
                return

    def _on_result_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None or self._selected_measurement is None:
            self._selected_result = None
            self._plot_widget.clear()
            self._plot_caption.setText("")
            self._outputs_label.setText("")
            self._clear_issues()
            return
        result_id = current.data(Qt.ItemDataRole.UserRole + 1)
        result = next(
            (r for r in self._selected_measurement.analysis_results if r.id == result_id),
            None,
        )
        if result is None:
            return
        self._selected_result = result
        self._render_result_outputs(result)
        self._render_result_plot(result)
        self._render_result_issues(result)

    def _render_result_outputs(self, result: AnalysisResult) -> None:
        lines = [f"{k}: {v}" for k, v in result.outputs.items()]
        if not lines:
            lines = ["(no scalar outputs)"]
        self._outputs_label.setText("\n".join(lines))

    def _render_result_plot(self, result: AnalysisResult) -> None:
        """Plot the result's derived arrays.

        Generic policy: pick the first array as x, the second as y. For
        UV-DRS Tauc, that puts photon energy on the x-axis and
        Kubelka-Munk on the y-axis by default — same convention the
        underlying analyzer chose when populating `derived_arrays`.

        Special case: if the result carries a `fit_line` array (Tauc's
        linear-fit overlay) and a `band_gap_ev` scalar output, overlay
        them on the same axes — that's the headline visualization the
        whole stage exists to produce.
        """
        self._plot_widget.clear()
        arrays = self._load_derived_arrays(result)
        if not arrays:
            self._plot_caption.setText("No derived arrays for this result.")
            return
        names = list(arrays.keys())
        x_name = names[0]
        # Prefer `tauc_y` if present (the Tauc analyzer's canonical
        # y-axis), otherwise the second array (if any), otherwise the
        # only array (degenerate but defensive).
        if "tauc_y" in arrays:
            y_name = "tauc_y"
        elif len(names) >= _MIN_DERIVED_ARRAYS_FOR_XY_PLOT:
            y_name = names[1]
        else:
            y_name = names[0]
        x = arrays[x_name]
        y = arrays[y_name]
        self._plot_widget.setLabel("bottom", x_name)
        self._plot_widget.setLabel("left", y_name)
        self._plot_widget.plot(x, y, pen=_PLOT_PEN, name=y_name)

        # Tauc-specific overlays. Cheap-and-defensive: only fire when
        # both the array and the scalar are present.
        if "fit_line" in arrays:
            self._plot_widget.plot(x, arrays["fit_line"], pen=_FIT_PEN, name="fit")
        band_gap = result.outputs.get("band_gap_ev")
        if isinstance(band_gap, int | float):
            line = pg.InfiniteLine(pos=float(band_gap), angle=90, pen=_BAND_GAP_PEN)
            self._plot_widget.addItem(line)
            self._plot_caption.setText(f"Eg = {float(band_gap):.3f} eV")
        else:
            self._plot_caption.setText(f"{y_name} vs {x_name}")

    def _load_derived_arrays(self, result: AnalysisResult) -> dict[str, np.ndarray]:
        """Read the derived-arrays Parquet sidecar.

        Returns an empty dict if the result has no sidecar (scalar-only
        analyzer) or if the file is unreadable for any reason. We
        don't crash the UI on a missing file — show the empty-plot
        caption and let the user re-run if they care.
        """
        if result.derived_arrays_path is None:
            return {}
        path = Path(result.derived_arrays_path)
        if not path.is_file():
            return {}
        try:
            import numpy as np  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415

            table = pq.read_table(path)  # type: ignore[no-untyped-call]
            return {
                name: np.asarray(table.column(name).to_numpy(zero_copy_only=False))
                for name in table.column_names
            }
        except Exception:
            # Parquet read can fail many ways (missing file, corrupt
            # footer, schema mismatch). We catch them all and surface
            # an empty-plot caption rather than crashing the UI; the
            # user can re-run the analysis to regenerate.
            return {}

    def _render_result_issues(self, result: AnalysisResult) -> None:
        self._clear_issues()
        if not result.issues:
            empty = BodyLabel("No issues from this run.", self._issues_container)
            empty.setObjectName("AnalysisIssuesEmpty")
            self._issues_layout.addWidget(empty)
            return
        for issue in result.issues:
            color = _SEVERITY_COLOR.get(issue.severity, "#888888")
            label = BodyLabel(
                f"{_SEVERITY_DOT}  {issue.severity.value.upper()} · {issue.field}: {issue.message}",
                self._issues_container,
            )
            label.setObjectName(f"AnalysisIssue_{issue.severity.value}")
            label.setStyleSheet(f"color: {color};")
            label.setWordWrap(True)
            self._issues_layout.addWidget(label)

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
