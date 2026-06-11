"""Tests for `latos.ui.pages.analysis.AnalysisPage`.

The page is tested with stub analyzers + a real `AnalysisService` over
an in-memory SQLite DB + a `tmp_path`-backed `ArrayStore`. UI tests
never touch real parser/instrument files; the project is seeded
directly via `ProjectRepository.save()` and arrays via Parquet writes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QLineEdit, QSpinBox

from latos.analysis import (
    AnalysisService,
    AnalyzerInputs,
    AnalyzerOutput,
    AnalyzerRegistry,
    BaseAnalyzer,
)
from latos.core.enums import FileRole, Severity, Technique
from latos.core.models import (
    FileRef,
    Measurement,
    Project,
    Sample,
    ValidationIssue,
)
from latos.ingestion.array_store import ArrayStore
from latos.persistence.db import (
    create_memory_engine,
    init_schema,
    make_session_factory,
)
from latos.persistence.repository import ProjectRepository
from latos.ui.pages.analysis import AnalysisPage

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.ui


# ---------------------------------------------------------------------------
# Builders (mirror the SampleReviewPage test conventions)
# ---------------------------------------------------------------------------


def _id(seed: int) -> str:
    return f"{seed:032x}"


def _fileref(name: str = "f.txt") -> FileRef:
    return FileRef(
        path=Path("/stub") / name,
        sha256="0" * 64,
        size_bytes=1,
        role=FileRole.RAW,
        scanned_at=datetime.now(UTC),
    )


def _measurement(
    seed: int,
    sample_id: str,
    *,
    technique: Technique = Technique.UV_DRS,
    instrument: str = "UV-DRS Test",
) -> Measurement:
    return Measurement(
        id=_id(seed),
        sample_id=sample_id,
        technique=technique,
        instrument=instrument,
        measured_at=None,
        parsed_at=datetime.now(UTC),
        parser_version="1.0.0",
        files=(_fileref(f"f{seed}.txt"),),
    )


def _project_with_one_uvdrs_measurement(
    seed: int = 1,
) -> tuple[Project, Sample, Measurement]:
    sample_id = _id(seed * 100)
    measurement = _measurement(seed, sample_id, technique=Technique.UV_DRS)
    sample = Sample(
        id=sample_id,
        project_id=_id(seed),
        canonical_name=f"S{seed}",
        measurements=(measurement,),
    )
    project = Project(
        id=_id(seed),
        name="TestProject",
        root_path=Path("/stub/project"),
        created_at=datetime.now(UTC),
        schema_version=3,
        samples=(sample,),
    )
    return project, sample, measurement


# ---------------------------------------------------------------------------
# Stub analyzers
# ---------------------------------------------------------------------------


class _DoublingAnalyzer(BaseAnalyzer):
    """Multiplies a single 1-D input array by 2; reports the mean.

    Mirrors the doubler used in `test_service.py` so the UI tests
    exercise the same surface analysis flow without depending on the
    Tauc analyzer's array-shape expectations.
    """

    name: ClassVar[str] = "doubler"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)
    default_params: ClassVar[dict[str, Any]] = {
        "multiplier": 2.0,
        "label": "default",
    }

    def accepts(self, measurement: Measurement) -> bool:
        return True

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        multiplier = float(inputs.params.get("multiplier", 2.0))
        x = inputs.arrays.get("x", np.array([], dtype=np.float64))
        scaled = x * multiplier
        return AnalyzerOutput(
            outputs={"mean": float(np.mean(scaled)) if scaled.size else 0.0},
            derived_arrays={"x": x, "scaled": scaled},
            issues=(
                ValidationIssue(
                    field="multiplier",
                    severity=Severity.INFO,
                    message=f"used multiplier={multiplier}",
                    detected_at=datetime.now(UTC),
                ),
            ),
        )


class _XrdOnlyAnalyzer(BaseAnalyzer):
    """Accepts only XRD measurements — used to verify the tree filter."""

    name: ClassVar[str] = "xrd-stub"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.XRD,)

    def accepts(self, measurement: Measurement) -> bool:
        return True

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        return AnalyzerOutput()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():  # type: ignore[no-untyped-def]
    eng = create_memory_engine()
    init_schema(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def repo(engine):  # type: ignore[no-untyped-def]
    return ProjectRepository(make_session_factory(engine))


@pytest.fixture
def array_store(tmp_path: Path) -> ArrayStore:
    return ArrayStore(tmp_path / "arrays")


@pytest.fixture
def service(
    repo: ProjectRepository,
    array_store: ArrayStore,
) -> AnalysisService:
    return AnalysisService(repository=repo, array_store=array_store)


@pytest.fixture
def page(qtbot: QtBot) -> AnalysisPage:  # type: ignore[no-untyped-def]
    p = AnalysisPage()
    qtbot.addWidget(p)
    return p


@pytest.fixture
def page_with_runtime(
    page: AnalysisPage,
    service: AnalysisService,
    array_store: ArrayStore,
) -> AnalysisPage:
    """An AnalysisPage with the doubler analyzer registered + runtime bound."""
    registry = AnalyzerRegistry([_DoublingAnalyzer()])
    page.bind_runtime(service=service, registry=registry, array_store=array_store)
    return page


def _seed_project_and_arrays(
    repo: ProjectRepository,
    array_store: ArrayStore,
    *,
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[Project, Sample, Measurement]:
    """Persist a Project with one UV-DRS measurement + optional arrays."""
    project, sample, measurement = _project_with_one_uvdrs_measurement()
    repo.save(project)
    if arrays is not None:
        table = pa.table({k: pa.array(v) for k, v in arrays.items()})
        target = array_store.directory / f"{measurement.id}.parquet"
        pq.write_table(table, target)  # type: ignore[no-untyped-call]
    return project, sample, measurement


# ---------------------------------------------------------------------------
# Empty / construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_constructs_in_empty_state(self, page: AnalysisPage) -> None:
        assert page.objectName() == "AnalysisPage"
        assert page.project is None
        assert page.selected_measurement is None
        assert page.selected_result is None

    def test_empty_state_disables_run_buttons(self, page: AnalysisPage) -> None:
        assert page._run_button.isEnabled() is False
        assert page._rerun_button.isEnabled() is False


# ---------------------------------------------------------------------------
# Tree population & filtering
# ---------------------------------------------------------------------------


class TestTreePopulation:
    def test_set_project_populates_tree(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
    ) -> None:
        project, _, _ = _seed_project_and_arrays(repo, array_store)
        page_with_runtime.set_project(project)
        # One sample top-level item with one measurement child.
        assert page_with_runtime._tree.topLevelItemCount() == 1
        top = page_with_runtime._tree.topLevelItem(0)
        assert top.childCount() == 1

    def test_tree_filters_out_non_analyzable_measurements(
        self,
        page: AnalysisPage,
        service: AnalysisService,
        array_store: ArrayStore,
    ) -> None:
        # Registry only handles XRD; UV-DRS measurement has no analyzer.
        registry = AnalyzerRegistry([_XrdOnlyAnalyzer()])
        page.bind_runtime(service=service, registry=registry, array_store=array_store)
        project, _, _ = _project_with_one_uvdrs_measurement()
        page.set_project(project)
        # UV-DRS measurement has no applicable analyzer → tree empty.
        assert page._tree.topLevelItemCount() == 0

    def test_clear_empties_tree(self, page_with_runtime: AnalysisPage) -> None:
        project, _, _ = _project_with_one_uvdrs_measurement()
        page_with_runtime.set_project(project)
        page_with_runtime.clear()
        assert page_with_runtime._tree.topLevelItemCount() == 0
        assert page_with_runtime.project is None


# ---------------------------------------------------------------------------
# Selection & detail rendering
# ---------------------------------------------------------------------------


class TestSelection:
    def test_selecting_measurement_shows_analyzer_combo(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
        qtbot: QtBot,
    ) -> None:
        project, _, _ = _seed_project_and_arrays(repo, array_store)
        page_with_runtime.set_project(project)
        top = page_with_runtime._tree.topLevelItem(0)
        page_with_runtime._tree.setCurrentItem(top.child(0))
        qtbot.wait(10)
        assert page_with_runtime.selected_measurement is not None
        assert page_with_runtime._analyzer_combo.count() == 1
        assert "doubler" in page_with_runtime._analyzer_combo.itemText(0)

    def test_param_form_widgets_match_default_types(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
        qtbot: QtBot,
    ) -> None:
        """Float default → DoubleSpinBox; str default → LineEdit."""
        project, _, _ = _seed_project_and_arrays(repo, array_store)
        page_with_runtime.set_project(project)
        page_with_runtime._tree.setCurrentItem(page_with_runtime._tree.topLevelItem(0).child(0))
        qtbot.wait(10)
        widgets = page_with_runtime._param_widgets
        assert isinstance(widgets["multiplier"], QDoubleSpinBox)
        assert widgets["multiplier"].value() == pytest.approx(2.0)
        assert isinstance(widgets["label"], QLineEdit)
        assert widgets["label"].text() == "default"

    def test_run_buttons_enabled_after_selection(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
        qtbot: QtBot,
    ) -> None:
        project, _, _ = _seed_project_and_arrays(repo, array_store)
        page_with_runtime.set_project(project)
        page_with_runtime._tree.setCurrentItem(page_with_runtime._tree.topLevelItem(0).child(0))
        qtbot.wait(10)
        assert page_with_runtime._run_button.isEnabled()
        assert page_with_runtime._rerun_button.isEnabled()


# ---------------------------------------------------------------------------
# Parameter widget dispatch
# ---------------------------------------------------------------------------


class TestParamWidgetDispatch:
    def test_bool_default_yields_checkbox(self, page: AnalysisPage) -> None:
        widget = page._make_param_widget(True)
        assert isinstance(widget, QCheckBox)
        assert widget.isChecked() is True

    def test_int_default_yields_spinbox(self, page: AnalysisPage) -> None:
        widget = page._make_param_widget(7)
        assert isinstance(widget, QSpinBox)
        assert widget.value() == 7

    def test_float_default_yields_doublespinbox(self, page: AnalysisPage) -> None:
        widget = page._make_param_widget(0.42)
        assert isinstance(widget, QDoubleSpinBox)
        assert widget.value() == pytest.approx(0.42)

    def test_str_default_yields_lineedit(self, page: AnalysisPage) -> None:
        widget = page._make_param_widget("hello")
        assert isinstance(widget, QLineEdit)
        assert widget.text() == "hello"


# ---------------------------------------------------------------------------
# End-to-end run flow
# ---------------------------------------------------------------------------


class TestRunFlow:
    def test_run_populates_results_list(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
        qtbot: QtBot,
    ) -> None:
        _seed_project_and_arrays(repo, array_store, arrays={"x": np.array([1.0, 2.0, 3.0])})
        # Re-load the persisted project; the seeded version's measurement
        # objects don't carry the analysis_results yet but that's fine —
        # the page itself reloads on run.
        project = repo.load_first()
        assert project is not None
        page_with_runtime.set_project(project)
        page_with_runtime._tree.setCurrentItem(page_with_runtime._tree.topLevelItem(0).child(0))
        qtbot.wait(10)
        page_with_runtime._run_button.click()
        qtbot.wait(10)
        # One result now visible in the list.
        assert page_with_runtime._results_list.count() == 1
        # And the result is auto-selected → outputs label populated.
        assert "mean" in page_with_runtime._outputs_label.text()

    def test_rerun_force_creates_new_result_replacing_prior(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
        qtbot: QtBot,
    ) -> None:
        _seed_project_and_arrays(repo, array_store, arrays={"x": np.array([1.0, 2.0, 3.0])})
        project = repo.load_first()
        assert project is not None
        page_with_runtime.set_project(project)
        page_with_runtime._tree.setCurrentItem(page_with_runtime._tree.topLevelItem(0).child(0))
        qtbot.wait(10)
        page_with_runtime._run_button.click()
        qtbot.wait(10)
        first_id = page_with_runtime.selected_result.id  # type: ignore[union-attr]
        page_with_runtime._rerun_button.click()
        qtbot.wait(10)
        # Service replaces same-key results, so list count stays at 1
        # but the visible result id changed.
        assert page_with_runtime._results_list.count() == 1
        assert page_with_runtime.selected_result is not None
        assert page_with_runtime.selected_result.id != first_id

    def test_run_writes_derived_arrays_and_renders_plot(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
        qtbot: QtBot,
    ) -> None:
        _seed_project_and_arrays(repo, array_store, arrays={"x": np.array([1.0, 2.0, 3.0])})
        project = repo.load_first()
        assert project is not None
        page_with_runtime.set_project(project)
        page_with_runtime._tree.setCurrentItem(page_with_runtime._tree.topLevelItem(0).child(0))
        qtbot.wait(10)
        page_with_runtime._run_button.click()
        qtbot.wait(10)
        result = page_with_runtime.selected_result
        assert result is not None
        # Sidecar Parquet exists.
        assert result.derived_arrays_path is not None
        assert result.derived_arrays_path.is_file()
        # Plot widget got at least one PlotDataItem (the curve).
        items = page_with_runtime._plot_widget.plotItem.listDataItems()
        assert len(items) >= 1

    def test_run_renders_analyzer_issue(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
        qtbot: QtBot,
    ) -> None:
        _seed_project_and_arrays(repo, array_store, arrays={"x": np.array([1.0, 2.0, 3.0])})
        project = repo.load_first()
        assert project is not None
        page_with_runtime.set_project(project)
        page_with_runtime._tree.setCurrentItem(page_with_runtime._tree.topLevelItem(0).child(0))
        qtbot.wait(10)
        page_with_runtime._run_button.click()
        qtbot.wait(10)
        # The doubler always emits an INFO issue → one issue label
        # should appear in the issues container.
        labels = page_with_runtime._issues_container.findChildren(
            type(page_with_runtime._meta_label)  # BodyLabel
        )
        # At least one issue label that's NOT the empty placeholder.
        issue_texts = [le.text() for le in labels if "INFO" in le.text()]
        assert issue_texts, f"Expected an INFO issue label; got {[le.text() for le in labels]}"


# ---------------------------------------------------------------------------
# Parameter collection
# ---------------------------------------------------------------------------


class TestCollectParams:
    def test_collect_returns_widgets_in_correct_types(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
        qtbot: QtBot,
    ) -> None:
        _seed_project_and_arrays(repo, array_store)
        project = repo.load_first()
        assert project is not None
        page_with_runtime.set_project(project)
        page_with_runtime._tree.setCurrentItem(page_with_runtime._tree.topLevelItem(0).child(0))
        qtbot.wait(10)
        params = page_with_runtime._collect_params()
        assert params == {"multiplier": 2.0, "label": "default"}

    def test_user_edits_flow_into_collected_params(
        self,
        page_with_runtime: AnalysisPage,
        repo: ProjectRepository,
        array_store: ArrayStore,
        qtbot: QtBot,
    ) -> None:
        _seed_project_and_arrays(repo, array_store)
        project = repo.load_first()
        assert project is not None
        page_with_runtime.set_project(project)
        page_with_runtime._tree.setCurrentItem(page_with_runtime._tree.topLevelItem(0).child(0))
        qtbot.wait(10)
        mult_widget = page_with_runtime._param_widgets["multiplier"]
        assert isinstance(mult_widget, QDoubleSpinBox)
        mult_widget.setValue(3.5)
        label_widget = page_with_runtime._param_widgets["label"]
        assert isinstance(label_widget, QLineEdit)
        label_widget.setText("user-input")
        params = page_with_runtime._collect_params()
        assert params["multiplier"] == pytest.approx(3.5)
        assert params["label"] == "user-input"
