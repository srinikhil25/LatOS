"""Tests for `latos.ui.pages.sample_review.SampleReviewPage`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from latos.core.enums import FileRole, Severity, Technique
from latos.core.models import (
    FileRef,
    Measurement,
    Project,
    Sample,
    ValidationIssue,
)
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.parsed_data import ParsedData
from latos.ui.pages.sample_review import SampleReviewPage

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.ui


# ---------------------------------------------------------------------------
# Builders (kept in sync with `tests/unit/ui/pages/test_overview.py`)
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


def _parsed_data(arrays: dict[str, np.ndarray]) -> ParsedData:
    return ParsedData(
        technique=Technique.XRD,
        arrays=arrays,
        metadata={},
        instrument="StubBox",
        measured_at=None,
        issues=(),
        parser_name="stub-parser",
        parser_version="0.0.1",
    )


def _measurement(
    seed: int,
    sample_id: str,
    technique: Technique = Technique.XRD,
    *,
    issues: tuple[ValidationIssue, ...] = (),
    instrument: str = "StubBox",
) -> Measurement:
    return Measurement(
        id=_id(seed),
        sample_id=sample_id,
        technique=technique,
        instrument=instrument,
        measured_at=None,
        parsed_at=datetime.now(UTC),
        parser_version="0.0.1",
        files=(_fileref(f"f{seed}.txt"),),
        issues=issues,
    )


def _sample(
    seed: int,
    project_id: str,
    *,
    name: str | None = None,
    measurements: tuple[Measurement, ...] = (),
) -> Sample:
    return Sample(
        id=_id(seed + 1000),
        project_id=project_id,
        canonical_name=name or f"S{seed}",
        measurements=measurements,
    )


def _project(root: Path, *, name: str = "ReviewProj", samples: tuple[Sample, ...] = ()) -> Project:
    return Project(
        id=_id(42),
        name=name,
        root_path=root,
        created_at=datetime.now(UTC),
        schema_version=1,
        samples=samples,
    )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_construction_sets_object_name(self, qtbot: QtBot):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        assert page.objectName() == "SampleReviewPage"
        assert page.project is None
        assert page.selected_measurement is None

    def test_empty_detail_text(self, qtbot: QtBot):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        assert page._title_label.text() == "Select a measurement"


# ---------------------------------------------------------------------------
# Tree population
# ---------------------------------------------------------------------------


class TestSetProject:
    def test_tree_has_one_top_level_item_per_sample(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        s1 = _sample(1, _id(42), name="A")
        s2 = _sample(2, _id(42), name="B")
        proj = _project(tmp_path, samples=(s1, s2))
        page.set_project(proj, array_store=ArrayStore(tmp_path / "arrays"))
        assert page._tree.topLevelItemCount() == 2

    def test_tree_has_measurement_children(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        s = _sample(
            1,
            _id(42),
            measurements=(
                _measurement(1, _id(1001)),
                _measurement(2, _id(1001), technique=Technique.XPS),
            ),
        )
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=ArrayStore(tmp_path / "arrays"))

        top = page._tree.topLevelItem(0)
        assert top is not None
        assert top.childCount() == 2

    def test_set_project_resets_to_empty_detail(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        s = _sample(1, _id(42), measurements=(_measurement(1, _id(1001)),))
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=ArrayStore(tmp_path / "arrays"))
        # No item is selected — detail stays on placeholder text.
        assert page.selected_measurement is None
        assert page._title_label.text() == "Select a measurement"


# ---------------------------------------------------------------------------
# Selection → detail
# ---------------------------------------------------------------------------


class TestSelection:
    def test_selecting_measurement_updates_detail(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)

        m = _measurement(1, _id(1001), technique=Technique.XRD)
        s = _sample(1, _id(42), name="MX-001", measurements=(m,))
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=ArrayStore(tmp_path / "arrays"))

        # Drill into the measurement leaf and select it.
        sample_node = page._tree.topLevelItem(0)
        assert sample_node is not None
        measurement_node = sample_node.child(0)
        page._tree.setCurrentItem(measurement_node)

        assert page.selected_measurement is not None
        assert page.selected_measurement.id == m.id
        # Title combines sample name + technique.
        assert "MX-001" in page._title_label.text()
        assert "xrd" in page._title_label.text().lower()
        # Files block shows the contributing file.
        assert "f1.txt" in page._files_label.text()

    def test_selecting_sample_node_clears_detail(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        s = _sample(1, _id(42), measurements=(_measurement(1, _id(1001)),))
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=ArrayStore(tmp_path / "arrays"))

        # First select a measurement so we have non-empty detail.
        sample_node = page._tree.topLevelItem(0)
        assert sample_node is not None
        page._tree.setCurrentItem(sample_node.child(0))
        assert page.selected_measurement is not None
        # Then click back up to the sample (header) node — detail clears.
        page._tree.setCurrentItem(sample_node)
        assert page.selected_measurement is None
        assert page._title_label.text() == "Select a measurement"


# ---------------------------------------------------------------------------
# Issues rendering
# ---------------------------------------------------------------------------


class TestIssues:
    def test_no_issues_renders_empty_marker(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        m = _measurement(1, _id(1001))
        s = _sample(1, _id(42), measurements=(m,))
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=ArrayStore(tmp_path / "arrays"))

        sample_node = page._tree.topLevelItem(0)
        assert sample_node is not None
        page._tree.setCurrentItem(sample_node.child(0))

        widgets = [
            page._issues_layout.itemAt(i).widget() for i in range(page._issues_layout.count())
        ]
        names = [w.objectName() for w in widgets if w is not None]
        assert "SampleReviewIssuesEmpty" in names

    def test_issue_severity_object_name(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        issue = ValidationIssue(
            field="zT",
            severity=Severity.WARNING,
            message="Looks suspicious",
            detected_at=datetime.now(UTC),
        )
        m = _measurement(1, _id(1001), issues=(issue,))
        s = _sample(1, _id(42), measurements=(m,))
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=ArrayStore(tmp_path / "arrays"))

        sample_node = page._tree.topLevelItem(0)
        assert sample_node is not None
        page._tree.setCurrentItem(sample_node.child(0))

        widgets = [
            page._issues_layout.itemAt(i).widget() for i in range(page._issues_layout.count())
        ]
        names = [w.objectName() for w in widgets if w is not None]
        assert "SampleReviewIssue_warning" in names


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


class TestPlot:
    def test_no_arrays_shows_caption_only(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        m = _measurement(1, _id(1001))
        s = _sample(1, _id(42), measurements=(m,))
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=ArrayStore(tmp_path / "arrays"))

        sample_node = page._tree.topLevelItem(0)
        assert sample_node is not None
        page._tree.setCurrentItem(sample_node.child(0))

        assert "no plottable arrays" in page._plot_caption.text().lower()
        items = page._plot_widget.getPlotItem().listDataItems()
        assert items == []

    def test_arrays_are_plotted(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)

        store = ArrayStore(tmp_path / "arrays")
        m = _measurement(1, _id(1001))
        x = np.linspace(10.0, 80.0, 30)
        y = (x - 30.0) ** 2
        store.write(m.id, _parsed_data({"two_theta": x, "intensity": y}))

        s = _sample(1, _id(42), measurements=(m,))
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=store)

        sample_node = page._tree.topLevelItem(0)
        assert sample_node is not None
        page._tree.setCurrentItem(sample_node.child(0))

        items = page._plot_widget.getPlotItem().listDataItems()
        assert len(items) == 1
        x_data, _y_data = items[0].getData()
        assert pytest.approx(x_data[0]) == x[0]
        assert "intensity vs two_theta" in page._plot_caption.text()


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_everything(self, qtbot: QtBot, tmp_path: Path):
        page = SampleReviewPage()
        qtbot.addWidget(page)
        s = _sample(1, _id(42), measurements=(_measurement(1, _id(1001)),))
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=ArrayStore(tmp_path / "arrays"))

        page.clear()
        assert page.project is None
        assert page.selected_measurement is None
        assert page._tree.topLevelItemCount() == 0
        assert page._title_label.text() == "Select a measurement"
