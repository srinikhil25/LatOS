"""Tests for `latos.ui.pages.overview.OverviewPage`.

The page reuses the rest of the ingestion stack (`Project`, `Sample`,
`Measurement`, `ArrayStore`) so the tests build small, real instances
of those rather than mocking them — that catches incompatibilities at
the page boundary that mocks would mask.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from latos.core.enums import FileRole, Technique
from latos.core.models import FileRef, Measurement, Project, Sample
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.parsed_data import ParsedData
from latos.ui.pages.overview import (
    OverviewPage,
    StatCard,
    _find_first_plottable,
)

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.ui


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _id(seed: int) -> str:
    return f"{seed:032x}"


def _parsed_data(arrays: dict[str, np.ndarray]) -> ParsedData:
    """Minimal `ParsedData` with the required positional fields stubbed."""
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
    technique: Technique = Technique.XRD,
    parsed_data_path: Path | None = None,
) -> Measurement:
    return Measurement(
        id=_id(seed),
        sample_id=sample_id,
        technique=technique,
        instrument="StubBox",
        measured_at=None,
        parsed_at=datetime.now(UTC),
        parser_version="stub@1",
        files=(_fileref(f"f{seed}.txt"),),
        parsed_data_path=parsed_data_path,
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


def _project(root: Path, *, name: str = "TestProj", samples: tuple[Sample, ...] = ()) -> Project:
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
    def test_construction_renders_empty_placeholder(self, qtbot: QtBot):
        page = OverviewPage()
        qtbot.addWidget(page)
        assert page.objectName() == "OverviewPage"
        assert page.project is None
        # Title shows the placeholder copy.
        assert page._title_label.text() == "No project open yet"

    def test_clear_resets_to_empty_state(self, qtbot: QtBot, tmp_path: Path):
        page = OverviewPage()
        qtbot.addWidget(page)
        proj = _project(tmp_path, name="Loaded")
        page.set_project(proj, array_store=ArrayStore(tmp_path))
        page.clear()
        assert page.project is None
        assert page._title_label.text() == "No project open yet"


# ---------------------------------------------------------------------------
# Populated state
# ---------------------------------------------------------------------------


class TestSetProject:
    def test_title_updates_to_project_name(self, qtbot: QtBot, tmp_path: Path):
        page = OverviewPage()
        qtbot.addWidget(page)
        proj = _project(tmp_path, name="Dhivya MXene")
        page.set_project(proj, array_store=ArrayStore(tmp_path))
        assert page._title_label.text() == "Dhivya MXene"
        assert page.project is proj

    def test_stat_cards_reflect_counts(self, qtbot: QtBot, tmp_path: Path):
        page = OverviewPage()
        qtbot.addWidget(page)

        s1 = _sample(
            1,
            _id(42),
            measurements=(
                _measurement(1, _id(1001)),
                _measurement(2, _id(1001), technique=Technique.XPS),
            ),
        )
        s2 = _sample(2, _id(42), measurements=(_measurement(3, _id(1002)),))
        proj = _project(tmp_path, samples=(s1, s2))
        page.set_project(proj, array_store=ArrayStore(tmp_path))

        # 2 samples, 3 measurements, 3 parsed (= measurements), 0 cached/failed.
        assert page._stat_cards["samples"]._value_label.text() == "2"
        assert page._stat_cards["measurements"]._value_label.text() == "3"
        assert page._stat_cards["parsed"]._value_label.text() == "3"
        assert page._stat_cards["cached"]._value_label.text() == "0"
        assert page._stat_cards["failed"]._value_label.text() == "0"

    def test_sample_rows_render_one_per_sample(self, qtbot: QtBot, tmp_path: Path):
        page = OverviewPage()
        qtbot.addWidget(page)

        s1 = _sample(1, _id(42), name="A")
        s2 = _sample(2, _id(42), name="B")
        proj = _project(tmp_path, samples=(s1, s2))
        page.set_project(proj, array_store=ArrayStore(tmp_path))

        from PySide6.QtWidgets import QLabel

        rows = [
            child
            for child in page._samples_container.findChildren(QLabel)
            if child.objectName() == "SampleRow"
        ]
        assert len(rows) == 2
        assert "A" in rows[0].text()
        assert "B" in rows[1].text()

    def test_no_samples_renders_empty_state_message(self, qtbot: QtBot, tmp_path: Path):
        page = OverviewPage()
        qtbot.addWidget(page)
        proj = _project(tmp_path)  # no samples
        page.set_project(proj, array_store=ArrayStore(tmp_path))

        # We query the layout (not `findChildren`) because the previous
        # empty-state widget from the initial render is `deleteLater()`'d
        # but not yet GC'd at this point — `findChildren` would see both
        # the old and new instance until the next event loop tick.
        live_widgets = [
            page._samples_layout.itemAt(i).widget() for i in range(page._samples_layout.count())
        ]
        empties = [w for w in live_widgets if w.objectName() == "SamplesEmptyState"]
        assert len(empties) == 1


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


class TestPreviewPlot:
    def test_no_arrays_shows_caption_only(self, qtbot: QtBot, tmp_path: Path):
        page = OverviewPage()
        qtbot.addWidget(page)
        s = _sample(1, _id(42), measurements=(_measurement(1, _id(1001)),))
        proj = _project(tmp_path, samples=(s,))
        # ArrayStore points at a freshly-created (empty) directory.
        store = ArrayStore(tmp_path / "arrays")
        page.set_project(proj, array_store=store)

        assert "No plottable arrays" in page._plot_caption.text()

    def test_plots_first_measurement_with_arrays(self, qtbot: QtBot, tmp_path: Path):
        page = OverviewPage()
        qtbot.addWidget(page)

        # Real ArrayStore + real ParsedData. Save() writes a Parquet file
        # the page's render pass loads back and plots.
        store = ArrayStore(tmp_path / "arrays")
        m = _measurement(1, _id(1001))
        x = np.linspace(10.0, 80.0, 50)
        y = np.sin(x / 5) + 1.0
        store.write(m.id, _parsed_data({"two_theta": x, "intensity": y}))

        s = _sample(1, _id(42), measurements=(m,))
        proj = _project(tmp_path, samples=(s,))
        page.set_project(proj, array_store=store)

        # Caption mentions the technique and the column names.
        caption = page._plot_caption.text()
        assert "intensity vs two_theta" in caption
        assert "xrd" in caption.lower()

        # The PlotWidget's PlotItem now has at least one curve.
        items = page._plot_widget.getPlotItem().listDataItems()
        assert len(items) == 1
        # The curve x range matches our input.
        x_data, y_data = items[0].getData()
        assert pytest.approx(x_data[0]) == x[0]
        assert pytest.approx(x_data[-1]) == x[-1]
        assert len(y_data) == len(y)


class TestFindFirstPlottable:
    def test_returns_none_when_no_arrays_anywhere(self, tmp_path: Path):
        store = ArrayStore(tmp_path / "arrays")
        proj = _project(
            tmp_path,
            samples=(_sample(1, _id(42), measurements=(_measurement(1, _id(1001)),)),),
        )
        assert _find_first_plottable(proj, store) is None

    def test_skips_metadata_only_measurements(self, tmp_path: Path):
        store = ArrayStore(tmp_path / "arrays")
        # m1: no Parquet; m2: has Parquet — should pick m2.
        m1 = _measurement(1, _id(1001))
        m2 = _measurement(2, _id(1001))
        store.write(
            m2.id,
            _parsed_data({"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])}),
        )
        s = _sample(1, _id(42), measurements=(m1, m2))
        proj = _project(tmp_path, samples=(s,))

        found = _find_first_plottable(proj, store)
        assert found is not None
        chosen, arrays = found
        assert chosen.id == m2.id
        assert set(arrays) == {"x", "y"}


# ---------------------------------------------------------------------------
# StatCard widget
# ---------------------------------------------------------------------------


class TestStatCard:
    def test_object_name_includes_label(self, qtbot: QtBot):
        card = StatCard(label="samples", value=7)
        qtbot.addWidget(card)
        assert card.objectName() == "StatCard_samples"
        assert card._value_label.text() == "7"

    def test_set_value_updates_display(self, qtbot: QtBot):
        card = StatCard(label="x", value=0)
        qtbot.addWidget(card)
        card.set_value(42)
        assert card._value_label.text() == "42"
