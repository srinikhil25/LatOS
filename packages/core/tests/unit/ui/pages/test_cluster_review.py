"""Tests for `latos.ui.pages.cluster_review.ClusterReviewPage`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from latos.ingestion.labeling.cluster import SampleCluster
from latos.ingestion.labeling.decisions import (
    DECISIONS_FILENAME,
    ClusterDecisions,
    load_decisions,
)
from latos.ui.pages.cluster_review import ClusterReviewPage

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.ui


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _cluster(
    canonical: str,
    *,
    aliases: tuple[str, ...] = (),
    files: tuple[str, ...] = (),
) -> SampleCluster:
    return SampleCluster(
        canonical=canonical,
        aliases=aliases or (canonical,),
        file_paths=tuple(Path(f) for f in files),
        normalized_forms=(),
    )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_object_name_set(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        assert page.objectName() == "ClusterReviewPage"

    def test_table_empty_and_buttons_disabled(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        assert page._table.rowCount() == 0
        assert not page._merge_btn.isEnabled()
        assert not page._apply_btn.isEnabled()
        assert not page._revert_btn.isEnabled()

    def test_summary_label_shows_no_project_message(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        assert "No project" in page._summary_label.text()


# ---------------------------------------------------------------------------
# set_clusters / clear
# ---------------------------------------------------------------------------


class TestSetClusters:
    def test_one_row_per_cluster(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        clusters = (
            _cluster("CS-1", files=("/p/a.csv",)),
            _cluster("CS-3", files=("/p/b.csv",)),
        )
        page.set_clusters(clusters)
        assert page._table.rowCount() == 2

    def test_canonical_cell_is_editable(self, qtbot: QtBot):
        # The flag set on column 0 should include `ItemIsEditable`.
        from PySide6.QtCore import Qt

        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))
        item = page._table.item(0, 0)
        assert item is not None
        assert item.flags() & Qt.ItemFlag.ItemIsEditable

    def test_aliases_column_joins_with_slash(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", aliases=("CS-1", "cs_1")),))
        cell = page._table.item(0, 1)
        assert cell is not None
        # Joiner is " / " so they're visually distinct from aliases that
        # contain slashes themselves.
        assert cell.text() == "CS-1 / cs_1"

    def test_file_count_cell(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv", "/p/b.csv", "/p/c.csv")),))
        cell = page._table.item(0, 2)
        assert cell is not None
        assert cell.text() == "3"

    def test_set_clusters_loads_existing_decisions_file(self, qtbot: QtBot, tmp_path: Path):
        # Pre-existing decisions on disk should be picked up
        # immediately so re-opening a project preserves edits.
        decisions_dir = tmp_path / ".latos"
        decisions_dir.mkdir()
        (decisions_dir / DECISIONS_FILENAME).write_text(
            '{"renames": {"CS-1": "MX-Renamed"}, "merges": [], "splits": {}}',
            encoding="utf-8",
        )

        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters(
            (_cluster("CS-1", files=("/p/a.csv",)),),
            project_root=tmp_path,
        )

        # The displayed canonical is the renamed form.
        cell = page._table.item(0, 0)
        assert cell is not None
        assert cell.text() == "MX-Renamed"

    def test_set_clusters_without_project_root_starts_empty_decisions(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))
        assert page.decisions == ClusterDecisions()
        assert page.project_root is None

    def test_set_clusters_does_not_emit_decisions_changed(self, qtbot: QtBot):
        # Initial wire-up isn't a user edit. No signal should fire so
        # the main window doesn't get spurious notifications.
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        with qtbot.assertNotEmitted(page.decisionsChanged):
            page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))

    def test_clear_resets_state(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))
        page.clear()
        assert page._table.rowCount() == 0
        assert page.clusters == ()
        assert page.project_root is None


# ---------------------------------------------------------------------------
# Rename via cell edit
# ---------------------------------------------------------------------------


class TestRename:
    def test_editing_cell_updates_decisions(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))

        # Editing the canonical cell should record a rename.
        item = page._table.item(0, 0)
        assert item is not None
        item.setText("MX-Renamed")

        assert page.decisions.renames == {"CS-1": "MX-Renamed"}
        # The displayed clusters reflect it.
        assert page.clusters[0].canonical == "MX-Renamed"

    def test_renaming_to_same_value_records_nothing(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))

        item = page._table.item(0, 0)
        assert item is not None
        item.setText("CS-1")

        assert page.decisions.renames == {}

    def test_renaming_to_blank_clears_prior_rename(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))

        item = page._table.item(0, 0)
        assert item is not None
        item.setText("X")
        assert page.decisions.renames == {"CS-1": "X"}

        # Clear by typing whitespace.
        item = page._table.item(0, 0)
        assert item is not None
        item.setText("   ")
        assert page.decisions.renames == {}

    def test_rename_emits_decisions_changed(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))

        with qtbot.waitSignal(page.decisionsChanged, timeout=500):
            item = page._table.item(0, 0)
            assert item is not None
            item.setText("MX-Renamed")

    def test_rename_round_trips_through_auto_canonical(self, qtbot: QtBot):
        # After a rename, editing the *displayed* (renamed) cell again
        # should still target the same auto canonical — not record a
        # second rename keyed by the renamed name.
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))

        item = page._table.item(0, 0)
        assert item is not None
        item.setText("X")
        item = page._table.item(0, 0)
        assert item is not None
        item.setText("Y")

        # Single rename keyed by the original name.
        assert page.decisions.renames == {"CS-1": "Y"}


# ---------------------------------------------------------------------------
# Merge selected
# ---------------------------------------------------------------------------


def _select_rows(page: ClusterReviewPage, rows: tuple[int, ...]) -> None:
    """Programmatically multi-select rows under ExtendedSelection mode.

    `QTableWidget.selectRow(n)` REPLACES the selection in
    ExtendedSelection mode (it mirrors a plain click), so the only way
    to build a >1-row selection from code is via the selection model
    with `Select | Rows` flags - the same operation Qt performs
    internally when the user Ctrl-clicks a row number.
    """
    from PySide6.QtCore import QItemSelectionModel

    model = page._table.selectionModel()
    model.clearSelection()
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    for r in rows:
        index = page._table.model().index(r, 0)
        model.select(index, flags)


class TestMerge:
    def test_merge_two_selected_rows(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters(
            (
                _cluster("CS-1", files=("/p/a.csv",)),
                _cluster("cs_1", files=("/p/b.csv",)),
            )
        )

        _select_rows(page, (0, 1))
        page.merge_selected()

        # Single resulting cluster.
        assert page._table.rowCount() == 1
        # Merged file count.
        cell = page._table.item(0, 2)
        assert cell is not None
        assert cell.text() == "2"

    def test_merge_with_zero_selection_is_noop(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters(
            (
                _cluster("A", files=("/p/a.csv",)),
                _cluster("B", files=("/p/b.csv",)),
            )
        )
        page._table.clearSelection()
        page.merge_selected()
        assert page.decisions == ClusterDecisions()

    def test_merge_with_one_selected_is_noop(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters(
            (
                _cluster("A", files=("/p/a.csv",)),
                _cluster("B", files=("/p/b.csv",)),
            )
        )
        _select_rows(page, (0,))
        page.merge_selected()
        assert page.decisions == ClusterDecisions()

    def test_merge_emits_decisions_changed(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters(
            (
                _cluster("A", files=("/p/a.csv",)),
                _cluster("B", files=("/p/b.csv",)),
            )
        )
        _select_rows(page, (0, 1))
        with qtbot.waitSignal(page.decisionsChanged, timeout=500):
            page.merge_selected()

    def test_extended_selection_mode_set(self, qtbot: QtBot):
        # Guard against accidental regression to MultiSelection. The
        # bug was that MultiSelection + editable cells caused single
        # clicks to drop into edit mode instead of extending the
        # selection - users saw "Merge does nothing" because the
        # button never saw >=2 selected rows.
        from PySide6.QtWidgets import QAbstractItemView

        page = ClusterReviewPage()
        qtbot.addWidget(page)
        assert page._table.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
        # Edit triggers should require an explicit double-click or
        # Enter so single-click selects the row instead of editing.
        triggers = page._table.editTriggers()
        assert triggers & QAbstractItemView.EditTrigger.DoubleClicked
        assert not (triggers & QAbstractItemView.EditTrigger.SelectedClicked)
        # Vertical header (row numbers) must not be hidden - it's the
        # primary affordance users click to select a row.
        # `isVisible()` is False until the widget is shown, so we use
        # `isHidden()` which reflects the configured visibility flag
        # regardless of paint state.
        assert not page._table.verticalHeader().isHidden()


# ---------------------------------------------------------------------------
# Apply / Revert
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_writes_decisions_to_disk(self, qtbot: QtBot, tmp_path: Path):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters(
            (_cluster("CS-1", files=("/p/a.csv",)),),
            project_root=tmp_path,
        )

        # Edit, then apply.
        item = page._table.item(0, 0)
        assert item is not None
        item.setText("MX-Renamed")
        path = page.apply()

        assert path is not None
        assert path == tmp_path / ".latos" / DECISIONS_FILENAME
        # Round-trip via load_decisions to confirm what's on disk.
        loaded = load_decisions(tmp_path)
        assert loaded.renames == {"CS-1": "MX-Renamed"}

    def test_apply_without_project_root_is_noop(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))
        assert page.apply() is None

    def test_apply_button_disabled_without_project_root(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))
        # No project root → Apply stays disabled even with data.
        assert not page._apply_btn.isEnabled()

    def test_apply_button_enabled_with_project_root(self, qtbot: QtBot, tmp_path: Path):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters(
            (_cluster("CS-1", files=("/p/a.csv",)),),
            project_root=tmp_path,
        )
        assert page._apply_btn.isEnabled()


class TestRevert:
    def test_revert_clears_decisions(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))

        item = page._table.item(0, 0)
        assert item is not None
        item.setText("X")
        assert page.decisions.renames == {"CS-1": "X"}

        page.revert()
        assert page.decisions == ClusterDecisions()
        # The displayed canonical reverts to the auto value.
        cell = page._table.item(0, 0)
        assert cell is not None
        assert cell.text() == "CS-1"

    def test_revert_with_no_edits_is_noop(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))
        # Should not raise even with no edits.
        page.revert()
        assert page.decisions == ClusterDecisions()

    def test_revert_emits_decisions_changed_only_when_changes_existed(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("CS-1", files=("/p/a.csv",)),))

        # First, no edits → no signal.
        with qtbot.assertNotEmitted(page.decisionsChanged):
            page.revert()

        # After an edit, revert fires the signal.
        item = page._table.item(0, 0)
        assert item is not None
        item.setText("X")
        with qtbot.waitSignal(page.decisionsChanged, timeout=500):
            page.revert()


# ---------------------------------------------------------------------------
# Summary label
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_shows_cluster_and_file_counts(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters(
            (
                _cluster("A", files=("/p/a.csv", "/p/b.csv")),
                _cluster("B", files=("/p/c.csv",)),
            )
        )
        text = page._summary_label.text()
        assert "2 cluster" in text
        assert "3 file" in text

    def test_summary_singular_grammar(self, qtbot: QtBot):
        page = ClusterReviewPage()
        qtbot.addWidget(page)
        page.set_clusters((_cluster("A", files=("/p/a.csv",)),))
        text = page._summary_label.text()
        # No plural "s" on either count when there's one of each.
        assert "1 cluster " in text or text.startswith("1 cluster ")
        assert "1 file" in text
