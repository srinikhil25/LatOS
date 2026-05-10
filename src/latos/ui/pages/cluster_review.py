"""`ClusterReviewPage` — review and edit auto-clustered samples (Stage 2D).

What this page is for
---------------------
Stage 2C produced one `SampleCluster` per logical sample. The
heuristic is good but not perfect — sometimes researchers know two
clusters are the same sample (and the algorithm split them), or two
clusters are different (and the algorithm over-merged). This page
gives them a place to fix it before the labels are persisted.

Layout
------
- Toolbar across the top: Apply / Revert / Merge selected.
- Table beneath, one row per cluster:
    | Canonical (editable) | Aliases | # files |
- The first column is editable in place — typing commits a rename
  on focus-out via the `clusterRenamed` signal.
- The aliases column shows the sorted alias set joined by " / "
  (read-only); the file-count column is the integer.
- Multi-select on row headers; "Merge selected" combines the chosen
  rows and recomputes the table from the new decisions state.

Page state
----------
- `_clusters`: the original (unedited) Stage 2C output. Held to
  enable Revert.
- `_decisions`: a `ClusterDecisions` representing the user's edits
  so far. Edits update `_decisions`, then we re-derive the displayed
  clusters via `apply_decisions(_clusters, _decisions)`.
- `_project_root`: where to load/save decisions from. `None` until
  a project is set.

Why this shape
--------------
We chose to keep this page strictly about cluster editing — it does
not show measurements, plots, or per-file metadata. The Sample
Review page is for that drill-down view. Keeping concerns separate
makes each page easier to test and the user mental-model simpler.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
    SubtitleLabel,
    TableWidget,
)

from latos.ingestion.labeling.decisions import (
    ClusterDecisions,
    apply_decisions,
    load_decisions,
    save_decisions,
)

if TYPE_CHECKING:
    from latos.ingestion.labeling.cluster import SampleCluster

__all__ = ["ClusterReviewPage"]


# Column indices for the table. Named so refactors don't have to track
# magic numbers across slot bodies.
_COL_CANONICAL = 0
_COL_ALIASES = 1
_COL_FILES = 2

# Header labels. Displayed across the top of the table.
_HEADERS: tuple[str, ...] = ("Sample name", "Aliases", "# files")

# Stable role for storing the *auto* canonical on each row, so the
# rename slot can find the cluster even after its display canonical
# was edited. Above `Qt.ItemDataRole.UserRole` to avoid Qt-internal
# clashes (same convention as `SampleReviewPage`).
_AUTO_CANONICAL_ROLE = Qt.ItemDataRole.UserRole + 1

# Below this many rows selected, a merge is a no-op. Two is the floor;
# merging one cluster with itself would be meaningless.
_MIN_MERGE_SELECTION = 2


class ClusterReviewPage(QWidget):
    """Editable table of auto-clustered samples with apply / revert."""

    # Emitted whenever the user-edited decisions change (rename / merge /
    # revert / load). The main window listens to keep the project +
    # other pages in sync. Carries the *applied* clusters (post-edit).
    decisionsChanged = Signal(tuple)  # noqa: N815  Qt convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ClusterReviewPage")

        self._clusters: tuple[SampleCluster, ...] = ()
        self._decisions: ClusterDecisions = ClusterDecisions()
        self._project_root: Path | None = None
        # Forward declarations for type checkers — set in `_build_layout`.
        self._title_label: SubtitleLabel
        self._summary_label: BodyLabel
        self._table: TableWidget
        self._merge_btn: PushButton
        self._apply_btn: PrimaryPushButton
        self._revert_btn: PushButton
        # Suppresses cell-changed reentrancy while we repopulate the
        # table programmatically. Without this, every `setItem` call
        # would fire `_on_cell_changed` and stomp the decisions state
        # we just rebuilt.
        self._suppress_cell_signal: bool = False

        self._build_layout()
        self._render_empty()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def clusters(self) -> tuple[SampleCluster, ...]:
        """The clusters currently displayed (post-decisions)."""
        return apply_decisions(self._clusters, self._decisions)

    @property
    def decisions(self) -> ClusterDecisions:
        """The user's current edits, as a `ClusterDecisions`."""
        return self._decisions

    @property
    def project_root(self) -> Path | None:
        """Project root currently driving load/save, or `None`."""
        return self._project_root

    def set_clusters(
        self,
        clusters: tuple[SampleCluster, ...],
        *,
        project_root: Path | None = None,
    ) -> None:
        """Display `clusters` and load any saved decisions for the project.

        Args:
            clusters: The Stage 2C output to review. Stored verbatim
                so Revert can restore them.
            project_root: When given, the page loads existing
                decisions from `<root>/.latos/cluster_decisions.json`
                and saves there on Apply. `None` means "no
                persistence" (used by tests that exercise edit
                behaviour without writing files).
        """
        self._clusters = tuple(clusters)
        self._project_root = project_root
        # Existing on-disk decisions roll forward. A first-time review
        # of a project starts empty.
        if project_root is not None:
            self._decisions = load_decisions(project_root)
        else:
            self._decisions = ClusterDecisions()
        self._refresh_table()
        # Do NOT emit `decisionsChanged` here — set_clusters is the
        # initial wire-up, not a user edit. Listeners (main window)
        # already have the original project; they get a signal only
        # when the user makes an edit.

    def clear(self) -> None:
        """Reset to the empty placeholder."""
        self._clusters = ()
        self._decisions = ClusterDecisions()
        self._project_root = None
        self._refresh_table()

    def merge_selected(self) -> None:
        """Public hook so tests / shortcuts can trigger the merge action."""
        self._on_merge_clicked()

    def revert(self) -> None:
        """Public hook so tests / shortcuts can trigger Revert."""
        self._on_revert_clicked()

    def apply(self) -> Path | None:
        """Public hook so tests / shortcuts can trigger Apply.

        Returns the path that was written, or `None` if no project
        root is set (in which case the call is a no-op).
        """
        return self._on_apply_clicked()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(12)

        self._title_label = SubtitleLabel("Cluster review", self)
        self._title_label.setObjectName("ClusterReviewTitle")
        outer.addWidget(self._title_label)

        self._summary_label = BodyLabel("", self)
        self._summary_label.setObjectName("ClusterReviewSummary")
        self._summary_label.setWordWrap(True)
        outer.addWidget(self._summary_label)

        # Toolbar row.
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self._merge_btn = PushButton("Merge selected", self)
        self._merge_btn.setObjectName("ClusterReviewMergeButton")
        self._merge_btn.clicked.connect(self._on_merge_clicked)
        toolbar.addWidget(self._merge_btn)

        self._revert_btn = PushButton("Revert", self)
        self._revert_btn.setObjectName("ClusterReviewRevertButton")
        self._revert_btn.clicked.connect(self._on_revert_clicked)
        toolbar.addWidget(self._revert_btn)

        toolbar.addStretch(1)

        self._apply_btn = PrimaryPushButton("Apply", self)
        self._apply_btn.setObjectName("ClusterReviewApplyButton")
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        toolbar.addWidget(self._apply_btn)

        outer.addLayout(toolbar)

        # Editable table.
        self._table = TableWidget(self)
        self._table.setObjectName("ClusterReviewTable")
        self._table.setColumnCount(len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._table.verticalHeader().setVisible(False)
        # The Aliases column eats the spare width; the others fit content.
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_CANONICAL, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_ALIASES, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_FILES, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemChanged.connect(self._on_cell_changed)
        outer.addWidget(self._table, 1)

        # A footer caption documenting how the editable column works,
        # so the user doesn't have to guess.
        hint = StrongBodyLabel(
            "Click a sample name to rename it. Select multiple rows and click Merge "
            "to combine clusters. Apply saves your edits to .latos/cluster_decisions.json.",
            self,
        )
        hint.setObjectName("ClusterReviewHint")
        hint.setWordWrap(True)
        outer.addWidget(hint)

    # ------------------------------------------------------------------
    # Render passes
    # ------------------------------------------------------------------

    def _render_empty(self) -> None:
        self._summary_label.setText("No project loaded yet.")
        self._table.setRowCount(0)
        self._merge_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._revert_btn.setEnabled(False)

    def _refresh_table(self) -> None:
        """Repopulate the table from `_clusters` + `_decisions`.

        We rebuild the whole table on every change because the row
        set itself changes when merges/splits happen - partial
        updates would be more code with no real benefit at our row
        counts (10s to low hundreds).
        """
        applied = apply_decisions(self._clusters, self._decisions)

        self._suppress_cell_signal = True
        try:
            self._table.setRowCount(len(applied))
            for row, cluster in enumerate(applied):
                self._set_row(row, cluster)
        finally:
            self._suppress_cell_signal = False

        n_clusters = len(applied)
        n_files = sum(len(c.file_paths) for c in applied)
        self._summary_label.setText(
            f"{n_clusters} cluster{'s' if n_clusters != 1 else ''} · "
            f"{n_files} file{'s' if n_files != 1 else ''}"
        )

        has_data = bool(self._clusters)
        self._merge_btn.setEnabled(has_data)
        self._revert_btn.setEnabled(has_data)
        # Apply needs both data *and* a place to write — no project
        # root means we silently keep edits in memory only.
        self._apply_btn.setEnabled(has_data and self._project_root is not None)

    def _set_row(self, row: int, cluster: SampleCluster) -> None:
        # Canonical (editable). The auto canonical for *this row* is the
        # one we'd reverse-look-up if the user renames it. Stored as
        # user-data on the canonical cell so we can pull it out from
        # the cell-changed slot without index gymnastics.
        canonical_item = QTableWidgetItem(cluster.canonical)
        canonical_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
        )
        canonical_item.setData(_AUTO_CANONICAL_ROLE, self._auto_canonical_for(cluster))
        self._table.setItem(row, _COL_CANONICAL, canonical_item)

        aliases_item = QTableWidgetItem(" / ".join(cluster.aliases))
        aliases_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        aliases_item.setToolTip("\n".join(cluster.aliases))
        self._table.setItem(row, _COL_ALIASES, aliases_item)

        files_item = QTableWidgetItem(str(len(cluster.file_paths)))
        files_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        files_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Show the actual file paths on hover so the user can verify
        # the cluster contents without a drill-down.
        files_item.setToolTip("\n".join(str(p) for p in cluster.file_paths))
        self._table.setItem(row, _COL_FILES, files_item)

    def _auto_canonical_for(self, cluster: SampleCluster) -> str:
        """Return the *original* canonical that maps to this row.

        After a rename, the displayed canonical changes but our
        decisions are keyed by the *auto* canonical (what Stage 2C
        produced). To find that back out we check the rename map: if
        the displayed name is a value in `renames`, the auto canonical
        is the matching key. Otherwise the displayed name *is* the
        auto canonical.
        """
        # Inverse map: applied name → auto canonical. Built fresh
        # each call rather than cached because the rename map mutates
        # on every edit; cache invalidation cost > rebuild cost.
        for auto_name, new_name in self._decisions.renames.items():
            if new_name == cluster.canonical:
                return auto_name
        return cluster.canonical

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_cell_changed(self, item: QTableWidgetItem) -> None:
        if self._suppress_cell_signal:
            return
        if item.column() != _COL_CANONICAL:
            return
        auto_canonical = item.data(_AUTO_CANONICAL_ROLE)
        if not isinstance(auto_canonical, str):
            return
        new_name = item.text().strip()
        # Empty rename just clears the rename without removing the
        # cluster — `with_rename("", "")` already handles this safely.
        self._decisions = self._decisions.with_rename(auto_canonical, new_name)
        self._refresh_table()
        self.decisionsChanged.emit(self.clusters)

    def _on_merge_clicked(self) -> None:
        rows = self._selected_rows()
        if len(rows) < _MIN_MERGE_SELECTION:
            # Nothing to do - Qt happily fires `clicked` even with
            # zero selection; we'd rather no-op than emit a signal
            # that listeners would have to filter.
            return
        canonicals: list[str] = []
        for row in rows:
            item = self._table.item(row, _COL_CANONICAL)
            if item is not None:
                canonicals.append(item.text().strip())
        canonicals = [c for c in canonicals if c]
        if len(canonicals) < _MIN_MERGE_SELECTION:
            return
        self._decisions = self._decisions.with_merge(canonicals)
        # Drop the selection — after the merge the row indices have
        # shifted, and a stale selection points at the wrong rows.
        self._table.clearSelection()
        self._refresh_table()
        self.decisionsChanged.emit(self.clusters)

    def _on_revert_clicked(self) -> None:
        if self._decisions == ClusterDecisions():
            return
        self._decisions = ClusterDecisions()
        self._refresh_table()
        self.decisionsChanged.emit(self.clusters)

    def _on_apply_clicked(self) -> Path | None:
        if self._project_root is None:
            return None
        path = save_decisions(self._project_root, self._decisions)
        # Refresh to surface the disk-truth back to the user — load
        # path round-trips equal to what we wrote, so visually
        # nothing changes, but it confirms persistence worked.
        self._decisions = load_decisions(self._project_root)
        self._refresh_table()
        self.decisionsChanged.emit(self.clusters)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selected_rows(self) -> list[int]:
        """Distinct row indices currently selected, sorted ascending."""
        rows: set[int] = set()
        for index in self._table.selectionModel().selectedRows():
            rows.add(index.row())
        return sorted(rows)
