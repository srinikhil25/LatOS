"""`IngestionProgressDialog` — modal progress UI around `IngestionWorker`.

Lifecycle
---------
The dialog owns both an `IngestionWorker` and the `QThread` it lives on.
On `exec()` it starts the thread, displays progress, and closes itself
when the worker emits one of its terminal signals (`finished`,
`failed`, `cancelled`).

After `exec()` returns:
- `ingestion_result()` returns the `IngestionResult` on success, else `None`.
  We avoid the name `result()` because `QDialog.result()` is already
  defined and returns the dialog's accept/reject `int` code.
- `failure()` returns an `IngestionFailure` on failure, else `None`.
- `was_cancelled()` is `True` only on user-cancel.

Caller pattern in MainWindow:

    dialog = IngestionProgressDialog(self, root)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        self._last_result = dialog.ingestion_result()

Why a dialog (vs. an embedded page)
-----------------------------------
Ingestion is a one-shot, blocking operation from the user's mental
model — they pick a folder, they wait, they get a project. A modal
dialog matches that flow and keeps the rest of the window from
appearing interactive while a slow IO-bound task is running. Stage 1E.4
will navigate to the Overview page after the dialog closes.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ProgressBar,
    PushButton,
    StrongBodyLabel,
    SubtitleLabel,
)

from latos.ingestion.orchestrator import IngestionResult
from latos.ui.services.ingestion_worker import (
    IngestionFailure,
    IngestionWorker,
    OrchestratorFactory,
)

__all__ = ["IngestionProgressDialog"]


class IngestionProgressDialog(QDialog):
    """Blocking progress dialog hosted around `IngestionWorker`.

    Construct, call `.exec()`, then read `result()` / `failure()` /
    `was_cancelled()`.
    """

    def __init__(
        self,
        root: Path,
        *,
        orchestrator_factory: OrchestratorFactory | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("IngestionProgressDialog")
        self.setWindowTitle(f"Ingesting {root.name}")
        self.setModal(True)
        self.setMinimumWidth(520)

        # Outcome holders. Exactly one is non-None after `exec()` returns.
        self._result: IngestionResult | None = None
        self._failure: IngestionFailure | None = None
        self._cancelled: bool = False

        self._build_layout(root)
        self._build_worker(root, orchestrator_factory)

    # ------------------------------------------------------------------
    # Public read-after-exec accessors
    # ------------------------------------------------------------------

    def ingestion_result(self) -> IngestionResult | None:
        """The successful result, or `None` if cancelled / failed.

        Named `ingestion_result` (not `result`) because `QDialog` already
        defines a `result()` method returning the int accept/reject code.
        """
        return self._result

    def failure(self) -> IngestionFailure | None:
        """The failure record, or `None` if cancelled / succeeded."""
        return self._failure

    def was_cancelled(self) -> bool:
        """`True` if the user clicked Cancel before completion."""
        return self._cancelled

    # ------------------------------------------------------------------
    # Internals — layout
    # ------------------------------------------------------------------

    def _build_layout(self, root: Path) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 20)
        outer.setSpacing(12)

        title = SubtitleLabel("Ingesting project", self)
        outer.addWidget(title)

        path_label = CaptionLabel(str(root), self)
        path_label.setWordWrap(True)
        outer.addWidget(path_label)

        # Status row: "12 / 161 — XPS_C0_S2.csv"
        self._count_label = StrongBodyLabel("Starting…", self)
        outer.addWidget(self._count_label)

        self._current_file_label = BodyLabel("", self)
        self._current_file_label.setObjectName("CurrentFileLabel")
        # Truncate visually rather than expanding the dialog horizontally
        # for a deep path.
        self._current_file_label.setMaximumWidth(480)
        self._current_file_label.setTextFormat(Qt.TextFormat.PlainText)
        outer.addWidget(self._current_file_label)

        # Indeterminate at first; switches to determinate once the crawl
        # has reported its first (idx, total) tick.
        self._progress_bar = ProgressBar(self)
        self._progress_bar.setRange(0, 0)
        outer.addWidget(self._progress_bar)

        # Buttons row.
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._cancel_button = PushButton("Cancel", self)
        self._cancel_button.setObjectName("CancelButton")
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        buttons.addWidget(self._cancel_button)
        outer.addLayout(buttons)

    # ------------------------------------------------------------------
    # Internals — worker / thread plumbing
    # ------------------------------------------------------------------

    def _build_worker(
        self,
        root: Path,
        orchestrator_factory: OrchestratorFactory | None,
    ) -> None:
        self._thread = QThread(self)
        self._worker = IngestionWorker(root, orchestrator_factory=orchestrator_factory)
        self._worker.moveToThread(self._thread)

        # Drive: thread starts → worker.start() runs on worker thread.
        self._thread.started.connect(self._worker.start)

        # Worker → dialog. Default Qt::AutoConnection becomes a queued
        # connection across threads, so these slots run on the GUI thread
        # — safe to update widgets directly.
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        # Tear down the thread after any terminal signal. We use queued
        # connections here too so the worker has finished its slot before
        # the thread quits.
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)

    def showEvent(self, event):  # type: ignore[no-untyped-def]  # noqa: N802 (Qt overrides)
        """Start the worker thread the moment the dialog is shown."""
        super().showEvent(event)
        # `showEvent` can fire more than once if the window manager
        # iconifies/restores the dialog; only kick off the thread on
        # the first show.
        if not self._thread.isRunning():
            self._thread.start()

    def closeEvent(self, event):  # type: ignore[no-untyped-def]  # noqa: N802 (Qt overrides)
        """Make sure the worker thread is stopped before we tear down."""
        # Triggered by user closing the dialog with the window-manager X
        # button. Treat as cancel-equivalent so the thread can exit
        # cleanly; the dialog rejection happens via the close itself.
        if self._thread.isRunning():
            self._worker.request_cancel()
            # Give the thread a moment to exit on its own. If it doesn't,
            # we still let the dialog close — the worker thread will
            # finish in the background and emit signals nobody is
            # listening to (harmless because the connections are scoped
            # to this dialog).
            self._thread.wait(2000)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Slots — worker callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, current: int, total: int, path: str) -> None:
        """Worker reported progress: update counts + current file."""
        # Switch from indeterminate to determinate the first time we get
        # a real (idx, total) pair. `total` can be 0 for an empty folder.
        if total > 0:
            if self._progress_bar.maximum() == 0:
                self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)

        self._count_label.setText(f"Processing file {current + 1} of {total}")
        # `Path.name` is enough for the user — the full path is shown in
        # the header line. Avoids a horizontally-scrolling label for deep
        # nested folders.
        self._current_file_label.setText(Path(path).name)

    def _on_finished(self, result: object) -> None:
        """Worker finished successfully — accept the dialog."""
        # Cast: the signal is typed `object` because Qt's meta-type
        # system can't carry a frozen dataclass across threads.
        self._result = result if isinstance(result, IngestionResult) else None
        self.accept()

    def _on_failed(self, failure: object) -> None:
        """Worker raised — record the failure and reject."""
        self._failure = failure if isinstance(failure, IngestionFailure) else None
        self.reject()

    def _on_cancelled(self) -> None:
        """Worker cancelled at our request — reject the dialog."""
        self._cancelled = True
        self.reject()

    def _on_cancel_clicked(self) -> None:
        """User clicked the Cancel button."""
        self._cancel_button.setEnabled(False)
        self._cancel_button.setText("Cancelling…")
        self._worker.request_cancel()
