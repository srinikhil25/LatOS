"""`IngestionWorker` — runs `Orchestrator.ingest()` on a worker thread.

Why a worker thread
-------------------
`Orchestrator.ingest()` walks the project folder, hashes every file,
runs each parser, and persists results to SQLite + Parquet. On a real
researcher's dataset this is seconds-to-minutes — running it on the GUI
thread would freeze the window and kill the perceived responsiveness of
the app.

PySide6 supports two threading patterns:

1.  Subclass `QThread` and override `run()`.
2.  A plain `QObject` with a `Slot` method, moved to a `QThread` via
    `QObject.moveToThread()`. The thread's `started` signal triggers
    the slot, signals from the worker cross threads via Qt's queued
    connections.

We use pattern #2: it cleanly separates "the work" from "the thread"
and is the pattern the Qt docs recommend for non-trivial cases. The
worker is also unit-testable without `moveToThread` — its `start()`
slot is synchronous, and tests just call it directly to verify the
state-machine logic without the threading layer.

Cancellation
------------
The orchestrator's `on_progress(idx, total, path)` callback is the only
point where we get a chance to interrupt mid-crawl. We poll a
`threading.Event` from the callback; if set, we raise
`_IngestionCancelledError`, which propagates up out of the orchestrator.
The worker catches it and emits `cancelled()` instead of `finished()`.

This means cancellation only takes effect during the crawl phase — by
the time parsing/persistence starts, the crawl loop has already exited.
That's an intentional trade-off: the parsing phase is fast (parsers are
already mapped 1:1 to crawled entries) and aborting it mid-write would
leave the SQLite + Parquet stores in inconsistent state.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Slot

from latos.ingestion.orchestrator import Orchestrator
from latos.ingestion.registry import default_registry

if TYPE_CHECKING:
    from latos.ingestion.registry import ParserRegistry

__all__ = [
    "IngestionFailure",
    "IngestionWorker",
    "OrchestratorFactory",
]


# Builder hook. Production passes `None` and we lazily construct
# `Orchestrator(default_registry())`. Tests pass a factory that returns
# a fake `Orchestrator` to keep the suite hermetic (no real SQLite,
# no Parquet, no parser invocation).
OrchestratorFactory = Callable[[], Orchestrator]


@dataclass(frozen=True, slots=True)
class IngestionFailure:
    """What the worker reports when ingestion raised an exception.

    The traceback string is captured at exception time on the worker
    thread; the dialog can display it to the user (or copy it for a bug
    report) without needing to re-execute anything.
    """

    error_type: str
    message: str
    traceback: str


class _IngestionCancelledError(Exception):
    """Internal sentinel raised from the progress callback on cancel."""


class IngestionWorker(QObject):
    """Runs one `Orchestrator.ingest(root)` call and reports back via signals.

    Lifecycle (typical use from a dialog):

        worker = IngestionWorker(root)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.start)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.start()

    All four terminal signals are mutually exclusive — exactly one fires
    per `start()` call.
    """

    # progress(current, total, path_str)
    # NOTE: Qt signals use mixedCase by Qt convention. Ruff's N815
    # default-rejects, so we suppress per-attribute below.
    progress = Signal(int, int, str)
    # finished(IngestionResult). Typed `object` because Qt's meta-type
    # system can't introspect frozen dataclasses across threads — we lose
    # the dataclass type at the boundary, callers must cast.
    finished = Signal(object)
    # failed(IngestionFailure)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(
        self,
        root: Path,
        *,
        orchestrator_factory: OrchestratorFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Build a worker for one ingestion of `root`.

        Args:
            root: Folder to ingest. Must exist.
            orchestrator_factory: Hook that returns the Orchestrator to
                use. `None` means "build one with the default registry".
                Tests inject a factory returning a stub Orchestrator.
            parent: Optional Qt parent for ownership; usually `None`
                because the worker is `moveToThread`'d.
        """
        super().__init__(parent)
        self._root = root
        self._orchestrator_factory = orchestrator_factory or _default_factory
        # Cancellation flag. `threading.Event` is safe across the GUI /
        # worker thread boundary; Qt's `Slot` decorator alone is not
        # enough because we want a non-blocking check inside a tight
        # `on_progress` loop.
        self._cancel_event = threading.Event()

    # ------------------------------------------------------------------
    # Public, thread-safe API (callable from any thread)
    # ------------------------------------------------------------------

    def request_cancel(self) -> None:
        """Ask the worker to stop at the next progress checkpoint.

        Safe to call from the GUI thread. The cancellation only takes
        effect during the crawl phase; if ingestion has already moved to
        parsing/persistence, the call is a no-op and the worker will
        complete normally.
        """
        self._cancel_event.set()

    def is_cancel_requested(self) -> bool:
        """True once `request_cancel()` has been called this lifetime."""
        return self._cancel_event.is_set()

    # ------------------------------------------------------------------
    # Slot — invoked by `QThread.started` (or directly in tests)
    # ------------------------------------------------------------------

    @Slot()
    def start(self) -> None:
        """Run ingestion. Emits exactly one terminal signal."""
        try:
            orchestrator = self._orchestrator_factory()
            result = orchestrator.ingest(self._root, on_progress=self._on_progress)
        except _IngestionCancelledError:
            self.cancelled.emit()
            return
        except Exception as exc:
            # We funnel every non-cancel exception into `failed` so the
            # GUI thread can show *something* rather than the worker
            # silently dying. The traceback string lets the user / a bug
            # report carry enough info to diagnose later.
            failure = IngestionFailure(
                error_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
            self.failed.emit(failure)
            return
        self.finished.emit(result)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_progress(self, idx: int, total: int, path: Path) -> None:
        """Forwarded to `Orchestrator.ingest(on_progress=...)`.

        Raises `_IngestionCancelledError` if the user has clicked Cancel; the
        crawler propagates that out, our `start()` catches it.
        """
        if self._cancel_event.is_set():
            raise _IngestionCancelledError
        # The crawler doesn't catch exceptions from the callback, so the
        # `progress.emit()` itself must not raise. Qt's emit is
        # thread-safe via queued connections by default when crossing
        # threads, so this is fine.
        self.progress.emit(idx, total, str(path))


def _default_factory() -> Orchestrator:
    """Production default: ingest with the auto-discovered parser registry."""
    registry: ParserRegistry = default_registry()
    return Orchestrator(registry)
