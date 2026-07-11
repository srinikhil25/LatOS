"""Server-side state: the one ingestion job and its progress stream.

The sidecar serves exactly one desktop window, so state is deliberately
simple: at most one ingestion runs at a time, and its progress events
flow through a queue that the SSE endpoint drains. No database, no
session handling — the persisted truth lives in the project's own
`.latos/` store, same as always.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from latos.core.models import Project
from latos.ingestion.orchestrator import IngestionResult, Orchestrator

if TYPE_CHECKING:
    from latos.ingestion.array_store import ArrayStore

__all__ = [
    "IngestStatus",
    "OrchestratorFactory",
    "ProgressEvent",
    "ServerState",
    "TerminalEvent",
]

OrchestratorFactory = Callable[[], Orchestrator]


class IngestStatus(StrEnum):
    """Lifecycle of the (single) ingestion job."""

    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One crawler tick: file `index` of `total`, currently `name`."""

    index: int
    total: int
    name: str


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    """Sentinel closing the progress stream (status: done | error)."""

    status: IngestStatus
    message: str | None = None


QueueItem = ProgressEvent | TerminalEvent


def _default_orchestrator() -> Orchestrator:
    """Production orchestrator with the auto-discovered parser registry."""
    # Local import: keeps server module import cheap and avoids pulling
    # every parser's dependencies at app-factory time.
    from latos.ingestion.registry import default_registry  # noqa: PLC0415

    return Orchestrator(registry=default_registry())


@dataclass
class ServerState:
    """Mutable state shared by the API endpoints.

    Thread model: `start_ingest` is called on the event-loop thread and
    spawns a worker thread; the worker only touches `_queue`, `status`,
    `result`, and `error` — each transition guarded by `_lock`.
    """

    orchestrator_factory: OrchestratorFactory = field(default=_default_orchestrator)
    status: IngestStatus = IngestStatus.IDLE
    result: IngestionResult | None = None
    error: str | None = None
    root: Path | None = None

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _queue: queue.Queue[QueueItem] = field(default_factory=queue.Queue, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    # ─── Commands ────────────────────────────────────────────────────
    def start_ingest(self, root: Path, *, project_name: str | None = None) -> bool:
        """Spawn the ingestion thread. Returns False if one is running."""
        with self._lock:
            if self.status is IngestStatus.RUNNING:
                return False
            self.status = IngestStatus.RUNNING
            self.result = None
            self.error = None
            self.root = root
            self._queue = queue.Queue()  # fresh stream per run

        def _run() -> None:
            try:
                orchestrator = self.orchestrator_factory()
                result = orchestrator.ingest(
                    root,
                    project_name=project_name,
                    on_progress=self._on_progress,
                )
            except Exception as exc:  # boundary: report, don't crash the server
                with self._lock:
                    self.status = IngestStatus.ERROR
                    self.error = f"{type(exc).__name__}: {exc}"
                self._queue.put(TerminalEvent(IngestStatus.ERROR, self.error))
                return
            # Apply the researcher's synthesis log (if one sits next to the
            # raw files) now that the project's samples exist. A bad log
            # must never fail an otherwise-good ingestion.
            try:
                from latos.server.synthesis_log import apply_log  # noqa: PLC0415

                apply_log(root, result.project)
            except Exception:
                logging.getLogger("latos.synthesis_log").exception(
                    "applying the synthesis log failed"
                )
            with self._lock:
                self.status = IngestStatus.DONE
                self.result = result
            self._queue.put(TerminalEvent(IngestStatus.DONE))

        self._thread = threading.Thread(target=_run, name="latos-ingest", daemon=True)
        self._thread.start()
        return True

    def reset(self) -> None:
        """Forget the open project — after its `.latos/` store is deleted.

        Leaves the server idle so a stale `result`/`root` can't be read back
        (which would 500 once the store is gone). A no-op if nothing is open.
        """
        with self._lock:
            self.status = IngestStatus.IDLE
            self.result = None
            self.error = None
            self.root = None
            self._queue = queue.Queue()

    def _on_progress(self, index: int, total: int, path: Path) -> None:
        """Crawler callback (worker thread) → progress queue."""
        self._queue.put(ProgressEvent(index=index, total=total, name=path.name))

    # ─── Stream consumption ──────────────────────────────────────────
    def drain_events(self, *, poll_seconds: float = 0.5) -> queue.Queue[QueueItem]:
        """The live progress queue for the SSE endpoint to drain."""
        # Exposed as a method (not the attribute) so the endpoint can't
        # accidentally replace the queue object.
        _ = poll_seconds  # reserved for a future timeout policy
        return self._queue

    def join(self, timeout: float | None = None) -> None:
        """Wait for the ingestion thread (tests use this to synchronize)."""
        if self._thread is not None:
            self._thread.join(timeout)

    def array_store(self) -> ArrayStore | None:
        """ArrayStore over the open project's arrays dir, or None pre-open."""
        if self.root is None:
            return None
        from latos.ingestion.array_store import ArrayStore  # noqa: PLC0415

        return ArrayStore(self.root / ".latos" / "arrays")

    def apply_edit(self, transform: Callable[[Project], Project]) -> Project:
        """Apply `transform` to the persisted project and refresh state.

        Loads the project fresh from its SQLite store (the source of
        truth), applies the pure transform, saves it back, and updates
        the in-memory `result.project`. The engine is disposed in a
        `finally` so SQLite file handles don't linger on Windows.

        Raises `LookupError` if no project is open.
        """
        if self.result is None or self.root is None:
            raise LookupError("No project is open")

        # Local imports keep SQLAlchemy out of the module import path.
        from latos.persistence.db import (  # noqa: PLC0415
            create_project_engine,
            init_schema,
            make_session_factory,
        )
        from latos.persistence.repository import ProjectRepository  # noqa: PLC0415

        engine = create_project_engine(self.root)
        try:
            init_schema(engine)
            repo = ProjectRepository(make_session_factory(engine))
            current = repo.load_first()
            if current is None:
                raise LookupError("No project in the database")
            updated = transform(current)
            repo.save(updated)
        finally:
            engine.dispose()

        self.result = replace(self.result, project=updated)
        return updated
