"""Tests for `latos.ui.services.ingestion_worker.IngestionWorker`.

These tests deliberately do NOT exercise the `moveToThread` path. The
worker's `start()` slot is callable on the main thread and has all the
state-machine logic — that's what we test. The threading layer (queued
signal connections, `QThread.started.connect`) is tested at the dialog
level, where the user-visible behavior is the contract.

Stubs avoid the real `Orchestrator` so the suite never touches SQLite,
Parquet, or any parser. Each test injects a tiny `MagicMock`-based
factory that drives the worker's signals exactly the way the real
orchestrator would.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from latos.core.models import Project
from latos.ingestion.orchestrator import IngestionResult, Orchestrator
from latos.ui.services.ingestion_worker import (
    IngestionFailure,
    IngestionWorker,
)

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.ui


def _empty_result(root: Path) -> IngestionResult:
    return IngestionResult(
        project=Project(
            id="0" * 32,
            name=root.name or "stub",
            root_path=root,
            created_at=datetime.now(UTC),
            schema_version=1,
        ),
        outcomes=(),
    )


def _orch_returning(result: IngestionResult) -> Orchestrator:
    """Build a MagicMock Orchestrator whose ingest() returns `result`."""
    orch = MagicMock(spec=Orchestrator)
    orch.ingest.return_value = result
    return orch


def _orch_emitting_progress(paths: list[Path], result: IngestionResult) -> Orchestrator:
    """Orchestrator whose ingest() calls `on_progress` for each path."""
    orch = MagicMock(spec=Orchestrator)

    def fake_ingest(_root: Path, *, on_progress=None, **_kw) -> IngestionResult:
        if on_progress is not None:
            for i, p in enumerate(paths):
                on_progress(i, len(paths), p)
        return result

    orch.ingest.side_effect = fake_ingest
    return orch


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestSuccess:
    def test_emits_finished_with_result(self, qtbot: QtBot, tmp_path: Path):
        result = _empty_result(tmp_path)
        worker = IngestionWorker(
            tmp_path,
            orchestrator_factory=lambda: _orch_returning(result),
        )

        with qtbot.waitSignal(worker.finished, timeout=1000) as blocker:
            worker.start()
        assert blocker.args[0] is result

    def test_emits_progress_for_each_file(self, qtbot: QtBot, tmp_path: Path):
        paths = [tmp_path / f"f{i}.txt" for i in range(3)]
        result = _empty_result(tmp_path)
        worker = IngestionWorker(
            tmp_path,
            orchestrator_factory=lambda: _orch_emitting_progress(paths, result),
        )

        progress_events: list[tuple[int, int, str]] = []
        worker.progress.connect(lambda c, t, p: progress_events.append((c, t, p)))

        with qtbot.waitSignal(worker.finished, timeout=1000):
            worker.start()

        assert progress_events == [
            (0, 3, str(paths[0])),
            (1, 3, str(paths[1])),
            (2, 3, str(paths[2])),
        ]

    def test_emits_only_finished_on_success(self, qtbot: QtBot, tmp_path: Path):
        result = _empty_result(tmp_path)
        worker = IngestionWorker(
            tmp_path,
            orchestrator_factory=lambda: _orch_returning(result),
        )
        # `failed` and `cancelled` must NOT fire on the success path.
        with (
            qtbot.assertNotEmitted(worker.failed),
            qtbot.assertNotEmitted(worker.cancelled),
            qtbot.waitSignal(worker.finished, timeout=1000),
        ):
            worker.start()


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestFailure:
    def test_orchestrator_raises_emits_failed(self, qtbot: QtBot, tmp_path: Path):
        def boom() -> Orchestrator:
            orch = MagicMock(spec=Orchestrator)
            orch.ingest.side_effect = RuntimeError("kaboom")
            return orch

        worker = IngestionWorker(tmp_path, orchestrator_factory=boom)

        with qtbot.waitSignal(worker.failed, timeout=1000) as blocker:
            worker.start()

        failure = blocker.args[0]
        assert isinstance(failure, IngestionFailure)
        assert failure.error_type == "RuntimeError"
        assert failure.message == "kaboom"
        assert "RuntimeError" in failure.traceback

    def test_failure_does_not_emit_finished(self, qtbot: QtBot, tmp_path: Path):
        def boom() -> Orchestrator:
            orch = MagicMock(spec=Orchestrator)
            orch.ingest.side_effect = ValueError("nope")
            return orch

        worker = IngestionWorker(tmp_path, orchestrator_factory=boom)
        with (
            qtbot.assertNotEmitted(worker.finished),
            qtbot.waitSignal(worker.failed, timeout=1000),
        ):
            worker.start()


# ---------------------------------------------------------------------------
# Cancellation path
# ---------------------------------------------------------------------------


class TestCancel:
    def test_request_cancel_before_start_emits_cancelled(self, qtbot: QtBot, tmp_path: Path):
        # The cancel flag is checked inside `_on_progress`; the stub
        # orchestrator must invoke it for the cancel signal to fire.
        paths = [tmp_path / f"f{i}.txt" for i in range(3)]
        worker = IngestionWorker(
            tmp_path,
            orchestrator_factory=lambda: _orch_emitting_progress(paths, _empty_result(tmp_path)),
        )
        worker.request_cancel()
        assert worker.is_cancel_requested() is True

        with qtbot.waitSignal(worker.cancelled, timeout=1000):
            worker.start()

    def test_cancel_path_does_not_emit_finished(self, qtbot: QtBot, tmp_path: Path):
        paths = [tmp_path / "f.txt"]
        worker = IngestionWorker(
            tmp_path,
            orchestrator_factory=lambda: _orch_emitting_progress(paths, _empty_result(tmp_path)),
        )
        worker.request_cancel()
        with (
            qtbot.assertNotEmitted(worker.finished),
            qtbot.waitSignal(worker.cancelled, timeout=1000),
        ):
            worker.start()

    def test_no_progress_callback_means_cancel_is_silent_noop(self, qtbot: QtBot, tmp_path: Path):
        # If ingestion finishes before `on_progress` is ever called, the
        # cancel never gets a chance to fire — the worker completes
        # normally. Documents the intentional limitation in the
        # docstring.
        result = _empty_result(tmp_path)
        worker = IngestionWorker(
            tmp_path,
            orchestrator_factory=lambda: _orch_returning(result),
        )
        worker.request_cancel()  # but no progress callback ever fires

        with qtbot.waitSignal(worker.finished, timeout=1000):
            worker.start()
