"""Tests for `latos.ui.dialogs.ingestion_progress.IngestionProgressDialog`.

Unlike the worker tests, these exercise the full thread plumbing —
`QThread.started.connect(worker.start)`. We rely on `qtbot.waitUntil`
to wait for the dialog's accept/reject without hanging the test.

Each test injects a stub `OrchestratorFactory` so no real ingestion
happens. The stubs control timing precisely so we can assert on the
intermediate state (progress label updates, cancel button label) before
the dialog accepts.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from latos.core.models import Project
from latos.ingestion.orchestrator import IngestionResult, Orchestrator
from latos.ui.dialogs.ingestion_progress import IngestionProgressDialog

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


def _success_factory(result: IngestionResult):
    """Factory whose orchestrator returns `result` immediately."""

    def factory() -> Orchestrator:
        orch = MagicMock(spec=Orchestrator)
        orch.ingest.return_value = result
        return orch

    return factory


def _slow_factory(result: IngestionResult, ticks: int = 5, sleep_s: float = 0.02):
    """Factory whose orchestrator emits progress slowly so cancel can trigger."""

    def factory() -> Orchestrator:
        orch = MagicMock(spec=Orchestrator)

        def fake_ingest(root: Path, *, on_progress=None, **_kw) -> IngestionResult:
            for i in range(ticks):
                if on_progress is not None:
                    on_progress(i, ticks, root / f"f{i}.txt")
                time.sleep(sleep_s)
            return result

        orch.ingest.side_effect = fake_ingest
        return orch

    return factory


def _failing_factory(error_type: type[Exception] = RuntimeError, msg: str = "oops"):
    def factory() -> Orchestrator:
        orch = MagicMock(spec=Orchestrator)
        orch.ingest.side_effect = error_type(msg)
        return orch

    return factory


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_object_name_and_modal(self, qtbot: QtBot, tmp_path: Path):
        dialog = IngestionProgressDialog(
            tmp_path,
            orchestrator_factory=_success_factory(_empty_result(tmp_path)),
        )
        qtbot.addWidget(dialog)
        assert dialog.objectName() == "IngestionProgressDialog"
        assert dialog.isModal() is True


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestSuccessPath:
    def test_accepts_and_exposes_result(self, qtbot: QtBot, tmp_path: Path):
        result = _empty_result(tmp_path)
        dialog = IngestionProgressDialog(tmp_path, orchestrator_factory=_success_factory(result))
        qtbot.addWidget(dialog)

        # `exec()` blocks; instead, drive the lifecycle by `show()` and
        # `qtbot.waitUntil` for the dialog to accept itself. This still
        # exercises the QThread + signal wiring end to end.
        dialog.show()
        qtbot.waitUntil(lambda: dialog.ingestion_result() is not None, timeout=2000)
        assert dialog.ingestion_result() is result
        assert dialog.failure() is None
        assert dialog.was_cancelled() is False

    def test_progress_label_updates(self, qtbot: QtBot, tmp_path: Path):
        result = _empty_result(tmp_path)
        dialog = IngestionProgressDialog(
            tmp_path,
            orchestrator_factory=_slow_factory(result, ticks=3, sleep_s=0.05),
        )
        qtbot.addWidget(dialog)

        dialog.show()
        # Wait until at least one progress tick has updated the label,
        # then until the dialog finishes.
        qtbot.waitUntil(lambda: dialog._count_label.text() != "Starting…", timeout=2000)
        qtbot.waitUntil(lambda: dialog.ingestion_result() is not None, timeout=2000)
        # Final state: count label mentions the last file index.
        assert "of 3" in dialog._count_label.text()


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestFailurePath:
    def test_rejects_and_exposes_failure(self, qtbot: QtBot, tmp_path: Path):
        dialog = IngestionProgressDialog(
            tmp_path,
            orchestrator_factory=_failing_factory(ValueError, "bang"),
        )
        qtbot.addWidget(dialog)

        dialog.show()
        qtbot.waitUntil(lambda: dialog.failure() is not None, timeout=2000)

        failure = dialog.failure()
        assert failure is not None
        assert failure.error_type == "ValueError"
        assert failure.message == "bang"
        assert dialog.ingestion_result() is None
        assert dialog.was_cancelled() is False


# ---------------------------------------------------------------------------
# Cancel path
# ---------------------------------------------------------------------------


class TestCancelPath:
    def test_cancel_button_marks_cancelled(self, qtbot: QtBot, tmp_path: Path):
        # Slow factory keeps the worker running long enough for us to
        # click cancel.
        result = _empty_result(tmp_path)
        dialog = IngestionProgressDialog(
            tmp_path,
            orchestrator_factory=_slow_factory(result, ticks=20, sleep_s=0.05),
        )
        qtbot.addWidget(dialog)

        dialog.show()
        # Wait until the worker has reported at least one tick — that
        # tells us the worker thread is definitely running its on_progress
        # loop, so the next cancel will be observed.
        qtbot.waitUntil(lambda: dialog._count_label.text() != "Starting…", timeout=2000)

        dialog._cancel_button.click()
        qtbot.waitUntil(dialog.was_cancelled, timeout=2000)
        assert dialog.ingestion_result() is None
        assert dialog.failure() is None
        # Cancel button label flips to indicate the cancel is in-flight.
        assert dialog._cancel_button.text() == "Cancelling…"
        assert dialog._cancel_button.isEnabled() is False
