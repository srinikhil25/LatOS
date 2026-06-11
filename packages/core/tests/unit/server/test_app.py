"""Tests for the `latos.server` sidecar skeleton.

Strategy: drive the FastAPI app through `TestClient` with a stub
orchestrator that returns a small, real `IngestionResult` (real core
models, no fakes of our own types). The stub emits two progress ticks
so the SSE stream is exercised end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from latos.core.enums import FileRole, Technique
from latos.core.models import (
    FileRef,
    Measurement,
    Project,
    Sample,
    new_id,
    utc_now,
)
from latos.ingestion.orchestrator import (
    FileOutcome,
    IngestionResult,
    Outcome,
)
from latos.server.app import create_app
from latos.server.state import ServerState

if TYPE_CHECKING:
    from latos.ingestion.crawler import ProgressCallback


def _tiny_result(root: Path) -> IngestionResult:
    """A real one-sample, one-measurement IngestionResult."""
    project_id = new_id()
    sample_id = new_id()
    measurement = Measurement(
        id=new_id(),
        sample_id=sample_id,
        technique=Technique.XRD,
        instrument="Rigaku",
        measured_at=None,
        parsed_at=utc_now(),
        parser_version="test-1",
        files=(
            FileRef(
                path=root / "CS-3.asc",
                sha256="0" * 64,
                size_bytes=10,
                role=FileRole.RAW,
                scanned_at=utc_now(),
            ),
        ),
        issues=(),
        parsed_data_path=None,
        analysis_results=(),
    )
    sample = Sample(
        id=sample_id,
        project_id=project_id,
        canonical_name="CS-3",
        aliases=("cs3",),
        measurements=(measurement,),
    )
    project = Project(
        id=project_id,
        name=root.name,
        root_path=root,
        created_at=utc_now(),
        schema_version=3,
        samples=(sample,),
        unassigned_files=(),
    )
    outcomes = (
        FileOutcome(
            path=root / "CS-3.asc",
            relative_path=Path("CS-3.asc"),
            sha256="0" * 64,
            outcome=Outcome.PARSED,
            sample_name="CS-3",
            parser_name="rigaku-asc",
            measurement_id=measurement.id,
            error=None,
        ),
        FileOutcome(
            path=root / "junk.bin",
            relative_path=Path("junk.bin"),
            sha256=None,
            outcome=Outcome.SKIPPED_UNCLASSIFIED,
            sample_name=None,
            parser_name=None,
            measurement_id=None,
            error=None,
        ),
    )
    return IngestionResult(project=project, outcomes=outcomes)


class StubOrchestrator:
    """Quacks like `Orchestrator.ingest` and ticks progress twice."""

    def ingest(
        self,
        root: Path,
        *,
        project_name: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        if on_progress is not None:
            on_progress(0, 2, root / "CS-3.asc")
            on_progress(1, 2, root / "junk.bin")
        return _tiny_result(root)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(orchestrator_factory=StubOrchestrator)  # type: ignore[arg-type]
    return TestClient(app)


def _state(client: TestClient) -> ServerState:
    state: ServerState = client.app.state.latos  # type: ignore[union-attr]
    return state


def _open_and_join(client: TestClient, root: Path) -> None:
    response = client.post("/project/open", json={"root": str(root)})
    assert response.status_code == 202
    _state(client).join(timeout=10)


class TestHealth:
    def test_health_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]


class TestOpenProject:
    def test_missing_folder_404(self, client: TestClient, tmp_path: Path):
        response = client.post(
            "/project/open",
            json={"root": str(tmp_path / "nope")},
        )
        assert response.status_code == 404

    def test_open_starts_and_completes(self, client: TestClient, tmp_path: Path):
        _open_and_join(client, tmp_path)
        assert _state(client).result is not None

    def test_project_before_open_is_404(self, client: TestClient):
        assert client.get("/project").status_code == 404

    def test_samples_before_open_is_404(self, client: TestClient):
        assert client.get("/samples").status_code == 404


class TestProjectSummary:
    def test_summary_counts(self, client: TestClient, tmp_path: Path):
        _open_and_join(client, tmp_path)
        body = client.get("/project").json()
        assert body["name"] == tmp_path.name
        assert body["samples"] == 1
        assert body["measurements"] == 1
        assert body["techniques"] == 1
        assert body["parsed"] == 1
        assert body["failed"] == 0
        assert body["unclassified"] == 1


class TestSamplesTree:
    def test_samples_shape(self, client: TestClient, tmp_path: Path):
        _open_and_join(client, tmp_path)
        body = client.get("/samples").json()
        assert len(body) == 1
        sample = body[0]
        assert sample["name"] == "CS-3"
        assert sample["aliases"] == ["cs3"]
        assert len(sample["measurements"]) == 1
        m = sample["measurements"][0]
        assert m["technique"] == "xrd"
        assert m["instrument"] == "Rigaku"
        assert m["filename"] == "CS-3.asc"


class TestIngestEvents:
    def test_events_before_open_409(self, client: TestClient):
        assert client.get("/ingest/events").status_code == 409

    def test_stream_carries_progress_then_done(self, client: TestClient, tmp_path: Path):
        response = client.post("/project/open", json={"root": str(tmp_path)})
        assert response.status_code == 202
        # Drain the SSE stream; TestClient buffers until the generator
        # returns, which it does on the terminal event.
        with client.stream("GET", "/ingest/events") as stream:
            text = "".join(stream.iter_text())
        assert "event: progress" in text
        assert '"name": "CS-3.asc"' in text
        assert "event: done" in text

    def test_late_subscriber_gets_terminal_event(self, client: TestClient, tmp_path: Path):
        _open_and_join(client, tmp_path)
        with client.stream("GET", "/ingest/events") as stream:
            text = "".join(stream.iter_text())
        assert "event: done" in text
