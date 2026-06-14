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
        FileOutcome(
            path=root / "CS-1.asc",
            relative_path=Path("CS-1.asc"),
            sha256="1" * 64,
            outcome=Outcome.SKIPPED_CACHED,
            sample_name="CS-1",
            parser_name="rigaku-asc",
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


def _image_result(root: Path) -> IngestionResult:
    """A result with one TEM measurement pointing at a real TIF on disk."""
    import numpy as np
    from PIL import Image

    tif = root / "tem_image.tif"
    Image.fromarray(
        np.random.default_rng(1).integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
    ).save(tif, format="TIFF")

    project_id = new_id()
    sample_id = new_id()
    measurement = Measurement(
        id=new_id(),
        sample_id=sample_id,
        technique=Technique.TEM,
        instrument="JEOL",
        measured_at=None,
        parsed_at=utc_now(),
        parser_version="test-1",
        files=(
            FileRef(
                path=tif,
                sha256="a" * 64,
                size_bytes=tif.stat().st_size,
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
        canonical_name="CS",
        aliases=(),
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
    return IngestionResult(project=project, outcomes=())


def _two_sample_project(root: Path) -> Project:
    """A persistable two-sample project (CS: 2 measurements, CS-3: 1)."""
    pid = new_id()
    s1_id, s2_id = new_id(), new_id()

    def meas(sample_id: str, tech: Technique) -> Measurement:
        return Measurement(
            id=new_id(),
            sample_id=sample_id,
            technique=tech,
            instrument=None,
            measured_at=None,
            parsed_at=utc_now(),
            parser_version="t-1",
            files=(),
            issues=(),
            parsed_data_path=None,
            analysis_results=(),
        )

    s1 = Sample(
        id=s1_id,
        project_id=pid,
        canonical_name="CS",
        aliases=(),
        measurements=(meas(s1_id, Technique.TEM), meas(s1_id, Technique.XRD)),
    )
    s2 = Sample(
        id=s2_id,
        project_id=pid,
        canonical_name="CS-3",
        aliases=(),
        measurements=(meas(s2_id, Technique.TEM),),
    )
    return Project(
        id=pid,
        name=root.name,
        root_path=root,
        created_at=utc_now(),
        schema_version=4,
        samples=(s1, s2),
        unassigned_files=(),
    )


class ImageStubOrchestrator:
    """Returns a project whose single TEM measurement has a real TIF."""

    def ingest(
        self,
        root: Path,
        *,
        project_name: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        return _image_result(root)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(orchestrator_factory=StubOrchestrator)  # type: ignore[arg-type]
    return TestClient(app)


@pytest.fixture()
def image_client(tmp_path: Path) -> TestClient:
    app = create_app(orchestrator_factory=ImageStubOrchestrator)  # type: ignore[arg-type]
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
        assert body["cached"] == 1
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


class TestMeasurementArrays:
    def test_before_open_404(self, client: TestClient):
        assert client.get("/measurements/abc123/arrays").status_code == 404

    def test_unknown_measurement_404(self, client: TestClient, tmp_path: Path):
        _open_and_join(client, tmp_path)
        assert client.get("/measurements/not-a-real-id/arrays").status_code == 404

    def test_known_measurement_without_arrays_404(self, client: TestClient, tmp_path: Path):
        _open_and_join(client, tmp_path)
        mid = client.get("/samples").json()[0]["measurements"][0]["id"]
        response = client.get(f"/measurements/{mid}/arrays")
        assert response.status_code == 404
        assert "No arrays" in response.json()["detail"]

    def test_arrays_round_trip_with_nan_gap(self, client: TestClient, tmp_path: Path):
        import numpy as np

        from latos.ingestion.array_store import ArrayStore
        from latos.ingestion.parsed_data import ParsedData

        _open_and_join(client, tmp_path)
        mid = client.get("/samples").json()[0]["measurements"][0]["id"]
        store = ArrayStore(tmp_path / ".latos" / "arrays")
        store.write(
            mid,
            ParsedData(
                technique=Technique.XRD,
                arrays={
                    "two_theta": np.array([10.0, 20.0, 30.0]),
                    "intensity": np.array([1.0, float("nan"), 3.0]),
                },
                metadata={},
                instrument=None,
                measured_at=None,
                issues=(),
                parser_name="test",
                parser_version="1.0.0",
            ),
        )
        body = client.get(f"/measurements/{mid}/arrays").json()
        assert body["measurement_id"] == mid
        assert body["names"] == ["two_theta", "intensity"]
        assert body["arrays"]["two_theta"] == [10.0, 20.0, 30.0]
        # NaN must arrive as a JSON null (trace gap), not break the payload.
        assert body["arrays"]["intensity"] == [1.0, None, 3.0]


class TestMeasurementImage:
    def test_before_open_404(self, image_client: TestClient):
        assert image_client.get("/measurements/abc/image").status_code == 404

    def test_unknown_measurement_404(self, image_client: TestClient, tmp_path: Path):
        _open_and_join(image_client, tmp_path)
        assert image_client.get("/measurements/nope/image").status_code == 404

    def test_non_image_technique_404(self, client: TestClient, tmp_path: Path):
        # The plain stub's measurement is XRD — not an image technique.
        _open_and_join(client, tmp_path)
        mid = client.get("/samples").json()[0]["measurements"][0]["id"]
        response = client.get(f"/measurements/{mid}/image")
        assert response.status_code == 404
        assert "image" in response.json()["detail"].lower()

    def test_renders_png(self, image_client: TestClient, tmp_path: Path):
        _open_and_join(image_client, tmp_path)
        mid = image_client.get("/samples").json()[0]["measurements"][0]["id"]
        response = image_client.get(f"/measurements/{mid}/image")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG\r\n\x1a\n")

    def test_missing_source_file_404(self, image_client: TestClient, tmp_path: Path):
        _open_and_join(image_client, tmp_path)
        mid = image_client.get("/samples").json()[0]["measurements"][0]["id"]
        # Delete the TIF the stub wrote, then request it.
        (tmp_path / "tem_image.tif").unlink()
        assert image_client.get(f"/measurements/{mid}/image").status_code == 404


class TestProjectReview:
    """End-to-end edit + confirm flow through the API.

    Unlike the other tests, these need the project actually persisted to
    SQLite (the edit endpoints reload from the DB), so we save a real
    two-sample project and point the server state at it.
    """

    @pytest.fixture()
    def review_client(self, tmp_path: Path) -> TestClient:
        from latos.ingestion.orchestrator import IngestionResult
        from latos.persistence.db import (
            create_project_engine,
            init_schema,
            make_session_factory,
        )
        from latos.persistence.repository import ProjectRepository

        project = _two_sample_project(tmp_path)
        engine = create_project_engine(tmp_path)
        init_schema(engine)
        ProjectRepository(make_session_factory(engine)).save(project)
        engine.dispose()

        app = create_app()
        state: ServerState = app.state.latos  # type: ignore[union-attr]
        state.root = tmp_path
        state.result = IngestionResult(project=project, outcomes=())
        return TestClient(app)

    def test_starts_needs_review(self, review_client: TestClient):
        assert review_client.get("/project").json()["review_status"] == "needs_review"

    def test_confirm_then_reopen(self, review_client: TestClient):
        assert review_client.post("/project/confirm").json()["review_status"] == "confirmed"
        # Persisted: a fresh GET still reads confirmed.
        assert review_client.get("/project").json()["review_status"] == "confirmed"
        assert review_client.post("/project/reopen").json()["review_status"] == "needs_review"

    def test_rename_resets_to_needs_review(self, review_client: TestClient):
        review_client.post("/project/confirm")
        sid = review_client.get("/samples").json()[0]["id"]
        body = review_client.post(f"/samples/{sid}/rename", json={"name": "CuSe"})
        assert body.status_code == 200
        assert body.json()["review_status"] == "needs_review"
        names = {s["name"] for s in review_client.get("/samples").json()}
        assert "CuSe" in names

    def test_set_technique(self, review_client: TestClient):
        samples = review_client.get("/samples").json()
        mid = samples[0]["measurements"][0]["id"]
        resp = review_client.post(
            f"/measurements/{mid}/technique",
            json={"technique": "stem"},
        )
        assert resp.status_code == 200
        techniques = {
            m["id"]: m["technique"]
            for s in review_client.get("/samples").json()
            for m in s["measurements"]
        }
        assert techniques[mid] == "stem"

    def test_set_unknown_technique_400(self, review_client: TestClient):
        mid = review_client.get("/samples").json()[0]["measurements"][0]["id"]
        resp = review_client.post(
            f"/measurements/{mid}/technique",
            json={"technique": "not-a-technique"},
        )
        assert resp.status_code == 400

    def test_merge_samples(self, review_client: TestClient):
        samples = review_client.get("/samples").json()
        target, source = samples[0]["id"], samples[1]["id"]
        resp = review_client.post(
            "/samples/merge",
            json={"source_ids": [source], "target_id": target},
        )
        assert resp.status_code == 200
        assert resp.json()["samples"] == 1

    def test_split_into_new_sample(self, review_client: TestClient):
        samples = review_client.get("/samples").json()
        mid = samples[0]["measurements"][0]["id"]
        resp = review_client.post(
            "/samples/split",
            json={"measurement_ids": [mid], "new_name": "CS-pure"},
        )
        assert resp.status_code == 200
        names = {s["name"] for s in review_client.get("/samples").json()}
        assert "CS-pure" in names

    def test_rename_unknown_sample_400(self, review_client: TestClient):
        assert review_client.post("/samples/nope/rename", json={"name": "X"}).status_code == 400

    def test_hard_gate_blocks_until_confirmed(self, review_client: TestClient):
        from fastapi import HTTPException

        from latos.server.app import _require_confirmed

        state: ServerState = review_client.app.state.latos  # type: ignore[union-attr]
        # NEEDS_REVIEW → gate raises 409.
        with pytest.raises(HTTPException) as exc:
            _require_confirmed(state)
        assert exc.value.status_code == 409
        # After confirm → gate passes.
        review_client.post("/project/confirm")
        _require_confirmed(state)  # no raise


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
