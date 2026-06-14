"""FastAPI application factory for the Latos sidecar.

The desktop shell (`apps/desktop`, Tauri) spawns this server as a
hidden child process, waits for `GET /health`, and then drives the
whole application through these endpoints. The server binds to
`127.0.0.1` only — see `__main__.py` — so nothing is reachable from
the network.

Endpoints (v0 skeleton):
- `GET  /health`         — liveness + version handshake
- `POST /project/open`   — start ingesting a folder (202; progress via SSE)
- `GET  /ingest/events`  — Server-Sent Events progress stream
- `GET  /project`        — summary of the ingested project
- `GET  /samples`        — samples → measurements tree
"""

from __future__ import annotations

import json
import math
import queue
from collections.abc import Callable, Iterator
from importlib import metadata
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from latos.core.enums import ReviewStatus, Technique
from latos.core.models import Measurement, Project
from latos.ingestion.orchestrator import IngestionResult
from latos.server import edits
from latos.server.edits import EditError
from latos.server.imaging import render_to_png
from latos.server.schemas import (
    HealthResponse,
    IngestStartedResponse,
    MeasurementArrays,
    MeasurementSummary,
    MergeSamplesRequest,
    OpenProjectRequest,
    ProjectSummary,
    RenameSampleRequest,
    SampleSummary,
    SetTechniqueRequest,
    SplitMeasurementsRequest,
)
from latos.server.state import (
    IngestStatus,
    OrchestratorFactory,
    ProgressEvent,
    ServerState,
    TerminalEvent,
)

# Techniques whose measurements carry a renderable image rather than
# plottable arrays.
_IMAGE_TECHNIQUES = frozenset({Technique.TEM, Technique.SEM, Technique.STEM})

__all__ = ["create_app"]


def _version() -> str:
    """Installed package version, or a dev marker when not installed."""
    try:
        return metadata.version("latos")
    except metadata.PackageNotFoundError:  # pragma: no cover — dev tree only
        return "0.0.0+dev"


def _sse(event: str, data: dict[str, object]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _event_stream(state: ServerState) -> Iterator[str]:
    """Yield SSE frames for the current ingestion until terminal.

    Drains whatever the worker has already queued — a fast ingestion
    may finish before the client subscribes, and its progress history
    should not be lost. Only when the queue is empty AND the job is
    over do we synthesize the terminal frame (true late re-subscriber).
    """
    events = state.drain_events()
    while True:
        try:
            item = events.get_nowait()
        except queue.Empty:
            if state.status in (IngestStatus.DONE, IngestStatus.ERROR):
                yield _sse(
                    state.status.value,
                    {"message": state.error} if state.error else {},
                )
                return
            # Still running: block until the worker's next event.
            item = events.get()
        if isinstance(item, TerminalEvent):
            payload: dict[str, object] = {}
            if item.message:
                payload["message"] = item.message
            yield _sse(item.status.value, payload)
            return
        if isinstance(item, ProgressEvent):
            yield _sse(
                "progress",
                {"index": item.index, "total": item.total, "name": item.name},
            )


def create_app(*, orchestrator_factory: OrchestratorFactory | None = None) -> FastAPI:
    """Build the sidecar app.

    Args:
        orchestrator_factory: Override for tests — returns the
            `Orchestrator` used by `POST /project/open`. `None` uses
            the production default (auto-discovered parser registry).
    """
    state = ServerState() if orchestrator_factory is None else ServerState(orchestrator_factory)
    app = FastAPI(title="latos-core", version=_version(), docs_url="/docs")
    # Stash for tests / introspection; FastAPI's `state` is meant for this.
    app.state.latos = state

    # The desktop WebView is a different *origin* than this server
    # (http://localhost:1420 in `tauri dev`, http://tauri.localhost /
    # tauri://localhost when packaged), so without these headers the
    # browser engine silently discards every response. The server still
    # binds 127.0.0.1 only — CORS here is about which local UI may read
    # the responses, not about network exposure.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:1420",
            "http://tauri.localhost",
            "https://tauri.localhost",
            "tauri://localhost",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=_version())

    @app.post("/project/open", status_code=202)
    def open_project(body: OpenProjectRequest) -> IngestStartedResponse:
        root = Path(body.root)
        if not root.is_dir():
            raise HTTPException(status_code=404, detail=f"Folder not found: {root}")
        started = state.start_ingest(root, project_name=body.project_name)
        if not started:
            raise HTTPException(status_code=409, detail="An ingestion is already running")
        return IngestStartedResponse(status="started")

    @app.get("/ingest/events")
    def ingest_events() -> StreamingResponse:
        if state.status is IngestStatus.IDLE:
            raise HTTPException(status_code=409, detail="No ingestion has been started")
        return StreamingResponse(_event_stream(state), media_type="text/event-stream")

    @app.get("/project")
    def project() -> ProjectSummary:
        result = state.result
        if result is None:
            raise HTTPException(status_code=404, detail="No project is open")
        return _project_summary(result)

    @app.post("/project/confirm")
    def confirm_project() -> ProjectSummary:
        return _apply(state, edits.confirm)

    @app.post("/project/reopen")
    def reopen_project() -> ProjectSummary:
        return _apply(state, edits.reopen)

    @app.post("/samples/{sample_id}/rename")
    def rename_sample(sample_id: str, body: RenameSampleRequest) -> ProjectSummary:
        return _apply(state, lambda p: edits.rename_sample(p, sample_id, body.name))

    @app.post("/measurements/{measurement_id}/technique")
    def set_technique(measurement_id: str, body: SetTechniqueRequest) -> ProjectSummary:
        try:
            technique = Technique(body.technique)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown technique: {body.technique}",
            ) from exc
        return _apply(
            state,
            lambda p: edits.set_measurement_technique(p, measurement_id, technique),
        )

    @app.post("/samples/merge")
    def merge_samples(body: MergeSamplesRequest) -> ProjectSummary:
        return _apply(
            state,
            lambda p: edits.merge_samples(p, body.source_ids, body.target_id),
        )

    @app.post("/samples/split")
    def split_measurements(body: SplitMeasurementsRequest) -> ProjectSummary:
        return _apply(
            state,
            lambda p: edits.move_measurements_to_new_sample(p, body.measurement_ids, body.new_name),
        )

    @app.get("/samples")
    def samples() -> list[SampleSummary]:
        result = state.result
        if result is None:
            raise HTTPException(status_code=404, detail="No project is open")
        out: list[SampleSummary] = []
        for sample in result.project.samples:
            rows = [
                MeasurementSummary(
                    id=m.id,
                    technique=m.technique.value,
                    instrument=m.instrument,
                    filename=m.files[0].path.name if m.files else None,
                )
                for m in sample.measurements
            ]
            out.append(
                SampleSummary(
                    id=sample.id,
                    name=sample.canonical_name,
                    aliases=list(sample.aliases),
                    measurements=rows,
                ),
            )
        return out

    @app.get("/measurements/{measurement_id}/arrays")
    def measurement_arrays(measurement_id: str) -> MeasurementArrays:
        result = state.result
        store = state.array_store()
        if result is None or store is None:
            raise HTTPException(status_code=404, detail="No project is open")
        if _find_measurement(result, measurement_id) is None:
            raise HTTPException(status_code=404, detail="Unknown measurement")
        arrays = store.load(measurement_id)
        if not arrays:
            raise HTTPException(
                status_code=404,
                detail="No arrays stored for this measurement",
            )
        # NaN/inf are not valid JSON — emit None so traces show gaps.
        payload = {
            name: [x if math.isfinite(x) else None for x in arr.tolist()]
            for name, arr in arrays.items()
        }
        return MeasurementArrays(
            measurement_id=measurement_id,
            names=list(payload.keys()),
            arrays=payload,
        )

    @app.get("/measurements/{measurement_id}/image")
    def measurement_image(measurement_id: str) -> Response:
        png = _render_measurement_image(state, measurement_id)
        return Response(content=png, media_type="image/png")

    return app


def _find_measurement(result: IngestionResult, measurement_id: str) -> Measurement | None:
    """Locate a measurement by id within a project, or None."""
    for sample in result.project.samples:
        for measurement in sample.measurements:
            if measurement.id == measurement_id:
                return measurement
    return None


def _project_summary(result: IngestionResult) -> ProjectSummary:
    """Build the hub summary from an IngestionResult."""
    proj = result.project
    measurements = [m for s in proj.samples for m in s.measurements]
    return ProjectSummary(
        id=proj.id,
        name=proj.name,
        root_path=str(proj.root_path),
        samples=len(proj.samples),
        measurements=len(measurements),
        techniques=len({m.technique for m in measurements}),
        parsed=result.parsed_count,
        cached=result.cached_count,
        failed=result.failed_count,
        unclassified=result.unclassified_count,
        review_status=proj.review_status.value,
    )


def _apply(
    state: ServerState,
    transform: Callable[[Project], Project],
) -> ProjectSummary:
    """Apply an edit transform, persist it, and return the new summary.

    Maps domain failures to HTTP: no project open → 404, a bad edit
    (unknown id, empty name) → 400.
    """
    try:
        state.apply_edit(transform)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EditError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    assert state.result is not None  # apply_edit refreshed it
    return _project_summary(state.result)


def _require_confirmed(state: ServerState) -> None:
    """Hard gate: raise 409 unless the open project is CONFIRMED.

    Downstream phases (analysis, correlation, optimization) call this
    first — property predictions are only valid on human-verified
    sample/technique identity.
    """
    if state.result is None:
        raise HTTPException(status_code=404, detail="No project is open")
    if state.result.project.review_status is not ReviewStatus.CONFIRMED:
        raise HTTPException(
            status_code=409,
            detail="Project must be confirmed before analysis. Review and confirm it first.",
        )


def _render_measurement_image(state: ServerState, measurement_id: str) -> bytes:
    """Resolve a measurement to PNG bytes, raising HTTP errors on the way."""
    result = state.result
    if result is None:
        raise HTTPException(status_code=404, detail="No project is open")
    measurement = _find_measurement(result, measurement_id)
    if measurement is None:
        raise HTTPException(status_code=404, detail="Unknown measurement")
    if measurement.technique not in _IMAGE_TECHNIQUES:
        raise HTTPException(status_code=404, detail="Not an image measurement")
    if not measurement.files:
        raise HTTPException(status_code=404, detail="No source file for this measurement")
    path = measurement.files[0].path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Source image file is missing")
    try:
        return render_to_png(path)
    except Exception as exc:  # boundary: any decode failure → 422, not a 500
        raise HTTPException(
            status_code=422,
            detail=f"Could not render image: {exc}",
        ) from exc
