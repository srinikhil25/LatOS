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
import queue
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from latos.server.schemas import (
    HealthResponse,
    IngestStartedResponse,
    MeasurementSummary,
    OpenProjectRequest,
    ProjectSummary,
    SampleSummary,
)
from latos.server.state import (
    IngestStatus,
    OrchestratorFactory,
    ProgressEvent,
    ServerState,
    TerminalEvent,
)

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

        def stream() -> Iterator[str]:
            # Drain whatever the worker has already queued — a fast
            # ingestion may finish before the client subscribes, and its
            # progress history should not be lost. Only when the queue
            # is empty AND the job is over do we synthesize the terminal
            # frame (true late re-subscriber).
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

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/project")
    def project() -> ProjectSummary:
        result = state.result
        if result is None:
            raise HTTPException(status_code=404, detail="No project is open")
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
            failed=result.failed_count,
            unclassified=result.unclassified_count,
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

    return app
