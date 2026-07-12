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
import os
import queue
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from latos import optimization
from latos.analysis.base_analyzer import AnalyzerInputs
from latos.analysis.registry import default_registry as analyzer_registry
from latos.analysis.transport import TransportError
from latos.core import physics
from latos.core.enums import ReviewStatus, Technique
from latos.core.models import Measurement, Project
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.labeling.anomalies import flag_anomalies
from latos.ingestion.labeling.suggestions import suggest_merges
from latos.ingestion.orchestrator import IngestionResult
from latos.optimization import (
    OptimizationError,
    OptimizationResult,
    Recommendation,
    freeze,
    length_scale_robustness,
    optimize,
)
from latos.server import edits, optimization_data, synthesis_store, transport_data
from latos.server.edits import EditError
from latos.server.imaging import render_to_png
from latos.server.schemas import (
    AnalyzerResultOut,
    DatasetPoint,
    DeleteProjectRequest,
    DeleteProjectResult,
    FreezeResult,
    HealthResponse,
    IngestStartedResponse,
    InputVariableOut,
    MeasurementArrays,
    MeasurementSummary,
    MergeSamplesRequest,
    MergeSuggestionOut,
    MoveMeasurementsRequest,
    OpenProjectRequest,
    OptimizationDataset,
    OptimizeResult,
    OptimizeRunRequest,
    OutcomeVerdictOut,
    PreregSummary,
    ProjectSummary,
    QualityFlagOut,
    RecommendationOut,
    RemoveMeasurementsRequest,
    RenameSampleRequest,
    SampleAnomalyOut,
    SampleParametersRequest,
    SampleSummary,
    SetTechniqueRequest,
    SkippedPoint,
    SpbCheckResult,
    SpbSampleOut,
    SplitMeasurementsRequest,
    ThermoelectricResult,
    ValidateOutcomeRequest,
)
from latos.server.state import (
    IngestStatus,
    OrchestratorFactory,
    ProgressEvent,
    ServerState,
    TerminalEvent,
)
from latos.server.trash import trash_path

# Techniques whose measurements carry a renderable image rather than
# plottable arrays.
_IMAGE_TECHNIQUES = frozenset({Technique.TEM, Technique.SEM, Technique.STEM})

# Minimum confirmed (param, target) points before a GP is worth fitting.
_MIN_OPTIMIZE_POINTS = 3

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

    @app.post("/project/delete")
    def delete_project(body: DeleteProjectRequest) -> DeleteProjectResult:
        """Recycle a project's derived ``.latos/`` store — raw files untouched.

        Idempotent: succeeds even if the store is already gone (e.g. the folder
        was renamed). Only ever removes the ``.latos/`` child, never the project
        folder. If the deleted project is the one currently open, the server
        forgets it so later reads don't 500 on a vanished store.
        """
        return _delete_project_store(state, Path(body.root))

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

    _register_review_routes(app, state)
    _register_optimization_data_routes(app, state)

    _register_sample_read_routes(app, state)

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


def _delete_project_store(state: ServerState, root: Path) -> DeleteProjectResult:
    """Recycle a project's ``.latos/`` store; forget it if it was the open one.

    Never touches the raw files. Idempotent when the store is already gone.
    """
    if root == Path(root.anchor):
        raise HTTPException(status_code=400, detail="Refusing to act on a filesystem root")
    store = root / ".latos"
    existed = store.is_dir()
    recycled = True
    if existed:
        try:
            recycled = trash_path(store)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not delete the store: {exc}"
            ) from exc
    if state.root is not None and os.path.normcase(str(state.root)) == os.path.normcase(str(root)):
        state.reset()
    return DeleteProjectResult(root=str(root), removed=existed, recycled=recycled)


def _json_safe(value: object) -> object:
    """Replace non-finite floats with None so the payload is valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _hall_cross_technique_overrides(
    result: IngestionResult, store: ArrayStore, measurement: Measurement
) -> dict[str, dict[str, object]] | None:
    """Cross-technique context for the Hall analyzer, from sibling measurements.

    Injects the sample's Seebeck sign (carrier-type check) and its R&S-derived
    conductivity (conductivity cross-check) so the Hall analyzer can compare
    itself against the independent transport measurements. None when not Hall
    or no sibling data exists.
    """
    if measurement.technique is not Technique.HALL:
        return None
    sample = next((s for s in result.project.samples if s.id == measurement.sample_id), None)
    if sample is None:
        return None
    hall_params: dict[str, object] = {}
    sign = transport_data.seebeck_sign(sample, store.load)
    if sign is not None:
        hall_params["seebeck_sign"] = sign
    rs_sigma = transport_data.rs_conductivity_s_cm(sample, store.load)
    if rs_sigma is not None:
        hall_params["rs_conductivity_s_cm"] = rs_sigma
    return {"hall-metrics": hall_params} if hall_params else None


def _run_analysis(
    measurement: Measurement,
    arrays: dict[str, np.ndarray],
    param_overrides: dict[str, dict[str, object]] | None = None,
) -> list[AnalyzerResultOut]:
    """Run every applicable analyzer on a measurement and serialize results.

    `param_overrides` maps analyzer name → caller-supplied parameters
    (e.g. the sample's Seebeck sign injected for the Hall analyzer's
    cross-technique check).
    """
    out: list[AnalyzerResultOut] = []
    for analyzer in analyzer_registry().find_for(measurement):
        overrides = (param_overrides or {}).get(analyzer.name)
        result = analyzer.analyze(
            AnalyzerInputs(
                measurement=measurement,
                arrays=arrays,
                params=analyzer.merge_params(overrides),
            )
        )
        out.append(
            AnalyzerResultOut(
                analyzer=analyzer.name,
                outputs={k: _json_safe(v) for k, v in result.outputs.items()},
                issues=[f"{i.severity.value}: {i.message}" for i in result.issues],
            )
        )
    return out


def _register_sample_read_routes(app: FastAPI, state: ServerState) -> None:
    """Register the read-only samples tree + review-insight endpoints.

    Kept out of `create_app` to keep that factory small. All three are
    pure reads over the current project; none mutate state.
    """

    @app.get("/samples")
    def samples() -> list[SampleSummary]:
        result = state.result
        if result is None:
            raise HTTPException(status_code=404, detail="No project is open")
        root = result.project.root_path
        out: list[SampleSummary] = []
        for sample in result.project.samples:
            rows = [
                MeasurementSummary(
                    id=m.id,
                    technique=m.technique.value,
                    instrument=m.instrument,
                    filename=m.files[0].path.name if m.files else None,
                    folder=_folder_of(m, root),
                    features=dict(m.features),
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

    @app.get("/samples/merge-suggestions")
    def merge_suggestions() -> list[MergeSuggestionOut]:
        """Suggest-only near-duplicate sample pairs for the review gate.

        Pure read: computes candidates from the current samples and
        returns them ranked. Nothing is merged — the user confirms each
        one via POST /samples/merge.
        """
        result = state.result
        if result is None:
            raise HTTPException(status_code=404, detail="No project is open")
        return [
            MergeSuggestionOut(
                target_id=s.target_id,
                target_name=s.target_name,
                source_id=s.source_id,
                source_name=s.source_name,
                score=s.score,
                confidence=s.confidence,
                reason=s.reason,
            )
            for s in suggest_merges(result.project.samples)
        ]

    @app.get("/samples/anomalies")
    def sample_anomalies() -> list[SampleAnomalyOut]:
        """Flag samples that probably aren't real samples (read-only)."""
        result = state.result
        if result is None:
            raise HTTPException(status_code=404, detail="No project is open")
        return [
            SampleAnomalyOut(
                sample_id=a.sample_id,
                sample_name=a.sample_name,
                kind=a.kind,
                message=a.message,
                related=list(a.related),
            )
            for a in flag_anomalies(result.project.samples)
        ]

    @app.get("/samples/{sample_id}/thermoelectric")
    def sample_thermoelectric(sample_id: str) -> ThermoelectricResult:
        """Derive zT(T) for a sample from its R&S + LFA measurements.

        422 when the sample lacks one of the two required measurements.
        """
        result = state.result
        store = state.array_store()
        if result is None or store is None:
            raise HTTPException(status_code=404, detail="No project is open")
        sample = next((s for s in result.project.samples if s.id == sample_id), None)
        if sample is None:
            raise HTTPException(status_code=404, detail="Unknown sample")
        try:
            zt = transport_data.sample_zt(sample, store.load)
        except TransportError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ThermoelectricResult(
            temperature_k=zt.temperature_k.tolist(),
            zt=zt.zt.tolist(),
            power_factor_uw_mk2=zt.power_factor_uw_mk2.tolist(),
            peak_zt=zt.peak_zt,
            peak_zt_temperature_k=zt.peak_zt_temperature_k,
            provenance=zt.provenance,
            warnings=zt.warnings,
        )

    @app.get("/measurements/{measurement_id}/analysis")
    def measurement_analysis(measurement_id: str) -> list[AnalyzerResultOut]:
        """Run the applicable analyzers on a measurement (stateless, on demand).

        Returns one entry per analyzer that accepts the measurement —
        e.g. a band gap for UV-DRS, fitted peaks for XRD. Empty list when
        no analyzer applies. Re-running these fits is cheap, so nothing
        is cached or persisted.
        """
        result = state.result
        store = state.array_store()
        if result is None or store is None:
            raise HTTPException(status_code=404, detail="No project is open")
        measurement = _find_measurement(result, measurement_id)
        if measurement is None:
            raise HTTPException(status_code=404, detail="Unknown measurement")
        overrides = _hall_cross_technique_overrides(result, store, measurement)
        return _run_analysis(measurement, store.load(measurement_id), overrides)


def _register_review_routes(app: FastAPI, state: ServerState) -> None:
    """Register the Review & Confirm edit endpoints on `app`.

    Kept out of `create_app` so that function stays small; these all
    funnel through `_apply`, which persists the edit and returns the
    refreshed project summary.
    """

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
        return _apply(state, lambda p: edits.merge_samples(p, body.source_ids, body.target_id))

    @app.post("/samples/split")
    def split_measurements(body: SplitMeasurementsRequest) -> ProjectSummary:
        return _apply(
            state,
            lambda p: edits.move_measurements_to_new_sample(p, body.measurement_ids, body.new_name),
        )

    @app.post("/measurements/move")
    def move_measurements(body: MoveMeasurementsRequest) -> ProjectSummary:
        return _apply(
            state,
            lambda p: edits.move_measurements_to_sample(
                p, body.measurement_ids, body.target_sample_id
            ),
        )

    @app.post("/measurements/remove")
    def remove_measurements(body: RemoveMeasurementsRequest) -> ProjectSummary:
        return _apply(state, lambda p: edits.remove_measurements(p, body.measurement_ids))


def _register_optimization_data_routes(app: FastAPI, state: ServerState) -> None:
    """Synthesis-parameter storage + the (x, y) dataset assembly (BO2)."""

    @app.post("/samples/{sample_id}/parameters")
    def set_parameters(sample_id: str, body: SampleParametersRequest) -> dict[str, str]:
        if state.result is None or state.root is None:
            raise HTTPException(status_code=404, detail="No project is open")
        known = {s.id for s in state.result.project.samples}
        if sample_id not in known:
            raise HTTPException(status_code=404, detail="Unknown sample")
        synthesis_store.set_sample_params(state.root, sample_id, body.parameters)
        return {"status": "ok"}

    @app.get("/parameters")
    def get_parameters() -> dict[str, dict[str, float]]:
        if state.root is None:
            raise HTTPException(status_code=404, detail="No project is open")
        return synthesis_store.load_params(state.root)

    @app.get("/optimize/targets")
    def optimize_targets() -> dict[str, list[str]]:
        result = state.result
        store = state.array_store()
        if result is None or store is None:
            raise HTTPException(status_code=404, detail="No project is open")
        return {"properties": optimization_data.list_target_properties(result.project, store)}

    @app.get("/optimize/inputs")
    def optimize_inputs() -> list[InputVariableOut]:
        """Available BO input axes: synthesis parameters + measured features.

        Measured features (e.g. the Hall carrier concentration) give a
        common physical axis when a sample set shares no synthesis knob.
        """
        result = state.result
        if result is None or state.root is None:
            raise HTTPException(status_code=404, detail="No project is open")
        params = synthesis_store.load_params(state.root)
        return [
            InputVariableOut(name=v.name, source=v.source, values=v.values)
            for v in optimization_data.list_input_variables(result.project, params)
        ]

    @app.get("/optimize/dataset")
    def optimize_dataset(input_variable: str, target_property: str) -> OptimizationDataset:
        result = state.result
        store = state.array_store()
        if result is None or store is None or state.root is None:
            raise HTTPException(status_code=404, detail="No project is open")
        params = synthesis_store.load_params(state.root)
        rows, skipped = optimization_data.build_dataset(
            result.project, store, params, input_variable, target_property
        )
        flags = optimization_data.quality_flags(
            result.project, rows, input_variable, target_property, store
        )
        return OptimizationDataset(
            input_variable=input_variable,
            target_property=target_property,
            points=[
                DatasetPoint(sample_id=r.sample_id, sample_name=r.sample_name, x=r.x, y=r.y)
                for r in rows
            ],
            skipped=[SkippedPoint(sample_name=s.sample_name, reason=s.reason) for s in skipped],
            quality_flags=[_flag_out(f) for f in flags],
        )

    @app.get("/optimize/spb")
    def optimize_spb() -> SpbCheckResult:
        """Single-parabolic-band physics read of the project's samples.

        For each sample that can derive zT, interpret its measured
        (Seebeck, zT) at the zT peak against SPB physics — where it sits
        versus its own zT optimum, or a multi-band / data flag when the
        pair is inconsistent with single-band transport. `best` is the
        highest-peak-zT sample.
        """
        result = state.result
        store = state.array_store()
        if result is None or store is None:
            raise HTTPException(status_code=404, detail="No project is open")
        return _assemble_spb_check(result.project, store)

    @app.post("/optimize/run")
    def optimize_run(body: OptimizeRunRequest) -> OptimizeResult:
        # Hard gate: optimization only runs on human-confirmed identity.
        asm = _assemble_optimization(state, body)
        try:
            res = optimize(
                asm.xs,
                asm.ys,
                bounds=asm.bounds,
                input_name=body.input_variable,
                target_name=asm.target_label,
                direction=asm.direction,
                y_transform=asm.y_transform,
                y_min=asm.y_min,
                y_max=asm.y_max,
            )
        except OptimizationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return OptimizeResult(
            input_variable=res.input_name,
            target_property=res.target_name,
            objective=body.objective,
            reliability_level=res.reliability.level if res.reliability else "unknown",
            reliability_note=res.reliability.note if res.reliability else "",
            quality_flags=asm.quality_flags,
            grid_x=list(res.grid_x),
            grid_mean=list(res.grid_mean),
            grid_ci95=list(res.grid_ci95),
            grid_lower=list(res.grid_lower),
            grid_upper=list(res.grid_upper),
            grid_ei=list(res.grid_ei),
            points=asm.points,
            best_x=res.best_x,
            best_y=res.best_y,
            recommendation=_rec_out(res.recommendation),
            max_ei=res.max_ei,
            noise_threshold=res.noise_threshold,
            converged=res.converged,
            verdict=_verdict(res),
        )

    @app.post("/optimize/freeze")
    def optimize_freeze(body: OptimizeRunRequest) -> FreezeResult:
        """Freeze the current recommendation into an auditable pre-registration record.

        Runs the optimizer + a kernel length-scale robustness sweep and writes a
        timestamped JSON (+ Markdown) under `<root>/.latos/prereg/` pinning the
        frozen config and the predicted value with its predictive interval — before
        the recommended sample is made, so it cannot be retuned afterwards.
        """
        return _freeze_recommendation(state, body)

    @app.get("/optimize/prereg")
    def list_prereg() -> list[PreregSummary]:
        """Every frozen pre-registration for the open project, newest first.

        Closes the loop's front half: each entry carries the committed
        prediction and, once its sample is measured and validated, the
        recorded outcome verdict.
        """
        if state.root is None:
            raise HTTPException(status_code=404, detail="No project is open")
        return [_prereg_summary(e) for e in optimization.list_preregistrations(state.root)]

    @app.post("/optimize/validate")
    def validate_prereg(body: ValidateOutcomeRequest) -> OutcomeVerdictOut:
        """Score a measured outcome against a frozen pre-registration.

        Reads the frozen record, judges calibration (inside the 95%
        interval?) and improvement (beat the prior best, in the optimized
        direction?), and writes the verdict to a ``*.outcome.json`` sibling
        so prediction and outcome stay side by side and auditable.
        """
        return _validate_prereg(state, body)


def _freeze_recommendation(state: ServerState, body: OptimizeRunRequest) -> FreezeResult:
    """Optimize, sweep kernel robustness, and write the pre-registration record."""
    asm = _assemble_optimization(state, body)
    assert state.root is not None  # _assemble_optimization guarantees an open project
    try:
        res = optimize(
            asm.xs,
            asm.ys,
            bounds=asm.bounds,
            input_name=body.input_variable,
            target_name=asm.target_label,
            direction=asm.direction,
            y_transform=asm.y_transform,
            y_min=asm.y_min,
            y_max=asm.y_max,
        )
        robustness = length_scale_robustness(
            asm.xs,
            asm.ys,
            bounds=asm.bounds,
            input_name=body.input_variable,
            target_name=asm.target_label,
            direction=asm.direction,
            y_transform=asm.y_transform,
            y_min=asm.y_min,
            y_max=asm.y_max,
            length_scales=(1.0, 2.0, 3.0, 4.0, 5.0),
        )
    except OptimizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stamp = res.config.created_at.strftime("%Y%m%dT%H%M%SZ")
    out_path = state.root / ".latos" / "prereg" / f"prereg_{stamp}.json"
    freeze(res, out_path, prior_best=res.best_y, robustness=robustness)
    return FreezeResult(
        path=str(out_path),
        recommendation=_rec_out(res.recommendation),
        prior_best=res.best_y,
        robustness_stable=robustness.stable,
        converged=res.converged,
        reliability_level=res.reliability.level if res.reliability else "unknown",
        reliability_note=res.reliability.note if res.reliability else "",
    )


@dataclass(frozen=True)
class _AssembledOptimization:
    """The (x, y) table plus the resolved objective, ready for the engine."""

    points: list[DatasetPoint]
    xs: np.ndarray
    ys: np.ndarray
    bounds: tuple[float, float]
    target_label: str
    direction: str  # what the engine runs: "maximize" | "minimize"
    quality_flags: list[QualityFlagOut]  # untrustworthy points (warn, don't block)
    # Physics layer: the fit space + physical clamp bounds for the target.
    y_transform: str  # "identity" | "log"
    y_min: float | None
    y_max: float | None


def _assemble_optimization(state: ServerState, body: OptimizeRunRequest) -> _AssembledOptimization:
    """Shared assembly for /optimize/run and /optimize/freeze.

    Resolves the input variable (synthesis parameter or measured feature),
    the target (array property, feature, or derived zT — optionally at a
    temperature), and the objective mode. "target" mode is implemented as
    exact minimization of |y - target_value|, relabelled so the chart and
    verdict speak in distance units.
    """
    _require_confirmed(state)
    result = state.result
    store = state.array_store()
    if result is None or store is None or state.root is None:
        raise HTTPException(status_code=404, detail="No project is open")

    if body.objective not in ("maximize", "minimize", "target"):
        raise HTTPException(
            status_code=400,
            detail=f"objective must be maximize, minimize or target; got {body.objective!r}",
        )

    params = synthesis_store.load_params(state.root)
    rows, _skipped = optimization_data.build_dataset(
        result.project,
        store,
        params,
        body.input_variable,
        body.target_property,
        at_temperature_k=body.at_temperature_k,
    )
    if len(rows) < _MIN_OPTIMIZE_POINTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least {_MIN_OPTIMIZE_POINTS} samples with both a "
                f"'{body.input_variable}' value and '{body.target_property}' data; "
                f"only {len(rows)} qualify."
            ),
        )

    xs = np.array([r.x for r in rows])
    ys = np.array([r.y for r in rows])
    target_label = body.target_property
    if body.target_property == optimization_data.DERIVED_ZT and body.at_temperature_k:
        target_label = f"{target_label} @ {body.at_temperature_k:g} K"

    direction = "maximize"
    if body.objective == "minimize":
        direction = "minimize"
    elif body.objective == "target":
        if body.target_value is None:
            raise HTTPException(status_code=400, detail="objective 'target' requires target_value")
        ys = np.abs(ys - body.target_value)
        target_label = f"|{target_label} - {body.target_value:g}|"
        direction = "minimize"

    points = [
        DatasetPoint(sample_id=r.sample_id, sample_name=r.sample_name, x=r.x, y=float(y))
        for r, y in zip(rows, ys.tolist(), strict=True)
    ]
    # Data-quality flags use the ORIGINAL rows and property (raw values), not
    # the target-mode transform, so a Hall-derived target is judged correctly.
    flags = [
        _flag_out(fl)
        for fl in optimization_data.quality_flags(
            result.project, rows, body.input_variable, body.target_property, store
        )
    ]
    # Physics layer: choose the fit space + clamp bounds from the property's
    # physics. Target mode optimizes a distance |y - t| (not the property
    # itself), so it stays linear/unclamped.
    y_transform, y_min, y_max = "identity", None, None
    prop = physics.lookup(body.target_property)
    if prop is not None and body.objective != "target":
        if prop.log_natural:
            y_transform = "log"
        if prop.positive:  # only clamp strictly-positive quantities
            y_min, y_max = prop.min_value, prop.max_value

    bounds = body.bounds or (float(xs.min()), float(xs.max()))
    return _AssembledOptimization(
        points=points,
        xs=xs,
        ys=ys,
        bounds=bounds,
        target_label=target_label,
        direction=direction,
        quality_flags=flags,
        y_transform=y_transform,
        y_min=y_min,
        y_max=y_max,
    )


def _flag_out(flag: optimization_data.QualityFlag) -> QualityFlagOut:
    """Map a core QualityFlag to the API shape."""
    return QualityFlagOut(
        sample_name=flag.sample_name,
        variable=flag.variable,
        value=flag.value,
        reason=flag.reason,
    )


def _assemble_spb_check(project: Project, store: ArrayStore) -> SpbCheckResult:
    """Single-parabolic-band read of every sample that can derive zT.

    `best` is the highest-peak-zT sample. Kept out of the route body so the
    route-registration function stays within its statement budget.
    """
    samples: list[SpbSampleOut] = []
    for sample in project.samples:
        g = transport_data.sample_spb_guidance(sample, store.load)
        if g is None:
            continue
        samples.append(
            SpbSampleOut(
                sample_name=sample.canonical_name,
                applicable=g.applicable,
                note=g.note,
                measured_seebeck_uv_k=g.measured_seebeck_uv_k,
                measured_zt=g.measured_zt,
                beta=g.beta,
                optimal_seebeck_uv_k=g.optimal_seebeck_uv_k,
                zt_ceiling=g.zt_ceiling,
                direction=g.direction,
            )
        )
    best = max(samples, key=lambda s: s.measured_zt, default=None)
    return SpbCheckResult(best=best, samples=samples)


def _validate_prereg(state: ServerState, body: ValidateOutcomeRequest) -> OutcomeVerdictOut:
    """Score a measured outcome against a frozen record and persist the verdict.

    The record path is confined to this project's ``.latos/prereg/``
    directory so the endpoint can only read/write inside the open project.
    """
    if state.root is None:
        raise HTTPException(status_code=404, detail="No project is open")
    expected_dir = (state.root / ".latos" / "prereg").resolve()
    try:
        resolved = Path(body.prereg_path).resolve()
        in_dir = resolved.parent == expected_dir
    except OSError:
        in_dir = False
    if not in_dir or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Unknown pre-registration for this project")
    try:
        record = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the record: {exc}") from exc
    verdict = optimization.validate_outcome(record, body.measured_value)
    optimization.write_outcome(resolved, verdict)
    return _verdict_out(verdict)


def _prereg_summary(entry: optimization.PreregEntry) -> PreregSummary:
    """Map a core PreregEntry to the API shape."""
    return PreregSummary(
        path=entry.path,
        created_at=entry.created_at,
        input_variable=entry.input_variable,
        property_name=entry.property_name,
        direction=entry.direction,
        recommended_x=entry.recommended_x,
        predicted_mean=entry.predicted_mean,
        predictive_interval_95=entry.predictive_interval_95,
        prior_best=entry.prior_best,
        reliability_level=entry.reliability_level,
        outcome=(_verdict_out_from_dict(entry.outcome) if entry.outcome else None),
    )


def _verdict_out(verdict: optimization.OutcomeVerdict) -> OutcomeVerdictOut:
    """Map a core OutcomeVerdict to the API shape."""
    return OutcomeVerdictOut(
        measured=verdict.measured,
        predicted_mean=verdict.predicted_mean,
        predictive_interval_95=verdict.predictive_interval_95,
        prior_best=verdict.prior_best,
        direction=verdict.direction,
        within_interval=verdict.within_interval,
        improved=verdict.improved,
        signed_error=verdict.signed_error,
        absolute_error=verdict.absolute_error,
        relative_error=verdict.relative_error,
        summary=verdict.summary,
        validated_at=verdict.validated_at,
    )


def _verdict_out_from_dict(data: dict[str, object]) -> OutcomeVerdictOut:
    """Rebuild an OutcomeVerdictOut from a persisted outcome payload."""
    interval = data.get("predictive_interval_95") or [0.0, 0.0]
    return OutcomeVerdictOut(
        measured=float(data["measured"]),  # type: ignore[arg-type]
        predicted_mean=float(data["predicted_mean"]),  # type: ignore[arg-type]
        predictive_interval_95=(float(interval[0]), float(interval[1])),  # type: ignore[index]
        prior_best=float(data["prior_best"]),  # type: ignore[arg-type]
        direction=str(data.get("direction", "maximize")),
        within_interval=bool(data["within_interval"]),
        improved=bool(data["improved"]),
        signed_error=float(data["signed_error"]),  # type: ignore[arg-type]
        absolute_error=float(data["absolute_error"]),  # type: ignore[arg-type]
        relative_error=(
            float(data["relative_error"])  # type: ignore[arg-type]
            if data.get("relative_error") is not None
            else None
        ),
        summary=str(data.get("summary", "")),
        validated_at=str(data.get("validated_at", "")),
    )


def _rec_out(rec: Recommendation) -> RecommendationOut:
    """Map an engine `Recommendation` to the API shape (both CIs + interval).

    The interval comes straight from the engine's `predictive_interval_95` —
    it is exact and physically-bounded (asymmetric for a log-space fit), not a
    symmetric ± half-width, which would be wrong for a positive quantity.
    """
    return RecommendationOut(
        x=rec.x,
        predicted_mean=rec.predicted_mean,
        ci95=rec.ci95,
        ci95_predictive=rec.ci95_predictive,
        predictive_interval_95=rec.predictive_interval_95,
    )


def _verdict(res: OptimizationResult) -> str:
    """Plain-language summary of an optimization result for the UI.

    No jargon — this is read by materials scientists, not CS people.
    """
    rec = res.recommendation
    best_word = "lowest" if res.config.direction == "minimize" else "best"
    if res.converged:
        return (
            f"Optimum reached within measurement precision. "
            f"{best_word.capitalize()} so far: {res.best_y:.3f} at "
            f"{res.input_name} = {res.best_x:g}. "
            f"A confirmatory run at {rec.x:.3g} is optional but unlikely to improve."
        )
    return (
        f"Recommended next experiment: {res.input_name} = {rec.x:.3g} "
        f"(predicted {res.target_name} {rec.predicted_mean:.2f} "
        f"+/- {rec.ci95_predictive:.2f}, 95% predictive). "
        f"A meaningful improvement over the current {best_word} ({res.best_y:.3f}) "
        f"is still expected."
    )


def _find_measurement(result: IngestionResult, measurement_id: str) -> Measurement | None:
    """Locate a measurement by id within a project, or None."""
    for sample in result.project.samples:
        for measurement in sample.measurements:
            if measurement.id == measurement_id:
                return measurement
    return None


def _folder_of(measurement: Measurement, root: Path) -> str | None:
    """Source file's directory relative to the project root, posix-style.

    `""` means the file sits directly in the project root; `None` means
    the measurement has no file (or lives outside the root, which
    shouldn't happen). Drives the folder tree in the review UI.
    """
    if not measurement.files:
        return None
    try:
        rel = measurement.files[0].path.parent.relative_to(root)
    except ValueError:
        return None
    posix = rel.as_posix()
    return "" if posix == "." else posix


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
