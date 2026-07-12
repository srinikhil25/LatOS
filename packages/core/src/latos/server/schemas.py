"""Pydantic response/request models for the Latos sidecar API.

These are the wire contract between `latos-core` and the desktop UI
(`apps/desktop`). Keep them flat and JSON-friendly — the React side
generates its TypeScript types from this shape via the OpenAPI schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = [
    "DeleteProjectRequest",
    "DeleteProjectResult",
    "HealthResponse",
    "IngestStartedResponse",
    "MeasurementSummary",
    "OpenProjectRequest",
    "ProjectSummary",
    "SampleSummary",
]


class HealthResponse(BaseModel):
    """GET /health — liveness + version handshake for the shell."""

    status: str
    version: str


class OpenProjectRequest(BaseModel):
    """POST /project/open — start ingesting a folder."""

    root: str
    project_name: str | None = None


class IngestStartedResponse(BaseModel):
    """202 body acknowledging the ingestion thread has started."""

    status: str


class DeleteProjectRequest(BaseModel):
    """POST /project/delete — recycle a project's derived ``.latos/`` store."""

    root: str


class DeleteProjectResult(BaseModel):
    """Outcome of a project delete. Raw files are never touched.

    Attributes:
        root: The project folder acted on.
        removed: True if a ``.latos/`` store existed and was removed.
        recycled: True if it went to the Recycle Bin (recoverable), False if it
            had to be permanently deleted.
    """

    root: str
    removed: bool
    recycled: bool


class ProjectSummary(BaseModel):
    """GET /project — headline numbers for the hub screen."""

    id: str
    name: str
    root_path: str
    samples: int
    measurements: int
    techniques: int
    parsed: int
    cached: int
    failed: int
    unclassified: int
    review_status: str  # "needs_review" | "confirmed"


class RenameSampleRequest(BaseModel):
    """POST /samples/{id}/rename."""

    name: str


class SetTechniqueRequest(BaseModel):
    """POST /measurements/{id}/technique."""

    technique: str


class MergeSamplesRequest(BaseModel):
    """POST /samples/merge — fold `source_ids` into `target_id`."""

    source_ids: list[str]
    target_id: str


class SplitMeasurementsRequest(BaseModel):
    """POST /samples/split — pull measurements into a new named sample."""

    measurement_ids: list[str]
    new_name: str


class MoveMeasurementsRequest(BaseModel):
    """POST /measurements/move — reassign to an existing sample."""

    measurement_ids: list[str]
    target_sample_id: str


class RemoveMeasurementsRequest(BaseModel):
    """POST /measurements/remove — drop from the project (soft delete)."""

    measurement_ids: list[str]


# ─── Optimization (BO) ───────────────────────────────────────────────
class SampleParametersRequest(BaseModel):
    """POST /samples/{id}/parameters — set a sample's synthesis inputs."""

    parameters: dict[str, float]


class DatasetPoint(BaseModel):
    """One usable (input, target) optimization point."""

    sample_id: str
    sample_name: str
    x: float
    y: float


class SkippedPoint(BaseModel):
    """A sample left out of the optimization dataset, with why."""

    sample_name: str
    reason: str


class QualityFlagOut(BaseModel):
    """A dataset point whose target/axis value is flagged untrustworthy.

    Raised when a Hall-derived carrier metric comes from an unreliable Hall
    measurement, or a value is physically impossible (e.g. negative mobility).
    The run still proceeds — this warns, it does not block.
    """

    sample_name: str
    variable: str
    value: float
    reason: str


class OptimizationDataset(BaseModel):
    """GET /optimize/dataset — the (x, y) table + what was skipped/flagged."""

    input_variable: str
    target_property: str
    points: list[DatasetPoint]
    skipped: list[SkippedPoint]
    quality_flags: list[QualityFlagOut] = []


class OptimizeRunRequest(BaseModel):
    """POST /optimize/run — run one BO round over a chosen variable/target.

    `objective` selects what "best" means:
    - "maximize" (default) / "minimize" — direction of optimization.
    - "target" — reach `target_value`; internally the distance
      |y − target_value| is minimized, and the result is labelled as a
      distance so the chart and verdict stay interpretable.
    `at_temperature_k` applies only to the derived-zT target: optimize zT
    at that temperature instead of the peak.
    """

    input_variable: str
    target_property: str
    bounds: tuple[float, float] | None = None  # default: observed data range
    objective: str = "maximize"  # "maximize" | "minimize" | "target"
    target_value: float | None = None  # required when objective == "target"
    at_temperature_k: float | None = None  # derived-zT only


class InputVariableOut(BaseModel):
    """One available BO input axis with its per-sample values.

    `source` is "synthesis" (researcher-entered, editable) or "measured"
    (from an instrument, e.g. Hall carrier concentration — read-only).
    """

    name: str
    source: str
    values: dict[str, float]  # sample_id -> value


class RecommendationOut(BaseModel):
    """The single recommended next experiment."""

    x: float
    predicted_mean: float
    ci95: float  # model (epistemic) 95% half-width
    ci95_predictive: float  # 95% half-width a new measurement should fall within
    predictive_interval_95: tuple[float, float]  # [low, high] — for calibration checks


class OptimizeResult(BaseModel):
    """POST /optimize/run — posterior curve + recommendation + verdict."""

    input_variable: str
    target_property: str  # display label (e.g. "|zT − 1.0|" in target mode)
    objective: str = "maximize"  # direction actually optimized
    # Self-assessed trustworthiness of the model's intervals, from the
    # data itself (count tier + leave-one-out): exploratory | indicative
    # | calibrated.
    reliability_level: str = "unknown"
    reliability_note: str = ""
    # Points whose target/axis value can't be trusted (e.g. an unreliable
    # Hall measurement). Non-empty means "warn before acting on this run".
    quality_flags: list[QualityFlagOut] = []
    # Posterior over the search range, for the curve.
    grid_x: list[float]
    grid_mean: list[float]
    grid_ci95: list[float]
    grid_ei: list[float]
    # Observed points, with sample names for labelling.
    points: list[DatasetPoint]
    best_x: float
    best_y: float
    recommendation: RecommendationOut
    max_ei: float
    noise_threshold: float
    converged: bool
    verdict: str  # plain-language summary for the UI


class FreezeResult(BaseModel):
    """POST /optimize/freeze — the committed pre-registration record.

    Writes a timestamped JSON (+ Markdown sibling) that pins the frozen model
    config and the predicted value with its predictive interval, *before* the
    recommended sample is made — so the later measurement can be checked for
    calibration and improvement against a prediction that could not have been
    retuned after the fact.
    """

    path: str  # where the JSON record was written (a .md sibling sits beside it)
    recommendation: RecommendationOut
    prior_best: float
    robustness_stable: bool
    converged: bool
    reliability_level: str = "unknown"  # exploratory | indicative | calibrated
    reliability_note: str = ""


class OutcomeVerdictOut(BaseModel):
    """The score of a measured outcome against a frozen prediction.

    `within_interval` is the calibration criterion (did the measurement
    land inside the committed 95% interval?); `improved` is the
    improvement criterion (did it beat the prior best, in the optimized
    direction?).
    """

    measured: float
    predicted_mean: float
    predictive_interval_95: tuple[float, float]
    prior_best: float
    direction: str
    within_interval: bool
    improved: bool
    signed_error: float
    absolute_error: float
    relative_error: float | None
    summary: str
    validated_at: str


class PreregSummary(BaseModel):
    """GET /optimize/prereg — one frozen pre-registration + its outcome."""

    path: str
    created_at: str
    input_variable: str
    property_name: str
    direction: str
    recommended_x: float
    predicted_mean: float
    predictive_interval_95: tuple[float, float]
    prior_best: float
    reliability_level: str
    outcome: OutcomeVerdictOut | None = None  # None until the sample is validated


class ValidateOutcomeRequest(BaseModel):
    """POST /optimize/validate — score a measured value against a record.

    `measured_value` must be in the same units as the frozen prediction
    (the property the freeze optimized).
    """

    prereg_path: str
    measured_value: float


class MeasurementSummary(BaseModel):
    """One measurement row in the samples tree."""

    id: str
    technique: str
    instrument: str | None
    filename: str | None
    folder: str | None  # source file's dir, relative to project root (posix)
    features: dict[str, float] = {}  # curated scalar features (Hall, EDS, …)


class SampleSummary(BaseModel):
    """One sample node in the samples tree."""

    id: str
    name: str
    aliases: list[str]
    measurements: list[MeasurementSummary]


class MergeSuggestionOut(BaseModel):
    """GET /samples/merge-suggestions — one proposed (suggest-only) merge.

    `target` is the cleaner spelling to keep; `source` would fold into it.
    Confirming calls POST /samples/merge with these ids.
    """

    target_id: str
    target_name: str
    source_id: str
    source_name: str
    score: float
    confidence: str  # "high" | "medium"
    reason: str


class AnalyzerResultOut(BaseModel):
    """One analyzer's result for GET /measurements/{id}/analysis.

    `outputs` is the analyzer's JSON-safe scalar payload (band gap, peak
    positions, …). `issues` are human-readable "severity: message" lines.
    Computed on demand — not persisted (re-running these fits is cheap).
    """

    analyzer: str
    outputs: dict[str, Any]
    issues: list[str]


class ThermoelectricResult(BaseModel):
    """GET /samples/{id}/thermoelectric — zT(T) derived from R&S + LFA.

    Arrays are aligned to `temperature_k`. `provenance` lists the
    derivation steps; `warnings` carries plausibility flags.
    """

    temperature_k: list[float]
    zt: list[float]
    power_factor_uw_mk2: list[float]
    peak_zt: float
    peak_zt_temperature_k: float
    provenance: list[str]
    warnings: list[str]


class SampleAnomalyOut(BaseModel):
    """GET /samples/anomalies — one sample flagged for human attention.

    `kind` is "mixed_samples" (files belong to other samples) or
    "non_sample_name" (the name looks like a folder/date). `related`
    lists the other sample names for the mixed case.
    """

    sample_id: str
    sample_name: str
    kind: str
    message: str
    related: list[str]


class MeasurementArrays(BaseModel):
    """GET /measurements/{id}/arrays — parsed columns for plotting.

    `arrays` values use `None` for non-finite samples (NaN/inf are not
    valid JSON); the UI treats them as gaps in the trace.
    """

    measurement_id: str
    names: list[str]
    arrays: dict[str, list[float | None]]
