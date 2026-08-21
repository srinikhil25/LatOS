"""Pydantic response/request models for the Latos sidecar API.

These are the wire contract between `latos-core` and the desktop UI
(`apps/desktop`). Keep them flat and JSON-friendly — the React side
generates its TypeScript types from this shape via the OpenAPI schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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


class SpbSampleOut(BaseModel):
    """One sample's single-parabolic-band read (Seebeck vs zT at its zT peak)."""

    sample_name: str
    applicable: bool
    note: str
    measured_seebeck_uv_k: float
    measured_zt: float
    beta: float | None = None
    optimal_seebeck_uv_k: float | None = None
    zt_ceiling: float | None = None
    direction: str | None = None  # increase_seebeck | decrease_seebeck | at_optimum


class SpbCheckResult(BaseModel):
    """GET /optimize/spb — physics-informed read of the project's best sample.

    The single-parabolic-band model interprets each sample's measured
    (Seebeck, zT) against thermoelectric physics: whether it sits below,
    at, or above its own zT optimum, or — when the pair is inconsistent
    with single-band transport — an explicit multi-band / data flag rather
    than a fabricated target. `best` is the sample with the highest peak zT.
    """

    best: SpbSampleOut | None
    samples: list[SpbSampleOut] = []


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
    # Posterior over the search range, for the curve. grid_lower/grid_upper are
    # the explicit 95% band in physical units (asymmetric for a log-space fit,
    # clamped to the physical domain); grid_ci95 is a symmetric approximation.
    grid_x: list[float]
    grid_mean: list[float]
    grid_ci95: list[float]
    grid_lower: list[float]
    grid_upper: list[float]
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
    # Probabilistic regret bound (Wilson, NeurIPS 2024): the chance that the
    # best measured point is already within `epsilon` of the true optimum,
    # *under this model*. Read it next to `reliability_level`, which says how
    # far the model itself can be trusted.
    epsilon: float = 0.0
    delta: float = 0.1
    prob_within_epsilon: float = 0.0
    epsilon_delta_met: bool = False
    # Observations down-weighted in the fit because a physics check rejected
    # them (they are never silently dropped).
    n_unreliable: int = 0
    # How many of those the researcher marked by hand, rather than a physics
    # check. Shown back so a ticked box visibly changes the run.
    n_distrusted: int = 0
    # Whether `noise_threshold` is the observed repeatability of repeat
    # measurements, or the assumed relative noise. The verdict is "the expected
    # gain is below the noise", so which one it is changes what that means.
    noise_measured: bool = False


class OptimizeRunNdRequest(BaseModel):
    """POST /optimize/run-nd — one BO round over several input axes at once.

    The multi-axis sibling of `OptimizeRunRequest`. Kept separate rather than
    widening that model because `OptimizeRunRequest` is also the body of
    `/optimize/freeze`, whose records are on disk in a one-variable shape.

    Samples missing a value on *any* chosen axis are dropped, so adding an axis
    can shrink the dataset — the response reports how many were lost. `bounds`,
    when given, must have one (low, high) pair per axis in the same order.
    """

    input_variables: list[str]
    target_property: str
    bounds: list[tuple[float, float]] | None = None  # default: observed range per axis
    objective: str = "maximize"  # "maximize" | "minimize" | "target"
    target_value: float | None = None  # required when objective == "target"
    at_temperature_k: float | None = None  # derived-zT only
    # Side of the lattice the posterior surface is reported on (2 axes only).
    # Bounded because the cost is the square: 256 is already 65k GP predictions
    # and finer than any screen resolves, so a larger request is a mistake worth
    # rejecting rather than a preference worth honouring.
    # Zero skips it, which is what a caller that only wants the recommendation
    # should send.
    surface_size: int = Field(default=48, ge=0, le=256)


class NdDatasetPoint(BaseModel):
    """One observed point in the multi-axis dataset; `x` is a coordinate."""

    sample_id: str
    sample_name: str
    x: list[float]
    y: float


class RecommendationNdOut(BaseModel):
    """The recommended next experiment, one value per input axis."""

    x: list[float]
    predicted_mean: float
    ci95: float
    ci95_predictive: float
    predictive_interval_95: tuple[float, float]


class AxisOut(BaseModel):
    """One input axis of a multi-dimensional run.

    `length_scale` is the fitted ARD value in normalized units, where each
    axis's search range spans the same distance, and it is the diagnostic worth
    reading. `pinned_at` says whether the fit ran into the range it was allowed:

      "high" — the model found no structure along this axis; it does not move
               the target over this range.
      "low"  — the model wants finer resolution than the floor permits, so it
               is under-resolving real structure. Seen on multi-axis fits,
               where the floor is inherited from the one-variable engine.
      None   — the scale settled in the interior, which is the healthy case.

    The two mean opposite things, which is why this is not a boolean.
    """

    name: str
    low: float
    high: float
    length_scale: float
    pinned_at: str | None = None  # "low" | "high" | None


class SurfaceOut(BaseModel):
    """The posterior on a regular lattice, for a 2-D contour or heat map.

    `mean[j][i]` is the value at `(axis_x[i], axis_y[j])`. Only present when
    exactly two axes were optimized and a non-zero `surface_size` was asked for.
    """

    axis_names: tuple[str, str]
    axis_x: list[float]
    axis_y: list[float]
    mean: list[list[float]]
    sd: list[list[float]]
    ei: list[list[float]]


class OptimizeNdResult(BaseModel):
    """POST /optimize/run-nd — multi-axis posterior + recommendation + verdict.

    Mirrors `OptimizeResult` field for field except where dimension forces a
    difference: the 1-D `grid_*` curve becomes `surface` (2-D only), and the
    scalar `x` fields become coordinates.
    """

    input_variables: list[str]
    target_property: str
    objective: str = "maximize"
    axes: list[AxisOut]
    kernel: str
    acquisition: str  # "sobol" or "sobol+lbfgsb"
    reliability_level: str = "unknown"
    reliability_note: str = ""
    # Geometric half of the grade: the radius of the largest unsampled hole,
    # and the largest hole the claimed tier tolerates. Counting points cannot
    # answer coverage once there is more than one axis; this can.
    fill_distance: float = 0.0
    fill_limit: float = 0.0
    quality_flags: list[QualityFlagOut] = []
    surface: SurfaceOut | None = None
    points: list[NdDatasetPoint]
    # Samples that have the target and the first axis but are missing a value on
    # one of the added axes. Adding an axis costs data, and it should be visible.
    n_dropped_for_missing_axis: int = 0
    best_x: list[float]
    best_y: float
    recommendation: RecommendationNdOut
    max_ei: float
    noise_threshold: float
    converged: bool
    verdict: str
    epsilon: float = 0.0
    delta: float = 0.1
    prob_within_epsilon: float = 0.0
    epsilon_delta_met: bool = False
    n_unreliable: int = 0
    n_distrusted: int = 0
    noise_measured: bool = False


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
    # Whether the frozen "we are already within epsilon of the optimum" claim
    # survived this measurement. None for records frozen before the claim
    # existed, so an older pre-registration still validates cleanly.
    stopping_claim_held: bool | None = None


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


class DriftStepOut(BaseModel):
    """One move of the recommendation between two consecutive freezes."""

    from_created_at: str
    to_created_at: str
    from_x: float
    to_x: float
    distance: float  # in the input variable's own units
    fraction_of_span: float  # the same move, relative to the search range


class CampaignDriftOut(BaseModel):
    """GET /optimize/drift — is the campaign still changing its mind?

    An out-of-model convergence check (Ishiyama et al., NPG Asia Mater. 16,
    17, 2024): the distance between successive frozen recommendations. Every
    other stopping signal asks the model about itself, so they all fail
    together when the model is wrong. This reads the records on disk instead.

    `settled` is None with fewer than two freezes — a single point cannot
    show movement.
    """

    input_variable: str
    property_name: str
    direction: str
    n_freezes: int
    steps: list[DriftStepOut]
    search_span: float | None = None
    latest_fraction: float | None = None
    settled: bool | None = None
    note: str = ""


class DistrustRequest(BaseModel):
    """POST /samples/{sample_id}/distrust — the researcher's own quality call.

    Marks a sample as not to be trusted. It is down-weighted in the fit, the
    same treatment a physics-flagged point gets, and never deleted.
    """

    distrusted: bool


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


# ─── Fit engine (Stage 4) ──────────────────────────────────────────────
class DetectPeaksRequest(BaseModel):
    """POST /fit/detect-peaks — auto-detect candidate peak centers."""

    x: list[float]
    y: list[float]
    max_peaks: int = 30
    min_prominence_frac: float = 0.02


class DetectPeaksResult(BaseModel):
    """Candidate peak centers (x-values), strongest first."""

    centers: list[float]


class FitBackgroundInput(BaseModel):
    """Background choice for a fit request."""

    kind: str = "linear"  # none|constant|linear|polynomial|shirley|als
    degree: int = 2
    lam: float = 1e5
    p: float = 0.01


class FitConstraintInput(BaseModel):
    """One inter-peak constraint (peaks referenced by index)."""

    type: str  # fixed_delta | fixed_ratio | shared_width
    ref: int
    target: int
    delta: float | None = None  # fixed_delta
    ratio: float | None = None  # fixed_ratio


class FitRequest(BaseModel):
    """POST /fit — fit N peaks of one shape over a background to (x, y)."""

    x: list[float]
    y: list[float]
    peak_shape: str = "pseudo_voigt"
    peaks: list[float]  # initial peak centers
    background: FitBackgroundInput = FitBackgroundInput()
    constraints: list[FitConstraintInput] = []


class FitParamOut(BaseModel):
    """A fitted parameter's value and 1σ uncertainty (None if unestimated)."""

    name: str
    value: float
    stderr: float | None


class FitComponentOut(BaseModel):
    """One fitted peak's headline numbers."""

    center: float
    amplitude: float
    sigma: float
    fwhm: float | None
    height: float | None


class FitResultOut(BaseModel):
    """POST /fit — the fit, its components, arrays for overlay, and a report."""

    success: bool
    r_squared: float
    chi_square: float
    reduced_chi_square: float
    components: list[FitComponentOut]
    params: list[FitParamOut]
    baseline: list[float]
    best_fit: list[float]
    residual: list[float]
    markdown: str


class FitPresetsOut(BaseModel):
    """GET /fit/presets — known XPS spin-orbit doublets: name -> [ΔBE, ratio]."""

    doublets: dict[str, list[float]]


# ─── Cross-correlation + reporting (Stage 6) ───────────────────────────
class FeatureCellOut(BaseModel):
    """One property's value for one sample, with provenance."""

    value: float
    unit: str
    source: str
    reliable: bool


class FeatureRowOut(BaseModel):
    """A sample's row in the feature table."""

    sample_id: str
    sample_name: str
    features: dict[str, FeatureCellOut]


class FeatureTableOut(BaseModel):
    """GET /features — the sample × property matrix with provenance."""

    properties: list[str]
    rows: list[FeatureRowOut]


class CorrelationOut(BaseModel):
    """One property-pair relationship."""

    property_a: str
    property_b: str
    pearson: float
    spearman: float
    n: int


class CorrelationsOut(BaseModel):
    """GET /correlations — Pearson matrix (heatmap) + ranked pairs."""

    properties: list[str]
    matrix: list[list[float | None]]
    pairs: list[CorrelationOut]
