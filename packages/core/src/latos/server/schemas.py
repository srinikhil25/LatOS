"""Pydantic response/request models for the Latos sidecar API.

These are the wire contract between `latos-core` and the desktop UI
(`apps/desktop`). Keep them flat and JSON-friendly — the React side
generates its TypeScript types from this shape via the OpenAPI schema.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = [
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


class OptimizationDataset(BaseModel):
    """GET /optimize/dataset — the (x, y) table + what was skipped."""

    input_variable: str
    target_property: str
    points: list[DatasetPoint]
    skipped: list[SkippedPoint]


class OptimizeRunRequest(BaseModel):
    """POST /optimize/run — run one BO round over a chosen variable/target."""

    input_variable: str
    target_property: str
    bounds: tuple[float, float] | None = None  # default: observed data range


class RecommendationOut(BaseModel):
    """The single recommended next experiment."""

    x: float
    predicted_mean: float
    ci95: float


class OptimizeResult(BaseModel):
    """POST /optimize/run — posterior curve + recommendation + verdict."""

    input_variable: str
    target_property: str
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


class MeasurementSummary(BaseModel):
    """One measurement row in the samples tree."""

    id: str
    technique: str
    instrument: str | None
    filename: str | None
    folder: str | None  # source file's dir, relative to project root (posix)


class SampleSummary(BaseModel):
    """One sample node in the samples tree."""

    id: str
    name: str
    aliases: list[str]
    measurements: list[MeasurementSummary]


class MeasurementArrays(BaseModel):
    """GET /measurements/{id}/arrays — parsed columns for plotting.

    `arrays` values use `None` for non-finite samples (NaN/inf are not
    valid JSON); the UI treats them as gaps in the trace.
    """

    measurement_id: str
    names: list[str]
    arrays: dict[str, list[float | None]]
