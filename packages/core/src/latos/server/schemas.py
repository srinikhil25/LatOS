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


class MeasurementSummary(BaseModel):
    """One measurement row in the samples tree."""

    id: str
    technique: str
    instrument: str | None
    filename: str | None


class SampleSummary(BaseModel):
    """One sample node in the samples tree."""

    id: str
    name: str
    aliases: list[str]
    measurements: list[MeasurementSummary]
