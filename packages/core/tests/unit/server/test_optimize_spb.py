"""Tests for GET /optimize/spb — the single-parabolic-band physics check.

The SPB math is unit-tested in tests/unit/optimization/test_spb.py; here we
check the endpoint wiring: pairing Seebeck at the zT-peak temperature, picking
the best sample, and surfacing the applicable vs multi-band-flag verdicts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from latos.core.enums import ReviewStatus, Technique
from latos.core.models import Measurement, Project, Sample, new_id, utc_now
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.orchestrator import IngestionResult
from latos.ingestion.parsed_data import ParsedData
from latos.server.app import create_app
from latos.server.state import ServerState


def _measurement(sample_id: str, store: ArrayStore, arrays: dict) -> Measurement:
    mid = new_id()
    store.write(
        mid,
        ParsedData(
            technique=Technique.THERMOELECTRIC,
            arrays=arrays,
            metadata={},
            instrument=None,
            measured_at=None,
            issues=(),
            parser_name="te",
            parser_version="1.0.0",
        ),
    )
    return Measurement(
        id=mid,
        sample_id=sample_id,
        technique=Technique.THERMOELECTRIC,
        instrument=None,
        measured_at=None,
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=(),
        issues=(),
        parsed_data_path=None,
        analysis_results=(),
    )


def _te_sample(
    pid: str, store: ArrayStore, name: str, seebeck_uv_k: float, resistivity_uohm_m: float
) -> Sample:
    """Sample whose zT peaks at 600 K with the given Seebeck and resistivity.

    R&S and LFA are separate measurements (as in real data). κ = 1 W/mK, so
    zT(600) = S² · 600 / (ρ · κ) with the chosen S (µV/K) and ρ (µΩ·m).
    """
    sid = new_id()
    temp = np.array([300.0, 600.0])
    rs = _measurement(
        sid,
        store,
        {
            "temperature_k": temp,
            "resistivity_uohm_m": np.array([resistivity_uohm_m, resistivity_uohm_m]),
            "seebeck_uv_k": np.array([seebeck_uv_k * 0.5, seebeck_uv_k]),
        },
    )
    lfa = _measurement(
        sid,
        store,
        {"temperature_k": temp, "thermal_conductivity": np.array([1.0, 1.0])},
    )
    return Sample(
        id=sid,
        project_id=pid,
        canonical_name=name,
        aliases=(),
        measurements=(rs, lfa),
    )


def _client(root: Path) -> TestClient:
    store = ArrayStore(root / ".latos" / "arrays")
    pid = new_id()
    samples = (
        # S = 200 µV/K, ρ = 30 µΩ·m -> zT(600) ≈ 0.8, SPB-applicable.
        _te_sample(pid, store, "APPLIC", seebeck_uv_k=200.0, resistivity_uohm_m=30.0),
        # S = 27 µV/K, ρ = 0.45 µΩ·m -> zT(600) ≈ 0.98 at a tiny Seebeck:
        # impossible for one band -> multi-band / data flag (the CS-3 case).
        _te_sample(pid, store, "CEILING", seebeck_uv_k=27.0, resistivity_uohm_m=0.45),
    )
    project = Project(
        id=pid,
        name=root.name,
        root_path=root,
        created_at=utc_now(),
        schema_version=4,
        samples=samples,
        unassigned_files=(),
        review_status=ReviewStatus.CONFIRMED,
        confirmed_at=utc_now(),
    )
    app = create_app()
    state: ServerState = app.state.latos  # type: ignore[union-attr]
    state.root = root
    state.result = IngestionResult(project=project, outcomes=())
    return TestClient(app)


class TestSpbEndpoint:
    def test_no_project_404(self, tmp_path: Path):
        app = create_app()
        assert TestClient(app).get("/optimize/spb").status_code == 404

    def test_returns_both_samples(self, tmp_path: Path):
        body = _client(tmp_path).get("/optimize/spb").json()
        names = {s["sample_name"] for s in body["samples"]}
        assert names == {"APPLIC", "CEILING"}

    def test_best_is_highest_zt_and_flags_ceiling(self, tmp_path: Path):
        body = _client(tmp_path).get("/optimize/spb").json()
        best = body["best"]
        # CEILING has the higher zT (~0.98) so it's the "best" sample...
        assert best["sample_name"] == "CEILING"
        # ...but it's physically inconsistent with a single band.
        assert best["applicable"] is False
        assert best["beta"] is None
        assert best["zt_ceiling"] is not None and best["zt_ceiling"] < best["measured_zt"]
        assert "multi-band" in best["note"]

    def test_applicable_sample_has_direction_and_beta(self, tmp_path: Path):
        body = _client(tmp_path).get("/optimize/spb").json()
        applic = next(s for s in body["samples"] if s["sample_name"] == "APPLIC")
        assert applic["applicable"] is True
        assert applic["beta"] is not None
        assert applic["optimal_seebeck_uv_k"] is not None
        assert applic["direction"] in {"increase_seebeck", "decrease_seebeck", "at_optimum"}
        assert applic["measured_seebeck_uv_k"] == pytest.approx(200.0)
