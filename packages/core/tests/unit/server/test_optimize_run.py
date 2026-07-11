"""Tests for POST /optimize/run (BO3) — the end-to-end optimization endpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from latos.core.enums import ReviewStatus, Technique
from latos.core.models import (
    Measurement,
    Project,
    Sample,
    new_id,
    utc_now,
)
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.orchestrator import IngestionResult
from latos.ingestion.parsed_data import ParsedData
from latos.server import synthesis_store
from latos.server.app import create_app
from latos.server.state import ServerState

# The real Dhivya peak-zT-vs-doping data.
_TE_DATA = [
    ("CS", 0.0, 0.587),
    ("CSCBI-1", 1.0, 0.362),
    ("CSCBI-3", 3.0, 0.967),
    ("CSCBI-5", 5.0, 0.482),
]


def _te_measurement(sample_id: str, store: ArrayStore, zt_peak: float) -> Measurement:
    mid = new_id()
    store.write(
        mid,
        ParsedData(
            technique=Technique.THERMOELECTRIC,
            arrays={
                "temperature_k": np.array([300.0, 600.0]),
                "zt": np.array([zt_peak * 0.5, zt_peak]),
            },
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


def _confirmed_te_project(root: Path) -> tuple[Project, dict[str, float]]:
    """4-sample TE project (CONFIRMED) + each sample's doping value."""
    store = ArrayStore(root / ".latos" / "arrays")
    pid = new_id()
    samples = []
    doping_by_id: dict[str, float] = {}
    for name, doping, zt in _TE_DATA:
        sid = new_id()
        doping_by_id[sid] = doping
        samples.append(
            Sample(
                id=sid,
                project_id=pid,
                canonical_name=name,
                aliases=(),
                measurements=(_te_measurement(sid, store, zt),),
            )
        )
    project = Project(
        id=pid,
        name=root.name,
        root_path=root,
        created_at=utc_now(),
        schema_version=4,
        samples=tuple(samples),
        unassigned_files=(),
        review_status=ReviewStatus.CONFIRMED,
        confirmed_at=utc_now(),
    )
    return project, doping_by_id


def _client(root: Path, *, confirmed: bool = True) -> TestClient:
    project, doping_by_id = _confirmed_te_project(root)
    if not confirmed:
        from dataclasses import replace

        project = replace(project, review_status=ReviewStatus.NEEDS_REVIEW, confirmed_at=None)
    # Persist each sample's doping value.
    for sid, doping in doping_by_id.items():
        synthesis_store.set_sample_params(root, sid, {"doping_pct": doping})
    app = create_app()
    state: ServerState = app.state.latos  # type: ignore[union-attr]
    state.root = root
    state.result = IngestionResult(project=project, outcomes=())
    return TestClient(app)


def _run(client: TestClient) -> dict:
    return client.post(
        "/optimize/run",
        json={"input_variable": "doping_pct", "target_property": "zt"},
    ).json()


class TestGate:
    def test_unconfirmed_project_blocked(self, tmp_path: Path):
        client = _client(tmp_path, confirmed=False)
        resp = client.post(
            "/optimize/run",
            json={"input_variable": "doping_pct", "target_property": "zt"},
        )
        assert resp.status_code == 409
        assert "confirm" in resp.json()["detail"].lower()


class TestRun:
    def test_recommends_near_peak(self, tmp_path: Path):
        body = _run(_client(tmp_path))
        assert 2.5 <= body["recommendation"]["x"] <= 4.0

    def test_best_is_3pct(self, tmp_path: Path):
        body = _run(_client(tmp_path))
        assert body["best_x"] == 3.0
        assert body["best_y"] == 0.967

    def test_returns_curve_and_verdict(self, tmp_path: Path):
        body = _run(_client(tmp_path))
        assert len(body["grid_x"]) == len(body["grid_mean"]) > 50
        assert len(body["points"]) == 4
        assert isinstance(body["verdict"], str) and body["verdict"]
        assert "noise_threshold" in body and "converged" in body

    def test_too_few_points_400(self, tmp_path: Path):
        client = _client(tmp_path)
        # Optimize on a property no sample has -> 0 usable points.
        resp = client.post(
            "/optimize/run",
            json={"input_variable": "doping_pct", "target_property": "nonexistent"},
        )
        assert resp.status_code == 400

    def test_run_returns_predictive_interval(self, tmp_path: Path):
        rec = _run(_client(tmp_path))["recommendation"]
        assert "ci95_predictive" in rec
        lo, hi = rec["predictive_interval_95"]
        assert lo <= rec["predicted_mean"] <= hi


class TestFreeze:
    def test_freeze_writes_prereg_record(self, tmp_path: Path):
        client = _client(tmp_path)
        resp = client.post(
            "/optimize/freeze",
            json={"input_variable": "doping_pct", "target_property": "zt"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert Path(body["path"]).exists()
        assert Path(body["path"]).with_suffix(".md").exists()
        assert "predictive_interval_95" in body["recommendation"]
        assert isinstance(body["robustness_stable"], bool)
        assert body["prior_best"] == 0.967

    def test_freeze_blocked_until_confirmed(self, tmp_path: Path):
        client = _client(tmp_path, confirmed=False)
        resp = client.post(
            "/optimize/freeze",
            json={"input_variable": "doping_pct", "target_property": "zt"},
        )
        assert resp.status_code == 409


# ─── Objective modes + input-variable listing (OP2/OP3) ─────────────


class TestObjectives:
    def test_minimize_reports_lowest(self, tmp_path: Path):
        client = _client(tmp_path)
        body = client.post(
            "/optimize/run",
            json={
                "input_variable": "doping_pct",
                "target_property": "zt",
                "objective": "minimize",
            },
        ).json()
        assert body["objective"] == "minimize"
        assert body["best_x"] == 1.0  # zt = 0.362, the lowest peak
        assert body["best_y"] == 0.362

    def test_target_mode_transforms_and_relabels(self, tmp_path: Path):
        client = _client(tmp_path)
        body = client.post(
            "/optimize/run",
            json={
                "input_variable": "doping_pct",
                "target_property": "zt",
                "objective": "target",
                "target_value": 0.5,
            },
        ).json()
        assert body["target_property"] == "|zt - 0.5|"
        # Distances: CS 0.087, CSCBI-1 0.138, CSCBI-3 0.467, CSCBI-5 0.018.
        assert body["best_x"] == 5.0
        assert abs(body["best_y"] - 0.018) < 1e-9

    def test_target_without_value_is_400(self, tmp_path: Path):
        client = _client(tmp_path)
        resp = client.post(
            "/optimize/run",
            json={
                "input_variable": "doping_pct",
                "target_property": "zt",
                "objective": "target",
            },
        )
        assert resp.status_code == 400
        assert "target_value" in resp.json()["detail"]

    def test_unknown_objective_is_400(self, tmp_path: Path):
        client = _client(tmp_path)
        resp = client.post(
            "/optimize/run",
            json={
                "input_variable": "doping_pct",
                "target_property": "zt",
                "objective": "upward",
            },
        )
        assert resp.status_code == 400


class TestInputVariablesEndpoint:
    def test_lists_synthesis_variables_with_values(self, tmp_path: Path):
        client = _client(tmp_path)
        body = client.get("/optimize/inputs").json()
        by_name = {v["name"]: v for v in body}
        assert by_name["doping_pct"]["source"] == "synthesis"
        assert len(by_name["doping_pct"]["values"]) == 4


class TestReliabilityGate:
    def test_run_reports_exploratory_for_small_series(self, tmp_path: Path):
        body = _run(_client(tmp_path))  # the 4-point TE fixture
        assert body["reliability_level"] == "exploratory"
        assert "4 measured points" in body["reliability_note"]
        assert "Leave-one-out" in body["reliability_note"]

    def test_freeze_records_reliability(self, tmp_path: Path):
        client = _client(tmp_path)
        body = client.post(
            "/optimize/freeze",
            json={"input_variable": "doping_pct", "target_property": "zt"},
        ).json()
        assert body["reliability_level"] == "exploratory"
        # The written prereg JSON carries the same self-assessment.
        import json as _json

        record = _json.loads(Path(body["path"]).read_text(encoding="utf-8"))
        assert record["reliability"]["level"] == "exploratory"
        assert record["reliability"]["loo_total"] == 4


class TestLoopCloser:
    def _freeze(self, client: TestClient) -> str:
        body = client.post(
            "/optimize/freeze",
            json={"input_variable": "doping_pct", "target_property": "zt"},
        ).json()
        return body["path"]

    def test_list_prereg_empty_before_freeze(self, tmp_path: Path):
        client = _client(tmp_path)
        assert client.get("/optimize/prereg").json() == []

    def test_freeze_then_list_shows_entry(self, tmp_path: Path):
        client = _client(tmp_path)
        self._freeze(client)
        listed = client.get("/optimize/prereg").json()
        assert len(listed) == 1
        entry = listed[0]
        assert entry["input_variable"] == "doping_pct"
        assert entry["direction"] == "maximize"
        assert entry["reliability_level"] == "exploratory"
        assert entry["outcome"] is None

    def test_validate_calibrated_and_no_improvement(self, tmp_path: Path):
        client = _client(tmp_path)
        path = self._freeze(client)
        # The frozen prediction interval is wide (exploratory); a measured
        # value near the prediction lands inside and does not beat 0.967.
        v = client.post(
            "/optimize/validate",
            json={"prereg_path": path, "measured_value": 0.95},
        ).json()
        assert v["within_interval"] is True
        assert v["improved"] is False
        assert "within" in v["summary"].lower()

    def test_validate_improvement(self, tmp_path: Path):
        client = _client(tmp_path)
        path = self._freeze(client)
        v = client.post(
            "/optimize/validate",
            json={"prereg_path": path, "measured_value": 1.10},
        ).json()
        assert v["improved"] is True

    def test_validate_persists_and_relist_attaches_outcome(self, tmp_path: Path):
        client = _client(tmp_path)
        path = self._freeze(client)
        client.post(
            "/optimize/validate",
            json={"prereg_path": path, "measured_value": 0.95},
        )
        listed = client.get("/optimize/prereg").json()
        assert len(listed) == 1  # the .outcome.json sibling is not a new row
        assert listed[0]["outcome"]["within_interval"] is True
        assert listed[0]["outcome"]["measured"] == 0.95

    def test_validate_rejects_path_outside_project(self, tmp_path: Path):
        client = _client(tmp_path)
        self._freeze(client)
        resp = client.post(
            "/optimize/validate",
            json={"prereg_path": str(tmp_path / "evil.json"), "measured_value": 1.0},
        )
        assert resp.status_code == 404
