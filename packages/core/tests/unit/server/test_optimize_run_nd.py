"""Tests for POST /optimize/run-nd (MV4) — the multi-axis optimization endpoint.

The route's whole job is to widen one column of x into d and hand the result to
`optimize_nd`, so the tests concentrate on the widening and on what it costs:
which samples survive the join, what happens when an added axis is missing or
constant, and whether the response says so rather than silently shrinking the
dataset.

The fixture is a 4 x 3 (doping, anneal temperature) grid because that is the
shape of the two-variable design this endpoint was built for — and because a
separable surface with a known peak makes the anisotropic kernel's answer
checkable: doping has a maximum, temperature only rises.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from latos.core.enums import ReviewStatus, Technique
from latos.core.models import Measurement, Project, Sample, new_id, utc_now
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.orchestrator import IngestionResult
from latos.ingestion.parsed_data import ParsedData
from latos.server import synthesis_store
from latos.server.app import create_app
from latos.server.state import ServerState

DOPINGS = (0.0, 1.0, 3.0, 5.0)
ANNEALS = (573.0, 673.0, 773.0)


def _zt(doping: float, anneal_k: float) -> float:
    """Peaks at 3% doping, rises monotonically with annealing temperature."""
    return math.exp(-((doping - 3.0) ** 2) / 2.0) * (anneal_k / 800.0)


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


def _client(
    root: Path,
    *,
    constant_anneal: bool = False,
    drop_anneal_on: tuple[str, ...] = (),
) -> TestClient:
    """A confirmed 12-sample project on a (doping, anneal) grid.

    Args:
        constant_anneal: give every sample the same annealing temperature, so
            the second axis has no range at all.
        drop_anneal_on: sample names to leave without an annealing value, to
            exercise the join that drops them.
    """
    store = ArrayStore(root / ".latos" / "arrays")
    pid = new_id()
    samples, params = [], {}
    for doping in DOPINGS:
        for anneal in ANNEALS:
            sid, name = new_id(), f"D{doping:g}-A{anneal:g}"
            samples.append(
                Sample(
                    id=sid,
                    project_id=pid,
                    canonical_name=name,
                    aliases=(),
                    measurements=(_te_measurement(sid, store, _zt(doping, anneal)),),
                )
            )
            record = {"doping_pct": doping}
            if name not in drop_anneal_on:
                record["anneal_k"] = ANNEALS[0] if constant_anneal else anneal
            params[sid] = record

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
    for sid, record in params.items():
        synthesis_store.set_sample_params(root, sid, record)

    app = create_app()
    state: ServerState = app.state.latos  # type: ignore[union-attr]
    state.root = root
    state.result = IngestionResult(project=project, outcomes=())
    return TestClient(app)


def _post(client: TestClient, **overrides: object):
    body = {
        "input_variables": ["doping_pct", "anneal_k"],
        "target_property": "zt",
        "surface_size": 16,
    }
    body.update(overrides)
    return client.post("/optimize/run-nd", json=body)


class TestGate:
    def test_unconfirmed_project_is_blocked_here_too(self, tmp_path: Path):
        """The identity gate is not something a new route may skip."""
        client = _client(tmp_path)
        state: ServerState = client.app.state.latos  # type: ignore[union-attr]
        from dataclasses import replace

        state.result = IngestionResult(
            project=replace(
                state.result.project,
                review_status=ReviewStatus.NEEDS_REVIEW,
                confirmed_at=None,
            ),
            outcomes=(),
        )
        assert _post(client).status_code == 409


class TestTwoAxisRun:
    def test_runs_over_both_axes(self, tmp_path: Path):
        body = _post(_client(tmp_path)).json()
        assert body["input_variables"] == ["doping_pct", "anneal_k"]
        assert len(body["points"]) == len(DOPINGS) * len(ANNEALS)
        assert len(body["points"][0]["x"]) == 2
        assert len(body["recommendation"]["x"]) == 2
        assert body["n_dropped_for_missing_axis"] == 0

    def test_recommendation_stays_inside_the_observed_box(self, tmp_path: Path):
        rec = _post(_client(tmp_path)).json()["recommendation"]["x"]
        assert min(DOPINGS) <= rec[0] <= max(DOPINGS)
        assert min(ANNEALS) <= rec[1] <= max(ANNEALS)

    def test_best_point_is_the_measured_optimum(self, tmp_path: Path):
        body = _post(_client(tmp_path)).json()
        assert body["best_x"] == [3.0, 773.0]
        assert body["best_y"] == max(_zt(d, a) for d in DOPINGS for a in ANNEALS)

    def test_axes_report_their_range_and_fitted_scale(self, tmp_path: Path):
        axes = _post(_client(tmp_path)).json()["axes"]
        assert [a["name"] for a in axes] == ["doping_pct", "anneal_k"]
        assert (axes[0]["low"], axes[0]["high"]) == (0.0, 5.0)
        assert (axes[1]["low"], axes[1]["high"]) == (573.0, 773.0)
        assert all(math.isfinite(a["length_scale"]) for a in axes)
        assert all(a["pinned_at"] in (None, "low", "high") for a in axes)

    def test_kernel_is_anisotropic(self, tmp_path: Path):
        body = _post(_client(tmp_path)).json()
        assert "ARD" in body["kernel"]

    def test_the_curved_axis_gets_the_shorter_length_scale(self, tmp_path: Path):
        """The point of an anisotropic kernel: zT peaks in doping but only
        rises in annealing temperature, so doping must be the axis the model
        says it needs to resolve more finely."""
        axes = _post(_client(tmp_path)).json()["axes"]
        doping, anneal = axes[0]["length_scale"], axes[1]["length_scale"]
        assert doping < anneal

    def test_verdict_names_both_axes(self, tmp_path: Path):
        verdict = _post(_client(tmp_path)).json()["verdict"]
        assert "doping_pct" in verdict
        assert "anneal_k" in verdict

    def test_reliability_reports_the_geometric_gate(self, tmp_path: Path):
        """Twelve points over a plane cannot cover it. The count tier alone
        would not notice; fill distance is what does."""
        body = _post(_client(tmp_path)).json()
        assert body["reliability_level"] == "exploratory"
        assert body["fill_distance"] > body["fill_limit"] > 0.0


class TestSurface:
    def test_lattice_is_square_and_oriented_row_by_y(self, tmp_path: Path):
        s = _post(_client(tmp_path)).json()["surface"]
        assert s["axis_names"] == ["doping_pct", "anneal_k"]
        assert len(s["axis_x"]) == len(s["axis_y"]) == 16
        assert len(s["mean"]) == 16  # one row per axis_y value
        assert all(len(row) == 16 for row in s["mean"])
        assert len(s["sd"]) == len(s["ei"]) == 16

    def test_lattice_spans_the_search_box(self, tmp_path: Path):
        s = _post(_client(tmp_path)).json()["surface"]
        assert (s["axis_x"][0], s["axis_x"][-1]) == (0.0, 5.0)
        assert (s["axis_y"][0], s["axis_y"][-1]) == (573.0, 773.0)

    def test_the_posterior_peak_sits_near_the_true_one(self, tmp_path: Path):
        """The surface is what the user will read a recommendation off, so it
        has to agree with the data: the peak belongs near 3% doping and at the
        hottest anneal, which is where the measurements say it is."""
        s = _post(_client(tmp_path)).json()["surface"]
        mean = np.array(s["mean"])
        j, i = np.unravel_index(int(np.argmax(mean)), mean.shape)
        assert abs(s["axis_x"][i] - 3.0) < 1.0
        assert s["axis_y"][j] > 700.0

    def test_zero_size_skips_it(self, tmp_path: Path):
        """A caller that only wants a recommendation should not pay for a
        lattice of predictions it will throw away."""
        assert _post(_client(tmp_path), surface_size=0).json()["surface"] is None

    def test_a_single_axis_run_has_no_surface(self, tmp_path: Path):
        body = _post(_client(tmp_path), input_variables=["doping_pct"]).json()
        assert body["surface"] is None
        assert len(body["recommendation"]["x"]) == 1
        assert "ARD" not in body["kernel"]


class TestWhatAddingAnAxisCosts:
    def test_samples_missing_the_added_axis_are_dropped_and_counted(self, tmp_path: Path):
        """Adding an axis can shrink the dataset. That has to be visible in the
        response, not inferred from a point count the caller has to know."""
        client = _client(tmp_path, drop_anneal_on=("D0-A573", "D1-A673"))
        body = _post(client).json()
        assert body["n_dropped_for_missing_axis"] == 2
        assert len(body["points"]) == len(DOPINGS) * len(ANNEALS) - 2

    def test_dropping_only_affects_the_multi_axis_run(self, tmp_path: Path):
        """The same samples still qualify for the one-variable route, which
        never asked for the annealing value."""
        client = _client(tmp_path, drop_anneal_on=("D0-A573", "D1-A673"))
        one = client.post(
            "/optimize/run",
            json={"input_variable": "doping_pct", "target_property": "zt"},
        ).json()
        assert len(one["points"]) == len(DOPINGS) * len(ANNEALS)

    def test_too_few_survivors_explains_which_axis_cost_them(self, tmp_path: Path):
        keep = ("D3-A773", "D5-A573")
        drop = tuple(
            f"D{d:g}-A{a:g}" for d in DOPINGS for a in ANNEALS if f"D{d:g}-A{a:g}" not in keep
        )
        resp = _post(_client(tmp_path, drop_anneal_on=drop))
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "anneal_k" in detail
        assert "missing axis value" in detail


class TestRequestGuards:
    def test_a_constant_axis_is_rejected_by_name(self, tmp_path: Path):
        resp = _post(_client(tmp_path, constant_anneal=True))
        assert resp.status_code == 400
        assert "anneal_k" in resp.json()["detail"]
        assert "does not vary" in resp.json()["detail"]

    def test_a_repeated_axis_is_rejected(self, tmp_path: Path):
        """Two identical columns are not two axes; the fit would be degenerate."""
        resp = _post(_client(tmp_path), input_variables=["doping_pct", "doping_pct"])
        assert resp.status_code == 400
        assert "once" in resp.json()["detail"]

    def test_no_axes_is_rejected(self, tmp_path: Path):
        assert _post(_client(tmp_path), input_variables=[]).status_code == 400

    def test_bounds_must_match_the_axis_count(self, tmp_path: Path):
        resp = _post(_client(tmp_path), bounds=[[0.0, 5.0]])
        assert resp.status_code == 400
        assert "one (low, high) pair per axis" in resp.json()["detail"]

    def test_an_empty_explicit_bound_blames_the_bound_not_the_data(self, tmp_path: Path):
        """Same guard, different fault. Telling a user their annealing values do
        not vary when in fact they sent low >= high sends them to fix the wrong
        thing."""
        resp = _post(_client(tmp_path), bounds=[[0.0, 5.0], [800.0, 700.0]])
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "bounds given for 'anneal_k'" in detail
        assert "does not vary" not in detail

    def test_explicit_bounds_are_honoured(self, tmp_path: Path):
        body = _post(_client(tmp_path), bounds=[[0.0, 8.0], [500.0, 900.0]]).json()
        assert [(a["low"], a["high"]) for a in body["axes"]] == [(0.0, 8.0), (500.0, 900.0)]
        assert body["surface"]["axis_x"][-1] == 8.0

    def test_an_unknown_objective_is_rejected(self, tmp_path: Path):
        assert _post(_client(tmp_path), objective="sideways").status_code == 400


class TestObjectiveModesCarryOver:
    """The objective handling is `_assemble_optimization`'s, reused verbatim.
    These check the reuse actually happened rather than being reimplemented."""

    def test_minimize_reports_the_lowest_measurement_as_best(self, tmp_path: Path):
        body = _post(_client(tmp_path), objective="minimize").json()
        assert body["best_y"] == min(_zt(d, a) for d in DOPINGS for a in ANNEALS)

    def test_target_mode_relabels_as_a_distance(self, tmp_path: Path):
        body = _post(_client(tmp_path), objective="target", target_value=0.5).json()
        assert body["target_property"].startswith("|zt")
        assert body["best_y"] >= 0.0

    def test_target_mode_needs_a_value(self, tmp_path: Path):
        assert _post(_client(tmp_path), objective="target").status_code == 400
