"""Tests for the BO (X, y) data layer — synthesis params + target extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from latos.core.enums import Technique
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
from latos.server import optimization_data, synthesis_store
from latos.server.app import create_app
from latos.server.state import ServerState


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


def _project_with_zt(root: Path) -> tuple[Project, ArrayStore, dict[str, str]]:
    """Two samples (CS, CS-3), each one TE measurement carrying a zt array."""
    store = ArrayStore(root / ".latos" / "arrays")
    pid = new_id()
    samples = []
    ids: dict[str, str] = {}
    for name, zt_peak in [("CS", 0.59), ("CS-3", 0.97)]:
        sid = new_id()
        ids[name] = sid
        samples.append(
            Sample(
                id=sid,
                project_id=pid,
                canonical_name=name,
                aliases=(),
                measurements=(_te_measurement(sid, store, zt_peak),),
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
    )
    return project, store, ids


class TestSynthesisStore:
    def test_set_and_load_round_trip(self, tmp_path: Path):
        synthesis_store.set_sample_params(tmp_path, "s1", {"doping_pct": 3.0})
        loaded = synthesis_store.load_params(tmp_path)
        assert loaded == {"s1": {"doping_pct": 3.0}}

    def test_empty_params_removes_sample(self, tmp_path: Path):
        synthesis_store.set_sample_params(tmp_path, "s1", {"doping_pct": 3.0})
        synthesis_store.set_sample_params(tmp_path, "s1", {})
        assert synthesis_store.load_params(tmp_path) == {}

    def test_missing_file_is_empty(self, tmp_path: Path):
        assert synthesis_store.load_params(tmp_path) == {}


class TestTargetExtraction:
    def test_peak_target_takes_max(self, tmp_path: Path):
        project, store, _ = _project_with_zt(tmp_path)
        cs3 = next(s for s in project.samples if s.canonical_name == "CS-3")
        assert optimization_data.peak_target(cs3, store, "zt") == 0.97

    def test_peak_target_missing_property_is_none(self, tmp_path: Path):
        project, store, _ = _project_with_zt(tmp_path)
        cs = project.samples[0]
        assert optimization_data.peak_target(cs, store, "not_a_prop") is None

    def test_list_target_properties(self, tmp_path: Path):
        project, store, _ = _project_with_zt(tmp_path)
        props = optimization_data.list_target_properties(project, store)
        assert "zt" in props
        # Axes/independent variables are filtered out — only objectives remain.
        assert "temperature_k" not in props


class TestBuildDataset:
    def test_assembles_xy_rows(self, tmp_path: Path):
        project, store, ids = _project_with_zt(tmp_path)
        params = {ids["CS"]: {"doping_pct": 0.0}, ids["CS-3"]: {"doping_pct": 3.0}}
        rows, skipped = optimization_data.build_dataset(project, store, params, "doping_pct", "zt")
        assert len(rows) == 2
        assert skipped == []
        by_name = {r.sample_name: (r.x, r.y) for r in rows}
        assert by_name["CS"] == (0.0, 0.59)
        assert by_name["CS-3"] == (3.0, 0.97)

    def test_missing_param_skips_sample(self, tmp_path: Path):
        project, store, ids = _project_with_zt(tmp_path)
        params = {ids["CS"]: {"doping_pct": 0.0}}  # CS-3 has no param
        rows, skipped = optimization_data.build_dataset(project, store, params, "doping_pct", "zt")
        assert len(rows) == 1
        assert len(skipped) == 1
        assert "doping_pct" in skipped[0].reason


class TestEndpoints:
    def _client(self, root: Path) -> TestClient:
        project, _store, _ids = _project_with_zt(root)
        app = create_app()
        state: ServerState = app.state.latos  # type: ignore[union-attr]
        state.root = root
        state.result = IngestionResult(project=project, outcomes=())
        return TestClient(app)

    def test_set_and_get_parameters(self, tmp_path: Path):
        client = self._client(tmp_path)
        sid = client.get("/samples").json()[0]["id"]
        resp = client.post(f"/samples/{sid}/parameters", json={"parameters": {"doping_pct": 2.0}})
        assert resp.status_code == 200
        assert client.get("/parameters").json()[sid] == {"doping_pct": 2.0}

    def test_set_parameters_unknown_sample_404(self, tmp_path: Path):
        client = self._client(tmp_path)
        resp = client.post("/samples/nope/parameters", json={"parameters": {"x": 1.0}})
        assert resp.status_code == 404

    def test_targets_lists_zt(self, tmp_path: Path):
        client = self._client(tmp_path)
        assert "zt" in client.get("/optimize/targets").json()["properties"]

    def test_dataset_endpoint(self, tmp_path: Path):
        client = self._client(tmp_path)
        samples = client.get("/samples").json()
        for i, s in enumerate(samples):
            client.post(f"/samples/{s['id']}/parameters", json={"parameters": {"doping_pct": i}})
        body = client.get(
            "/optimize/dataset",
            params={"input_variable": "doping_pct", "target_property": "zt"},
        ).json()
        assert len(body["points"]) == 2
        assert body["skipped"] == []


# ─── Derived zT target ──────────────────────────────────────────────
def _lfa_meas(sample_id: str, store: ArrayStore) -> Measurement:
    mid = new_id()
    store.write(
        mid,
        ParsedData(
            technique=Technique.THERMOELECTRIC,
            arrays={
                "temperature_k": np.array([300.0, 400.0, 500.0, 600.0]),
                "thermal_conductivity": np.array([5.0, 4.5, 4.2, 4.0]),
            },
            metadata={},
            instrument=None,
            measured_at=None,
            issues=(),
            parser_name="lfa-xlsx",
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


def _rs_meas(sample_id: str, store: ArrayStore) -> Measurement:
    mid = new_id()
    store.write(
        mid,
        ParsedData(
            technique=Technique.THERMOELECTRIC,
            arrays={
                "temperature_k": np.array([316.0, 400.0, 500.0, 600.0]),
                "resistivity_uohm_m": np.array([0.12, 0.18, 0.24, 0.29]),
                "seebeck_uv_k": np.array([8.0, 14.0, 22.0, 32.0]),
            },
            metadata={},
            instrument=None,
            measured_at=None,
            issues=(),
            parser_name="resistivity-seebeck-xlsx",
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


def _project_with_te_pair(root: Path) -> tuple[Project, ArrayStore, str]:
    store = ArrayStore(root / ".latos" / "arrays")
    pid = new_id()
    sid = new_id()
    sample = Sample(
        id=sid,
        project_id=pid,
        canonical_name="CS",
        aliases=(),
        measurements=(_lfa_meas(sid, store), _rs_meas(sid, store)),
    )
    project = Project(
        id=pid,
        name=root.name,
        root_path=root,
        created_at=utc_now(),
        schema_version=4,
        samples=(sample,),
        unassigned_files=(),
    )
    return project, store, sid


class TestDerivedZt:
    def test_offered_first_when_pair_present(self, tmp_path: Path):
        project, store, _ = _project_with_te_pair(tmp_path)
        props = optimization_data.list_target_properties(project, store)
        assert props[0] == optimization_data.DERIVED_ZT

    def test_not_offered_without_pair(self, tmp_path: Path):
        project, store, _ = _project_with_zt(tmp_path)  # only a `zt` column
        props = optimization_data.list_target_properties(project, store)
        assert optimization_data.DERIVED_ZT not in props

    def test_build_dataset_uses_derived_zt(self, tmp_path: Path):
        project, store, sid = _project_with_te_pair(tmp_path)
        rows, skipped = optimization_data.build_dataset(
            project,
            store,
            {sid: {"doping_pct": 3.0}},
            "doping_pct",
            optimization_data.DERIVED_ZT,
        )
        assert len(rows) == 1
        assert 0.0 < rows[0].y < 1.5
        assert skipped == []

    def test_missing_pair_reported(self, tmp_path: Path):
        project, store, _ = _project_with_zt(tmp_path)
        rows, skipped = optimization_data.build_dataset(
            project,
            store,
            {s.id: {"doping_pct": 1.0} for s in project.samples},
            "doping_pct",
            optimization_data.DERIVED_ZT,
        )
        assert rows == []
        assert all("derive zT" in s.reason for s in skipped)


# ─── Measured features as inputs/targets + zT at temperature (OP2) ──


def _hall_measurement(sample_id: str, n_cm3: float) -> Measurement:
    """A features-only Hall measurement (no arrays)."""
    return Measurement(
        id=new_id(),
        sample_id=sample_id,
        technique=Technique.HALL,
        instrument=None,
        measured_at=None,
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=(),
        issues=(),
        parsed_data_path=None,
        analysis_results=(),
        features={"carrier_concentration_cm3": n_cm3, "mobility_cm2_vs": 100.0},
    )


def _project_with_hall(root: Path) -> tuple[Project, ArrayStore, dict[str, str]]:
    """Samples carrying a zt array AND a Hall features measurement."""
    store = ArrayStore(root / ".latos" / "arrays")
    pid = new_id()
    samples = []
    ids: dict[str, str] = {}
    series = [("CS-1", 0.42, 7.3e18), ("CS-3", 0.97, 2.1e19), ("CS-5", 0.51, 4.4e19)]
    for name, zt_peak, n in series:
        sid = new_id()
        ids[name] = sid
        samples.append(
            Sample(
                id=sid,
                project_id=pid,
                canonical_name=name,
                aliases=(),
                measurements=(
                    _te_measurement(sid, store, zt_peak),
                    _hall_measurement(sid, n),
                ),
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
    )
    return project, store, ids


class TestMeasuredInputVariables:
    def test_list_input_variables_includes_both_sources(self, tmp_path: Path):
        project, _store, ids = _project_with_hall(tmp_path)
        params = {ids["CS-1"]: {"doping_pct": 1.0}}
        out = optimization_data.list_input_variables(project, params)
        by_name = {v.name: v for v in out}
        assert by_name["doping_pct"].source == "synthesis"
        assert by_name["carrier_concentration_cm3"].source == "measured"
        assert by_name["carrier_concentration_cm3"].values[ids["CS-3"]] == 2.1e19

    def test_build_dataset_resolves_x_from_features(self, tmp_path: Path):
        project, store, _ids = _project_with_hall(tmp_path)
        rows, skipped = optimization_data.build_dataset(
            project, store, {}, "carrier_concentration_cm3", "zt"
        )
        assert not skipped
        xs = {r.sample_name: r.x for r in rows}
        assert xs["CS-1"] == 7.3e18
        assert {r.sample_name: r.y for r in rows}["CS-3"] == 0.97

    def test_feature_as_target(self, tmp_path: Path):
        project, store, ids = _project_with_hall(tmp_path)
        params = {
            sid: {"doping_pct": d}
            for sid, d in [(ids["CS-1"], 1.0), (ids["CS-3"], 3.0), (ids["CS-5"], 5.0)]
        }
        rows, skipped = optimization_data.build_dataset(
            project, store, params, "doping_pct", "mobility_cm2_vs"
        )
        assert not skipped
        assert all(r.y == 100.0 for r in rows)

    def test_targets_list_includes_curated_features(self, tmp_path: Path):
        project, store, _ids = _project_with_hall(tmp_path)
        props = optimization_data.list_target_properties(project, store)
        assert "carrier_concentration_cm3" in props
        assert "mobility_cm2_vs" in props


def _zt_pair_measurements(sample_id: str, store: ArrayStore) -> tuple[Measurement, ...]:
    """An R&S + LFA pair with an analytically known zT(T): 0.3 at 300 K, 2.4 at 600 K."""
    out = []
    for arrays in (
        {
            "temperature_k": np.array([300.0, 600.0]),
            "resistivity_uohm_m": np.array([10.0, 10.0]),
            "seebeck_uv_k": np.array([100.0, 200.0]),
        },
        {
            "temperature_k": np.array([300.0, 600.0]),
            "thermal_conductivity": np.array([1.0, 1.0]),
        },
    ):
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
        out.append(
            Measurement(
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
        )
    return tuple(out)


class TestDerivedZtAtTemperature:
    def _sample(self, tmp_path: Path):
        store = ArrayStore(tmp_path / ".latos" / "arrays")
        sid = new_id()
        sample = Sample(
            id=sid,
            project_id=new_id(),
            canonical_name="CS",
            aliases=(),
            measurements=_zt_pair_measurements(sid, store),
        )
        return sample, store

    def test_interpolates_between_grid_points(self, tmp_path: Path):
        sample, store = self._sample(tmp_path)
        # zT is 0.3 at 300 K and 2.4 at 600 K -> linear interp at 450 K.
        value = optimization_data.derived_zt_at(sample, store, 450.0)
        assert value is not None
        assert abs(value - 1.35) < 1e-6

    def test_outside_measured_range_is_none(self, tmp_path: Path):
        sample, store = self._sample(tmp_path)
        assert optimization_data.derived_zt_at(sample, store, 200.0) is None
        assert optimization_data.derived_zt_at(sample, store, 700.0) is None


# ─── Optimization data-quality flags (DQ2) ──────────────────────────


def _hall_with_cross(sample_id, *, mobility, ac, bd, n=1e19):
    """A features-only Hall measurement carrying the AC/BD cross coefficients."""
    return Measurement(
        id=new_id(),
        sample_id=sample_id,
        technique=Technique.HALL,
        instrument=None,
        measured_at=None,
        parsed_at=utc_now(),
        parser_version="1.2.0",
        files=(),
        issues=(),
        parsed_data_path=None,
        analysis_results=(),
        features={
            "carrier_concentration_cm3": n,
            "mobility_cm2_vs": mobility,
            "conductivity_s_cm": 3000.0,
            "hall_ac_cross_cm3_c": ac,
            "hall_bd_cross_cm3_c": bd,
        },
    )


def _project_hall_cross(root):
    """3 samples: CS-1 & CS-3 unreliable (sign-disagree; CS-3 also negative mobility), CS-5 questionable."""  # noqa: E501
    store = ArrayStore(root / ".latos" / "arrays")
    pid = new_id()
    spec = [
        ("CS-1", 2494.0, -0.009, 1.71),  # unreliable (sign disagree)
        ("CS-3", -771.0, 0.084, -0.418),  # unreliable + negative mobility
        ("CS-5", 327.0, 0.18, 0.025),  # questionable (7x, same sign)
    ]
    samples = []
    ids = {}
    for name, mob, ac, bd in spec:
        sid = new_id()
        ids[name] = sid
        samples.append(
            Sample(
                id=sid,
                project_id=pid,
                canonical_name=name,
                aliases=(),
                measurements=(
                    _te_measurement(sid, store, 0.5),
                    _hall_with_cross(sid, mobility=mob, ac=ac, bd=bd),
                ),
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
    )
    return project, store, ids


class TestQualityFlags:
    def _rows(self, project, store, target):
        rows, _ = optimization_data.build_dataset(project, store, {}, "doping_pct", target)
        return rows

    def test_mobility_flags_unreliable_and_negative(self, tmp_path):
        project, store, _ = _project_hall_cross(tmp_path)
        # doping values so build_dataset yields rows
        for s in project.samples:
            synthesis_store.set_sample_params(tmp_path, s.id, {"doping_pct": 1.0})
        rows, _ = optimization_data.build_dataset(
            project, store, synthesis_store.load_params(tmp_path), "doping_pct", "mobility_cm2_vs"
        )
        flags = optimization_data.quality_flags(
            project, rows, "doping_pct", "mobility_cm2_vs", store
        )
        flagged = {f.sample_name for f in flags}
        assert "CS-1" in flagged and "CS-3" in flagged  # both sign-disagree
        assert "CS-5" not in flagged  # questionable, not unreliable
        cs3 = next(f for f in flags if f.sample_name == "CS-3")
        assert "mobility cannot be negative" in cs3.reason
        assert "disagree in sign" in cs3.reason

    def test_conductivity_target_never_flagged(self, tmp_path):
        project, store, _ = _project_hall_cross(tmp_path)
        for s in project.samples:
            synthesis_store.set_sample_params(tmp_path, s.id, {"doping_pct": 1.0})
        rows, _ = optimization_data.build_dataset(
            project, store, synthesis_store.load_params(tmp_path), "doping_pct", "conductivity_s_cm"
        )
        assert (
            optimization_data.quality_flags(project, rows, "doping_pct", "conductivity_s_cm", store)
            == []
        )

    def test_zt_target_never_flagged(self, tmp_path):
        project, store, _ = _project_hall_cross(tmp_path)
        for s in project.samples:
            synthesis_store.set_sample_params(tmp_path, s.id, {"doping_pct": 1.0})
        rows, _ = optimization_data.build_dataset(
            project, store, synthesis_store.load_params(tmp_path), "doping_pct", "zt"
        )
        assert optimization_data.quality_flags(project, rows, "doping_pct", "zt", store) == []

    def test_derived_zt_flags_wiedemann_franz_violation(self, tmp_path):
        # A sample whose derived zT rests on a metallic σ with an impossibly
        # low κ (Wiedemann–Franz violation) must be flagged when the target
        # is the DERIVED zT.
        store = ArrayStore(tmp_path / ".latos" / "arrays")
        pid = new_id()

        def _te(sid: str, arrays: dict) -> Measurement:
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
                sample_id=sid,
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

        sid = new_id()
        t = np.array([500.0, 600.0])
        rs = _te(
            sid,
            {
                "temperature_k": t,
                "resistivity_uohm_m": np.array([0.28, 0.30]),  # metallic
                "seebeck_uv_k": np.array([26.0, 30.0]),
            },
        )
        lfa = _te(sid, {"temperature_k": t, "thermal_conductivity": np.array([1.4, 1.5])})
        sample = Sample(
            id=sid,
            project_id=pid,
            canonical_name="WF-BAD",
            aliases=(),
            measurements=(rs, lfa),
        )
        project = Project(
            id=pid,
            name=tmp_path.name,
            root_path=tmp_path,
            created_at=utc_now(),
            schema_version=4,
            samples=(sample,),
            unassigned_files=(),
        )
        synthesis_store.set_sample_params(tmp_path, sid, {"doping_pct": 1.0})
        rows, _ = optimization_data.build_dataset(
            project,
            store,
            synthesis_store.load_params(tmp_path),
            "doping_pct",
            optimization_data.DERIVED_ZT,
        )
        assert rows, "sample should yield a derived-zT row"
        flags = optimization_data.quality_flags(
            project, rows, "doping_pct", optimization_data.DERIVED_ZT, store
        )
        assert any("Wiedemann" in f.reason for f in flags)
