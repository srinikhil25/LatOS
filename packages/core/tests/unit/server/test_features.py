"""Tests for the cross-sample feature-extraction layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from latos.core.enums import ReviewStatus, Technique
from latos.core.models import Measurement, Project, Sample, new_id, utc_now
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.parsed_data import ParsedData
from latos.server.features import extract_features


def _te(sample_id: str, store: ArrayStore, arrays: dict) -> Measurement:
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


def _transport_sample(pid: str, store: ArrayStore, name: str, resistivity: float) -> Sample:
    sid = new_id()
    t = np.array([300.0, 600.0])
    rs = _te(
        sid,
        store,
        {
            "temperature_k": t,
            "resistivity_uohm_m": np.array([resistivity, resistivity]),
            "seebeck_uv_k": np.array([100.0, 200.0]),
        },
    )
    lfa = _te(sid, store, {"temperature_k": t, "thermal_conductivity": np.array([1.5, 1.5])})
    return Sample(id=sid, project_id=pid, canonical_name=name, aliases=(), measurements=(rs, lfa))


def _project(root: Path, resistivity: float) -> tuple[Project, ArrayStore]:
    store = ArrayStore(root / ".latos" / "arrays")
    pid = new_id()
    sample = _transport_sample(pid, store, "S1", resistivity)
    project = Project(
        id=pid,
        name=root.name,
        root_path=root,
        created_at=utc_now(),
        schema_version=4,
        samples=(sample,),
        unassigned_files=(),
        review_status=ReviewStatus.CONFIRMED,
        confirmed_at=utc_now(),
    )
    return project, store


class TestExtractFeatures:
    def test_derives_transport_features(self, tmp_path: Path):
        # ρ=30 µΩ·m, S~200 µV/K, κ=1.5 -> a physically consistent zT.
        project, store = _project(tmp_path, resistivity=30.0)
        table = extract_features(project, store)
        assert len(table.rows) == 1
        feats = table.rows[0].features
        assert "peak_zt" in feats
        assert "peak_seebeck_uv_k" in feats
        assert "peak_thermal_conductivity" in feats
        assert feats["peak_seebeck_uv_k"].value == 200.0
        assert "peak_zt" in table.properties

    def test_wiedemann_franz_violation_flags_zt_unreliable(self, tmp_path: Path):
        # Metallic ρ=0.3 µΩ·m with κ=1.5 -> WF-violating -> zT unreliable.
        project, store = _project(tmp_path, resistivity=0.3)
        table = extract_features(project, store)
        zt = table.rows[0].features["peak_zt"]
        assert zt.reliable is False
        assert zt.source.startswith("derived")

    def test_consistent_transport_is_reliable(self, tmp_path: Path):
        project, store = _project(tmp_path, resistivity=30.0)
        table = extract_features(project, store)
        assert table.rows[0].features["peak_zt"].reliable is True

    def test_empty_sample_yields_empty_row(self, tmp_path: Path):
        store = ArrayStore(tmp_path / ".latos" / "arrays")
        pid = new_id()
        bare = Sample(id=pid, project_id=pid, canonical_name="bare", aliases=(), measurements=())
        project = Project(
            id=pid,
            name="p",
            root_path=tmp_path,
            created_at=utc_now(),
            schema_version=4,
            samples=(bare,),
            unassigned_files=(),
        )
        table = extract_features(project, store)
        assert table.rows[0].features == {}
        assert table.properties == []
