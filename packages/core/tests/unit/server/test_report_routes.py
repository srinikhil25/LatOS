"""Tests for the Stage-6 /features and /correlations endpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from latos.core.enums import ReviewStatus, Technique
from latos.core.models import Measurement, Project, Sample, new_id, utc_now
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.orchestrator import IngestionResult
from latos.ingestion.parsed_data import ParsedData
from latos.server.app import create_app
from latos.server.state import ServerState


def _te(sid: str, store: ArrayStore, arrays: dict) -> Measurement:
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


def _sample(
    pid: str, store: ArrayStore, name: str, rho: float, seebeck: float, kappa: float
) -> Sample:
    sid = new_id()
    t = np.array([300.0, 600.0])
    rs = _te(
        sid,
        store,
        {
            "temperature_k": t,
            "resistivity_uohm_m": np.array([rho, rho]),
            "seebeck_uv_k": np.array([seebeck * 0.5, seebeck]),
        },
    )
    lfa = _te(sid, store, {"temperature_k": t, "thermal_conductivity": np.array([kappa, kappa])})
    return Sample(id=sid, project_id=pid, canonical_name=name, aliases=(), measurements=(rs, lfa))


def _client(root: Path) -> TestClient:
    store = ArrayStore(root / ".latos" / "arrays")
    pid = new_id()
    # Four samples with distinct transport so features vary across samples.
    samples = tuple(
        _sample(pid, store, name, rho, seebeck, kappa)
        for name, rho, seebeck, kappa in [
            ("A", 20.0, 150.0, 1.2),
            ("B", 30.0, 180.0, 1.4),
            ("C", 40.0, 210.0, 1.6),
            ("D", 55.0, 240.0, 1.9),
        ]
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


class TestFeaturesEndpoint:
    def test_no_project_404(self):
        assert TestClient(create_app()).get("/features").status_code == 404

    def test_returns_table_with_provenance(self, tmp_path: Path):
        body = _client(tmp_path).get("/features").json()
        assert "peak_zt" in body["properties"]
        assert len(body["rows"]) == 4
        cell = body["rows"][0]["features"]["peak_seebeck_uv_k"]
        assert cell["unit"] == "µV/K"
        assert "source" in cell and "reliable" in cell


class TestCorrelationsEndpoint:
    def test_returns_matrix_and_pairs(self, tmp_path: Path):
        body = _client(tmp_path).get("/correlations").json()
        props = body["properties"]
        assert "peak_seebeck_uv_k" in props
        # Square matrix, unit diagonal.
        assert len(body["matrix"]) == len(props)
        i = props.index("peak_seebeck_uv_k")
        assert body["matrix"][i][i] == 1.0
        # Seebeck rises monotonically with our samples -> some strong pair exists.
        assert any(abs(p["pearson"]) > 0.9 for p in body["pairs"])

    def test_reliable_only_flag_accepted(self, tmp_path: Path):
        resp = _client(tmp_path).get("/correlations", params={"reliable_only": True})
        assert resp.status_code == 200


class TestFigureEndpoints:
    def test_lists_journal_styles(self, tmp_path: Path):
        styles = _client(tmp_path).get("/report/styles").json()["styles"]
        assert {"nature", "acs", "rsc", "thesis", "presentation"} <= set(styles)

    def test_heatmap_svg(self, tmp_path: Path):
        resp = _client(tmp_path).get("/report/figure", params={"kind": "heatmap", "fmt": "svg"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/svg+xml"
        assert b"<svg" in resp.content[:400] or resp.content[:5] == b"<?xml"

    def test_scatter_png(self, tmp_path: Path):
        resp = _client(tmp_path).get(
            "/report/figure",
            params={"kind": "scatter", "x": "peak_seebeck_uv_k", "y": "peak_zt", "fmt": "png"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_scatter_requires_x_and_y(self, tmp_path: Path):
        resp = _client(tmp_path).get("/report/figure", params={"kind": "scatter"})
        assert resp.status_code == 400

    def test_bad_style_and_kind_and_format_400(self, tmp_path: Path):
        c = _client(tmp_path)
        assert (
            c.get("/report/figure", params={"kind": "heatmap", "style": "vogue"}).status_code == 400
        )
        assert c.get("/report/figure", params={"kind": "banana"}).status_code == 400
        assert c.get("/report/figure", params={"kind": "heatmap", "fmt": "gif"}).status_code == 400
