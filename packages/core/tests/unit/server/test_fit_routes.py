"""Tests for the Stage-4 fit endpoints (stateless — no project needed)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from latos.server.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _two_peak_spectrum():
    x = np.linspace(0.0, 100.0, 500)
    height1 = 50.0 / (3.0 * np.sqrt(2.0 * np.pi))
    height2 = 80.0 / (5.0 * np.sqrt(2.0 * np.pi))
    y = (
        10.0
        + 0.1 * x
        + height1 * np.exp(-0.5 * ((x - 30.0) / 3.0) ** 2)
        + height2 * np.exp(-0.5 * ((x - 60.0) / 5.0) ** 2)
    )
    rng = np.random.default_rng(0)
    return x, y + rng.normal(0.0, 0.05, size=x.size)


class TestDetectPeaks:
    def test_finds_the_two_peaks(self):
        x, y = _two_peak_spectrum()
        resp = _client().post("/fit/detect-peaks", json={"x": x.tolist(), "y": y.tolist()})
        assert resp.status_code == 200
        centers = sorted(resp.json()["centers"])
        assert len(centers) == 2
        assert centers[0] == 30.0 or abs(centers[0] - 30.0) < 1.0
        assert abs(centers[1] - 60.0) < 1.0


class TestFit:
    def test_fits_and_returns_overlay_arrays_and_report(self):
        x, y = _two_peak_spectrum()
        body = {
            "x": x.tolist(),
            "y": y.tolist(),
            "peak_shape": "gaussian",
            "peaks": [30.0, 60.0],
            "background": {"kind": "linear"},
        }
        resp = _client().post("/fit", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["r_squared"] > 0.999
        assert len(data["components"]) == 2
        # Overlay arrays align with the input length.
        assert len(data["best_fit"]) == len(x)
        assert len(data["residual"]) == len(x)
        assert len(data["baseline"]) == len(x)
        assert "R²" in data["markdown"]
        centers = sorted(c["center"] for c in data["components"])
        assert abs(centers[0] - 30.0) < 0.3
        assert abs(centers[1] - 60.0) < 0.3

    def test_xps_doublet_constraints_via_endpoint(self):
        x = np.linspace(925.0, 960.0, 700)
        h = 100.0 / (1.1 * np.sqrt(2.0 * np.pi))
        y = (
            45.0
            + h * np.exp(-0.5 * ((x - 932.6) / 1.1) ** 2)
            + 0.5 * h * np.exp(-0.5 * ((x - 952.4) / 1.1) ** 2)
        )
        body = {
            "x": x.tolist(),
            "y": y.tolist(),
            "peak_shape": "gaussian",
            "peaks": [932.6, 952.4],
            "background": {"kind": "shirley"},
            "constraints": [
                {"type": "fixed_delta", "ref": 0, "target": 1, "delta": 19.8},
                {"type": "fixed_ratio", "ref": 0, "target": 1, "ratio": 0.5},
                {"type": "shared_width", "ref": 0, "target": 1},
            ],
        }
        data = _client().post("/fit", json=body).json()
        centers = sorted(c["center"] for c in data["components"])
        assert (centers[1] - centers[0]) == pytest.approx(19.8, abs=1e-4)

    def test_bad_shape_returns_400(self):
        x, y = _two_peak_spectrum()
        resp = _client().post(
            "/fit",
            json={"x": x.tolist(), "y": y.tolist(), "peak_shape": "banana", "peaks": [30.0]},
        )
        assert resp.status_code == 400

    def test_presets_lists_doublets(self):
        data = _client().get("/fit/presets").json()
        assert "Cu 2p" in data["doublets"]
        assert data["doublets"]["Cu 2p"] == [19.8, 0.5]
