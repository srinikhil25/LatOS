"""Tests for `latos.optimization.prereg` — the auditable pre-registration record."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latos.optimization import (
    build_record,
    freeze,
    length_scale_robustness,
    optimize,
)


def _result():
    x = np.array([0.0, 1.0, 3.0, 5.0])
    y = np.array([0.587, 0.362, 0.967, 0.482])
    res = optimize(
        x, y, bounds=(0.0, 5.0), input_name="doping_pct", target_name="peak_zt", seed=0
    )
    return res, x, y


class TestBuildRecord:
    def test_pins_config_and_prediction(self):
        res, _, _ = _result()
        record = build_record(res, prior_best=res.best_y)
        assert record["kind"] == "latos.bo.prereg"
        assert record["objective"]["property"] == "peak_zt"
        assert record["frozen_config"]["seed"] == 0
        assert record["frozen_config"]["length_scale_fitted"] is True
        assert record["prior_best"] == res.best_y

    def test_prediction_lies_within_its_own_interval(self):
        res, _, _ = _result()
        pred = build_record(res, prior_best=res.best_y)["prediction_at_recommendation"]
        lo, hi = pred["predictive_interval_95"]
        assert lo <= pred["predicted_mean"] <= hi
        assert pred["ci95_predictive"] >= pred["ci95_model"]


class TestWrite:
    def test_freeze_emits_json_and_markdown(self, tmp_path: Path):
        res, x, y = _result()
        robustness = length_scale_robustness(
            x,
            y,
            bounds=(0.0, 5.0),
            input_name="doping_pct",
            target_name="peak_zt",
            length_scales=(1.0, 2.0, 3.0),
        )
        out = freeze(
            res, tmp_path / "sub" / "prereg.json", prior_best=res.best_y, robustness=robustness
        )
        assert out.exists()
        assert out.with_suffix(".md").exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "robustness" in data
        assert isinstance(data["robustness"]["stable"], bool)
        # the human-readable note mentions the frozen config and prediction
        md = out.with_suffix(".md").read_text(encoding="utf-8")
        assert "pre-registration" in md.lower()
        assert "predictive interval" in md.lower()
