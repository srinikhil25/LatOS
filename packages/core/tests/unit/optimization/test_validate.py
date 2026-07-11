"""Tests for outcome validation (`latos.optimization.validate`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from latos.optimization.validate import (
    list_preregistrations,
    outcome_path_for,
    validate_outcome,
    write_outcome,
)


def _record(
    *,
    predicted_mean: float = 0.93,
    interval: tuple[float, float] = (0.68, 1.18),
    prior_best: float = 0.985,
    direction: str = "maximize",
    x: float = 3.45,
    property_name: str = "zT (derived)",
    reliability: str = "exploratory",
) -> dict[str, Any]:
    """A minimal but realistic frozen prereg record dict."""
    return {
        "kind": "latos.bo.prereg",
        "created_at": "2026-07-11T17:21:33+00:00",
        "objective": {
            "property": property_name,
            "direction": direction,
            "aggregation": "peak",
            "input_variable": "doping_pct",
            "search_bounds": [0.0, 5.0],
        },
        "prediction_at_recommendation": {
            "x": x,
            "predicted_mean": predicted_mean,
            "predictive_interval_95": list(interval),
        },
        "prior_best": prior_best,
        "reliability": {"level": reliability},
    }


class TestCalibration:
    def test_measured_inside_interval_is_calibrated(self):
        v = validate_outcome(_record(), measured=0.95)
        assert v.within_interval is True
        assert "within" in v.summary.lower()

    def test_measured_outside_interval_is_overconfident(self):
        v = validate_outcome(_record(), measured=1.40)
        assert v.within_interval is False
        assert "outside" in v.summary.lower()

    def test_interval_bounds_are_inclusive(self):
        v = validate_outcome(_record(interval=(0.68, 1.18)), measured=1.18)
        assert v.within_interval is True


class TestImprovement:
    def test_maximize_beats_prior_best(self):
        v = validate_outcome(_record(prior_best=0.985), measured=1.05)
        assert v.improved is True

    def test_maximize_does_not_beat_prior_best(self):
        v = validate_outcome(_record(prior_best=0.985), measured=0.95)
        assert v.improved is False
        assert "did not beat" in v.summary.lower()

    def test_minimize_improvement_is_lower(self):
        # For a minimize objective (e.g. thermal conductivity) a *lower*
        # measured value is the improvement.
        rec = _record(direction="minimize", prior_best=1.5, property_name="kappa")
        assert validate_outcome(rec, measured=1.2).improved is True
        assert validate_outcome(rec, measured=1.9).improved is False


class TestErrors:
    def test_signed_and_absolute_error(self):
        v = validate_outcome(_record(predicted_mean=0.93), measured=1.00)
        assert round(v.signed_error, 6) == 0.07
        assert round(v.absolute_error, 6) == 0.07

    def test_relative_error(self):
        v = validate_outcome(_record(predicted_mean=0.90), measured=1.00)
        assert v.relative_error is not None
        assert round(v.relative_error, 4) == 0.1

    def test_relative_error_none_when_measured_zero(self):
        v = validate_outcome(_record(predicted_mean=0.0), measured=0.0)
        assert v.relative_error is None


class TestPersistenceAndListing:
    def _freeze_dir(self, root: Path) -> Path:
        d = root / ".latos" / "prereg"
        d.mkdir(parents=True)
        return d

    def test_outcome_path_is_sibling(self, tmp_path: Path):
        p = tmp_path / "prereg_20260101T000000Z.json"
        assert outcome_path_for(p).name == "prereg_20260101T000000Z.outcome.json"

    def test_write_and_relist_attaches_outcome(self, tmp_path: Path):
        d = self._freeze_dir(tmp_path)
        prereg = d / "prereg_20260711T172133Z.json"
        prereg.write_text(json.dumps(_record()), encoding="utf-8")

        listed = list_preregistrations(tmp_path)
        assert len(listed) == 1
        assert listed[0].outcome is None
        assert listed[0].reliability_level == "exploratory"
        assert listed[0].recommended_x == 3.45

        verdict = validate_outcome(_record(), measured=0.95)
        write_outcome(prereg, verdict)

        relisted = list_preregistrations(tmp_path)
        assert len(relisted) == 1  # the .outcome.json sibling is not a new entry
        assert relisted[0].outcome is not None
        assert relisted[0].outcome["within_interval"] is True
        assert relisted[0].outcome["measured"] == 0.95

    def test_newest_first(self, tmp_path: Path):
        d = self._freeze_dir(tmp_path)
        older = _record()
        older["created_at"] = "2026-07-10T00:00:00+00:00"
        newer = _record()
        newer["created_at"] = "2026-07-12T00:00:00+00:00"
        (d / "prereg_a.json").write_text(json.dumps(older), encoding="utf-8")
        (d / "prereg_b.json").write_text(json.dumps(newer), encoding="utf-8")
        listed = list_preregistrations(tmp_path)
        assert [e.created_at for e in listed] == [
            "2026-07-12T00:00:00+00:00",
            "2026-07-10T00:00:00+00:00",
        ]

    def test_malformed_record_skipped(self, tmp_path: Path):
        d = self._freeze_dir(tmp_path)
        (d / "prereg_bad.json").write_text("{not json", encoding="utf-8")
        (d / "prereg_ok.json").write_text(json.dumps(_record()), encoding="utf-8")
        assert len(list_preregistrations(tmp_path)) == 1

    def test_no_prereg_dir_returns_empty(self, tmp_path: Path):
        assert list_preregistrations(tmp_path) == []


class TestPreregRecordCarriesDirection:
    def test_build_record_includes_direction(self):
        """Regression: the frozen record must carry the objective direction
        so validation scores improvement correctly."""
        import numpy as np

        from latos.optimization.engine import optimize
        from latos.optimization.prereg import build_record

        x = np.array([1.0, 3.0, 5.0])
        y = np.array([2.0, 1.0, 1.8])  # a loss to minimize
        res = optimize(
            x, y, bounds=(1.0, 5.0), input_name="d", target_name="loss",
            direction="minimize",
        )
        record = build_record(res, prior_best=res.best_y)
        assert record["objective"]["direction"] == "minimize"
