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
    res = optimize(x, y, bounds=(0.0, 5.0), input_name="doping_pct", target_name="peak_zt", seed=0)
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


class TestStoppingClaimIsFrozen:
    """The (epsilon, delta) claim must be recorded, not just displayed."""

    @staticmethod
    def _result():
        import numpy as np

        from latos.optimization.engine import optimize

        x = np.array([40.169, 44.946, 50.025, 54.956])
        y = np.array([71.6667, 57.0, 36.6667, 57.0])
        return optimize(
            x,
            y,
            bounds=(40.169, 54.956),
            input_name="particle_wt_pct",
            target_name="peak_force_n",
            direction="minimize",
        )

    def test_record_carries_the_claim(self):
        from latos.optimization.prereg import build_record

        res = self._result()
        rec = build_record(res, prior_best=res.best_y)
        claim = rec["stopping_claim"]
        assert claim["epsilon"] == res.epsilon
        assert claim["probability_within_epsilon"] == res.prob_within_epsilon
        assert claim["best_measured"] == res.best_y
        # And the criterion is stated in advance, like the other two.
        assert "stopping_claim" in rec["validation_criteria"]

    def test_markdown_states_the_claim_and_how_to_falsify_it(self, tmp_path):
        from latos.optimization.prereg import freeze

        res = self._result()
        path = freeze(res, tmp_path / "p.json", prior_best=res.best_y)
        md = path.with_suffix(".md").read_text(encoding="utf-8")
        assert "Stopping claim" in md
        assert "Falsifiable by" in md


class TestStoppingClaimValidation:
    """A later measurement either survives the frozen claim or breaks it."""

    @staticmethod
    def _record(epsilon=4.0, best=36.667, direction="minimize"):
        return {
            "objective": {"direction": direction, "y_transform": "identity"},
            "prediction_at_recommendation": {
                "predicted_mean": 44.3,
                "predictive_interval_95": [31.2, 57.4],
            },
            "prior_best": best,
            "stopping_claim": {"epsilon": epsilon, "best_measured": best},
        }

    def test_claim_holds_when_nothing_much_better_turns_up(self):
        from latos.optimization.validate import validate_outcome

        v = validate_outcome(self._record(), measured=51.0)  # worse than best
        assert v.stopping_claim_held is True
        assert "held" in v.summary

    def test_claim_breaks_on_a_big_improvement(self):
        from latos.optimization.validate import validate_outcome

        # Minimizing: 30.0 beats 36.667 by 6.7, more than epsilon = 4.0.
        v = validate_outcome(self._record(), measured=30.0)
        assert v.stopping_claim_held is False
        assert "did NOT hold" in v.summary

    def test_small_improvement_inside_epsilon_still_holds(self):
        from latos.optimization.validate import validate_outcome

        v = validate_outcome(self._record(), measured=34.0)  # beats by 2.7 < 4.0
        assert v.stopping_claim_held is True

    def test_maximize_direction_is_handled(self):
        from latos.optimization.validate import validate_outcome

        rec = self._record(direction="maximize", best=1.0, epsilon=0.1)
        assert validate_outcome(rec, measured=1.05).stopping_claim_held is True
        assert validate_outcome(rec, measured=1.5).stopping_claim_held is False

    def test_older_records_without_a_claim_still_validate(self):
        from latos.optimization.validate import validate_outcome

        rec = self._record()
        del rec["stopping_claim"]
        v = validate_outcome(rec, measured=51.0)
        assert v.stopping_claim_held is None
        assert "stopping claim" not in v.summary.lower()
