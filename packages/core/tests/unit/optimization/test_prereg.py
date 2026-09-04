"""Tests for `latos.optimization.prereg` — the auditable pre-registration record."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latos import __version__
from latos.optimization import (
    build_record,
    freeze,
    length_scale_robustness,
    observations_digest,
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


class TestObservationsDigest:
    """The digest identifies the training set. Everything here is about that."""

    X = (0.0, 1.0, 3.0, 5.0)
    Y = (0.587, 0.362, 0.967, 0.482)

    def test_the_same_data_hashes_the_same(self):
        assert observations_digest(self.X, self.Y) == observations_digest(self.X, self.Y)

    def test_row_order_does_not_change_it(self):
        """Deliberate: a reordering is the same set of measurements.

        An auditor recomputing the digest from their own notebook will not
        list the rows in the order the code happened to hold them, and a check
        that cried mismatch over that would be a check nobody trusts.
        """
        order = [2, 0, 3, 1]
        shuffled_x = tuple(self.X[i] for i in order)
        shuffled_y = tuple(self.Y[i] for i in order)
        assert observations_digest(shuffled_x, shuffled_y) == observations_digest(self.X, self.Y)

    def test_pairing_is_preserved_when_sorting(self):
        """Sorting must not decouple x from its own y."""
        swapped_y = (self.Y[1], self.Y[0], self.Y[2], self.Y[3])
        assert observations_digest(self.X, swapped_y) != observations_digest(self.X, self.Y)

    def test_a_changed_measurement_changes_it(self):
        nudged = (self.Y[0] + 1e-9, *self.Y[1:])
        assert observations_digest(self.X, nudged) != observations_digest(self.X, self.Y)

    def test_weights_are_part_of_the_training_set(self):
        """Same (x, y), different per-point noise, is a different fit."""
        bare = observations_digest(self.X, self.Y)
        weighted = observations_digest(self.X, self.Y, sigma=(0.1, 0.1, 0.1, 0.1))
        other = observations_digest(self.X, self.Y, sigma=(0.1, 0.2, 0.1, 0.1))
        assert bare != weighted != other
        assert bare != other

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="x has 4 points and y has 3"):
            observations_digest(self.X, self.Y[:3])
        with pytest.raises(ValueError, match="sigma has 2 points"):
            observations_digest(self.X, self.Y, sigma=(0.1, 0.2))


class TestTheRecordIdentifiesItsTrainingSet:
    """The gap this closes: `n_observations` is a count, not an identity."""

    def test_the_digest_recomputes_from_the_observations(self):
        res, x, y = _result()
        record = build_record(res, prior_best=res.best_y)
        assert record["training_data"]["sha256"] == observations_digest(x, y)
        assert record["training_data"]["n_observations"] == len(x)

    def test_the_record_alone_is_enough_to_recompute_the_digest(self):
        """No workbook, no re-run, no rounding: just the JSON on disk.

        Recomputing from the values printed in a report does NOT work, because
        the digest is over exact float64 and a report rounds. That is why the
        observations are stored here at full precision.
        """
        res, _, _ = _result()
        record = json.loads(json.dumps(build_record(res, prior_best=res.best_y)))
        data = record["training_data"]
        assert (
            observations_digest(
                data["x"], data["y"], sigma=record["frozen_config"]["point_noise_scale"]
            )
            == data["sha256"]
        )

    def test_rounded_values_do_not_reproduce_the_digest(self):
        """The trap this guards against, stated as a test.

        A fitted slope is a full-precision float. Typing the 4-decimal figure
        off a report and hashing that gives a mismatch which means nothing —
        hence storing the observations rather than expecting a human to
        re-enter them.
        """
        x = (0.0, 1.0, 3.0)
        y = (0.5871234567890123, 0.3621111111111111, 0.9673333333333333)
        rounded = [round(v, 4) for v in y]
        assert rounded != list(y)  # the fixture must actually exercise this
        assert observations_digest(x, rounded) != observations_digest(x, y)

    def test_two_records_on_different_data_are_now_distinguishable(self):
        """Before this, only the timestamps differed."""
        res_a, _, _ = _result()
        x = np.array([0.0, 1.0, 3.0, 5.0])
        y = np.array([0.587, 0.362, 0.967, 0.9])  # one sample measured differently
        res_b = optimize(
            x, y, bounds=(0.0, 5.0), input_name="doping_pct", target_name="peak_zt", seed=0
        )
        a = build_record(res_a, prior_best=res_a.best_y)
        b = build_record(res_b, prior_best=res_b.best_y)
        assert a["frozen_config"]["n_observations"] == b["frozen_config"]["n_observations"]
        assert a["training_data"]["sha256"] != b["training_data"]["sha256"]

    def test_the_digest_covers_the_weights_when_they_were_used(self):
        x = np.array([0.0, 1.0, 3.0, 5.0])
        y = np.array([0.587, 0.362, 0.967, 0.482])
        kw = {"bounds": (0.0, 5.0), "input_name": "doping_pct", "target_name": "peak_zt", "seed": 0}
        even = optimize(x, y, point_noise=np.full(4, 0.05), **kw)
        uneven = optimize(x, y, point_noise=np.array([0.05, 0.2, 0.05, 0.05]), **kw)
        rec_even = build_record(even, prior_best=even.best_y)["training_data"]
        rec_uneven = build_record(uneven, prior_best=uneven.best_y)["training_data"]
        assert rec_even["digest_covers_point_noise"] is True
        assert rec_even["sha256"] != rec_uneven["sha256"]

    def test_the_recorded_weights_make_the_digest_recomputable(self):
        """A hash advertised as falsifiable must ship every input to it."""
        x = np.array([0.0, 1.0, 3.0, 5.0])
        y = np.array([0.587, 0.362, 0.967, 0.482])
        res = optimize(
            x,
            y,
            bounds=(0.0, 5.0),
            input_name="doping_pct",
            target_name="peak_zt",
            seed=0,
            point_noise=np.array([0.05, 0.2, 0.05, 0.05]),
        )
        record = build_record(res, prior_best=res.best_y)
        weights = record["frozen_config"]["point_noise_scale"]
        assert weights is not None
        # Recompute using only what the record itself carries.
        assert (
            observations_digest(res.observed_x, res.observed_y, sigma=weights)
            == record["training_data"]["sha256"]
        )

    def test_equal_weighting_says_so(self):
        res, _, _ = _result()
        data = build_record(res, prior_best=res.best_y)["training_data"]
        assert data["digest_covers_point_noise"] is False
        assert data["point_noise_used"] is False
        record = build_record(_result()[0], prior_best=0.0)
        assert record["frozen_config"]["point_noise_scale"] is None


class TestTheRecordNamesItsBuild:
    def test_the_version_is_recorded(self):
        res, _, _ = _result()
        assert build_record(res, prior_best=res.best_y)["latos_version"] == __version__

    def test_the_note_shows_the_digest_and_the_version(self, tmp_path: Path):
        res, _, _ = _result()
        out = freeze(res, tmp_path / "prereg.json", prior_best=res.best_y)
        note = out.with_suffix(".md").read_text(encoding="utf-8")
        record = json.loads(out.read_text(encoding="utf-8"))
        assert record["training_data"]["sha256"] in note
        assert __version__ in note

    def test_a_record_written_before_this_still_renders(self, tmp_path: Path):
        """Frozen records are immutable, so the renderer must read older ones."""
        from latos.optimization.prereg import _to_markdown

        res, _, _ = _result()
        record = build_record(res, prior_best=res.best_y)
        del record["training_data"]
        del record["latos_version"]
        note = _to_markdown(record)
        assert "unknown" in note
