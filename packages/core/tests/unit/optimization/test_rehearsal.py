"""Tests for the campaign rehearsal harness.

A rehearsal answers two questions before a sample is spent: roughly how many
experiments will this take, and is the plan sound. The second is the valuable
one, because it is normally settled by taste — a physics-informed prior *feels*
like free information, and three separate ones have now been measured not to be.

Most of what follows is therefore about the harness giving the right verdict on
priors whose effect is known by construction, and about it not overclaiming.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization.rehearsal import (
    CAVEAT,
    HARMS,
    HELPS,
    NEUTRAL,
    Shape,
    default_shapes,
    rehearse,
)

BOUNDS = (0.0, 1.0)
FAST = {"n_seeds": 8, "budget": 9, "noise": 0.06}


def _sharp_peak() -> Shape:
    """One narrow optimum at x = 0.72, away from every seed point."""

    def fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 0.3 + 1.2 * np.exp(-((x - 0.72) ** 2) / (2 * 0.09**2))

    return Shape("sharp peak", fn, "hardest case")


class TestTheReport:
    def test_it_reports_one_outcome_per_shape(self):
        report = rehearse(bounds=BOUNDS, **FAST)
        assert len(report.outcomes) == len(default_shapes(BOUNDS))

    def test_the_caveat_travels_with_every_report(self):
        """A budget quoted without it manufactures the confidence we exist to remove."""
        report = rehearse(bounds=BOUNDS, **FAST)
        assert report.caveat == CAVEAT
        assert "not evidence that the real system follows any of them" in report.caveat
        assert CAVEAT in report.summary()

    def test_the_summary_is_readable_without_the_object(self):
        report = rehearse(bounds=BOUNDS, **FAST)
        text = report.summary()
        assert "Median experiments" in text or "did not reach the tolerance" in text
        for outcome in report.outcomes:
            assert outcome.name in text

    def test_it_is_deterministic(self):
        """Same inputs, same answer — a rehearsal that drifts cannot be quoted."""
        first = rehearse(bounds=BOUNDS, **FAST)
        second = rehearse(bounds=BOUNDS, **FAST)
        assert [o.median_experiments for o in first.outcomes] == [
            o.median_experiments for o in second.outcomes
        ]
        assert first.solved_fraction == second.solved_fraction


class TestTheEndpointTrap:
    """The design flaw that made an earlier version of this experiment useless."""

    def test_endpoint_optima_are_marked_as_not_discriminating(self):
        """A seed design that measures both ends solves those shapes for free.

        Reporting them alongside the hard cases without saying so is how 54.5 %
        of an early Starrydata benchmark came to mean nothing.
        """
        report = rehearse(bounds=BOUNDS, **FAST)
        by_name = {o.name: o for o in report.outcomes}
        assert by_name["ideal / linear"].discriminating is False
        assert by_name["saturating"].discriminating is False
        assert by_name["interior maximum"].discriminating is True
        assert by_name["sharp peak"].discriminating is True

    def test_a_sign_flip_puts_the_optimum_at_an_endpoint(self):
        """Documented behaviour in ionic thermoelectrics, and worth its own check.

        When a mixing ratio carries the coefficient through zero, the magnitude
        has an interior *minimum*, so both endpoints beat everything between.
        """
        report = rehearse(bounds=BOUNDS, **FAST)
        by_name = {o.name: o for o in report.outcomes}
        assert by_name["sign flip"].discriminating is False

    def test_the_headline_ignores_the_free_wins(self):
        """Easy shapes must not flatter the budget the lab is asked to plan for."""
        report = rehearse(bounds=BOUNDS, **FAST)
        hard = [o for o in report.outcomes if o.discriminating]
        assert report.solved_fraction == pytest.approx(
            float(np.mean([o.solved_fraction for o in hard]))
        )

    def test_a_family_of_only_endpoint_shapes_reports_nothing_useful(self):
        """Better to say "no answer" than to average the trivial cases."""

        def rising(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=float)

        report = rehearse(bounds=BOUNDS, shapes=(Shape("rising", rising, "monotonic"),), **FAST)
        assert report.median_experiments is None
        assert report.solved_fraction == 0.0


class TestAuditioningAPrior:
    def test_no_prior_means_no_verdict(self):
        report = rehearse(bounds=BOUNDS, **FAST)
        assert report.prior_verdict is None
        assert report.prior_outcomes == ()

    def test_a_prior_that_knows_the_answer_helps(self):
        """The control: a prior equal to the truth must be recognised as good.

        If the harness cannot detect a prior this good, its HARMS verdicts mean
        nothing either.
        """
        peak = _sharp_peak()
        report = rehearse(
            bounds=BOUNDS, shapes=(peak,), prior_mean=peak.fn, n_seeds=8, budget=9, noise=0.06
        )
        assert report.prior_verdict == HELPS

    def test_a_prior_pointing_the_wrong_way_harms(self):
        """A straight line across a range whose optimum is a narrow interior peak.

        This is the shape of the mixing-law prior that dropped the success rate
        on exactly this objective, and the harness should say so unprompted.
        """
        report = rehearse(
            bounds=BOUNDS,
            shapes=(_sharp_peak(),),
            prior_mean=lambda x: 0.3 + 1.2 * np.asarray(x, dtype=float),
            n_seeds=10,
            budget=9,
            noise=0.06,
        )
        assert report.prior_verdict == HARMS
        assert "suppresses the search" in report.prior_detail

    def test_a_verdict_always_carries_its_numbers(self):
        peak = _sharp_peak()
        report = rehearse(
            bounds=BOUNDS, shapes=(peak,), prior_mean=peak.fn, n_seeds=8, budget=9, noise=0.06
        )
        assert "%" in report.prior_detail
        assert report.prior_verdict in {HELPS, NEUTRAL, HARMS}

    def test_the_unprimed_outcomes_are_kept_alongside_the_primed_ones(self):
        """Both halves of the comparison stay on the report, not just the verdict."""
        peak = _sharp_peak()
        report = rehearse(
            bounds=BOUNDS, shapes=(peak,), prior_mean=peak.fn, n_seeds=8, budget=9, noise=0.06
        )
        assert len(report.outcomes) == 1
        assert len(report.prior_outcomes) == 1


class TestScaling:
    def test_shapes_follow_the_bounds_they_are_given(self):
        """A campaign on doping percent is not a campaign on [0, 1]."""
        wide = default_shapes((0.0, 100.0))
        peak = next(s for s in wide if s.name == "sharp peak")
        grid = np.linspace(0.0, 100.0, 1001)
        assert grid[int(np.argmax(peak.fn(grid)))] == pytest.approx(72.0, abs=1.0)

    def test_a_rehearsal_runs_on_an_arbitrary_range(self):
        report = rehearse(bounds=(300.0, 700.0), **FAST)
        assert len(report.outcomes) == 5


class TestRefusals:
    """Nonsense in must not produce a plausible-looking budget out."""

    def test_a_budget_below_two_is_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            rehearse(bounds=BOUNDS, noise=0.05, budget=1)

    def test_noise_outside_a_fraction_is_rejected(self):
        with pytest.raises(ValueError, match="fraction of the signal"):
            rehearse(bounds=BOUNDS, noise=8.0, budget=6)

    def test_a_tolerance_outside_a_fraction_is_rejected(self):
        with pytest.raises(ValueError, match="tolerance is a fraction"):
            rehearse(bounds=BOUNDS, noise=0.05, budget=6, tolerance=5.0)

    def test_a_seed_design_larger_than_the_budget_is_rejected(self):
        with pytest.raises(ValueError, match="leaving nothing for the optimizer"):
            rehearse(bounds=BOUNDS, noise=0.05, budget=3, seed_design=(0.0, 0.25, 0.5, 0.75, 1.0))


class TestScoring:
    def test_a_run_is_scored_on_the_sample_made_not_the_reading(self):
        """Noise must not be able to win a rehearsal.

        With a huge scatter the readings are meaningless, but the score looks at
        the true value where samples were actually made. A harness scored on the
        noisy reading would report success for a campaign that found nothing.
        """
        report = rehearse(bounds=BOUNDS, shapes=(_sharp_peak(),), n_seeds=6, budget=5, noise=0.45)
        assert 0.0 <= report.solved_fraction <= 1.0

    def test_a_generous_budget_solves_more_often_than_a_tight_one(self):
        peak = _sharp_peak()
        tight = rehearse(bounds=BOUNDS, shapes=(peak,), n_seeds=8, budget=4, noise=0.06)
        roomy = rehearse(bounds=BOUNDS, shapes=(peak,), n_seeds=8, budget=12, noise=0.06)
        assert roomy.solved_fraction >= tight.solved_fraction
