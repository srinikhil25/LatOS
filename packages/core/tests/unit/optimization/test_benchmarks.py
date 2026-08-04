"""Closed-loop validation on functions whose optimum is published (MV5).

Everything else is judged on real experiments where nobody knows the right
answer. These are the only tests that can say the loop *works* rather than that
it behaves plausibly.

Campaigns are short and run with `with_reliability=False`: the leave-one-out
grade costs n extra GP fits per round and simple regret does not depend on it.
The grade is exercised separately, once, in `TestReliabilityDuringACampaign`.

Anything that runs a *comparison* — several seeds times two strategies — is
marked `slow`. A single campaign is roughly six seconds, so the comparison
classes alone would add minutes to a suite that already takes six, and the
project convention is that CI can deselect with -m "not slow". The cheap tests
that follow still cover every code path; only the statistical claims are gated.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization.benchmarks import (
    BENCHMARKS,
    branin,
    hartmann3,
    run_campaign,
)

BRANIN_OPTIMA = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
HARTMANN3_OPTIMUM = np.array([[0.114614, 0.555649, 0.852547]])


class TestTheBenchmarksThemselves:
    """If the test functions are wrong, every result below is meaningless."""

    def test_branin_reaches_its_published_value_at_all_three_optima(self):
        got = branin(BRANIN_OPTIMA)
        assert got == pytest.approx([0.397887] * 3, abs=1e-5)

    def test_hartmann3_reaches_its_published_value(self):
        assert hartmann3(HARTMANN3_OPTIMUM)[0] == pytest.approx(-3.86278, abs=1e-4)

    def test_no_sampled_point_beats_the_published_optimum(self):
        """A random sweep must never find something better than the stated
        minimum — that would mean the constant, not the search, is wrong."""
        rng = np.random.default_rng(0)
        for name, lo, hi in (
            ("branin", [-5.0, 0.0], [10.0, 15.0]),
            ("hartmann3", [0.0] * 3, [1.0] * 3),
        ):
            bench = BENCHMARKS[name]
            pts = rng.uniform(lo, hi, size=(4000, bench.n_dims))
            assert float(bench.fn(pts).min()) >= bench.optimum - 1e-6

    def test_shapes_and_orientation(self):
        assert branin(np.array([[1.0, 2.0], [3.0, 4.0]])).shape == (2,)
        assert hartmann3(np.zeros((5, 3))).shape == (5,)


class TestCampaignMechanics:
    def test_history_is_monotone_and_matches_the_best(self):
        r = run_campaign("branin", n_initial=8, n_rounds=5, seed=0)
        h = np.asarray(r.history)
        assert len(h) == r.n_rounds + 1
        assert np.all(np.diff(h) <= 1e-12)  # best-so-far can only improve
        assert h[-1] == pytest.approx(r.best_y)
        assert r.regret == pytest.approx(r.best_y - BENCHMARKS["branin"].optimum)

    def test_regret_is_never_negative(self):
        for name in ("branin", "hartmann3"):
            r = run_campaign(name, n_initial=8, n_rounds=5, seed=1)
            assert r.regret >= -1e-9

    def test_same_seed_reproduces(self):
        a = run_campaign("branin", n_initial=8, n_rounds=6, seed=3)
        b = run_campaign("branin", n_initial=8, n_rounds=6, seed=3)
        assert a.best_x == b.best_x
        assert a.regret == pytest.approx(b.regret)

    def test_rejects_an_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            run_campaign("branin", n_rounds=1, strategy="greedy")


@pytest.mark.slow
class TestItFindsKnownOptima:
    """The headline claim. Simple regret against a published minimum."""

    def test_branin_gets_close(self):
        best = min(
            run_campaign("branin", n_initial=8, n_rounds=12, seed=s).regret for s in (0, 1, 2)
        )
        assert best < 0.2

    def test_hartmann3_gets_close(self):
        best = min(
            run_campaign("hartmann3", n_initial=8, n_rounds=12, seed=s).regret for s in (0, 1, 2)
        )
        assert best < 0.15

    def test_branin_lands_near_one_of_the_three_minima(self):
        r = run_campaign("branin", n_initial=8, n_rounds=12, seed=0)
        d = np.linalg.norm(BRANIN_OPTIMA - np.asarray(r.best_x), axis=1)
        assert d.min() < 1.5


@pytest.mark.slow
class TestItBeatsRandomSearch:
    """Without this comparison, a good result may be the sampling budget's
    doing rather than the method's."""

    @pytest.mark.parametrize("name", ["branin", "hartmann3"])
    def test_median_regret_is_lower_than_random(self, name: str):
        seeds = (0, 1, 2)
        bo = np.median([run_campaign(name, seed=s, n_initial=8, n_rounds=12).regret for s in seeds])
        rand = np.median(
            [
                run_campaign(name, seed=s, n_initial=8, n_rounds=12, strategy="random").regret
                for s in seeds
            ]
        )
        assert bo < rand

    def test_the_advantage_grows_with_dimension(self):
        """Random search degrades faster than Bayesian optimization as the box
        gains axes. This is the quantitative form of why one-variable BO is
        unimpressive and multi-variable BO is worth building."""

        def ratio(name: str) -> float:
            seeds = (0, 1, 2)
            bo = np.median(
                [run_campaign(name, seed=s, n_initial=8, n_rounds=12).regret for s in seeds]
            )
            rand = np.median(
                [
                    run_campaign(name, seed=s, n_initial=8, n_rounds=12, strategy="random").regret
                    for s in seeds
                ]
            )
            return float(rand / max(bo, 1e-9))

        assert ratio("hartmann3") > ratio("branin")


@pytest.mark.slow
class TestLengthScaleFloor:
    """Evidence for the limitation recorded in `optimize_nd`'s docstring: the
    1-D length-scale floor under-resolves multi-dimensional structure."""

    def test_a_lower_floor_rescues_a_stalled_branin_seed(self):
        stalled = run_campaign("branin", n_initial=8, n_rounds=12, seed=2)
        freed = run_campaign(
            "branin",
            n_initial=8,
            n_rounds=12,
            seed=2,
            length_scale_bounds=(0.3, 5.0),
        )
        assert stalled.regret > 1.0
        assert freed.regret < stalled.regret / 2


@pytest.mark.slow
class TestAcquisitionPolishPaysOff:
    """Measured over 8 seeds, not 3. On three seeds the refinement appeared to
    hurt Hartmann-3; over eight it improves the median there too. Three noisy
    campaigns are not enough to set a default on."""

    def test_polish_improves_the_worst_case_on_branin(self):
        seeds = tuple(range(8))
        grid = np.array(
            [
                run_campaign("branin", seed=s, n_initial=8, n_rounds=12, polish=False).regret
                for s in seeds
            ]
        )
        fine = np.array(
            [
                run_campaign("branin", seed=s, n_initial=8, n_rounds=12, polish=True).regret
                for s in seeds
            ]
        )
        assert np.median(fine) < np.median(grid)
        # The important one: refinement removes the catastrophic stall.
        assert fine.max() < grid.max() / 2


class TestReliabilityDuringACampaign:
    def test_the_grade_is_recorded_each_round(self):
        r = run_campaign("branin", n_initial=8, n_rounds=3, seed=0, with_reliability=True)
        assert len(r.reliability_levels) == r.n_rounds
        assert set(r.reliability_levels) <= {"exploratory", "indicative", "calibrated"}

    def test_a_short_two_dimensional_campaign_stays_exploratory(self):
        """Eleven points over a plane cannot cover it, and the fill-distance
        gate from MV3 should keep saying so however the count grows."""
        r = run_campaign("branin", n_initial=8, n_rounds=3, seed=0, with_reliability=True)
        assert set(r.reliability_levels) == {"exploratory"}
