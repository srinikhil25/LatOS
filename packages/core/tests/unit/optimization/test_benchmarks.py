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

    def test_the_advantage_is_large_on_both(self):
        """Bayesian optimization beats random by more than an order of magnitude
        on both problems.

        This assertion used to be stronger and more interesting — that the
        advantage *grows* with dimension, `ratio(hartmann3) > ratio(branin)` —
        and it was true when written. MV2's acquisition polish then improved the
        two-dimensional case enough that the ratios converged (~56x and ~55x
        over three seeds) and the ordering stopped reproducing. It is being
        recorded rather than quietly relaxed: at this budget both problems are
        nearly solved after twenty evaluations, so the ratio is governed by how
        badly random does rather than by dimension. Demonstrating the dimension
        effect needs a budget where BO is still working — an AX question, not an
        assertion to weaken until it goes green.
        """

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

        assert ratio("branin") > 10.0
        assert ratio("hartmann3") > 10.0


@pytest.mark.slow
class TestLengthScaleFloor:
    """Evidence for the `_LS_BOUNDS` floor, which AX1 lowered from 1.0 to 0.2.

    The predecessor of this class asserted that the old floor caused a
    catastrophic stall on Branin seed 2 (regret 2.32, rescued to 0.33 by a lower
    floor). That was true when measured and is no longer: MV2's continuous
    acquisition polish removed the stall independently, and seed 2 now finishes
    at 0.058 whatever the floor. The stall was Sobol quantisation, not the
    length-scale — so the original diagnosis was wrong even though the fix
    helped. What the floor actually costs is smaller, real, and measured below.
    """

    def test_the_old_floor_is_measurably_worse(self):
        """Why the default moved. Eight seeds, because three were not enough to
        set a default on the last time this was attempted."""
        seeds = tuple(range(8))
        old = np.median(
            [
                run_campaign(
                    "branin", seed=s, n_initial=8, n_rounds=12, length_scale_bounds=(1.0, 5.0)
                ).regret
                for s in seeds
            ]
        )
        new = np.median(
            [run_campaign("branin", seed=s, n_initial=8, n_rounds=12).regret for s in seeds]
        )
        assert new < old

    def test_lowering_the_floor_further_buys_nothing(self):
        """The reason 0.2 and not 0.1: below the saturation point the floor
        stops binding, so a smaller value only gives up guard rail for free.
        Halving it must not measurably change the answer."""
        seeds = tuple(range(4))
        default = [run_campaign("branin", seed=s, n_initial=8, n_rounds=12).regret for s in seeds]
        lower = [
            run_campaign(
                "branin", seed=s, n_initial=8, n_rounds=12, length_scale_bounds=(0.1, 5.0)
            ).regret
            for s in seeds
        ]
        assert np.allclose(default, lower, atol=1e-3)


@pytest.mark.slow
class TestAcquisitionPolishPaysOff:
    """Measured over 8 seeds, not 3. On three seeds the refinement appeared to
    hurt Hartmann-3; over eight it improves the median there too. Three noisy
    campaigns are not enough to set a default on.

    This class used to demand `fine.max() < grid.max() / 2`, and AX1 broke it in
    an informative way. Both numbers, on Branin over eight seeds:

                            grid (no polish)      fine (polish)
        floor 1.0 (old)     median 0.429  max 2.322    median 0.126  max 0.540
        floor 0.2 (now)     median 0.190  max 0.744    median 0.094  max 0.413

    The catastrophic 2.32 stall the halving criterion was calibrated against was
    an *unpolished* campaign at the old floor — and lowering the floor removes it
    too, independently (2.32 -> 0.74). So the polish and the length-scale floor
    were fixing the same pathology from opposite directions: the acquisition
    maximum falling into a Sobol gap because the surrogate was too smooth to
    place it precisely. With both fixed, each one's marginal contribution is
    necessarily smaller, and the arbitrary factor of two no longer holds (it is
    now 1.80x).

    The assertions below therefore test the claim that is actually being made —
    refinement improves both the typical and the worst case — rather than a
    ratio tuned to a failure mode that no longer exists.
    """

    def _regrets(self, *, polish: bool) -> np.ndarray:
        return np.array(
            [
                run_campaign("branin", seed=s, n_initial=8, n_rounds=12, polish=polish).regret
                for s in range(8)
            ]
        )

    def test_polish_improves_the_typical_case_on_branin(self):
        assert np.median(self._regrets(polish=True)) < np.median(self._regrets(polish=False))

    def test_polish_improves_the_worst_case_on_branin(self):
        assert self._regrets(polish=True).max() < self._regrets(polish=False).max()


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
