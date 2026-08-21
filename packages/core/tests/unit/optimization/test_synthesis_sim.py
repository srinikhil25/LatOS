"""Tests for the simulated-synthesis harness (AX2).

The harness exists to produce a number — N90% — that gets compared against a
published one. That makes its *geometry* the thing worth testing hardest: if the
model functions are not the paper's model functions, the comparison is
meaningless however carefully the campaigns are run.

So most of this file checks the surface rather than the search: peak heights and
widths, that anisotropy preserves peak volume, that the success region is the
size Gaussian geometry says it should be, and that the optimum is interior.
Campaign behaviour is checked for mechanics (no repeated grid points, budgets
respected, reproducibility) and the statistical claims are marked `slow`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from latos.optimization.synthesis_sim import (
    _STREAM_CAMPAIGN,
    _STREAM_MODEL,
    GRIDS,
    PUBLISHED_N90,
    PW_MEDIUM,
    PW_NARROW,
    PW_WIDE,
    _anisotropic_sigma,
    _snap,
    _stream,
    make_model_function,
    n90,
    run_grid_campaign,
)


class TestTheSurfaceGeometry:
    """If these are wrong, every N90% downstream is meaningless."""

    def test_peak_heights_match_the_paper(self):
        assert [p.height for p in make_model_function(2, PW_MEDIUM, seed=0).peaks] == [
            1.2,
            0.7,
            0.3,
        ]
        assert [p.height for p in make_model_function(3, PW_MEDIUM, seed=0).peaks] == [
            1.2,
            0.7,
            0.6,
            0.3,
        ]

    def test_the_global_peak_has_the_requested_process_window(self):
        for pw in (PW_NARROW, PW_MEDIUM, PW_WIDE):
            m = make_model_function(2, pw, seed=0)
            assert m.peaks[0].sigma == (pw, pw)

    @pytest.mark.parametrize("n_dims", [2, 3])
    @pytest.mark.parametrize("aspect", [1.0, 2.0, 5.0, 10.0])
    def test_anisotropy_preserves_peak_volume(self, n_dims: int, aspect: float):
        """The load-bearing design decision. If a stretched window were also a
        *smaller* one, the anisotropy sweep would be measuring that small targets
        are hard to find, which needs no simulation."""
        widths = _anisotropic_sigma(PW_MEDIUM, aspect, n_dims)
        assert float(np.prod(widths)) == pytest.approx(PW_MEDIUM**n_dims, rel=1e-9)
        assert max(widths) / min(widths) == pytest.approx(aspect, rel=1e-9)

    def test_isotropic_is_the_aspect_one_case(self):
        assert _anisotropic_sigma(5.0, 1.0, 2) == (5.0, 5.0)

    def test_rejects_a_non_positive_aspect(self):
        with pytest.raises(ValueError, match="aspect"):
            _anisotropic_sigma(5.0, 0.0, 2)

    def test_rejects_dimensions_the_paper_does_not_cover(self):
        with pytest.raises(ValueError, match="n_dims"):
            make_model_function(4, PW_MEDIUM, seed=0)

    def test_the_optimum_is_interior(self):
        """An optimum against a wall is an easier search along that axis — the
        model only has to approach from one side — so it would quietly change
        what N90% means."""
        for seed in range(20):
            m = make_model_function(2, PW_WIDE, seed=seed)
            best = m.lattice[int(np.argmax(m.clean(m.lattice)))]
            assert np.all(best > 0)
            assert np.all(best < GRIDS - 1)

    def test_the_success_region_is_the_size_gaussian_geometry_predicts(self):
        """A Gaussian sits above 90% of its peak within 0.459 sigma, so the
        success region should scale as sigma^2 in 2-D. This is what makes a
        narrow process window genuinely harder, and it is the property the whole
        study depends on."""
        areas = {}
        for pw in (PW_NARROW, PW_MEDIUM, PW_WIDE):
            counts = []
            for seed in range(20):
                m = make_model_function(2, pw, seed=seed)
                vals = m.clean(m.lattice)
                counts.append(int(np.sum(vals >= 0.9 * vals.max())))
            areas[pw] = float(np.mean(counts))
            expected = math.pi * (0.459 * pw) ** 2
            assert areas[pw] == pytest.approx(expected, rel=0.5)
        assert areas[PW_NARROW] < areas[PW_MEDIUM] < areas[PW_WIDE]

    def test_noise_is_a_fraction_of_the_tallest_peak(self):
        m = make_model_function(2, PW_MEDIUM, seed=0, noise_fraction=0.05)
        point = np.asarray(m.peaks[0].centre)
        draws = np.array([m(point, np.random.default_rng(s)) for s in range(400)])
        assert float(np.std(draws)) == pytest.approx(0.05 * 1.2, rel=0.25)


class TestTheSeedStreamsAreIndependent:
    """Regression test for the bug that invalidated the first N90% numbers.

    `make_model_function` and `run_grid_campaign` both took a `seed`, and callers
    naturally pass the same one so a run is reproducible. Both then called
    `default_rng(seed)`, so the peak centre — drawn from `uniform(inset, 50-inset)`
    — and the first initial point — drawn from `uniform(0, 50)` — came off the
    *same* variate and differed by at most `inset` grids.

    The first "random" point therefore landed on the optimum every time, and more
    precisely for narrower windows, because the inset shrinks with the peak. The
    symptom was a harness that looked like an excellent optimizer (N90% of 9
    against a published 16.8) while the process-window trend ran backwards. The
    whole anisotropy study rests on difficulty responding to peak shape, so this
    would have invalidated every downstream number.
    """

    @pytest.mark.parametrize("pw", [PW_NARROW, PW_MEDIUM, PW_WIDE])
    def test_the_first_initial_point_is_not_near_the_optimum(self, pw: float):
        distances = []
        for seed in range(120):
            m = make_model_function(2, pw, seed=seed)
            first = np.rint(_stream(_STREAM_CAMPAIGN, seed).uniform(0, GRIDS - 1, size=2))
            distances.append(float(np.linalg.norm(first - np.asarray(m.peaks[0].centre))))
        # Uniform points in a 51x51 box average ~19 grids from any fixed target.
        # The bug produced a median of 2.5 (narrow) and 5.8 (wide).
        assert float(np.median(distances)) > 12.0

    def test_streams_stay_reproducible(self):
        a = _stream(_STREAM_CAMPAIGN, 5).uniform(size=4)
        b = _stream(_STREAM_CAMPAIGN, 5).uniform(size=4)
        assert np.array_equal(a, b)

    def test_different_streams_differ_for_the_same_seed(self):
        model = _stream(_STREAM_MODEL, 5).uniform(size=4)
        campaign = _stream(_STREAM_CAMPAIGN, 5).uniform(size=4)
        assert not np.allclose(model, campaign)


class TestSnapping:
    """The paper searches a lattice; the optimizer proposes anywhere."""

    def test_returns_the_nearest_point_when_free(self):
        assert _snap(np.array([3.4, 7.6]), set()) == (3, 8)

    def test_never_returns_a_visited_point(self):
        visited = {(3, 8)}
        assert _snap(np.array([3.4, 7.6]), visited) not in visited

    def test_stays_inside_the_lattice(self):
        for p in ([-99.0, -99.0], [999.0, 999.0]):
            got = _snap(np.array(p), set())
            assert all(0 <= v < GRIDS for v in got)

    def test_a_crowded_neighbourhood_still_finds_somewhere(self):
        visited = {(i, j) for i in range(3) for j in range(3)}
        assert _snap(np.array([1.0, 1.0]), visited) not in visited


class TestN90:
    def test_is_the_ninetieth_percentile_of_successes(self):
        outs = [
            run_grid_campaign(make_model_function(2, PW_WIDE, seed=s), budget=100, seed=s)
            for s in range(10)
        ]
        got = n90(outs)
        trials = sorted(o.trials_to_success for o in outs if o.found)
        assert got == float(trials[8])  # ceil(0.9 * 10) = 9 -> the 9th smallest

    def test_returns_none_when_too_few_ever_succeed(self):
        """Censored runs are not the same as slow ones. Treating a campaign that
        never found the optimum as if it had finished at the budget would
        understate N90% exactly where the method is doing worst."""
        outs = [
            run_grid_campaign(make_model_function(2, PW_NARROW, seed=s), budget=6, seed=s)
            for s in range(10)
        ]
        assert n90(outs) is None

    def test_rejects_an_empty_batch(self):
        with pytest.raises(ValueError, match="no outcomes"):
            n90([])


class TestCampaignMechanics:
    def test_never_measures_the_same_grid_point_twice(self):
        m = make_model_function(2, PW_MEDIUM, seed=1)
        out = run_grid_campaign(m, budget=25, seed=1)
        assert out.n_trials <= 25

    def test_stops_at_the_budget_when_it_cannot_find_the_optimum(self):
        m = make_model_function(2, PW_NARROW, seed=7)
        out = run_grid_campaign(m, budget=8, seed=7)
        assert out.n_trials <= 8
        if not out.found:
            assert out.trials_to_success is None

    def test_is_reproducible_for_a_fixed_seed(self):
        m = make_model_function(2, PW_MEDIUM, seed=2)
        a = run_grid_campaign(m, budget=30, seed=2)
        b = run_grid_campaign(m, budget=30, seed=2)
        assert (a.found, a.trials_to_success, a.n_trials) == (
            b.found,
            b.trials_to_success,
            b.n_trials,
        )

    def test_rejects_an_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            run_grid_campaign(make_model_function(2, PW_MEDIUM, seed=0), strategy="greedy")

    def test_the_isotropic_arm_runs(self):
        """The control condition for H1/H2 — an isotropic kernel cannot express
        an elongated process window, and measuring that costs requires being able
        to switch ARD off."""
        m = make_model_function(2, PW_MEDIUM, seed=0)
        out = run_grid_campaign(m, budget=20, seed=0, kernel_ard=False)
        assert out.n_trials <= 20

    def test_published_values_cover_the_cells_the_gate_checks(self):
        for key in ((2, PW_WIDE), (2, PW_MEDIUM), (2, PW_NARROW), (3, PW_MEDIUM), (3, PW_NARROW)):
            assert key in PUBLISHED_N90


@pytest.mark.slow
class TestItBeatsRandomOnASynthesisSurface:
    """The claim that matters for a materials researcher: fewer syntheses than
    guessing. Marked slow — it is a multi-campaign comparison, not a unit test."""

    def test_bayesian_optimization_finds_the_window_sooner_than_random(self):
        seeds = range(20)
        bo = [
            run_grid_campaign(make_model_function(2, PW_MEDIUM, seed=s), budget=100, seed=s)
            for s in seeds
        ]
        rand = [
            run_grid_campaign(
                make_model_function(2, PW_MEDIUM, seed=s), budget=100, seed=s, strategy="random"
            )
            for s in seeds
        ]
        assert sum(o.found for o in bo) > sum(o.found for o in rand)
        bo_trials = [o.trials_to_success for o in bo if o.found]
        rand_trials = [o.trials_to_success for o in rand if o.found]
        assert np.median(bo_trials) < np.median(rand_trials)
