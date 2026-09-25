"""Tests for the acquisition functions and the incumbent problem behind them."""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization.acquisitions import (
    ACQUISITIONS,
    DEFAULT_ACQUISITION,
    AcquisitionInputs,
    acquire,
    expected_improvement,
)
from latos.optimization.acquisitions import _expected_max_affine as expected_max_affine


def _inputs(**over) -> AcquisitionInputs:
    """A small posterior with an interior bump the arms can disagree about."""
    grid = np.linspace(0.0, 1.0, 101)
    mu = 1.0 + 0.5 * np.exp(-(((grid - 0.7) / 0.15) ** 2))
    sigma = 0.05 + 0.25 * np.exp(-(((grid - 0.3) / 0.2) ** 2))
    base = {
        "mu": mu,
        "sigma": sigma,
        "f_best_observed": 1.4,
        "f_best_plugin": 1.2,
        "noise_std": 0.1,
        "xi": 0.01,
        "cov": None,
    }
    base.update(over)
    return AcquisitionInputs(**base)


class TestTheExpectedMaximumOfLines:
    """`_expected_max_affine` is the only closed form here, so it is pinned hard."""

    def test_a_single_line_returns_its_intercept(self):
        # E[a + bZ] = a for any slope, because E[Z] = 0.
        assert expected_max_affine(np.array([3.0]), np.array([9.0])) == pytest.approx(3.0)

    def test_the_positive_part_of_a_normal(self):
        # E[max(0, Z)] = phi(0).
        got = expected_max_affine(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        assert got == pytest.approx(1.0 / np.sqrt(2 * np.pi), rel=1e-12)

    def test_equal_slopes_keep_only_the_best_intercept(self):
        got = expected_max_affine(np.array([1.0, 5.0, 2.0]), np.array([2.0, 2.0, 2.0]))
        assert got == pytest.approx(5.0)

    def test_flat_lines_return_their_maximum(self):
        got = expected_max_affine(np.array([1.5, -3.0, 0.0]), np.zeros(3))
        assert got == pytest.approx(1.5)

    def test_it_matches_monte_carlo(self):
        rng = np.random.default_rng(4)
        z = rng.standard_normal(400_000)
        for _ in range(5):
            a = rng.normal(0.0, 2.0, 8)
            b = rng.normal(0.0, 1.5, 8)
            mc = np.max(a[:, None] + b[:, None] * z[None, :], axis=0).mean()
            assert expected_max_affine(a, b) == pytest.approx(mc, abs=0.01)

    def test_it_is_never_below_the_best_intercept(self):
        # A convex function of Z, so Jensen puts the expectation at or above
        # the value at E[Z] = 0, which is max(a).
        rng = np.random.default_rng(5)
        for _ in range(20):
            a = rng.normal(0.0, 1.0, 6)
            b = rng.normal(0.0, 1.0, 6)
            assert expected_max_affine(a, b) >= float(np.max(a)) - 1e-9


class TestEveryArmIsWellFormed:
    def test_all_arms_score_the_whole_grid(self):
        inputs = _inputs()
        for name in ACQUISITIONS:
            scores = acquire(name, inputs)
            assert scores.shape == inputs.mu.shape, name
            assert np.all(np.isfinite(scores)), name

    def test_an_unknown_arm_is_refused(self):
        with pytest.raises(ValueError, match="acquisition must be one of"):
            acquire("gut feeling", _inputs())

    def test_the_default_is_the_shipped_behaviour(self):
        # Changing this changes every recommendation the tool makes, so it is
        # pinned rather than left to whatever order the tuple happens to have.
        assert DEFAULT_ACQUISITION == "ei"

    def test_zero_posterior_sd_does_not_divide_by_zero(self):
        inputs = _inputs(sigma=np.zeros(101))
        for name in ACQUISITIONS:
            assert np.all(np.isfinite(acquire(name, inputs))), name


class TestTheIncumbentIsWhatSeparatesTheArms:
    """The whole module exists because `max(y)` is biased upward."""

    def test_ei_and_ei_plugin_differ_only_in_the_incumbent(self):
        inputs = _inputs()
        assert np.allclose(
            acquire("ei", inputs),
            expected_improvement(inputs.mu, inputs.sigma, inputs.f_best_observed, inputs.xi),
        )
        assert np.allclose(
            acquire("ei_plugin", inputs),
            expected_improvement(inputs.mu, inputs.sigma, inputs.f_best_plugin, inputs.xi),
        )

    def test_a_lower_incumbent_can_only_raise_ei(self):
        # EI is decreasing in f_best, so the plug-in incumbent — which is never
        # above max(y) — cannot make the surface smaller anywhere.
        inputs = _inputs()
        assert inputs.f_best_plugin <= inputs.f_best_observed
        assert np.all(acquire("ei_plugin", inputs) >= acquire("ei", inputs) - 1e-15)

    def test_an_inflated_incumbent_turns_ei_into_a_variance_seeker(self):
        """The measured failure, and it is worse than a loss of scale.

        `mu` peaks at index 70 and `sigma` peaks at index 30. With an honest
        incumbent EI recommends the mean bump, which is what an exploitation
        rule should do. Raise the incumbent above the entire posterior mean —
        which one lucky noisy reading does — and `improvement` is negative
        everywhere, so `EI -> sigma * pdf(improvement / sigma)`, whose argmax is
        driven by `sigma`. EI silently stops being an exploitation rule and
        becomes a variance seeker, and the switch is invisible because by then
        its magnitude is 1e-60 and means nothing to anyone reading it.
        """
        honest = acquire("ei", _inputs(f_best_observed=1.0))
        assert int(np.argmax(honest)) == 70  # the posterior-mean bump

        inflated = acquire("ei", _inputs(f_best_observed=5.0))
        assert int(np.argmax(inflated)) == 30  # the posterior-sd bump
        assert inflated.max() < 1e-12  # far below any plausible noise floor

    def test_ucb_has_no_incumbent_so_the_bias_cannot_reach_it(self):
        a = acquire("ucb", _inputs(f_best_observed=1.4, f_best_plugin=1.2))
        b = acquire("ucb", _inputs(f_best_observed=99.0, f_best_plugin=98.0))
        assert np.allclose(a, b)


class TestAugmentedEiDiscountsIrreducibleNoise:
    def test_it_never_exceeds_the_ei_it_is_built_from(self):
        inputs = _inputs()
        assert np.all(acquire("aei", inputs) <= acquire("ei_plugin", inputs) + 1e-15)

    def test_noiseless_measurement_removes_the_discount(self):
        inputs = _inputs(noise_std=0.0)
        assert np.allclose(acquire("aei", inputs), acquire("ei_plugin", inputs))

    def test_the_discount_bites_hardest_where_the_model_is_certain(self):
        inputs = _inputs()
        ratio = acquire("aei", inputs) / np.maximum(acquire("ei_plugin", inputs), 1e-300)
        # sigma is largest near x = 0.3 by construction, so the surviving
        # fraction of EI must be larger there than at the certain right edge.
        assert ratio[30] > ratio[-1]


class TestKnowledgeGradient:
    def _cov(self, n: int = 101, scale: float = 0.2) -> np.ndarray:
        g = np.linspace(0.0, 1.0, n)
        return 0.09 * np.exp(-((g[:, None] - g[None, :]) ** 2) / (2 * scale**2))

    def test_without_a_covariance_it_reports_nothing_rather_than_zero_gain(self):
        # Zeros are the documented "unavailable" signal; the caller must not
        # read them as "no observation could help".
        assert np.all(acquire("kg", _inputs(cov=None)) == 0.0)

    def test_the_gain_is_never_negative_where_it_was_evaluated(self):
        inputs = _inputs(cov=self._cov())
        scores = acquire("kg", inputs)
        assert scores.max() >= 0.0

    def test_unevaluated_points_can_never_win(self):
        """KG strides its candidates for cost. The gaps must be unable to take
        the argmax, or a coarse grid would silently recommend a point KG never
        scored."""
        n = 1001
        grid = np.linspace(0.0, 1.0, n)
        inputs = _inputs(
            mu=1.0 + 0.5 * np.exp(-(((grid - 0.7) / 0.15) ** 2)),
            sigma=0.05 + 0.25 * np.exp(-(((grid - 0.3) / 0.2) ** 2)),
            cov=self._cov(n=n),
        )
        scores = acquire("kg", inputs)
        # Strided candidates are the only entries above the sentinel floor, so
        # the argmax can only land on a point that was actually scored.
        assert scores[int(np.argmax(scores))] > scores.min()
        assert np.sum(scores > scores.min()) <= 129

    def test_a_covariance_on_the_wrong_grid_is_refused(self):
        with pytest.raises(ValueError, match="cov must be"):
            acquire("kg", _inputs(cov=self._cov(n=1001)))

    def test_a_tie_across_every_candidate_is_reported_as_no_preference(self):
        """Measured 2026-09-10, and the reason this guard exists.

        When the surrogate over-smooths — a fitted length-scale pinned at its
        ceiling, which is exactly what a prior mean provokes — the rank-one
        update direction is constant across the grid, so
        `max_i(mu_i + s * Z) = max(mu) + s * Z` and the expectation is `max(mu)`
        for every candidate. KG is then identically zero, and the sentinel
        scheme would leave `argmax` returning the first scored index: the tool
        would synthesise the low end of the search range and report that a
        decision procedure had chosen it.
        """
        g = np.linspace(0.0, 1.0, 101)
        flat = np.full((101, 101), 0.09)  # perfectly correlated: s constant in i
        scores = acquire("kg", _inputs(mu=np.full(101, 1.0), sigma=np.full(101, 0.3), cov=flat))
        assert np.all(scores == 0.0)
        # And nothing in the surface may prefer one end over the other.
        assert float(np.ptp(scores)) == 0.0
        assert len(g) == scores.size

    def test_more_measurement_noise_cannot_increase_the_gain(self):
        cov = self._cov()
        quiet = acquire("kg", _inputs(cov=cov, noise_std=0.01)).max()
        loud = acquire("kg", _inputs(cov=cov, noise_std=1.0)).max()
        assert loud <= quiet + 1e-12


class TestMaxValueEntropySearch:
    def test_information_is_non_negative(self):
        assert np.all(acquire("mes", _inputs()) >= -1e-9)

    def test_it_prefers_the_uncertain_region_over_a_pinned_one(self):
        grid = np.linspace(0.0, 1.0, 101)
        mu = np.full(101, 1.0)
        sigma = np.where(grid < 0.5, 0.3, 1e-6)  # left half unknown, right half known
        scores = acquire("mes", _inputs(mu=mu, sigma=sigma))
        assert scores[:50].mean() > scores[50:].mean()

    def test_it_is_deterministic(self):
        # The Gumbel draws use a fixed seed: this is a property of the
        # posterior, and a recommendation that moved between identical calls
        # could not be pre-registered.
        inputs = _inputs()
        assert np.allclose(acquire("mes", inputs), acquire("mes", inputs))


class TestANonFinitePosteriorIsNeverTheAnswer:
    """Re-found 2026-09-17; first found and fixed 2026-07-24 on a branch that
    was never merged.

    `argmax` ranks NaN above every number, so one NaN on the grid used to become
    the recommendation. Rare — it needs an ill-conditioned fit — and silent when
    it happens, which is the combination worth a test.
    """

    NAN_AT = 20  # a point with a low mean, so a correct arm has no reason to pick it

    def _poisoned(self, field: str) -> AcquisitionInputs:
        clean = _inputs()
        values = np.array(getattr(clean, field), dtype=float)
        values[self.NAN_AT] = np.nan
        return _inputs(**{field: values})

    @pytest.mark.parametrize("arm", ACQUISITIONS)
    @pytest.mark.parametrize("field", ["sigma", "mu"])
    def test_no_arm_returns_nan_or_picks_the_poisoned_point(self, arm, field):
        scores = acquire(arm, self._poisoned(field))
        assert np.all(np.isfinite(scores))
        assert int(np.argmax(scores)) != self.NAN_AT

    def test_expected_improvement_scores_the_point_as_no_gain(self):
        inputs = self._poisoned("mu")
        ei = expected_improvement(inputs.mu, inputs.sigma, inputs.f_best_observed, inputs.xi)
        assert ei[self.NAN_AT] == 0.0
        assert np.all(ei >= 0.0)

    def test_a_nan_sigma_is_read_as_known_at_its_mean(self):
        """Flooring, not discarding: the point keeps whatever its mean earns."""
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.1, np.nan])
        ei = expected_improvement(mu, sigma, f_best=1.5, xi=0.0)
        assert ei[1] == pytest.approx(0.5)

    def test_a_surface_with_nothing_finite_is_flat(self):
        """No information is no preference, which callers already handle."""
        nan = np.full(101, np.nan)
        assert np.ptp(acquire("ucb", _inputs(mu=nan))) == 0.0
