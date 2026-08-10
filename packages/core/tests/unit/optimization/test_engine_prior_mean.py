"""Tests for the physics prior mean and the relative exploration sweetener.

Both were added on 2026-08-10 and they are entangled, which is why they share a
file. A zero-mean GP has nothing to extrapolate along, so a sparse campaign
recommends wherever it happens to know least; `prior_mean` replaces that flat
default with a physical curve. Wiring it in immediately exposed that `xi` was an
absolute constant in the target's units — on a zT-scale objective (~0.07) a
sweetener of 0.01 exceeds any real improvement, so Expected Improvement was
identically zero and the prior could never influence a recommendation. Fixing
one without the other proves nothing, so both are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization import optimize, optimize_nd
from latos.optimization.engine import _xi_absolute

# A zT-scale doping series with an interior peak at 3 %. Small magnitudes are
# the whole point: this is the regime where an absolute xi silently wins.
DOP = np.array([0.0, 1.0, 3.0, 5.0])
ZT = np.array([0.0289, 0.0377, 0.0701, 0.0409])
BOUNDS = (0.0, 8.0)


def _peak_at(centre: float, amplitude: float = 0.070, width: float = 1.6):
    """A prior shaped like the single-parabolic-band result: an interior peak."""

    def prior(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return amplitude * np.exp(-((x - centre) ** 2) / (2 * width**2))

    return prior


def _ei_peak(result) -> float:
    grid_x = np.asarray(result.grid_x)
    return float(grid_x[int(np.argmax(np.asarray(result.grid_ei)))])


class TestXiIsRelative:
    """`xi` must mean the same thing whatever units the target is recorded in."""

    def test_scales_with_the_targets_magnitude(self):
        small = _xi_absolute(0.01, ZT, noise_std=1e-6)
        large = _xi_absolute(0.01, ZT * 1000.0, noise_std=1e-6)
        assert large == pytest.approx(small * 1000.0, rel=1e-9)

    def test_equals_a_fraction_of_the_observed_spread(self):
        assert _xi_absolute(0.01, ZT, noise_std=1e-9) == pytest.approx(0.01 * float(np.std(ZT)))

    def test_noise_floors_a_degenerate_spread(self):
        """All-identical measurements are scatter, not a scale.

        Without the floor a degenerate first round drives xi to zero and makes
        EI perfectly greedy at exactly the moment the surrogate deserves least
        trust.
        """
        flat = np.full(4, 0.05)
        assert _xi_absolute(0.01, flat, noise_std=0.004) == pytest.approx(0.01 * 0.004)

    def test_ei_survives_on_a_zt_scale_target(self):
        """The regression that started this: EI was exactly zero here.

        With xi fixed at an absolute 0.01, `mu - f_best - xi` was negative
        everywhere on a target whose entire range is about 0.04, so every
        recommendation fell through to the exploration fallback.
        """
        r = optimize(DOP, ZT, bounds=BOUNDS, input_name="doping_pct", target_name="zT")
        assert r.max_ei > 0.0
        assert r.config.xi_absolute is not None
        assert r.config.xi_absolute < 0.01


class TestPriorMeanIsInert:
    """Absent or trivial priors must not perturb the existing engine at all."""

    def test_none_matches_omitting_the_argument(self):
        a = optimize(DOP, ZT, bounds=BOUNDS, input_name="d", target_name="zT")
        b = optimize(DOP, ZT, bounds=BOUNDS, input_name="d", target_name="zT", prior_mean=None)
        assert a.recommendation.x == b.recommendation.x
        assert a.max_ei == b.max_ei


class TestPriorMeanShapesTheSearch:
    def test_moves_expected_improvement_onto_the_physical_optimum(self):
        """A correct prior should point at the peak, not at the far edge.

        The samples stop at 2.5 % while the true peak sits at 3 %. Without
        structure the GP reverts to a flat mean beyond the data and the
        acquisition drifts outward; the prior is what pulls it back.
        """
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 2.5, 12)
        y = 0.070 * np.exp(-((x - 3.0) ** 2) / (2 * 1.6**2)) + rng.normal(0, 0.001, 12)

        with_prior = optimize(
            x,
            y,
            bounds=BOUNDS,
            input_name="doping_pct",
            target_name="zT",
            prior_mean=_peak_at(3.0),
        )
        assert _ei_peak(with_prior) == pytest.approx(3.0, abs=0.25)
        assert with_prior.recommendation.x == pytest.approx(3.0, abs=0.25)

    def test_a_wrong_prior_misleads_visibly_rather_than_silently(self):
        """A bad prior must fail loudly, by pointing somewhere checkable.

        This is the honest failure mode and it is worth pinning: the danger of
        a prior is that it gets quietly absorbed and the recommendation looks
        the same as it always did. It should not. Here a prior that puts the
        peak at 6.5 drags the recommendation there, where one experiment
        refutes it.
        """
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 2.5, 12)
        y = 0.070 * np.exp(-((x - 3.0) ** 2) / (2 * 1.6**2)) + rng.normal(0, 0.001, 12)

        misled = optimize(
            x,
            y,
            bounds=BOUNDS,
            input_name="doping_pct",
            target_name="zT",
            prior_mean=_peak_at(6.5, amplitude=0.060),
        )
        assert misled.recommendation.x > 5.0

    def test_lowers_the_posterior_where_the_physics_says_it_is_bad(self):
        plain = optimize(DOP, ZT, bounds=BOUNDS, input_name="d", target_name="zT")
        shaped = optimize(
            DOP,
            ZT,
            bounds=BOUNDS,
            input_name="d",
            target_name="zT",
            prior_mean=_peak_at(3.0),
        )
        # At the far edge the prior predicts a collapse; the flat GP does not.
        assert float(np.asarray(shaped.grid_mean)[-1]) < float(np.asarray(plain.grid_mean)[-1])

    def test_multi_dimensional_path_accepts_a_prior(self):
        x = np.array([[0.0, 10.0], [1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        y = np.array([0.03, 0.04, 0.07, 0.04])

        def prior(points: np.ndarray) -> np.ndarray:
            points = np.asarray(points, dtype=float)
            return 0.07 * np.exp(-((points[:, 0] - 3.0) ** 2) / (2 * 1.6**2))

        r = optimize_nd(
            x,
            y,
            bounds=[(0.0, 8.0), (5.0, 35.0)],
            input_names=["doping_pct", "time_min"],
            target_name="zT",
            prior_mean=prior,
        )
        assert len(r.recommendation.x) == 2
        assert r.config.xi_absolute is not None


class TestPriorMeanRejectsBadInput:
    """Silent coercion of a broken prior would poison the fit invisibly."""

    def test_wrong_length_is_an_error(self):
        with pytest.raises(ValueError, match="returned"):
            optimize(
                DOP,
                ZT,
                bounds=BOUNDS,
                input_name="d",
                target_name="zT",
                prior_mean=lambda x: np.zeros(3),
            )

    def test_non_finite_is_an_error(self):
        with pytest.raises(ValueError, match="non-finite"):
            optimize(
                DOP,
                ZT,
                bounds=BOUNDS,
                input_name="d",
                target_name="zT",
                prior_mean=lambda x: np.full(np.asarray(x).shape[0], np.nan),
            )

    def test_non_positive_prior_under_a_log_fit_is_an_error(self):
        """log(<=0) is undefined, and falling back to a linear fit would mean
        the residual was taken against a different prior than the one reported.
        """
        with pytest.raises(ValueError, match="log space"):
            optimize(
                DOP,
                ZT,
                bounds=BOUNDS,
                input_name="d",
                target_name="zT",
                y_transform="log",
                prior_mean=lambda x: np.zeros(np.asarray(x).shape[0]),
            )
