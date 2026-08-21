"""Per-observation measurement noise reaching the surrogate as a variance.

Reliability used to arrive as one bit per datapoint: a physics check either
flagged an observation or it did not, and a flagged one had its error bar
widened by a fixed factor. The analysis layer already computes better than that
— a fitted slope carries a standard error, a derived quantity carries propagated
uncertainty — and this is where that number starts being used.

What these tests pin down is that supplying it actually changes the fit in the
direction it should, that it composes with the existing flags rather than
replacing them, and that the frozen config records which of the two happened.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization.engine import OptimizationError, optimize, optimize_nd

BOUNDS = (0.0, 10.0)


def _run(x, y, **kwargs):
    return optimize(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        bounds=BOUNDS,
        input_name="doping_pct",
        target_name="zt",
        with_reliability=False,
        seed=0,
        **kwargs,
    )


class TestItChangesTheFit:
    """A variance the model was not told about cannot change anything."""

    def test_a_precise_outlier_pulls_harder_than_a_vague_one(self):
        """Same numbers, opposite confidence, different surface.

        The point at x = 5 sits well above its neighbours. Told it was measured
        precisely, the model should follow it; told it was barely measured at
        all, the model should largely ignore it. If both fits agree the per-
        point variance never reached the kernel.
        """
        x = [0.0, 2.5, 5.0, 7.5, 10.0]
        y = [1.0, 1.1, 2.0, 1.2, 1.3]

        trusted = _run(x, y, point_noise=[0.3, 0.3, 0.01, 0.3, 0.3])
        doubted = _run(x, y, point_noise=[0.05, 0.05, 2.0, 0.05, 0.05])

        peak_trusted = max(trusted.grid_mean)
        peak_doubted = max(doubted.grid_mean)
        assert peak_trusted > peak_doubted

    def test_uniform_point_noise_matches_the_scalar_form(self):
        """Equal uncertainties carry no per-point information.

        Supplying the same sigma everywhere should reproduce the scalar path,
        because the multipliers are then all 1. A difference here would mean the
        new route is doing something beyond weighting.
        """
        x = [0.0, 2.5, 5.0, 7.5, 10.0]
        y = [1.0, 1.4, 1.9, 1.6, 1.2]

        scalar = _run(x, y, measured_noise=0.1)
        per_point = _run(x, y, point_noise=[0.1] * 5)

        assert per_point.recommendation.x == pytest.approx(scalar.recommendation.x, abs=1e-6)
        assert np.allclose(per_point.grid_mean, scalar.grid_mean, atol=1e-9)

    def test_the_median_sigma_sets_the_shared_noise_level(self):
        """The scalar machinery still needs one number, and gets a robust one.

        Mean would let a single very uncertain point raise the convergence floor
        for the whole campaign.
        """
        x = [0.0, 2.5, 5.0, 7.5, 10.0]
        y = [1.0, 1.4, 1.9, 1.6, 1.2]
        result = _run(x, y, point_noise=[0.1, 0.1, 0.1, 0.1, 50.0])
        assert result.config.noise_std == pytest.approx(0.1, rel=1e-6)

    def test_an_explicit_measured_noise_is_not_overridden(self):
        x = [0.0, 2.5, 5.0, 7.5, 10.0]
        y = [1.0, 1.4, 1.9, 1.6, 1.2]
        result = _run(x, y, point_noise=[0.1] * 5, measured_noise=0.4)
        assert result.config.noise_std == pytest.approx(0.4, rel=1e-6)


class TestCompositionWithFlags:
    """Two different claims about one observation, both honoured."""

    def test_a_flagged_point_is_distrusted_further_than_its_error_bar_alone(self):
        """Precision and plausibility are separate questions.

        A tightly-measured value that physics says is impossible should be
        trusted *less* than the same value unflagged, not more. If the flag were
        replaced by the per-point sigma rather than multiplied into it, the two
        fits would coincide.
        """
        x = [0.0, 2.5, 5.0, 7.5, 10.0]
        y = [1.0, 1.1, 2.0, 1.2, 1.3]
        sigma = [0.3, 0.3, 0.02, 0.3, 0.3]
        flags = [False, False, True, False, False]

        plain = _run(x, y, point_noise=sigma)
        flagged = _run(x, y, point_noise=sigma, unreliable=flags)

        assert max(flagged.grid_mean) < max(plain.grid_mean)
        assert flagged.n_unreliable == 1


class TestTheFrozenRecord:
    """Two runs that fit differently must not carry identical configs."""

    def test_the_config_records_that_per_point_noise_was_used(self):
        x = [0.0, 2.5, 5.0, 7.5, 10.0]
        y = [1.0, 1.4, 1.9, 1.6, 1.2]
        assert _run(x, y, point_noise=[0.1] * 5).config.point_noise_used is True
        assert _run(x, y, measured_noise=0.1).config.point_noise_used is False

    def test_per_point_noise_counts_as_measured(self):
        x = [0.0, 2.5, 5.0, 7.5, 10.0]
        y = [1.0, 1.4, 1.9, 1.6, 1.2]
        assert _run(x, y, point_noise=[0.1] * 5).noise_measured is True


class TestRejections:
    """Bad uncertainties are refused, never repaired.

    Substituting something plausible for a malformed standard deviation would
    hand the surrogate a confidence nobody computed, which is the failure this
    whole feature exists to remove.
    """

    def test_wrong_length_is_rejected(self):
        with pytest.raises(OptimizationError, match="one entry per observation"):
            _run([0.0, 5.0, 10.0], [1.0, 2.0, 1.5], point_noise=[0.1, 0.1])

    def test_negative_sigma_is_rejected(self):
        with pytest.raises(OptimizationError, match="non-negative"):
            _run([0.0, 5.0, 10.0], [1.0, 2.0, 1.5], point_noise=[0.1, -0.2, 0.1])

    def test_non_finite_sigma_is_rejected(self):
        with pytest.raises(OptimizationError, match="finite"):
            _run([0.0, 5.0, 10.0], [1.0, 2.0, 1.5], point_noise=[0.1, np.nan, 0.1])

    def test_all_zero_sigma_is_rejected(self):
        """Claiming every measurement is exact is a caller bug, not a stance."""
        with pytest.raises(OptimizationError, match="perfect data"):
            _run([0.0, 5.0, 10.0], [1.0, 2.0, 1.5], point_noise=[0.0, 0.0, 0.0])

    def test_a_single_zero_sigma_is_floored_rather_than_rejected(self):
        """One exact-looking point is survivable; a GP told it is noiseless is not.

        Interpolating a point exactly lets one optimistic error bar dominate
        every neighbouring prediction, so the multiplier is floored relative to
        the campaign median instead.
        """
        result = _run(
            [0.0, 2.5, 5.0, 7.5, 10.0],
            [1.0, 1.1, 2.0, 1.2, 1.3],
            point_noise=[0.3, 0.3, 0.0, 0.3, 0.3],
        )
        assert np.all(np.isfinite(result.grid_mean))
        assert np.all(np.asarray(result.grid_upper) >= np.asarray(result.grid_mean))


class TestLogSpace:
    """An absolute sigma is a different fraction at each magnitude."""

    def test_per_point_noise_survives_a_log_fit(self):
        """The conversion has to be per point, which is the whole argument.

        In log space the GP sees fractional error, so the same absolute sigma
        means something quite different on a value of 1 than on a value of 1000.
        Dividing by the series mean, as the scalar path does, would erase that.
        """
        x = [0.0, 2.5, 5.0, 7.5, 10.0]
        y = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        result = _run(x, y, y_transform="log", point_noise=[0.1, 1.0, 10.0, 100.0, 1000.0])
        assert result.config.y_transform == "log"
        assert np.all(np.isfinite(result.grid_mean))


class TestMultiDimensional:
    """The d-D path must behave identically; it has drifted before."""

    def test_optimize_nd_accepts_and_records_point_noise(self):
        x = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0], [2.5, 2.5]])
        y = np.array([1.0, 1.2, 1.1, 1.3, 2.0])
        result = optimize_nd(
            x,
            y,
            bounds=((0.0, 5.0), (0.0, 5.0)),
            input_names=("a", "b"),
            target_name="zt",
            point_noise=[0.2, 0.2, 0.2, 0.2, 0.02],
            with_reliability=False,
            seed=0,
        )
        assert result.config.point_noise_used is True
        assert result.noise_measured is True

    def test_optimize_nd_rejects_a_wrong_length(self):
        x = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
        y = np.array([1.0, 1.2, 1.1, 1.3])
        with pytest.raises(OptimizationError, match="one entry per observation"):
            optimize_nd(
                x,
                y,
                bounds=((0.0, 5.0), (0.0, 5.0)),
                input_names=("a", "b"),
                target_name="zt",
                point_noise=[0.2, 0.2],
                with_reliability=False,
            )


class TestBackwardCompatibility:
    """Every existing caller must be untouched."""

    def test_omitting_point_noise_reproduces_the_previous_behaviour(self):
        x = [0.0, 2.5, 5.0, 7.5, 10.0]
        y = [1.0, 1.4, 1.9, 1.6, 1.2]
        result = _run(x, y)
        assert result.config.point_noise_used is False
        assert result.noise_measured is False
        assert np.all(np.isfinite(result.grid_mean))
