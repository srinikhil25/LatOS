"""Tests for `latos.optimization.engine`."""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization import OptimizationError, length_scale_robustness, optimize


def _real_te_data() -> tuple[np.ndarray, np.ndarray]:
    """Dhivya's measured peak zT vs Cs3Bi2I9 doping (the real demo case)."""
    x = np.array([0.0, 1.0, 3.0, 5.0])
    y = np.array([0.587, 0.362, 0.967, 0.482])
    return x, y


class TestGuards:
    def test_too_few_points_raises(self):
        with pytest.raises(OptimizationError):
            optimize(
                np.array([0.0, 1.0]),
                np.array([0.5, 0.6]),
                bounds=(0, 5),
                input_name="doping_pct",
                target_name="peak_zt",
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(OptimizationError):
            optimize(
                np.array([0.0, 1.0, 3.0]),
                np.array([0.5, 0.6]),
                bounds=(0, 5),
                input_name="x",
                target_name="y",
            )

    def test_degenerate_bounds_raises(self):
        x, y = _real_te_data()
        with pytest.raises(OptimizationError):
            optimize(x, y, bounds=(3, 3), input_name="x", target_name="y")


class TestRealTeCase:
    @pytest.fixture(scope="class")
    def result(self):
        x, y = _real_te_data()
        return optimize(
            x,
            y,
            bounds=(0.0, 5.0),
            input_name="doping_pct",
            target_name="peak_zt",
        )

    def test_best_observed_is_3pct(self, result):
        assert result.best_x == 3.0
        assert result.best_y == pytest.approx(0.967, abs=1e-3)

    def test_recommends_near_the_peak(self, result):
        # The recommendation should refine near the 3% optimum, not wander
        # off to an endpoint.
        assert 2.5 <= result.recommendation.x <= 4.0

    def test_recommendation_has_uncertainty(self, result):
        assert result.recommendation.ci95 > 0
        assert result.recommendation.predicted_mean > 0

    def test_posterior_grid_shapes_match(self, result):
        n = len(result.grid_x)
        assert n == len(result.grid_mean) == len(result.grid_ci95) == len(result.grid_ei)
        assert n > 50  # a smooth curve

    def test_observed_echoed_back(self, result):
        assert result.observed_x == (0.0, 1.0, 3.0, 5.0)
        assert len(result.observed_y) == 4

    def test_max_ei_is_positive(self, result):
        # With a sharp peak and only 4 points there's still improvement to chase.
        assert result.max_ei > 0

    def test_reports_noise_threshold(self, result):
        assert result.noise_threshold > 0


class TestConvergence:
    def test_converges_when_improvement_below_noise(self):
        # Real TE case at default 8% noise: best expected improvement
        # (~0.02) is below the measurement-noise floor (~0.05) -> the tool
        # reports it has reached the optimum within measurement precision.
        x, y = _real_te_data()
        result = optimize(x, y, bounds=(0.0, 5.0), input_name="doping_pct", target_name="peak_zt")
        assert result.max_ei < result.noise_threshold
        assert result.converged is True

    def test_not_converged_when_improvement_beats_noise(self):
        # Low measurement noise + a clear rising trend with headroom: a new
        # experiment can reliably improve, so keep going.
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.2, 0.5, 0.9])
        result = optimize(x, y, bounds=(0, 4), input_name="x", target_name="y", rel_noise=0.02)
        assert result.converged is False


class TestPredictiveCiAndConfig:
    @pytest.fixture(scope="class")
    def result(self):
        x, y = _real_te_data()
        return optimize(
            x, y, bounds=(0.0, 5.0), input_name="doping_pct", target_name="peak_zt", seed=0
        )

    def test_predictive_ci_includes_measurement_noise(self, result):
        # The predictive interval adds measurement noise on top of the model
        # uncertainty, so it is never narrower than the model CI.
        rec = result.recommendation
        assert rec.predictive_sd > 0
        assert rec.ci95_predictive >= rec.ci95

    def test_config_freezes_the_setup(self, result):
        cfg = result.config
        assert cfg.objective == "peak_zt"
        assert cfg.input_name == "doping_pct"
        assert cfg.bounds == (0.0, 5.0)
        assert cfg.length_scale_fitted is True
        assert cfg.seed == 0
        assert cfg.n_observations == 4
        assert cfg.noise_std > 0

    def test_fixed_length_scale_is_recorded_as_such(self):
        x, y = _real_te_data()
        r = optimize(
            x, y, bounds=(0.0, 5.0), input_name="d", target_name="z", length_scale=2.0
        )
        assert r.config.length_scale == 2.0
        assert r.config.length_scale_fitted is False

    def test_same_seed_is_reproducible(self):
        x, y = _real_te_data()
        a = optimize(x, y, bounds=(0.0, 5.0), input_name="d", target_name="z", seed=7)
        b = optimize(x, y, bounds=(0.0, 5.0), input_name="d", target_name="z", seed=7)
        assert a.recommendation.x == b.recommendation.x
        assert a.recommendation.predicted_mean == b.recommendation.predicted_mean


class TestRobustness:
    def test_report_is_wellformed(self):
        x, y = _real_te_data()
        report = length_scale_robustness(
            x,
            y,
            bounds=(0.0, 5.0),
            input_name="doping_pct",
            target_name="peak_zt",
            length_scales=(1.0, 2.0, 3.0, 4.0, 5.0),
        )
        assert len(report.entries) == 5
        assert report.search_span == 5.0
        assert report.tolerance == pytest.approx(0.5)
        assert report.recommended_x_spread >= 0.0
        assert isinstance(report.stable, bool)
        for entry in report.entries:
            assert 0.0 <= entry.recommended_x <= 5.0
            assert entry.ci95_predictive >= 0.0

    def test_stable_flag_tracks_spread(self):
        # A smooth, densely-sampled single peak: the recommendation should not
        # depend much on the length-scale, so spread <= tolerance -> stable.
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([0.1, 0.3, 0.7, 1.0, 0.7, 0.3, 0.1])
        report = length_scale_robustness(
            x,
            y,
            bounds=(0.0, 6.0),
            input_name="d",
            target_name="z",
            length_scales=(2.0, 2.5, 3.0),
        )
        assert report.stable == (report.recommended_x_spread <= report.tolerance)
