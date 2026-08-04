"""The noise floor should come from repeat measurements when they exist.

The convergence verdict is "the expected gain is below the noise", so the
noise is not a detail: it is the thing the whole claim rests on. It used to be
`rel_noise * mean(|y|)`, an assumed 8%, even for techniques that record the
scatter of their own repeats.
"""

from __future__ import annotations

import numpy as np

from latos.optimization.engine import _REL_NOISE, _noise_std, optimize

X = np.array([40.0, 45.0, 50.0, 55.0])
Y = np.array([71.667, 57.0, 36.667, 57.0])


def _run(**kwargs):
    return optimize(
        X,
        Y,
        bounds=(40.0, 55.0),
        input_name="wt_pct",
        target_name="peak_force_n",
        direction="minimize",
        **kwargs,
    )


class TestNoiseStd:
    def test_assumed_scales_with_the_data(self):
        y = np.array([10.0, 20.0, 30.0])
        assert _noise_std(y, 0.08, "identity") == 0.08 * 20.0

    def test_measured_wins_over_assumed(self):
        y = np.array([10.0, 20.0, 30.0])
        assert _noise_std(y, 0.08, "identity", measured_noise=5.0) == 5.0

    def test_measured_ignored_when_not_positive(self):
        # A zero or negative scatter is not a measurement, it is a bug or an
        # empty field. Falling back beats trusting it.
        y = np.array([10.0, 20.0, 30.0])
        assert _noise_std(y, 0.08, "identity", measured_noise=0.0) == 0.08 * 20.0
        assert _noise_std(y, 0.08, "identity", measured_noise=-2.0) == 0.08 * 20.0

    def test_log_space_converts_absolute_to_relative(self):
        # In log space the GP sees d(ln y) = dy/y, so an absolute scatter has
        # to be divided by the signal scale before it means anything.
        y_lin = np.array([10.0, 20.0, 30.0])
        got = _noise_std(np.log(y_lin), 0.08, "log", measured_noise=4.0, y_linear=y_lin)
        assert got == 4.0 / 20.0

    def test_log_space_without_a_scale_falls_back(self):
        assert _noise_std(np.array([1.0]), 0.08, "log", measured_noise=4.0) == 0.08


class TestOptimizeUsesIt:
    def test_default_is_the_assumption(self):
        res = _run()
        assert res.noise_measured is False
        assert res.noise_threshold == _REL_NOISE * float(np.mean(np.abs(Y)))

    def test_measured_sets_the_threshold_and_is_reported(self):
        res = _run(measured_noise=5.951)
        assert res.noise_measured is True
        assert res.noise_threshold == 5.951

    def test_a_larger_noise_floor_makes_stopping_easier(self):
        # The stop test is max_ei < noise_threshold. Raising the floor cannot
        # make the tool less willing to stop.
        low = _run(measured_noise=0.5)
        high = _run(measured_noise=20.0)
        assert high.noise_threshold > low.noise_threshold
        assert not (low.max_ei < low.noise_threshold and high.max_ei >= high.noise_threshold)

    def test_predictive_interval_widens_with_the_noise(self):
        # The predictive interval folds in the measurement noise, so a bigger
        # measured scatter has to produce a wider interval a new measurement
        # is expected to land in.
        narrow = _run(measured_noise=1.0).recommendation
        wide = _run(measured_noise=12.0).recommendation
        assert wide.ci95_predictive > narrow.ci95_predictive
