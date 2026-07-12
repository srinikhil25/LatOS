"""Tests for the fit-engine background library."""

from __future__ import annotations

import numpy as np
import pytest

from latos.fitting import backgrounds


def _gaussian(x, center, sigma, height):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


class TestSimpleBaselines:
    def test_constant_is_the_minimum(self):
        y = np.array([3.0, 1.0, 5.0, 2.0])
        assert np.allclose(backgrounds.constant_baseline(y), 1.0)

    def test_linear_recovers_a_line_exactly(self):
        x = np.linspace(0.0, 10.0, 50)
        y = 2.0 + 0.5 * x
        assert np.allclose(backgrounds.linear_baseline(x, y), y)

    def test_linear_uses_endpoints_only(self):
        x = np.linspace(0.0, 10.0, 11)
        y = np.array([0.0] + [99.0] * 9 + [10.0])  # bump in the middle
        base = backgrounds.linear_baseline(x, y)
        assert base[0] == pytest.approx(0.0)
        assert base[-1] == pytest.approx(10.0)
        assert base[5] == pytest.approx(5.0, abs=1e-9)  # ignores the bump

    def test_polynomial_recovers_a_quadratic(self):
        x = np.linspace(-5.0, 5.0, 60)
        y = 1.0 - 0.3 * x + 0.2 * x**2
        assert np.allclose(backgrounds.polynomial_baseline(x, y, degree=2), y, atol=1e-6)


class TestShirley:
    def test_endpoints_and_monotonic_under_a_peak(self):
        x = np.linspace(0.0, 100.0, 400)
        # Step-up background 100 -> 200 plus a peak in the middle.
        bg = np.linspace(100.0, 200.0, x.size)
        y = bg + _gaussian(x, 50.0, 4.0, 300.0)
        base = backgrounds.shirley_baseline(y)
        assert base[0] == pytest.approx(y[0], rel=1e-3)
        assert base[-1] == pytest.approx(y[-1], rel=1e-3)
        # The Shirley background is monotonic between anchors.
        assert np.all(np.diff(base) >= -1e-6)
        # Peak-subtracted trace is ~0 at the edges and positive at the peak.
        corrected = y - base
        assert abs(corrected[0]) < 1.0
        assert abs(corrected[-1]) < 1.0
        assert corrected[np.argmin(np.abs(x - 50.0))] > 100.0

    def test_short_input_is_safe(self):
        assert backgrounds.shirley_baseline(np.array([5.0])).shape == (1,)


class TestALS:
    def test_recovers_smooth_baseline_beneath_peaks(self):
        x = np.linspace(0.0, 100.0, 500)
        smooth = 20.0 + 0.1 * x  # gentle sloping background
        peaks = _gaussian(x, 30.0, 2.0, 200.0) + _gaussian(x, 70.0, 3.0, 150.0)
        y = smooth + peaks
        base = backgrounds.als_baseline(y, lam=1e5, p=0.001)
        # Baseline hugs the smooth trend, not the peaks.
        off_peak = (np.abs(x - 30.0) > 12) & (np.abs(x - 70.0) > 12)
        assert np.max(np.abs(base[off_peak] - smooth[off_peak])) < 3.0
        # At the peak centres the baseline sits far below the data.
        assert base[np.argmin(np.abs(x - 30.0))] < y[np.argmin(np.abs(x - 30.0))] - 100.0
