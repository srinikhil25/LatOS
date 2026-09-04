"""Tests for automatic peak detection."""

from __future__ import annotations

import numpy as np
import pytest

from latos.fitting import detect_peaks
from latos.fitting.peak_finder import detect_peaks_detailed, measure_widths


def _gaussian(x, center, sigma, height):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


class TestDetectPeaks:
    def test_finds_three_peaks_ranked_by_prominence(self):
        x = np.linspace(0.0, 100.0, 1000)
        y = (
            2.0
            + _gaussian(x, 20.0, 1.5, 30.0)
            + _gaussian(x, 50.0, 2.0, 80.0)  # tallest
            + _gaussian(x, 80.0, 1.5, 50.0)
        )
        rng = np.random.default_rng(0)
        y = y + rng.normal(0.0, 0.2, size=x.size)
        centers = detect_peaks(x, y)
        assert len(centers) == 3
        # Strongest first: the 50-unit peak leads.
        assert centers[0] == pytest.approx(50.0, abs=0.5)
        assert sorted(centers) == pytest.approx([20.0, 50.0, 80.0], abs=0.5)

    def test_respects_max_peaks(self):
        x = np.linspace(0.0, 200.0, 2000)
        y = np.full_like(x, 1.0)
        for c in range(10, 200, 15):
            y = y + _gaussian(x, float(c), 1.0, 20.0)
        assert len(detect_peaks(x, y, max_peaks=3)) == 3

    def test_ignores_noise_only_trace(self):
        x = np.linspace(0.0, 100.0, 500)
        rng = np.random.default_rng(1)
        y = 5.0 + rng.normal(0.0, 0.1, size=x.size)
        # No peak clears the 3σ-noise prominence floor.
        assert detect_peaks(x, y) == []

    def test_flat_and_short_traces_are_safe(self):
        assert detect_peaks(np.arange(3.0), np.ones(3)) == []
        assert detect_peaks(np.linspace(0, 10, 50), np.zeros(50)) == []


class TestWidths:
    """Widths are the input clustering needs, and they must describe the PEAK.

    Measured with an unbounded prominence window, a sharp peak sitting on broad
    structure takes its prominence from the far side of that structure and comes
    back tens of units wide. That is a property of the background, and it is what
    made a first attempt at cluster-fitting group 30 peaks into 3.
    """

    @staticmethod
    def _sharp_peaks_on_a_hump():
        x = np.linspace(0.0, 100.0, 4000)
        hump = 800.0 * np.exp(-0.5 * ((x - 25.0) / 12.0) ** 2)
        y = hump + 50.0
        for centre in (40.0, 55.0, 70.0, 85.0):
            y = y + 400.0 * np.exp(-0.5 * ((x - centre) / 0.25) ** 2)
        return x, y

    def test_width_describes_the_peak_not_the_hump(self):
        x, y = self._sharp_peaks_on_a_hump()
        found = detect_peaks_detailed(x, y)
        sharp = [c for c in found if c.center > 35.0]
        assert sharp, "the sharp peaks should be detected"
        # True FWHM is 2.355 * 0.25 ~= 0.59; the hump is ~28 units wide.
        for candidate in sharp:
            assert candidate.width < 3.0

    def test_measure_widths_matches_detection(self):
        x, y = self._sharp_peaks_on_a_hump()
        found = detect_peaks_detailed(x, y)
        again = measure_widths(x, y, [c.center for c in found])
        for candidate, width in zip(found, again, strict=True):
            assert width == pytest.approx(candidate.width, rel=0.25)

    def test_detect_peaks_is_the_centers_of_detailed(self):
        x, y = self._sharp_peaks_on_a_hump()
        assert detect_peaks(x, y) == [c.center for c in detect_peaks_detailed(x, y)]

    def test_widths_are_positive_and_finite(self):
        x, y = self._sharp_peaks_on_a_hump()
        assert all(0.0 < c.width < 100.0 for c in detect_peaks_detailed(x, y))

    def test_measure_widths_on_a_flat_trace_is_harmless(self):
        x = np.linspace(0.0, 10.0, 50)
        assert measure_widths(x, np.zeros(50), [5.0]) == [0.0]

    def test_measure_widths_with_no_centers(self):
        x = np.linspace(0.0, 10.0, 50)
        assert measure_widths(x, np.sin(x), []) == []
