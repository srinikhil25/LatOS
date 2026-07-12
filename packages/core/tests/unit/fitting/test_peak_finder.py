"""Tests for automatic peak detection."""

from __future__ import annotations

import numpy as np
import pytest

from latos.fitting import detect_peaks


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
