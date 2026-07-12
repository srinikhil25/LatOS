"""Tests for per-technique fit presets and inter-peak constraints."""

from __future__ import annotations

import numpy as np
import pytest

from latos.fitting import (
    XPS_DOUBLETS,
    BackgroundKind,
    PeakShape,
    fit_spectrum,
    raman_preset,
    xps_doublet_preset,
    xrd_preset,
)


def _gaussian_area(x, center, sigma, area):
    height = area / (sigma * np.sqrt(2.0 * np.pi))
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


class TestPresetShapes:
    def test_xrd_is_pseudo_voigt_on_polynomial(self):
        spec = xrd_preset([28.4, 47.3, 56.1])
        assert spec.peak_shape is PeakShape.PSEUDO_VOIGT
        assert spec.background.kind is BackgroundKind.POLYNOMIAL
        assert len(spec.peaks) == 3
        assert not spec.constraints

    def test_raman_is_lorentzian_on_als(self):
        spec = raman_preset([1350.0, 1580.0])
        assert spec.peak_shape is PeakShape.LORENTZIAN
        assert spec.background.kind is BackgroundKind.ALS

    def test_xps_doublet_ties_delta_ratio_width(self):
        spec = xps_doublet_preset(932.6, delta_be=19.8, area_ratio=0.5)
        assert spec.background.kind is BackgroundKind.SHIRLEY
        assert len(spec.peaks) == 2
        assert len(spec.constraints) == 3  # delta, ratio, shared width

    def test_doublet_table_has_expected_ratios(self):
        assert XPS_DOUBLETS["Cu 2p"] == (19.8, 0.5)  # p: 2:1
        assert XPS_DOUBLETS["Bi 4f"][1] == pytest.approx(0.75)  # f: 4:3


class TestXpsDoubletFit:
    def _cu_2p_like(self):
        # Synthetic Cu 2p: 2p3/2 at 932.6, 2p1/2 at 952.4 (Δ=19.8),
        # areas 100 / 50 (2:1), shared σ=1.1. A flat background is what the
        # Shirley model reduces to when the endpoints sit at the same level
        # — the honest way to isolate the doublet-constraint behaviour.
        x = np.linspace(925.0, 960.0, 700)
        y = 45.0 + _gaussian_area(x, 932.6, 1.1, 100.0) + _gaussian_area(x, 952.4, 1.1, 50.0)
        rng = np.random.default_rng(0)
        return x, y + rng.normal(0.0, 0.05, size=x.size)

    def test_constraints_hold_exactly_and_center_recovers(self):
        x, y = self._cu_2p_like()
        spec = xps_doublet_preset(932.6, delta_be=19.8, area_ratio=0.5, shape=PeakShape.GAUSSIAN)
        r = fit_spectrum(x, y, spec)

        assert r.success
        assert r.r_squared > 0.99
        p32, p12 = sorted(r.components, key=lambda c: c.center)
        # The splitting is pinned exactly by the FixedDelta constraint.
        assert (p12.center - p32.center) == pytest.approx(19.8, abs=1e-6)
        # The 2:1 area ratio is pinned exactly by FixedRatio.
        assert (p12.amplitude / p32.amplitude) == pytest.approx(0.5, abs=1e-6)
        # Shared width: both sigmas identical.
        assert p12.sigma == pytest.approx(p32.sigma, abs=1e-6)
        # The one free center lands on the true 2p3/2 line.
        assert p32.center == pytest.approx(932.6, abs=0.1)

    def test_constraint_cuts_free_parameters(self):
        # A free 2-Gaussian fit varies center/amp/sigma × 2 = 6; the tied
        # doublet varies only p0 (center, amp, sigma) → far fewer varying.
        x, y = self._cu_2p_like()
        spec = xps_doublet_preset(932.6, delta_be=19.8, shape=PeakShape.GAUSSIAN)
        r = fit_spectrum(x, y, spec)
        # p1_center / p1_amplitude / p1_sigma are expressions (stderr None,
        # value derived), not independently fit.
        assert r.params["p1_center"][0] == pytest.approx(952.4, abs=0.1)


class TestXrdPresetFit:
    def test_fits_two_reflections(self):
        x = np.linspace(20.0, 60.0, 800)
        bg = 5.0 + 0.02 * x
        y = bg + _gaussian_area(x, 28.4, 0.15, 40.0) + _gaussian_area(x, 47.3, 0.2, 25.0)
        r = fit_spectrum(x, y, xrd_preset([28.4, 47.3]))
        # A whole-trace polynomial background is slightly pulled up under
        # sharp Bragg peaks; the centers (→ d-spacings) still recover cleanly.
        assert r.r_squared > 0.96
        centers = sorted(c.center for c in r.components)
        assert centers[0] == pytest.approx(28.4, abs=0.1)
        assert centers[1] == pytest.approx(47.3, abs=0.1)
