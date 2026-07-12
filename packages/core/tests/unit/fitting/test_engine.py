"""Tests for the universal peak-fit engine."""

from __future__ import annotations

import numpy as np
import pytest

from latos.fitting import (
    BackgroundKind,
    BackgroundSpec,
    FitError,
    FitSpec,
    PeakInit,
    PeakShape,
    fit_spectrum,
    peak_model,
)

_GAUSS_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _gaussian_area(x, center, sigma, area):
    """A Gaussian parameterised by area (lmfit's `amplitude`)."""
    height = area / (sigma * np.sqrt(2.0 * np.pi))
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _two_peak_spectrum(seed: int = 0):
    x = np.linspace(0.0, 100.0, 600)
    background = 10.0 + 0.1 * x
    y = background + _gaussian_area(x, 30.0, 3.0, 50.0) + _gaussian_area(x, 60.0, 5.0, 80.0)
    rng = np.random.default_rng(seed)
    y = y + rng.normal(0.0, 0.05, size=x.size)  # tiny, reproducible noise
    return x, y


class TestPeakModelVocabulary:
    @pytest.mark.parametrize("shape", list(PeakShape))
    def test_every_shape_builds_with_core_params(self, shape):
        model = peak_model(shape, prefix="p0_")
        params = model.make_params()
        for core in ("p0_amplitude", "p0_center", "p0_sigma"):
            assert core in params


class TestFitSpectrum:
    def test_recovers_two_gaussians_over_linear_background(self):
        x, y = _two_peak_spectrum()
        spec = FitSpec(
            peak_shape=PeakShape.GAUSSIAN,
            peaks=[PeakInit(center=30.0), PeakInit(center=60.0)],
            background=BackgroundSpec(kind=BackgroundKind.LINEAR),
        )
        r = fit_spectrum(x, y, spec)

        assert r.success
        assert r.r_squared > 0.999
        assert len(r.components) == 2

        by_center = sorted(r.components, key=lambda c: c.center)
        assert by_center[0].center == pytest.approx(30.0, abs=0.2)
        assert by_center[1].center == pytest.approx(60.0, abs=0.2)
        assert by_center[0].amplitude == pytest.approx(50.0, rel=0.05)
        assert by_center[1].amplitude == pytest.approx(80.0, rel=0.05)
        # FWHM = 2√(2ln2)·σ for a Gaussian.
        assert by_center[0].fwhm == pytest.approx(_GAUSS_FWHM_PER_SIGMA * 3.0, rel=0.05)

    def test_reports_uncertainties(self):
        x, y = _two_peak_spectrum()
        spec = FitSpec(PeakShape.GAUSSIAN, [PeakInit(30.0), PeakInit(60.0)])
        r = fit_spectrum(x, y, spec)
        # Every fitted parameter has a finite 1σ error on clean data.
        stderrs = [se for _, se in r.params.values()]
        assert all(se is not None and se >= 0 for se in stderrs)

    def test_residual_and_bestfit_align_with_data(self):
        x, y = _two_peak_spectrum()
        spec = FitSpec(PeakShape.GAUSSIAN, [PeakInit(30.0), PeakInit(60.0)])
        r = fit_spectrum(x, y, spec)
        assert r.best_fit.shape == y.shape
        assert np.allclose(r.residual, y - r.best_fit)
        assert np.max(np.abs(r.residual)) < 1.0  # tight fit on clean data

    def test_pseudo_voigt_also_fits(self):
        x, y = _two_peak_spectrum()
        spec = FitSpec(PeakShape.PSEUDO_VOIGT, [PeakInit(30.0), PeakInit(60.0)])
        r = fit_spectrum(x, y, spec)
        assert r.r_squared > 0.99


class TestGuards:
    def test_no_peaks_raises(self):
        x, y = _two_peak_spectrum()
        with pytest.raises(FitError, match="no peaks"):
            fit_spectrum(x, y, FitSpec(PeakShape.GAUSSIAN, []))

    def test_length_mismatch_raises(self):
        with pytest.raises(FitError, match="differ in length"):
            fit_spectrum(
                np.arange(10.0), np.arange(9.0), FitSpec(PeakShape.GAUSSIAN, [PeakInit(5.0)])
            )

    def test_too_few_points_raises(self):
        with pytest.raises(FitError, match="Too few points"):
            fit_spectrum(
                np.array([0.0, 1.0]),
                np.array([1.0, 2.0]),
                FitSpec(PeakShape.GAUSSIAN, [PeakInit(0.0), PeakInit(1.0)]),
            )
