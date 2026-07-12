"""Universal peak-fit engine — *peaks on a background*, one call.

`fit_spectrum` is the reusable core behind Latos's "replace Origin" fitting:
give it x, y and a `FitSpec` (a line shape + initial peak positions + a
background choice) and it returns a `FitResult` carrying every number a
publication figure needs — fitted parameters with ±1σ, the component
decomposition, the background, the residual, and goodness-of-fit (R², χ²,
reduced χ²).

The background is computed first (see `backgrounds`) and subtracted; peaks
are fit to the corrected trace; the reported envelope adds the background
back so it overlays the raw data. Per-technique presets and an interactive
editor build on top of this — they only assemble a `FitSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from lmfit.model import Model
from lmfit.parameter import Parameters
from numpy.typing import NDArray

from latos.fitting import backgrounds
from latos.fitting.peak_shapes import PeakShape, peak_model

__all__ = [
    "BackgroundKind",
    "BackgroundSpec",
    "FitError",
    "FitResult",
    "FitSpec",
    "FittedComponent",
    "PeakInit",
    "compute_baseline",
    "fit_spectrum",
]

# Gaussian area ≈ height · σ · √(2π); used only to seed the amplitude when
# the caller gives a position but no starting amplitude.
_GAUSS_AREA_PER_HEIGHT_SIGMA = float(np.sqrt(2.0 * np.pi))

# Default starting σ as a fraction of the x-range when none is supplied —
# a few percent of the window is a neutral guess lmfit refines quickly.
_DEFAULT_SIGMA_RANGE_FRAC = 0.02


class FitError(ValueError):
    """Raised when a fit cannot be set up (no peaks, degenerate data)."""


class BackgroundKind(StrEnum):
    """Which background model to subtract before fitting peaks."""

    NONE = "none"
    CONSTANT = "constant"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    SHIRLEY = "shirley"
    ALS = "als"


@dataclass(frozen=True)
class BackgroundSpec:
    """Background choice plus its tuning knobs (only the relevant ones apply)."""

    kind: BackgroundKind = BackgroundKind.LINEAR
    degree: int = 2  # polynomial
    lam: float = 1e5  # ALS smoothness
    p: float = 0.01  # ALS asymmetry


@dataclass(frozen=True)
class PeakInit:
    """Starting guess for one peak. Only `center` is required."""

    center: float
    amplitude: float | None = None  # peak area; estimated if None
    sigma: float | None = None  # estimated if None


@dataclass(frozen=True)
class FitSpec:
    """A complete fit recipe: one shared line shape, N peaks, a background."""

    peak_shape: PeakShape
    peaks: list[PeakInit]
    background: BackgroundSpec = field(default_factory=BackgroundSpec)


@dataclass(frozen=True)
class FittedComponent:
    """One fitted peak's headline numbers (fwhm/height absent for some shapes)."""

    prefix: str
    center: float
    amplitude: float  # area under the peak
    sigma: float
    fwhm: float | None
    height: float | None


@dataclass(frozen=True)
class FitResult:
    """Everything a report or an overlay needs from one fit."""

    success: bool
    r_squared: float
    chi_square: float
    reduced_chi_square: float
    components: list[FittedComponent]
    baseline: NDArray[np.float64]
    best_fit: NDArray[np.float64]  # peaks + baseline (overlays raw data)
    residual: NDArray[np.float64]  # y - best_fit
    # name -> (value, 1σ stderr | None) for every fitted parameter.
    params: dict[str, tuple[float, float | None]]


def compute_baseline(
    x: NDArray[np.float64], y: NDArray[np.float64], spec: BackgroundSpec
) -> NDArray[np.float64]:
    """Baseline array for `y` under the chosen background model."""
    y = np.asarray(y, dtype=float)
    kind = spec.kind
    if kind is BackgroundKind.NONE:
        return np.zeros_like(y)
    if kind is BackgroundKind.CONSTANT:
        return backgrounds.constant_baseline(y)
    if kind is BackgroundKind.LINEAR:
        return backgrounds.linear_baseline(x, y)
    if kind is BackgroundKind.POLYNOMIAL:
        return backgrounds.polynomial_baseline(x, y, degree=spec.degree)
    if kind is BackgroundKind.SHIRLEY:
        return backgrounds.shirley_baseline(y)
    return backgrounds.als_baseline(y, lam=spec.lam, p=spec.p)


def _seed_params(
    model: Model, x: NDArray[np.float64], y_corr: NDArray[np.float64], spec: FitSpec
) -> Parameters:
    """Build lmfit params, seeding center/sigma/amplitude for each peak."""
    for m in model.components:
        m.set_param_hint("sigma", min=1e-12)
        m.set_param_hint("amplitude", min=0.0)
        m.set_param_hint("center", min=float(x.min()), max=float(x.max()))
    params = model.make_params()
    default_sigma = max((float(x.max()) - float(x.min())) * _DEFAULT_SIGMA_RANGE_FRAC, 1e-9)
    for i, peak in enumerate(spec.peaks):
        prefix = f"p{i}_"
        sigma = peak.sigma if peak.sigma is not None else default_sigma
        if peak.amplitude is not None:
            amplitude = peak.amplitude
        else:
            height = float(np.interp(peak.center, x, y_corr))
            amplitude = max(height, 0.0) * sigma * _GAUSS_AREA_PER_HEIGHT_SIGMA
        params[f"{prefix}center"].set(value=peak.center)
        params[f"{prefix}sigma"].set(value=sigma)
        params[f"{prefix}amplitude"].set(value=max(amplitude, 1e-9))
    return params


def fit_spectrum(x: NDArray[np.float64], y: NDArray[np.float64], spec: FitSpec) -> FitResult:
    """Fit `spec`'s peaks over its background to (x, y).

    Raises:
        FitError: if the spec has no peaks or the arrays are too short /
            mismatched to fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise FitError(f"x and y differ in length ({x.size} vs {y.size}).")
    if not spec.peaks:
        raise FitError("FitSpec has no peaks to fit.")
    if x.size < 2 * len(spec.peaks):
        raise FitError(f"Too few points ({x.size}) to fit {len(spec.peaks)} peak(s).")

    baseline = compute_baseline(x, y, spec.background)
    y_corr = y - baseline

    model = peak_model(spec.peak_shape, prefix="p0_")
    for i in range(1, len(spec.peaks)):
        model = model + peak_model(spec.peak_shape, prefix=f"p{i}_")

    params = _seed_params(model, x, y_corr, spec)
    result = model.fit(y_corr, params, x=x)

    components: list[FittedComponent] = []
    for i in range(len(spec.peaks)):
        prefix = f"p{i}_"
        fwhm = result.params.get(f"{prefix}fwhm")
        height = result.params.get(f"{prefix}height")
        components.append(
            FittedComponent(
                prefix=prefix,
                center=float(result.params[f"{prefix}center"].value),
                amplitude=float(result.params[f"{prefix}amplitude"].value),
                sigma=float(result.params[f"{prefix}sigma"].value),
                fwhm=None if fwhm is None else float(fwhm.value),
                height=None if height is None else float(height.value),
            )
        )

    best_fit = np.asarray(result.best_fit, dtype=float) + baseline
    param_table: dict[str, tuple[float, float | None]] = {
        name: (float(par.value), None if par.stderr is None else float(par.stderr))
        for name, par in result.params.items()
    }
    return FitResult(
        success=bool(getattr(result, "success", True)),
        r_squared=float(result.rsquared),
        chi_square=float(result.chisqr),
        reduced_chi_square=float(result.redchi),
        components=components,
        baseline=baseline,
        best_fit=best_fit,
        residual=y - best_fit,
        params=param_table,
    )
