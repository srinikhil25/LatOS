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
from latos.fitting.constraints import Constraint, apply_constraints
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

# Evaluation ceiling, so a badly-posed fit reports failure in seconds rather
# than spinning for a quarter of an hour. Measured on a 30-peak XRD pattern:
# 120 free parameters ran past 40,000 evaluations without converging, and the
# screen sat on "Fitting..." for fifteen minutes before saying so.
#
# lmfit's own default is 2000*(p+1); this is a tenth of that, generous for a
# well-conditioned fit and merciful for one that will never converge. Small fits
# keep lmfit's default: they are cheap, and they occasionally need the room.
_MAX_NFEV_PER_PARAM = 200
_UNCAPPED_BELOW_PARAMS = 20


def _eval_budget(n_varied: int) -> int | None:
    """Evaluation cap for a fit with `n_varied` free parameters, or None."""
    if n_varied < _UNCAPPED_BELOW_PARAMS:
        return None
    return _MAX_NFEV_PER_PARAM * (n_varied + 1)


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
    """A complete fit recipe: one shared line shape, N peaks, a background.

    `constraints` tie peaks together (spin-orbit splitting, area ratios,
    shared widths); peaks are referenced by their index in `peaks`.
    """

    peak_shape: PeakShape
    peaks: list[PeakInit]
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    constraints: list[Constraint] = field(default_factory=list)


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
    """Build lmfit params, seeding center/sigma/amplitude for each peak.

    Bounds are set directly on the parameters (not via `set_param_hint`,
    which a composite model does not always propagate) so a peak whose
    amplitude collapses can never let its center run off to infinity.
    """
    params = model.make_params()
    x_lo, x_hi = float(x.min()), float(x.max())
    span = x_hi - x_lo or 1.0
    default_sigma = max(span * _DEFAULT_SIGMA_RANGE_FRAC, 1e-9)
    for i, peak in enumerate(spec.peaks):
        prefix = f"p{i}_"
        sigma = peak.sigma if peak.sigma is not None else default_sigma
        if peak.amplitude is not None:
            amplitude = peak.amplitude
        else:
            height = float(np.interp(peak.center, x, y_corr))
            amplitude = max(height, 0.0) * sigma * _GAUSS_AREA_PER_HEIGHT_SIGMA
        params[f"{prefix}center"].set(value=peak.center, min=x_lo, max=x_hi)
        params[f"{prefix}sigma"].set(value=sigma, min=1e-12, max=span)
        params[f"{prefix}amplitude"].set(value=max(amplitude, 1e-9), min=0.0)
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

    # Fit on an ascending-x copy: baselines anchor on endpoints, and lmfit /
    # np.interp both assume increasing x (XPS binding energy runs descending).
    # Results are mapped back to the caller's original order at the end.
    order = np.argsort(x, kind="stable")
    x_asc, y_asc = x[order], y[order]

    baseline_asc = compute_baseline(x_asc, y_asc, spec.background)
    y_corr = y_asc - baseline_asc
    x = x_asc  # everything below fits on the ascending grid

    model = peak_model(spec.peak_shape, prefix="p0_")
    for i in range(1, len(spec.peaks)):
        model = model + peak_model(spec.peak_shape, prefix=f"p{i}_")

    params = _seed_params(model, x, y_corr, spec)
    apply_constraints(params, spec.constraints)
    budget = _eval_budget(sum(1 for par in params.values() if par.vary))
    result = model.fit(y_corr, params, x=x, **({} if budget is None else {"max_nfev": budget}))

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

    # Map the fit arrays back to the caller's original x order.
    inverse = np.argsort(order, kind="stable")
    baseline = baseline_asc[inverse]
    best_fit_asc = np.asarray(result.best_fit, dtype=float) + baseline_asc
    best_fit = best_fit_asc[inverse]
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
