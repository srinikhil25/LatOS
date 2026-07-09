"""Bayesian-optimization engine — the brain of Latos's closed loop.

Given a handful of (synthesis-parameter, measured-property) pairs, this
module answers two questions a researcher cares about:

1. **"What should I make next?"** — the synthesis that is most worth
   running, via the Expected-Improvement acquisition function.
2. **"Am I done?"** — whether any untried experiment is still expected
   to meaningfully beat the best result so far (the stopping signal).

Method
------
A **Gaussian Process (GP)** surrogate models the property as a smooth
function of the synthesis parameter, with calibrated uncertainty
(wide where there's no data, narrow near measurements). We inject a
realistic **measurement-noise floor** so the GP passes *near* — not
exactly through — the points; zero-noise interpolation of experimental
data is wrong.

**Expected Improvement (EI)** scores every candidate parameter by how
much it could beat the current best, weighing predicted value against
uncertainty:

    EI(x) = (mu(x) - f_best - xi)·Phi(Z) + sigma(x)·phi(Z),
    Z = (mu(x) - f_best - xi) / sigma(x)

The recommendation is `argmax EI`. **Convergence** is a *noise-aware
heuristic* (not a formal guarantee): we flag it when the largest EI
anywhere falls below the measurement-noise floor — no experiment is then
expected to improve things by more than we can measure.

Prospective, auditable recommendations
---------------------------------------
Every `optimize()` call also returns a frozen `BoConfig` — kernel,
length-scale (fitted or fixed), exploration `xi`, noise model, objective,
search bounds and RNG seed. Emitting that *with* the recommendation is
what lets a reviewer confirm the recommendation was not retuned after the
answer was known. `length_scale_robustness()` additionally checks that the
recommended point does not swing as the kernel length-scale is varied.

Implementation note: this uses scikit-learn's `GaussianProcessRegressor`
— light and dependable on a memory-constrained machine. GPyTorch/BoTorch
(the `ml` extra) are the GPU-scale production target; the GP method is
identical.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime

# Single-threaded BLAS keeps memory tiny for these small problems and
# avoids OpenBLAS over-allocating on constrained boxes. Set before numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

__all__ = [
    "BoConfig",
    "OptimizationError",
    "OptimizationResult",
    "Recommendation",
    "RobustnessEntry",
    "RobustnessReport",
    "length_scale_robustness",
    "optimize",
]

# Defaults
_REL_NOISE = 0.08  # ~8% relative measurement noise (zT-typical, defensible)
_XI = 0.01  # EI exploration sweetener
_GRID_SIZE = 200
_MIN_POINTS = 3  # a GP over fewer than 3 points isn't worth trusting
_LS_INIT = 1.5  # RBF length-scale starting point when fitted
_LS_BOUNDS = (1.0, 5.0)  # bounds the length-scale is fitted within
_N_RESTARTS = 8  # marginal-likelihood restarts when the length-scale is fitted
# A recommendation is "robust" if it moves by <= this fraction of the search
# span as the kernel length-scale is varied (one-line kernel-artifact defense).
_ROBUSTNESS_TOL_FRAC = 0.1
_CI95 = 1.96  # 95% Gaussian half-width in standard deviations


class OptimizationError(ValueError):
    """The optimization could not be run (too few points, bad inputs)."""


@dataclass(frozen=True, slots=True)
class Recommendation:
    """The single most-worth-running next experiment.

    Two uncertainty bands are reported, and the distinction matters when a
    prediction is later checked against a real measurement:

    * ``ci95`` — the **model** (epistemic) 95% half-width, ``1.96·sigma_GP``.
      How unsure the surrogate is about its own mean.
    * ``ci95_predictive`` — the **predictive** 95% half-width,
      ``1.96·sqrt(sigma_GP^2 + noise^2)``. What a *new measurement* at this
      point should fall within. This is the band to test calibration against.
    """

    x: float  # recommended parameter value
    predicted_mean: float  # GP-predicted property there
    ci95: float  # +/- 95% model half-width on that prediction
    predictive_sd: float  # sqrt(model variance + measurement noise variance)
    ci95_predictive: float  # +/- 95% half-width a new measurement should land in


@dataclass(frozen=True, slots=True)
class BoConfig:
    """Frozen record of exactly how a recommendation was produced.

    Emitting this *with* the recommendation is what makes the recommendation
    prospective and auditable: kernel, length-scale (and whether it was fitted
    or held fixed), the exploration sweetener, the noise model, the objective,
    the search bounds and the RNG seed are all pinned. Nothing can be quietly
    retuned after the answer is known.
    """

    objective: str  # the property being maximized
    objective_aggregation: str  # how y was reduced per sample (e.g. "peak")
    input_name: str  # the synthesis variable being optimized
    bounds: tuple[float, float]  # search range
    kernel: str  # human-readable kernel description
    length_scale: float  # RBF length-scale actually used
    length_scale_fitted: bool  # True if fitted from data, False if held fixed
    length_scale_bounds: tuple[float, float]  # bounds used when fitting
    xi: float  # EI exploration sweetener
    rel_noise: float  # relative measurement-noise level fed to the GP
    noise_std: float  # absolute measurement-noise std, in objective units
    n_observations: int  # number of measured points the GP was fit to
    grid_size: int  # resolution of the posterior/acquisition grid
    seed: int  # RNG seed (GP restarts) — makes the fit reproducible
    created_at: datetime  # when this recommendation was produced


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Everything the Optimize screen needs to render the loop.

    The `grid_*` arrays are the smooth posterior over the search range
    (for the curve + band + acquisition plot). The scalars are the
    headline answers. `config` is the frozen, auditable record of how the
    recommendation was produced.
    """

    input_name: str
    target_name: str
    # Posterior over the search range (1-D), for plotting.
    grid_x: tuple[float, ...]
    grid_mean: tuple[float, ...]
    grid_ci95: tuple[float, ...]
    grid_ei: tuple[float, ...]
    # The observed data echoed back (so the UI can plot the points).
    observed_x: tuple[float, ...]
    observed_y: tuple[float, ...]
    # Headline outputs.
    best_x: float
    best_y: float
    recommendation: Recommendation
    max_ei: float
    noise_threshold: float  # measurement-noise floor EI is compared against
    converged: bool
    config: BoConfig  # frozen record of the exact BO configuration


def _build_gp(
    y: np.ndarray, rel_noise: float, length_scale: float | None, seed: int
) -> tuple[GaussianProcessRegressor, float]:
    """A GP with a smooth RBF trend and a realistic measurement-noise floor.

    When `length_scale` is None the length-scale is fitted by marginal
    likelihood (within `_LS_BOUNDS`); when a value is given it is held
    fixed — which is what `length_scale_robustness()` sweeps over. Returns
    the (unfitted) GP and the absolute noise std used, in `y`'s units.
    """
    noise_std = rel_noise * float(np.mean(np.abs(y)))
    alpha = (noise_std / max(float(np.std(y)), 1e-9)) ** 2
    if length_scale is None:
        rbf = RBF(length_scale=_LS_INIT, length_scale_bounds=_LS_BOUNDS)
        n_restarts = _N_RESTARTS
    else:
        rbf = RBF(length_scale=length_scale, length_scale_bounds="fixed")
        n_restarts = 0
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * rbf
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )
    return gp, noise_std


def _fitted_length_scale(gp: GaussianProcessRegressor) -> float:
    """Read the RBF length-scale back out of a fitted GP kernel."""
    rbf = getattr(gp.kernel_, "k2", None)
    length_scale = getattr(rbf, "length_scale", None)
    if length_scale is None:
        return float("nan")
    try:
        return float(length_scale)
    except (TypeError, ValueError):
        return float("nan")


def _expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float
) -> np.ndarray:
    """Expected improvement over `f_best` (maximization)."""
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - f_best - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.asarray(ei, dtype=float)


def optimize(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bounds: tuple[float, float],
    input_name: str,
    target_name: str,
    length_scale: float | None = None,
    rel_noise: float = _REL_NOISE,
    xi: float = _XI,
    grid_size: int = _GRID_SIZE,
    seed: int = 0,
    objective_aggregation: str = "peak",
    created_at: datetime | None = None,
) -> OptimizationResult:
    """Run one round of Bayesian optimization over a 1-D parameter.

    Args:
        x: Observed parameter values, shape (n,).
        y: Observed property values to maximize, shape (n,).
        bounds: (low, high) search range for the recommendation.
        input_name: Label of the parameter (e.g. "doping_pct").
        target_name: Label of the property (e.g. "peak_zt").
        length_scale: Fix the RBF length-scale to this value; if None it is
            fitted from the data. Fixing it is how `length_scale_robustness`
            probes whether the recommendation is a kernel artifact.
        rel_noise: Relative measurement noise injected into the GP. Also
            sets the convergence floor: when the best expected improvement
            falls below this noise level, no experiment can *reliably* do
            better, so we report converged (a heuristic, not a guarantee).
        xi: Exploration sweetener in EI.
        grid_size: Resolution of the posterior curve.
        seed: RNG seed for the GP restarts — makes the fit reproducible.
        objective_aggregation: How each sample's `y` was reduced (recorded
            in the frozen config for auditability).
        created_at: Timestamp for the frozen config; defaults to now (UTC).

    Returns:
        An `OptimizationResult` with the posterior, the recommendation,
        the convergence verdict, and the frozen `BoConfig`.

    Raises:
        OptimizationError: Fewer than 3 points, mismatched shapes, or a
            degenerate (zero-width) bound range.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise OptimizationError(f"x and y length mismatch: {x.shape} vs {y.shape}")
    if x.size < _MIN_POINTS:
        raise OptimizationError(
            f"Need at least {_MIN_POINTS} measured points to optimize; got {x.size}"
        )
    lo, hi = bounds
    if not hi > lo:
        raise OptimizationError(f"bounds must have high > low; got {bounds}")

    gp, noise_std = _build_gp(y, rel_noise, length_scale, seed)
    # We deliberately bound the length-scale to keep a handful of points
    # from overfitting; sklearn then warns when the optimizer sits on that
    # bound. That's expected, not a problem — suppress it locally.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x.reshape(-1, 1), y)

    grid = np.linspace(lo, hi, grid_size)
    mean, std = gp.predict(grid.reshape(-1, 1), return_std=True)

    f_best = float(np.max(y))
    best_x = float(x[int(np.argmax(y))])
    ei = _expected_improvement(mean, std, f_best, xi)

    rec_i = int(np.argmax(ei))
    rec_sigma = float(std[rec_i])
    # Predictive uncertainty adds measurement noise to the model uncertainty:
    # this is the band a *new* measurement at the recommended point should
    # fall within, and the one to test calibration against.
    predictive_sd = float(np.sqrt(rec_sigma**2 + noise_std**2))
    recommendation = Recommendation(
        x=float(grid[rec_i]),
        predicted_mean=float(mean[rec_i]),
        ci95=_CI95 * rec_sigma,
        predictive_sd=predictive_sd,
        ci95_predictive=_CI95 * predictive_sd,
    )
    max_ei = float(ei[rec_i])
    # Stopping rule: when the best possible expected improvement is
    # smaller than the measurement noise, no experiment can *reliably*
    # do better. A noise-aware heuristic for "stop and publish", not a
    # formal optimality guarantee.
    noise_threshold = rel_noise * float(np.mean(np.abs(y)))
    converged = max_ei < noise_threshold

    config = BoConfig(
        objective=target_name,
        objective_aggregation=objective_aggregation,
        input_name=input_name,
        bounds=(float(lo), float(hi)),
        kernel="ConstantKernel * RBF",
        length_scale=(
            float(length_scale) if length_scale is not None else _fitted_length_scale(gp)
        ),
        length_scale_fitted=length_scale is None,
        length_scale_bounds=_LS_BOUNDS,
        xi=xi,
        rel_noise=rel_noise,
        noise_std=noise_std,
        n_observations=int(x.size),
        grid_size=grid_size,
        seed=seed,
        created_at=created_at if created_at is not None else datetime.now(UTC),
    )

    return OptimizationResult(
        input_name=input_name,
        target_name=target_name,
        grid_x=tuple(grid.tolist()),
        grid_mean=tuple(mean.tolist()),
        grid_ci95=tuple((_CI95 * std).tolist()),
        grid_ei=tuple(ei.tolist()),
        observed_x=tuple(x.tolist()),
        observed_y=tuple(y.tolist()),
        best_x=best_x,
        best_y=f_best,
        recommendation=recommendation,
        max_ei=max_ei,
        noise_threshold=noise_threshold,
        converged=converged,
        config=config,
    )


@dataclass(frozen=True, slots=True)
class RobustnessEntry:
    """The recommendation produced at one fixed length-scale."""

    length_scale: float
    recommended_x: float
    predicted_mean: float
    ci95_predictive: float


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    """Does the recommendation swing as the kernel length-scale changes?

    `stable` is the one-line defense against "it's just a kernel artifact":
    if the recommended point moves by no more than `tolerance` (a fraction
    of the search span) across the swept length-scales, the pick is robust.
    An unstable report is itself a useful signal — the data is too sparse to
    recommend confidently, which is worth knowing *before* synthesizing.
    """

    entries: tuple[RobustnessEntry, ...]
    recommended_x_spread: float
    search_span: float
    tolerance: float
    stable: bool


def length_scale_robustness(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bounds: tuple[float, float],
    input_name: str,
    target_name: str,
    length_scales: tuple[float, ...],
    rel_noise: float = _REL_NOISE,
    xi: float = _XI,
    grid_size: int = _GRID_SIZE,
    seed: int = 0,
    tol_frac: float = _ROBUSTNESS_TOL_FRAC,
) -> RobustnessReport:
    """Re-run the optimization at several fixed length-scales and compare.

    Returns a `RobustnessReport` whose `stable` flag is True when the
    recommended point varies by at most `tol_frac` of the search span.
    """
    entries: list[RobustnessEntry] = []
    for ls in length_scales:
        result = optimize(
            x,
            y,
            bounds=bounds,
            input_name=input_name,
            target_name=target_name,
            length_scale=float(ls),
            rel_noise=rel_noise,
            xi=xi,
            grid_size=grid_size,
            seed=seed,
        )
        rec = result.recommendation
        entries.append(
            RobustnessEntry(
                length_scale=float(ls),
                recommended_x=rec.x,
                predicted_mean=rec.predicted_mean,
                ci95_predictive=rec.ci95_predictive,
            )
        )
    rec_xs = [e.recommended_x for e in entries]
    spread = float(max(rec_xs) - min(rec_xs)) if rec_xs else 0.0
    span = float(bounds[1] - bounds[0])
    tolerance = tol_frac * span
    return RobustnessReport(
        entries=tuple(entries),
        recommended_x_spread=spread,
        search_span=span,
        tolerance=tolerance,
        stable=spread <= tolerance,
    )
