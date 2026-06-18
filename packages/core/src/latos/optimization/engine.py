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

The recommendation is `argmax EI`. **Convergence** is declared when the
largest EI anywhere falls below a small fraction of the current best —
i.e. no experiment is expected to improve things by more than that, so
you can stop and publish.

Implementation note: this uses scikit-learn's `GaussianProcessRegressor`
— light and dependable on a memory-constrained machine. GPyTorch/BoTorch
(the `ml` extra) are the GPU-scale production target; the GP method is
identical.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

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
    "OptimizationError",
    "OptimizationResult",
    "Recommendation",
    "optimize",
]

# Defaults
_REL_NOISE = 0.08  # ~8% relative measurement noise (zT-typical, defensible)
_XI = 0.01  # EI exploration sweetener
_GRID_SIZE = 200
_MIN_POINTS = 3  # a GP over fewer than 3 points isn't worth trusting


class OptimizationError(ValueError):
    """The optimization could not be run (too few points, bad inputs)."""


@dataclass(frozen=True, slots=True)
class Recommendation:
    """The single most-worth-running next experiment."""

    x: float  # recommended parameter value
    predicted_mean: float  # GP-predicted property there
    ci95: float  # +/- 95% half-width on that prediction


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Everything the Optimize screen needs to render the loop.

    The `grid_*` arrays are the smooth posterior over the search range
    (for the curve + band + acquisition plot). The scalars are the
    headline answers.
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


def _build_gp(y: np.ndarray, rel_noise: float) -> GaussianProcessRegressor:
    """A GP with a smooth RBF trend and a realistic measurement-noise floor."""
    noise_std = rel_noise * float(np.mean(np.abs(y)))
    alpha = (noise_std / max(float(np.std(y)), 1e-9)) ** 2
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=1.5, length_scale_bounds=(1.0, 5.0)
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=0,
    )


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
    rel_noise: float = _REL_NOISE,
    xi: float = _XI,
    grid_size: int = _GRID_SIZE,
) -> OptimizationResult:
    """Run one round of Bayesian optimization over a 1-D parameter.

    Args:
        x: Observed parameter values, shape (n,).
        y: Observed property values to maximize, shape (n,).
        bounds: (low, high) search range for the recommendation.
        input_name: Label of the parameter (e.g. "doping_pct").
        target_name: Label of the property (e.g. "peak_zt").
        rel_noise: Relative measurement noise injected into the GP. Also
            sets the convergence floor: when the best expected
            improvement falls below this noise level, no experiment can
            *reliably* do better, so we report converged.
        xi: Exploration sweetener in EI.
        grid_size: Resolution of the posterior curve.

    Returns:
        An `OptimizationResult` with the posterior, the recommendation,
        and the convergence verdict.

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

    gp = _build_gp(y, rel_noise)
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
    recommendation = Recommendation(
        x=float(grid[rec_i]),
        predicted_mean=float(mean[rec_i]),
        ci95=float(1.96 * std[rec_i]),
    )
    max_ei = float(ei[rec_i])
    # Stopping rule: when the best possible expected improvement is
    # smaller than the measurement noise, no experiment can *reliably*
    # do better — you've reached the optimum within your ability to
    # measure it, so stop and publish.
    noise_threshold = rel_noise * float(np.mean(np.abs(y)))
    converged = max_ei < noise_threshold

    return OptimizationResult(
        input_name=input_name,
        target_name=target_name,
        grid_x=tuple(grid.tolist()),
        grid_mean=tuple(mean.tolist()),
        grid_ci95=tuple((1.96 * std).tolist()),
        grid_ei=tuple(ei.tolist()),
        observed_x=tuple(x.tolist()),
        observed_y=tuple(y.tolist()),
        best_x=best_x,
        best_y=f_best,
        recommendation=recommendation,
        max_ei=max_ei,
        noise_threshold=noise_threshold,
        converged=converged,
    )
