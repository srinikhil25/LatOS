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
    "ReliabilityReport",
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
# X is normalized internally so the search span maps to this many units.
# 4.0 makes the canonical doping series (bounds 1–5, span 4) numerically
# identical to the historical unnormalized behaviour, while inputs of any
# magnitude (e.g. carrier concentration ~1e19 cm⁻³) land in the same range
# the length-scale bounds were designed for.
_SPAN_UNITS = 4.0
_DIRECTIONS = ("maximize", "minimize")

# Reliability tiers by observation count, grounded in the retrospective
# calibration study on real experimental data (P3HT/CNT, 233 samples):
# the nominal 95% predictive interval actually covered ~50% of held-out
# points when fit to 6 points, ~85% at 25, and 92% at 70+. Below these
# counts the model is over-confident and its intervals must be read as
# exploratory, not settled.
_RELIABILITY_INDICATIVE_N = 10  # below this: "exploratory"
_RELIABILITY_CALIBRATED_N = 25  # at or above this: "calibrated"
# A leave-one-out coverage this poor forces "exploratory" regardless of n —
# the model demonstrably cannot predict its own data points.
_LOO_FORCE_EXPLORATORY = 0.5
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

    objective: str  # the property being optimized
    direction: str  # "maximize" or "minimize"
    objective_aggregation: str  # how y was reduced per sample (e.g. "peak")
    input_name: str  # the synthesis variable being optimized
    bounds: tuple[float, float]  # search range
    kernel: str  # human-readable kernel description
    x_scale: float  # raw-x units per normalized unit (span / _SPAN_UNITS)
    length_scale: float  # RBF length-scale actually used (normalized-x units)
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
class ReliabilityReport:
    """How much the model's own uncertainty can be trusted — from the data.

    Two independent signals combine into `level`:

    * **Observation count** — the calibration study's tiers (see the
      constants above): "exploratory" below 10 points, "indicative" from
      10, "calibrated" from 25.
    * **Leave-one-out self-check** — refit without each point in turn and
      ask whether the held-out measurement lands inside its own 95%
      predictive interval. Coverage below 50% forces "exploratory": the
      model demonstrably cannot predict its own data.

    A small-n LOO *success* is weak evidence (few folds), but a small-n
    LOO *failure* is strong evidence — hence the asymmetric rule.
    """

    level: str  # "exploratory" | "indicative" | "calibrated"
    n_observations: int
    loo_inside: int  # held-out points inside their 95% predictive interval
    loo_total: int
    loo_coverage: float
    note: str  # plain-language explanation for the UI / prereg record


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
    # How trustworthy the intervals are, from the data itself. None only
    # when the caller skipped the assessment (e.g. the robustness sweep).
    reliability: ReliabilityReport | None = None


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


def _assess_reliability(
    x_norm: np.ndarray,
    y: np.ndarray,
    *,
    rel_noise: float,
    length_scale: float,
    seed: int,
) -> ReliabilityReport:
    """Count-tier + leave-one-out reliability of the model's intervals.

    Each LOO fold refits with the full fit's length-scale held FIXED — the
    cheap, standard approximation (re-optimizing hyperparameters per fold
    buys little at these sizes and costs n × restarts GP fits). The check
    asks the only question that matters: does the model's own 95%
    predictive interval contain the point it didn't see?
    """
    if not np.isfinite(length_scale):
        length_scale = _LS_INIT
    n = int(x_norm.size)
    inside = 0
    for i in range(n):
        mask = np.arange(n) != i
        gp, noise_std = _build_gp(y[mask], rel_noise, length_scale, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(x_norm[mask].reshape(-1, 1), y[mask])
        mu, sd = gp.predict(np.asarray([[x_norm[i]]]), return_std=True)
        half = _CI95 * float(np.sqrt(sd[0] ** 2 + noise_std**2))
        if abs(float(y[i]) - float(mu[0])) <= half:
            inside += 1
    coverage = inside / n if n else 0.0

    if n < _RELIABILITY_INDICATIVE_N:
        level = "exploratory"
        note = (
            f"Exploratory: only {n} measured points. Intervals from so few points "
            f"are typically over-confident — treat the recommendation as a guide, "
            f"not a settled answer. Leave-one-out: {inside}/{n} inside the 95% band."
        )
    elif n < _RELIABILITY_CALIBRATED_N:
        level = "indicative"
        note = (
            f"Indicative: {n} measured points. The trend is meaningful but the "
            f"95% intervals may still be somewhat narrow. "
            f"Leave-one-out: {inside}/{n} inside the 95% band."
        )
    else:
        level = "calibrated"
        note = (
            f"Calibrated: {n} measured points — enough for trustworthy intervals "
            f"per the calibration study. Leave-one-out: {inside}/{n} inside the "
            f"95% band."
        )
    if coverage < _LOO_FORCE_EXPLORATORY and level != "exploratory":
        level = "exploratory"
        note = (
            f"Exploratory: leave-one-out coverage is only {inside}/{n} — the model "
            f"cannot predict its own data points, so its intervals should not be "
            f"trusted regardless of the data count."
        )
    return ReliabilityReport(
        level=level,
        n_observations=n,
        loo_inside=inside,
        loo_total=n,
        loo_coverage=round(coverage, 3),
        note=note,
    )


def optimize(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bounds: tuple[float, float],
    input_name: str,
    target_name: str,
    direction: str = "maximize",
    length_scale: float | None = None,
    rel_noise: float = _REL_NOISE,
    xi: float = _XI,
    grid_size: int = _GRID_SIZE,
    seed: int = 0,
    objective_aggregation: str = "peak",
    created_at: datetime | None = None,
    with_reliability: bool = True,
) -> OptimizationResult:
    """Run one round of Bayesian optimization over a 1-D parameter.

    Args:
        x: Observed parameter values, shape (n,). Any magnitude — x is
            normalized internally so the search span maps to a fixed range
            (carrier concentrations ~1e19 work as well as doping 1–5).
        y: Observed property values, shape (n,).
        bounds: (low, high) search range for the recommendation.
        input_name: Label of the parameter (e.g. "doping_pct").
        target_name: Label of the property (e.g. "peak_zt").
        direction: "maximize" (default) or "minimize". Minimization is exact —
            the engine optimizes -y internally and reports back in original
            units; all displayed quantities (posterior mean, best, prediction)
            stay in the property's real scale.
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
        with_reliability: Run the leave-one-out reliability self-check and
            attach a `ReliabilityReport` (default). The robustness sweep
            passes False — n extra GP fits per swept length-scale would
            buy nothing there.

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
    if direction not in _DIRECTIONS:
        raise OptimizationError(f"direction must be one of {_DIRECTIONS}; got {direction!r}")
    lo, hi = bounds
    if not hi > lo:
        raise OptimizationError(f"bounds must have high > low; got {bounds}")

    # Normalize x so the search span maps to _SPAN_UNITS regardless of the
    # variable's magnitude. The RBF is stationary, so the shift is free; the
    # scale keeps the length-scale bounds meaningful for any input.
    x_scale = (hi - lo) / _SPAN_UNITS
    x_norm = (x - lo) / x_scale

    # Minimization is exact negation: the GP and EI work on y_int; every
    # displayed quantity is mapped back to the property's real scale.
    sign = 1.0 if direction == "maximize" else -1.0
    y_int = sign * y

    gp, noise_std = _build_gp(y_int, rel_noise, length_scale, seed)
    # We deliberately bound the length-scale to keep a handful of points
    # from overfitting; sklearn then warns when the optimizer sits on that
    # bound. That's expected, not a problem — suppress it locally.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x_norm.reshape(-1, 1), y_int)

    grid = np.linspace(lo, hi, grid_size)
    grid_norm = (grid - lo) / x_scale
    mean_int, std = gp.predict(grid_norm.reshape(-1, 1), return_std=True)

    best_i = int(np.argmax(y_int))
    f_best_int = float(y_int[best_i])
    best_x = float(x[best_i])
    ei = _expected_improvement(mean_int, std, f_best_int, xi)

    rec_i = int(np.argmax(ei))
    rec_sigma = float(std[rec_i])
    # Predictive uncertainty adds measurement noise to the model uncertainty:
    # this is the band a *new* measurement at the recommended point should
    # fall within, and the one to test calibration against.
    predictive_sd = float(np.sqrt(rec_sigma**2 + noise_std**2))
    recommendation = Recommendation(
        x=float(grid[rec_i]),
        predicted_mean=float(sign * mean_int[rec_i]),
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
        direction=direction,
        objective_aggregation=objective_aggregation,
        input_name=input_name,
        bounds=(float(lo), float(hi)),
        kernel="ConstantKernel * RBF",
        x_scale=float(x_scale),
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

    reliability: ReliabilityReport | None = None
    if with_reliability:
        reliability = _assess_reliability(
            x_norm,
            y_int,
            rel_noise=rel_noise,
            length_scale=config.length_scale,
            seed=seed,
        )

    return OptimizationResult(
        input_name=input_name,
        target_name=target_name,
        grid_x=tuple(grid.tolist()),
        grid_mean=tuple((sign * mean_int).tolist()),
        grid_ci95=tuple((_CI95 * std).tolist()),
        grid_ei=tuple(ei.tolist()),
        observed_x=tuple(x.tolist()),
        observed_y=tuple(y.tolist()),
        best_x=best_x,
        best_y=float(y[best_i]),
        recommendation=recommendation,
        max_ei=max_ei,
        noise_threshold=noise_threshold,
        converged=converged,
        config=config,
        reliability=reliability,
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
    direction: str = "maximize",
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
            direction=direction,
            length_scale=float(ls),
            rel_noise=rel_noise,
            xi=xi,
            grid_size=grid_size,
            seed=seed,
            with_reliability=False,
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
