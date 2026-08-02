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

import math
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
# A measurement our physics checks reject is treated as this many times
# noisier rather than deleted: the GP still sees it, but stops chasing it.
_UNRELIABLE_NOISE_FACTOR = 3.0
# Posterior draws behind the (epsilon, delta) statement. 512 puts the standard
# error of the reported probability near 2%, which is finer than we quote it.
_N_POSTERIOR_DRAWS = 512
_DEFAULT_DELTA = 0.1  # report "within epsilon" at 90% confidence by default

# y-space transforms (the physics layer). Strictly-positive order-of-magnitude
# quantities (mobility, conductivity, …) are fit in log space so the surrogate
# can never predict a negative value and its multiplicative noise is modelled
# correctly. Everything else is fit linearly and clamped to its physical domain.
_IDENTITY = "identity"
_LOG = "log"
_TRANSFORMS = (_IDENTITY, _LOG)


def _forward(y: np.ndarray, transform: str) -> np.ndarray:
    """Map property values into the space the GP is fit in."""
    return np.log(y) if transform == _LOG else y


def _inverse(v: np.ndarray, transform: str) -> np.ndarray:
    """Map GP-space values back to physical (property) units."""
    return np.exp(v) if transform == _LOG else v


def _clamp(v: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    """Clip to a physical domain (None = unbounded on that side)."""
    if lo is not None:
        v = np.maximum(v, lo)
    if hi is not None:
        v = np.minimum(v, hi)
    return v


def _to_phys(v: float, transform: str, lo: float | None, hi: float | None) -> float:
    """Inverse-transform and clamp a single fit-space value to physical units."""
    r = math.exp(v) if transform == _LOG else v
    if lo is not None:
        r = max(r, lo)
    if hi is not None:
        r = min(r, hi)
    return float(r)


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
    predicted_mean: float  # GP-predicted property there (physical units)
    ci95: float  # +/- 95% model half-width (approx, physical units)
    predictive_sd: float  # sqrt(model variance + noise variance), in fit space
    ci95_predictive: float  # +/- 95% predictive half-width (approx, physical units)
    # Explicit [low, high] predictive interval in physical units. Carried
    # separately because a log-space fit gives an ASYMMETRIC interval that a
    # single half-width can't represent — this is the band to test calibration
    # against, and it is always inside the property's physical domain.
    predictive_interval_95: tuple[float, float]


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
    y_transform: str  # fit space for the target: "identity" | "log"
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
    # Posterior over the search range (1-D), for plotting. `grid_lower` /
    # `grid_upper` are the explicit 95% model band in physical units — correct
    # (and asymmetric) for a log-space fit, and clamped to the physical domain.
    # `grid_ci95` is a symmetric-half-width approximation kept for compatibility.
    grid_x: tuple[float, ...]
    grid_mean: tuple[float, ...]
    grid_ci95: tuple[float, ...]
    grid_lower: tuple[float, ...]
    grid_upper: tuple[float, ...]
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
    # Probabilistic regret bound: how likely it is, under this model, that the
    # best measured point is already within `epsilon` of the true optimum.
    # `epsilon` is in fit-space units (the same scale as `noise_threshold`).
    epsilon: float = 0.0
    delta: float = _DEFAULT_DELTA
    prob_within_epsilon: float = 0.0
    epsilon_delta_met: bool = False
    # How many observations the physics checks flagged as unreliable, and so
    # were down-weighted in the fit.
    n_unreliable: int = 0
    # Whether `noise_threshold` came from repeat measurements or from the
    # assumed relative noise. The whole convergence verdict is "the expected
    # gain is below the noise", so a reader is entitled to know which.
    noise_measured: bool = False


def _physical_band(
    mean_work: np.ndarray,
    std: np.ndarray,
    transform: str,
    y_min: float | None,
    y_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map the fit-space posterior back to physical units, clamped.

    Returns `(mean, lower, upper, ci95)`. The band is exact even when the log
    inverse makes it asymmetric; `ci95` is the symmetric half-width kept for
    compatibility with older payloads.
    """
    mean = _clamp(_inverse(mean_work, transform), y_min, y_max)
    lower = _clamp(_inverse(mean_work - _CI95 * std, transform), y_min, y_max)
    upper = _clamp(_inverse(mean_work + _CI95 * std, transform), y_min, y_max)
    return mean, lower, upper, (upper - lower) / 2.0


def _noise_scale(
    unreliable: np.ndarray | None, n_observations: int
) -> tuple[np.ndarray | None, int]:
    """Per-observation noise multipliers from the physics-check verdicts.

    Returns `(scale, n_flagged)`. `scale` is None when nothing was flagged,
    which keeps the single shared noise level and so reproduces earlier runs
    exactly.
    """
    if unreliable is None:
        return None, 0
    flags = np.asarray(unreliable, dtype=bool).reshape(-1)
    if flags.size != n_observations:
        raise ValueError("unreliable must have one entry per observation")
    n_flagged = int(flags.sum())
    if not n_flagged:
        return None, 0
    return np.where(flags, _UNRELIABLE_NOISE_FACTOR, 1.0), n_flagged


def _prob_within_epsilon(
    gp: GaussianProcessRegressor,
    grid_norm: np.ndarray,
    best_x_norm: float,
    epsilon: float,
    seed: int,
    n_draws: int = _N_POSTERIOR_DRAWS,
) -> float:
    """P(the best measured point is within `epsilon` of the true optimum).

    This is the probabilistic-regret-bound criterion of Wilson (NeurIPS 2024),
    estimated by Monte Carlo. We draw joint sample paths from the posterior;
    for each path the regret of the incumbent is `max(f) - f(x_best)`, and the
    answer is the fraction of paths where that regret is at most `epsilon`.

    Everything here is in the engine's internal space, where larger is always
    better, so `epsilon` is in the same units as `noise_std`.

    Wilson evaluates this with a random-feature approximation because exact
    sampling is cubic in the number of points. Our grid is small and 1-D, so
    we sample exactly and skip the approximation. Note his own caveat: the
    probability is conditional *on the model*. That is precisely why Latos
    reports it alongside the data-sufficiency grade rather than instead of it.
    """
    points = np.concatenate([grid_norm, [best_x_norm]]).reshape(-1, 1)
    draws = gp.sample_y(points, n_samples=n_draws, random_state=seed)
    incumbent = draws[-1, :]
    best_possible = draws[:-1, :].max(axis=0)
    regret = best_possible - incumbent
    return float(np.mean(regret <= epsilon))


def _noise_std(
    y_work: np.ndarray,
    rel_noise: float,
    transform: str,
    measured_noise: float | None = None,
    y_linear: np.ndarray | None = None,
) -> float:
    """Measurement-noise std in the GP's fit space.

    In log space a *relative* measurement error becomes a constant *additive*
    error (d(ln y) = dy/y), so the noise floor is simply `rel_noise`; in linear
    space it scales with the data magnitude.

    `measured_noise` is an observed repeatability in the property's own units,
    typically the scatter of repeat measurements on one sample. When the caller
    has that, it beats any assumed percentage, so it wins. It still has to be
    carried into the fit space: an absolute scatter is already right for a
    linear fit, but a log fit needs it as a fraction of the signal.
    """
    if measured_noise is not None and measured_noise > 0:
        if transform != _LOG:
            return float(measured_noise)
        scale = float(np.mean(np.abs(y_linear))) if y_linear is not None else 0.0
        if scale > 0:
            return float(measured_noise) / scale
        # Nothing sane to divide by; fall through to the assumption rather
        # than inventing a scale.
    if transform == _LOG:
        return rel_noise
    return rel_noise * float(np.mean(np.abs(y_work)))


def _build_gp(
    y: np.ndarray,
    noise_std: float,
    length_scale: float | None,
    seed: int,
    noise_scale: np.ndarray | None = None,
) -> GaussianProcessRegressor:
    """A GP with a smooth RBF trend and a realistic measurement-noise floor.

    `noise_std` is the absolute noise in `y`'s (fit-space) units — the caller
    computes it via `_noise_std` so log-space fits get the right floor. When
    `length_scale` is None it is fitted by marginal likelihood (within
    `_LS_BOUNDS`); a fixed value is what `length_scale_robustness()` sweeps.

    `noise_scale` (shape (n,)) multiplies the assumed measurement noise of
    individual observations. A point our physics checks flagged as
    implausible is not discarded — discarding data silently is its own kind
    of dishonesty — it is simply trusted less, which is what a larger error
    bar means. `None` keeps the single shared noise level, exactly as before.
    """
    alpha_scalar = (noise_std / max(float(np.std(y)), 1e-9)) ** 2
    alpha = alpha_scalar if noise_scale is None else alpha_scalar * noise_scale**2
    if length_scale is None:
        rbf = RBF(length_scale=_LS_INIT, length_scale_bounds=_LS_BOUNDS)
        n_restarts = _N_RESTARTS
    else:
        rbf = RBF(length_scale=length_scale, length_scale_bounds="fixed")
        n_restarts = 0
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * rbf
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )


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


def _recommend(
    grid: np.ndarray,
    rec_i: int,
    mean_int: np.ndarray,
    std: np.ndarray,
    *,
    sign: float,
    noise_std: float,
    transform: str,
    y_min: float | None,
    y_max: float | None,
) -> Recommendation:
    """Build the recommendation at grid index `rec_i`, mapped to physical units.

    The model and predictive intervals are formed in fit space, then inverse-
    transformed and clamped, so both are exact (and asymmetric under log) and
    inside the property's physical domain.
    """
    rec_sigma = float(std[rec_i])
    predictive_sd = float(np.sqrt(rec_sigma**2 + noise_std**2))
    pm_work = float(sign * mean_int[rec_i])
    lo_pred = _to_phys(pm_work - _CI95 * predictive_sd, transform, y_min, y_max)
    hi_pred = _to_phys(pm_work + _CI95 * predictive_sd, transform, y_min, y_max)
    lo_model = _to_phys(pm_work - _CI95 * rec_sigma, transform, y_min, y_max)
    hi_model = _to_phys(pm_work + _CI95 * rec_sigma, transform, y_min, y_max)
    return Recommendation(
        x=float(grid[rec_i]),
        predicted_mean=_to_phys(pm_work, transform, y_min, y_max),
        ci95=(hi_model - lo_model) / 2.0,
        predictive_sd=predictive_sd,
        ci95_predictive=(hi_pred - lo_pred) / 2.0,
        predictive_interval_95=(lo_pred, hi_pred),
    )


def _assess_reliability(
    x_norm: np.ndarray,
    y: np.ndarray,
    *,
    noise_std: float,
    seed: int,
) -> ReliabilityReport:
    """Count-tier + leave-one-out reliability of the model's intervals.

    Runs in the GP's fit space (`y` and `noise_std` are already transformed),
    so the coverage check is consistent with a log-space fit. The check asks
    the only question that matters: does the model's own 95% predictive
    interval contain the point it didn't see?

    Each fold refits the length-scale from scratch on the n-1 points it is
    allowed to see. Earlier this reused the length-scale fitted on the FULL
    series, which is cheaper but leaks: the held-out point helped choose the
    hyper-parameter that the fold then predicts it with, so the fold has
    partly seen the answer. That biases coverage OPTIMISTIC, which is the one
    direction a check against over-confidence must not be biased in. On the
    five-sample drop-impact series the leak was worth a whole fold, 4/5
    against 3/5.

    The cost is n hyper-parameter fits instead of n cheap ones. That is real
    but small, and correctness of this particular number is the product.
    """
    n = int(x_norm.size)
    inside = 0
    for i in range(n):
        mask = np.arange(n) != i
        gp = _build_gp(y[mask], noise_std, None, seed)
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
    y_transform: str = _IDENTITY,
    y_min: float | None = None,
    y_max: float | None = None,
    length_scale: float | None = None,
    rel_noise: float = _REL_NOISE,
    measured_noise: float | None = None,
    xi: float = _XI,
    grid_size: int = _GRID_SIZE,
    seed: int = 0,
    objective_aggregation: str = "peak",
    created_at: datetime | None = None,
    with_reliability: bool = True,
    unreliable: np.ndarray | None = None,
    epsilon: float | None = None,
    delta: float = _DEFAULT_DELTA,
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
        y_transform: "identity" (default) or "log". "log" fits a strictly-
            positive, order-of-magnitude target (mobility, conductivity, …) in
            log space, so the prediction and its interval can never be negative
            and the multiplicative noise is modelled correctly. Falls back to
            "identity" if the data contains a non-positive value.
        y_min: lower physical bound for the target (clamps the band/interval),
            or None. From the physics registry; keeps a positive property's band
            from ever dipping below zero.
        y_max: upper physical bound for the target, or None.
        length_scale: Fix the RBF length-scale to this value; if None it is
            fitted from the data. Fixing it is how `length_scale_robustness`
            probes whether the recommendation is a kernel artifact.
        rel_noise: Relative measurement noise injected into the GP. Also
            sets the convergence floor: when the best expected improvement
            falls below this noise level, no experiment can *reliably* do
            better, so we report converged (a heuristic, not a guarantee).
            Only consulted when `measured_noise` is absent.
        measured_noise: Observed repeatability of the measurement, in the
            target's own units, typically the pooled scatter of repeats on the
            same sample. Overrides `rel_noise` when given: a measured noise
            floor is evidence, a percentage is a guess, and the convergence
            verdict rests entirely on this number.
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
        unreliable: Optional bool mask, shape (n,), True where a physics
            check rejected that measurement. Flagged points are fitted with
            a larger assumed noise rather than dropped, so the recommendation
            stops chasing numbers the physics layer does not believe.
        epsilon: Tolerance for the "already good enough" statement, in
            fit-space units. Defaults to the measurement-noise floor, i.e.
            "within one measurement noise of the optimum".
        delta: Risk level for that statement; `epsilon_delta_met` is True
            when the probability reaches 1 - delta.

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

    # Physics layer: fit a strictly-positive, order-of-magnitude target in log
    # space so the surrogate can never predict a negative value. Fall back to
    # linear if a log target carries a non-positive value (a bad measurement,
    # flagged elsewhere) so the fit never crashes.
    transform = y_transform if y_transform in _TRANSFORMS else _IDENTITY
    if transform == _LOG and bool(np.any(y <= 0)):
        transform = _IDENTITY
    y_work = _forward(y, transform)
    noise_std = _noise_std(y_work, rel_noise, transform, measured_noise, y)

    # Minimization is exact negation in the (possibly log) fit space.
    sign = 1.0 if direction == "maximize" else -1.0
    y_int = sign * y_work

    noise_scale, n_unreliable = _noise_scale(unreliable, y_int.size)
    gp = _build_gp(y_int, noise_std, length_scale, seed, noise_scale=noise_scale)
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

    # Undo the direction flip, then map the posterior back to physical units,
    # clamped to the property's domain. The band [lower, upper] is exact even
    # when the log inverse makes it asymmetric.
    mean_work = sign * mean_int
    grid_mean, grid_lower, grid_upper, grid_ci95 = _physical_band(
        mean_work, std, transform, y_min, y_max
    )

    ei_i = int(np.argmax(ei))
    max_ei = float(ei[ei_i])
    # Stopping rule (in fit space): the improvement signal is *exhausted* when
    # the best expected improvement is smaller than the measurement-noise floor
    # — no experiment can then reliably do better. This is necessary for
    # convergence but not sufficient (see the reliability gate below).
    noise_threshold = noise_std
    signal_exhausted = max_ei < noise_threshold

    # How likely it is that we are already done, stated as a probability
    # rather than as a yes/no. "Within one measurement noise of the optimum"
    # is the natural tolerance for an experimentalist, so that is the default.
    eps = float(epsilon) if epsilon is not None else noise_std
    prob_within = _prob_within_epsilon(gp, grid_norm, float(x_norm[best_i]), eps, seed)

    config = BoConfig(
        objective=target_name,
        direction=direction,
        y_transform=transform,
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
            noise_std=noise_std,
            seed=seed,
        )

    # Reliability-aware convergence and exploration. When the improvement
    # signal is exhausted but the data is still exploratory, a flat EI does not
    # mean "optimum found" — the surrogate is simply uninformative in the gaps
    # it never sampled. Two consequences: (1) do not report convergence (a
    # false stop), and (2) recommend the point of greatest posterior
    # uncertainty (the largest unmeasured gap), which is the most informative
    # next experiment, rather than the max-EI point sitting beside the current
    # best. Otherwise recommend the max-EI (exploit) point. When reliability
    # was not assessed (the robustness sweep, which reads neither field), fall
    # back to the plain max-EI pick.
    is_exploratory = reliability is not None and reliability.level == "exploratory"
    converged = signal_exhausted and not is_exploratory
    explore = signal_exhausted and is_exploratory
    rec_i = int(np.argmax(std)) if explore else ei_i
    recommendation = _recommend(
        grid,
        rec_i,
        mean_int,
        std,
        sign=sign,
        noise_std=noise_std,
        transform=transform,
        y_min=y_min,
        y_max=y_max,
    )

    return OptimizationResult(
        input_name=input_name,
        target_name=target_name,
        grid_x=tuple(grid.tolist()),
        grid_mean=tuple(grid_mean.tolist()),
        grid_ci95=tuple(grid_ci95.tolist()),
        grid_lower=tuple(grid_lower.tolist()),
        grid_upper=tuple(grid_upper.tolist()),
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
        epsilon=eps,
        delta=delta,
        prob_within_epsilon=prob_within,
        epsilon_delta_met=prob_within >= 1.0 - delta,
        n_unreliable=n_unreliable,
        noise_measured=bool(measured_noise is not None and measured_noise > 0),
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
    y_transform: str = _IDENTITY,
    y_min: float | None = None,
    y_max: float | None = None,
    rel_noise: float = _REL_NOISE,
    xi: float = _XI,
    grid_size: int = _GRID_SIZE,
    seed: int = 0,
    tol_frac: float = _ROBUSTNESS_TOL_FRAC,
) -> RobustnessReport:
    """Re-run the optimization at several fixed length-scales and compare.

    Returns a `RobustnessReport` whose `stable` flag is True when the
    recommended point varies by at most `tol_frac` of the search span. The
    y-transform/domain must match the main run so the sweep is comparable.
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
            y_transform=y_transform,
            y_min=y_min,
            y_max=y_max,
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
