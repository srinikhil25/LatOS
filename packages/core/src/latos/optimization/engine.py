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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

# Single-threaded BLAS keeps memory tiny for these small problems and
# avoids OpenBLAS over-allocating on constrained boxes. Set before numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

__all__ = [
    "BoConfig",
    "OptimizationError",
    "OptimizationResult",
    "Recommendation",
    "ReliabilityReport",
    "RobustnessEntry",
    "RobustnessReport",
    "StoppingVerdict",
    "length_scale_robustness",
    "optimize",
]

# Defaults
_REL_NOISE = 0.08  # ~8% relative measurement noise (zT-typical, defensible)
_XI = 0.01  # EI exploration sweetener
_GRID_SIZE = 200
_MIN_POINTS = 3  # a GP over fewer than 3 points isn't worth trusting
_LS_INIT = 1.5  # RBF length-scale starting point when fitted
# Bounds the length-scale is fitted within, in units where a search span is
# `_SPAN_UNITS`. The floor is a guard rail: it stops a GP fitting wiggles that
# sparse data cannot support.
#
# One dimension keeps the historical 1.0. That is not caution for its own sake —
# lowering it visibly changes the one-variable answer. On the frozen 4-point
# drop-impact record a 0.2 floor lets the fit fall to l = 0.29, which widens the
# 95% predictive interval from [31.2, 57.4] to [30.4, 79.1] and drops
# leave-one-out coverage from 3-of-4 to 2-of-4. Wider intervals and worse
# calibration on four points is the model overfitting, not resolving: there is
# no 1-D evidence for a lower floor, and those numbers are pre-registered.
_LS_BOUNDS = (1.0, 5.0)

# Multi-dimensional fits want a much lower floor, and here there IS evidence.
# Simple regret over eight seeds, floor swept with everything else held fixed:
#
#            Branin (2-D)              Hartmann-3 (3-D)
#   floor    median    max             median    max      mean
#   1.00     0.1258    0.5403          0.0170    0.7748   0.1351
#   0.30     0.0939    0.4131          0.0115    0.8351   0.1131
#   0.20     0.0939    0.4132          0.0105    0.3944   0.0574
#   0.10     0.0939    0.4132          0.0112    0.3937   0.0575
#
# Each benchmark saturates — Branin by 0.3, Hartmann-3 by 0.2 — and below its
# saturation point the floor stops binding, so nothing further changes. 0.2 is
# therefore the *largest* value that captures the whole measured gain, which is
# the one to want: maximum guard rail for zero cost. Worst-case regret on
# Hartmann-3 roughly halves (0.775 -> 0.394).
#
# That higher dimensions want a lower floor is consistent with structure
# appearing finer along each normalized axis as axes are added, and with Xu et
# al. (STAM Methods 3, 2210251, 2023), who independently measured ~2% of span as
# optimal for 2-D/3-D synthesis. 0.2/4.0 is 5% — the same order.
_LS_BOUNDS_ND = (0.2, 5.0)
_N_RESTARTS = 8  # marginal-likelihood restarts when the length-scale is fitted

# Kernel family for the stationary factor. RBF is what the engine shipped with.
# Matern 5/2 is what the field uses almost universally (Snoek, Liang, Rohr,
# Makarova, Hvarfner, Ishibashi, Shields) and two results argue for it here:
# Wang, Tuo & Wu prove a correlation function no smoother than the truth is
# more robust under misspecification, and Srinivas et al.'s GP-UCB regret bound
# needs nu > 2 — 5/2 is the smallest half-integer satisfying both. This is a
# switch rather than a swap because the claim is testable and the shipped
# behaviour has to stay reproducible while it is being tested.
_KERNELS = ("rbf", "matern52")
_MATERN_NU = 2.5
# The shipped default. Every entry point and helper reads this one name, so the
# 1-D and N-D paths cannot drift apart.
#
# Kept at RBF on 2026-09-02, against the field's near-universal preference for
# Matern 5/2, because the in-house harness measured the opposite (8 seeds per
# arm, simple regret, median / worst):
#
#     branin      rbf 0.0708 / 0.3509    matern52 0.1665 / 1.1385
#     hartmann3   rbf 0.0079 / 0.7836    matern52 0.0028 / 0.7739
#
# Matern wins 7/8 seeds on Hartmann-3 but only 2/8 on Branin, where it triples
# the worst case. Rohr's warning applies directly - the floor for deleterious
# effects is deeper than the ceiling for gain - so a 3x worse tail is not paid
# for by a median gain on one benchmark of two. The theory argument for Matern
# (Wang/Tuo/Wu robustness under misspecification, Srinivas' nu > 2) is noted and
# not acted on; AX4 settles it across process-window shapes.
_DEFAULT_KERNEL = "rbf"

# What to recommend when the improvement signal is exhausted but the data is
# still exploratory (the branch near the end of `optimize`):
#   "max_std"  argmax posterior sd — the largest unmeasured gap. Shipped.
#   "ei"       no fallback: keep the max-EI pick and let EI decide alone.
#   "ucb"      argmax(mean + _UCB_LAMBDA * sd) — optimism, between the two.
# Note this governs only *where to point*. Whether the engine declares
# convergence stays gated on the reliability tier either way, which is the part
# three independent stopping papers support.
_EXPLORE_POLICIES = ("max_std", "ei", "ucb")
# Kept at "max_std" on 2026-09-02, after a measurement that REFUTED the case for
# changing it. Four reviewed papers (Borg, Rohr, Srinivas, Shields) measure pure
# uncertainty sampling as the weakest available policy, and that was read here as
# an argument to switch the fallback to "ei". Measured on Forrester 1-D from
# n_initial = 4 - the tier this branch actually fires in, and it fired on 96/96
# rounds - with 12 seeds, simple regret, median / worst:
#
#     max_std  0.0220 / 0.1735
#     ei       0.0559 / 6.0212
#     ucb      0.0938 / 6.0212
#
# A worst case of 6.02 on a function whose range is ~6.02 means those campaigns
# never left the flat region. The papers measure pure exploration as a WHOLE
# STRATEGY over long campaigns; here it is a tie-break that fires only when EI is
# already flat and the data is still exploratory, and at n = 4-12 in one
# dimension "go to the biggest unmeasured gap" is simply the right move. The
# literature result does not transfer to this branch, and the switch would have
# been a regression with a 35x worse tail.
_DEFAULT_EXPLORE_POLICY = "max_std"
_UCB_LAMBDA = 2.0

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

# The count tiers above are dimension-blind, and counting is the wrong measure
# the moment there is more than one axis: ten points along a line is sparse,
# ten points scattered over a plane is very much sparser, and ten points
# clustered in one corner tells you nothing about the rest of the box however
# many there are.
#
# Fill distance is the quantity that actually governs it — the largest distance
# from anywhere in the search box to the nearest observation, i.e. the radius of
# the biggest unsampled hole. It is what bounds interpolation error in scattered
# data theory, it is insensitive to how the points are counted, and it scales
# with dimension on its own: filling a box to a given radius needs a number of
# points that grows as the volume does.
#
# The two limits below are the *same* thresholds as the counts, restated
# geometrically. n evenly spaced points spanning _SPAN_UNITS (endpoints
# included) leave a fill distance of half the gap, _SPAN_UNITS / (2(n-1)), so
# in one dimension with well-spread data this rule and the count rule agree by
# construction. In higher dimensions they diverge, which is the point.
_FILL_INDICATIVE = _SPAN_UNITS / (2 * (_RELIABILITY_INDICATIVE_N - 1))
_FILL_CALIBRATED = _SPAN_UNITS / (2 * (_RELIABILITY_CALIBRATED_N - 1))
_FILL_PROBE = 2**14  # probe points for the multi-dimensional fill estimate
# Boundary slack. Because the limits are derived to agree with the counts
# exactly, n evenly spaced points land precisely ON their own threshold, and
# float arithmetic then settles the tier on the last bit of the mantissa —
# ten evenly spaced points would fail the tier they define. Comparing against
# (1 + tol) makes the two rules agree at the boundary instead of fighting over
# one ULP.
_FILL_TOL = 1e-6
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
# Floor on a per-point noise multiplier, as a fraction of the median multiplier
# in the same campaign. A reported standard error of zero claims one observation
# is exact; a GP handed a noiseless point interpolates it exactly and lets that
# single optimistic error bar dominate every neighbouring prediction. Small
# uncertainties are believed, impossible ones are not.
_MIN_POINT_NOISE_FRACTION = 0.05
# Posterior draws behind the (epsilon, delta) statement. 512 puts the standard
# error of the reported probability near 2%, which is finer than we quote it.
_N_POSTERIOR_DRAWS = 512
# Points the joint posterior draw is taken over. Exact sampling is cubic in this
# number, and the 1-D grid (200) sits below the cap so that path is untouched;
# the multi-variable candidate set (2048) is cut down to it. See
# `_prob_within_epsilon` for why the selection is by upper confidence bound.
# 512 rather than a smaller, faster cap because that is where the measurement
# put it: averaged over six seeds the capped estimate sits within 0.5x the
# estimator's own Monte-Carlo error of the uncapped one, whereas 256 drifted to
# 1.5x on sparse data. Cheap is only worth having if the answer survives.
_PROB_MAX_POINTS = 512
_PROB_UCB_SIGMAS = 4.0  # a candidate this far above its mean is still competitive
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
    xi: float  # EI exploration sweetener, as a FRACTION of the observed spread
    rel_noise: float  # relative measurement-noise level fed to the GP
    noise_std: float  # absolute measurement-noise std, in objective units
    n_observations: int  # number of measured points the GP was fit to
    grid_size: int  # resolution of the posterior/acquisition grid
    seed: int  # RNG seed (GP restarts) — makes the fit reproducible
    created_at: datetime  # when this recommendation was produced
    # `xi` was an absolute value in objective units until 2026-08-10, so a bare
    # `xi = 0.01` is ambiguous across versions. Recording what it resolved to
    # disambiguates every record: absent means the old absolute reading.
    xi_absolute: float | None = None  # `xi` in objective units, as applied
    # True when per-observation standard deviations were supplied. Without it
    # two runs could carry byte-identical configs and still have produced
    # different recommendations, because a heteroscedastic fit weighs the same
    # points differently. A frozen record that cannot tell those apart is not a
    # record of how the answer was produced.
    point_noise_used: bool = False
    # ...and the weights themselves, in the same order as the observations.
    # The flag above says the fit was heteroscedastic; this says with what, so
    # the pre-registration's training-data digest covers the weighting too.
    # Effective per-point sd is `noise_std * point_noise_scale[i]`.
    point_noise_scale: tuple[float, ...] | None = None


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
    # Input dimensionality the grade was computed over.
    n_dims: int = 1
    # Radius of the largest unsampled hole in the search box, in normalized
    # units where every search range spans `_SPAN_UNITS`. This is what makes
    # the grade dimension-aware: counting points cannot tell a well-covered
    # plane from a line of points inside one, and `fill_distance` can.
    # `fill_limit` is the threshold the level was held to, so a reader can see
    # how far off the data is without knowing the constants.
    fill_distance: float = 0.0
    fill_limit: float = 0.0


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
    # The headline: one action, with the reasoning behind it. The four
    # fields above are its inputs and stay for callers that want them.
    stopping: StoppingVerdict | None = None
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


def _validate_point_noise(point_noise: np.ndarray | None, n_observations: int) -> np.ndarray | None:
    """Per-observation measurement uncertainties, checked and returned as floats.

    Rejects rather than repairs. A negative or non-finite standard deviation is
    not a value to clamp, it is a caller bug, and quietly substituting something
    plausible would hand the surrogate a confidence nobody computed.
    """
    if point_noise is None:
        return None
    sigma = np.asarray(point_noise, dtype=float).reshape(-1)
    if sigma.size != n_observations:
        raise OptimizationError(
            f"point_noise must have one entry per observation: got {sigma.size} "
            f"for {n_observations} points."
        )
    if not np.all(np.isfinite(sigma)):
        raise OptimizationError("point_noise must be finite; got NaN or infinity.")
    if np.any(sigma < 0):
        raise OptimizationError("point_noise must be non-negative; got a negative value.")
    if not np.any(sigma > 0):
        raise OptimizationError("point_noise is zero everywhere, which claims perfect data.")
    return sigma


def _point_noise_scale(
    sigma: np.ndarray | None,
    *,
    noise_std: float,
    transform: str,
    y_linear: np.ndarray,
) -> np.ndarray | None:
    """Turn measured per-point uncertainties into multipliers on the shared noise.

    Reliability has reached the surrogate as one bit per datapoint: a physics
    check either flagged an observation or it did not, and a flagged one had its
    error bar widened by a fixed factor. That throws away a number the analysis
    layer already computes. A fitted slope arrives with a standard error, a
    derived quantity carries propagated uncertainty, and a spread across
    modelling choices is itself a measurement of how well a value is known.
    Passing those through makes the Gaussian process heteroscedastic by
    construction rather than by category.

    The multiplier is `σᵢ / noise_std`, because `_build_gp` forms the diagonal
    as `(noise_std / std(y))² · scale²`; the shared term then cancels and each
    point contributes exactly `(σᵢ / std(y))²`.

    Both terms have to live in the same space. In log space a GP sees fractional
    error, so an absolute `σᵢ` is divided by that observation's own magnitude
    rather than by the series mean — which is the point of doing this per point.

    A σ of zero is lifted to a small floor. Zero would assert that one
    observation is exact, and a GP handed a noiseless point interpolates it
    exactly, letting a single optimistic error bar dominate the fit.
    """
    if sigma is None:
        return None

    if transform == _LOG:
        magnitude = np.abs(np.asarray(y_linear, dtype=float))
        safe = np.where(magnitude > 0, magnitude, 1.0)
        sigma_fit = sigma / safe
    else:
        sigma_fit = sigma

    if noise_std <= 0:
        return None
    scale = sigma_fit / noise_std
    floor = _MIN_POINT_NOISE_FRACTION * float(np.median(scale[scale > 0]))
    return np.maximum(scale, floor)


STOP = "stop"
CONFIRM = "confirm"
CONTINUE = "continue"


@dataclass(frozen=True, slots=True)
class StoppingVerdict:
    """Should another experiment be run? One answer, with the grounds for it.

    Everything needed to answer this was already computed, spread across four
    fields a caller had to combine correctly. In a project whose premise is
    spending very few experiments, "am I done?" deserves to be the headline of a
    recommendation rather than something reconstructed from `converged`,
    `max_ei`, `prob_within_epsilon` and the reliability grade.

    Two independent lines of evidence bear on it, and they can disagree:

    * the **probabilistic regret bound** -- how likely the best sample already
      taken is to sit within `epsilon` of the true optimum
    * the **data-sufficiency grade** -- whether the model that produced that
      probability has enough coverage to be believed at all

    When they agree the answer is easy. When the probability is high and the
    grade is still exploratory, neither "stop" nor "keep exploring" is honest:
    the model says it has found the answer, and separately says it is not yet
    trustworthy enough for that claim to stand alone. That case is CONFIRM --
    repeat the incumbent and let the two lines settle it, which costs one
    experiment where continued exploration costs several.

    Measured behaviour that motivated this: on a single-peak objective sampled
    at six points including the peak, the engine reported probability 0.992,
    signal exhausted, `converged=False`, and recommended the far edge of the
    search space. Every number was right and the advice was wrong.
    """

    action: str  # STOP, CONFIRM or CONTINUE
    probability: float  # P(incumbent is within `epsilon` of the optimum)
    epsilon: float  # tolerance the probability is stated against
    delta: float  # risk level; the claim holds at 1 - delta confidence
    signal_exhausted: bool  # no expected improvement left above the noise floor
    data_sufficient: bool  # the reliability grade is past "exploratory"
    reason: str  # one sentence, addressed to the experimentalist

    @property
    def should_stop(self) -> bool:
        """True only for an unambiguous stop, never for a contested one."""
        return self.action == STOP


def _stopping_verdict(
    *,
    probability: float,
    epsilon: float,
    delta: float,
    signal_exhausted: bool,
    reliability: ReliabilityReport | None,
    best_label: str,
) -> StoppingVerdict:
    """Turn the four stopping signals into one statement a person can act on."""
    met = probability >= 1.0 - delta
    data_sufficient = reliability is not None and reliability.level != "exploratory"

    if signal_exhausted and met and data_sufficient:
        action = STOP
        reason = (
            f"Stop. The best sample so far ({best_label}) is within {epsilon:.3g} of the "
            f"optimum with probability {probability:.2f}, no remaining experiment offers "
            "improvement above the measurement noise, and the data supports the claim."
        )
    elif met and not data_sufficient:
        action = CONFIRM
        reason = (
            f"Confirm before stopping. The model puts the best sample so far "
            f"({best_label}) within {epsilon:.3g} of the optimum with probability "
            f"{probability:.2f}, but there is too little data for that claim to stand on "
            "its own. Repeating the incumbent settles it in one experiment; continued "
            "exploration would cost several."
        )
    elif signal_exhausted:
        action = CONTINUE
        reason = (
            f"Keep going. Expected improvement has fallen below the noise floor, but the "
            f"best sample so far ({best_label}) is within {epsilon:.3g} of the optimum "
            f"with probability only {probability:.2f}. A flat acquisition here means the "
            "model is uninformative in the gaps it never sampled, not that the optimum "
            "is found."
        )
    else:
        action = CONTINUE
        reason = (
            f"Keep going. Expected improvement still exceeds the measurement noise, and "
            f"the best sample so far ({best_label}) is within {epsilon:.3g} of the "
            f"optimum with probability {probability:.2f}."
        )

    return StoppingVerdict(
        action=action,
        probability=float(probability),
        epsilon=float(epsilon),
        delta=float(delta),
        signal_exhausted=bool(signal_exhausted),
        data_sufficient=bool(data_sufficient),
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class _NoiseModel:
    """How much each observation is trusted, and on what grounds.

    Extracted so the two entry points cannot disagree. Both need the same five
    facts, and assembling them inline twice is how the 1-D and d-D paths drifted
    apart before `_fit_surrogate` was pulled out for the same reason.
    """

    std: float  # the shared level, in fit-space units
    scale: np.ndarray | None  # per-observation multipliers on it
    n_unreliable: int  # observations a physics check rejected
    measured: bool  # the level came from data, not from an assumed percentage
    per_point: bool  # per-observation standard deviations were supplied


def _noise_model(
    y: np.ndarray,
    y_work: np.ndarray,
    *,
    rel_noise: float,
    measured_noise: float | None,
    point_noise: np.ndarray | None,
    unreliable: np.ndarray | None,
    transform: str,
) -> _NoiseModel:
    """Assemble the shared noise level and the per-observation multipliers.

    A per-point series also has to supply the single number the rest of the
    engine reasons with, since the exploration sweetener, the convergence floor
    and the epsilon statement are all scalar. The median is the representative
    one, chosen over the mean so that a single very uncertain observation cannot
    inflate the level the whole campaign is judged against.
    """
    sigma = _validate_point_noise(point_noise, y.size)
    if sigma is not None and measured_noise is None:
        measured_noise = float(np.median(sigma))

    noise_std = _noise_std(y_work, rel_noise, transform, measured_noise, y)
    from_flags, n_unreliable = _noise_scale(unreliable, y.size)
    scale = _merge_noise_scales(
        from_flags,
        _point_noise_scale(sigma, noise_std=noise_std, transform=transform, y_linear=y),
    )
    return _NoiseModel(
        std=noise_std,
        scale=scale,
        n_unreliable=n_unreliable,
        measured=bool(measured_noise is not None and measured_noise > 0),
        per_point=sigma is not None,
    )


def _merge_noise_scales(
    from_flags: np.ndarray | None, from_points: np.ndarray | None
) -> np.ndarray | None:
    """Combine the two reasons an observation's error bar might be widened.

    They are not the same claim, so they multiply rather than compete. A per-
    point standard error says how repeatable a measurement was; a physics flag
    says the value is inconsistent with something that must hold regardless of
    how carefully it was taken. A precisely-measured impossible number deserves
    both penalties — the precision is exactly what makes it worth distrusting.
    """
    if from_flags is None:
        return from_points
    if from_points is None:
        return from_flags
    return np.asarray(from_flags * from_points, dtype=float)


def _prob_within_epsilon(
    gp: GaussianProcessRegressor,
    grid_norm: np.ndarray,
    best_x_norm: float,
    epsilon: float,
    seed: int,
    *,
    n_draws: int = _N_POSTERIOR_DRAWS,
    max_points: int = _PROB_MAX_POINTS,
) -> float:
    """P(the best measured point is within `epsilon` of the true optimum).

    This is the probabilistic-regret-bound criterion of Wilson (NeurIPS 2024),
    estimated by Monte Carlo. We draw joint sample paths from the posterior;
    for each path the regret of the incumbent is `max(f) - f(x_best)`, and the
    answer is the fraction of paths where that regret is at most `epsilon`.

    Everything here is in the engine's internal space, where larger is always
    better, so `epsilon` is in the same units as `noise_std`.

    Wilson evaluates this with a random-feature approximation because exact
    sampling is cubic in the number of points. Note his own caveat: the
    probability is conditional *on the model*. That is precisely why Latos
    reports it alongside the data-sufficiency grade rather than instead of it.

    Rather than approximate the sampler, we shrink what is sampled over. Only
    candidates that could plausibly *be* the maximum affect `max(f)`; one whose
    posterior sits far below the incumbent cannot change the answer however many
    paths are drawn. So when the candidate set is large it is cut to the top
    `max_points` by an upper confidence bound, which keeps exactly the points
    that drive the statistic.

    This is not a micro-optimisation. `optimize()` evaluates on a 200-point grid,
    where exact sampling is free, and the original version of this function was
    written for that case. `optimize_nd` maximises over 2048 Sobol candidates,
    and 2048^3 against 200^3 is roughly a thousandfold more work in the cubic
    term — enough that the multi-variable endpoint spent most of its time here,
    computing a number many callers never read. The 1-D path is unaffected: 200
    is already below the cap, so its results are bit-identical.

    Selecting by upper confidence bound rather than at random matters. A random
    subsample would usually miss the peak region entirely and report a
    confidently wrong probability; taking the most competitive points keeps the
    bias at or below half the estimator's own Monte-Carlo error, measured over
    six seeds at n = 10, 25 and 60 observations.
    """
    grid = np.asarray(grid_norm, dtype=float)
    grid = grid.reshape(grid.shape[0], -1)
    best = np.asarray(best_x_norm, dtype=float).reshape(1, -1)
    if grid.shape[0] > max_points:
        mean, sd = gp.predict(grid, return_std=True)
        competitive = np.argpartition(mean + _PROB_UCB_SIGMAS * sd, -max_points)[-max_points:]
        grid = grid[competitive]
    points = np.vstack([grid, best])
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


def _kernel_label(kernel: str, *, ard: bool = False) -> str:
    """How a fitted kernel is named in a result.

    One helper for both entry points: `optimize_nd` used to hardcode "RBF" here,
    so flipping the default would have made the result describe a model it was
    not using.
    """
    family = "Matern(nu=5/2)" if kernel == "matern52" else "RBF"
    return f"ConstantKernel * {family}{'(ARD)' if ard else ''}"


def _stationary(
    kernel: str,
    length_scale: float | list[float],
    bounds: tuple[float, float] | str,
) -> Any:
    """The stationary factor of the kernel — RBF, or Matern 5/2.

    Both take the same `length_scale` / `length_scale_bounds` contract, so
    everything downstream (`_fitted_length_scales`, the robustness sweep, the
    ARD list form) is unchanged by the choice.
    """
    if kernel == "matern52":
        return Matern(length_scale=length_scale, length_scale_bounds=bounds, nu=_MATERN_NU)
    return RBF(length_scale=length_scale, length_scale_bounds=bounds)


def _build_gp(
    y: np.ndarray,
    noise_std: float,
    length_scale: float | None,
    seed: int,
    *,
    noise_scale: np.ndarray | None = None,
    n_dims: int = 1,
    ls_bounds: tuple[float, float] = _LS_BOUNDS,
    kernel: str = _DEFAULT_KERNEL,
) -> GaussianProcessRegressor:
    """A GP with a smooth stationary trend and a realistic measurement-noise floor.

    `noise_std` is the absolute noise in `y`'s (fit-space) units — the caller
    computes it via `_noise_std` so log-space fits get the right floor. When
    `length_scale` is None it is fitted by marginal likelihood (within
    `_LS_BOUNDS`); a fixed value is what `length_scale_robustness()` sweeps.

    `noise_scale` (shape (n,)) multiplies the assumed measurement noise of
    individual observations. A point our physics checks flagged as
    implausible is not discarded — discarding data silently is its own kind
    of dishonesty — it is simply trusted less, which is what a larger error
    bar means. `None` keeps the single shared noise level, exactly as before.

    `n_dims` > 1 switches the kernel to ARD: one length-scale per input axis
    instead of one shared value. That is what lets the model say *which*
    variable the property actually responds to, and an isotropic kernel
    cannot express it. `n_dims == 1` reproduces the scalar kernel exactly, so
    every existing caller is bit-for-bit unaffected.

    `kernel` selects the stationary factor (`_KERNELS`). The default "rbf" is
    the shipped behaviour, bit-for-bit.
    """
    alpha_scalar = (noise_std / max(float(np.std(y)), 1e-9)) ** 2
    alpha = alpha_scalar if noise_scale is None else alpha_scalar * noise_scale**2
    if length_scale is None:
        start = min(max(_LS_INIT, ls_bounds[0]), ls_bounds[1])
        init = start if n_dims == 1 else [start] * n_dims
        shape = _stationary(kernel, init, ls_bounds)
        n_restarts = _N_RESTARTS
    else:
        fixed = length_scale if n_dims == 1 else [length_scale] * n_dims
        shape = _stationary(kernel, fixed, "fixed")
        n_restarts = 0
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-2, 1e2)) * shape,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )


def _fitted_length_scale(gp: GaussianProcessRegressor) -> float:
    """Read the RBF length-scale back out of a fitted GP kernel.

    Scalar for an isotropic kernel. An ARD kernel holds one per axis; this
    returns the first so the 1-D contract is unchanged — `_fitted_length_scales`
    is what multi-dimensional callers want.
    """
    scales = _fitted_length_scales(gp)
    return scales[0] if scales else float("nan")


def _fitted_length_scales(gp: GaussianProcessRegressor) -> tuple[float, ...]:
    """Every fitted RBF length-scale, one per input axis (ARD) or one total."""
    rbf = getattr(gp.kernel_, "k2", None)
    length_scale = getattr(rbf, "length_scale", None)
    if length_scale is None:
        return ()
    try:
        return tuple(float(v) for v in np.atleast_1d(length_scale))
    except (TypeError, ValueError):
        return ()


def _prior_offsets(
    prior_mean: Callable[[np.ndarray], np.ndarray],
    x_norm: np.ndarray,
    *,
    lows: np.ndarray,
    x_scales: np.ndarray,
    sign: float,
    transform: str,
    ravel_input: bool,
) -> np.ndarray:
    """Evaluate a physical prior at normalized coordinates, in internal space.

    The caller writes `prior_mean` in the units they think in — real parameter
    values in, real property values out. Everything the engine does to the
    target afterwards (the log transform, the sign flip for minimization) has
    to be applied to the prior too, or the residual the GP fits is a mixture of
    two different spaces. Doing that conversion in one place is why this is a
    function rather than three lines at each call site.
    """
    pts = np.asarray(x_norm, dtype=float)
    pts = pts.reshape(pts.shape[0], -1)
    x_orig = lows + pts * x_scales
    raw = np.asarray(prior_mean(x_orig[:, 0] if ravel_input else x_orig), dtype=float).ravel()

    if raw.shape[0] != pts.shape[0]:
        raise ValueError(f"prior_mean returned {raw.shape[0]} values for {pts.shape[0]} points.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("prior_mean returned a non-finite value.")
    # A log-space fit of a prior that predicts zero or negative is not a warning
    # case: log(<=0) is undefined and silently falling back to linear would fit
    # the residual against a *different* prior than the one reported.
    if transform == _LOG and bool(np.any(raw <= 0)):
        raise ValueError(
            "prior_mean returned a non-positive value but the target is fitted "
            "in log space. Give a strictly-positive prior, or pass "
            'y_transform="identity".'
        )
    return sign * _forward(raw, transform)


def _prior_offsets_at(
    prior_mean: Callable[[np.ndarray], np.ndarray],
    *,
    lows: np.ndarray,
    x_scales: np.ndarray,
    sign: float,
    transform: str,
    ravel_input: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    """Bind a prior to one problem's geometry, leaving a function of points."""

    def offsets(x_norm: np.ndarray) -> np.ndarray:
        return _prior_offsets(
            prior_mean,
            x_norm,
            lows=lows,
            x_scales=x_scales,
            sign=sign,
            transform=transform,
            ravel_input=ravel_input,
        )

    return offsets


class _Surrogate(Protocol):
    """The three things every consumer in this module asks of a fitted model.

    `_fit_surrogate` hands back either a bare `GaussianProcessRegressor` or a
    `_PriorMeanGP` wrapping one, and nothing downstream cares which. That
    contract used to be typed as `object`, which is true but useless: it made
    every `gp.predict(...)` an attribute error under strict typing. Naming it
    keeps the return honest and documents, in one place, exactly how narrow the
    surface is that the prior wrapper has to reproduce.

    `predict` is annotated loosely on purpose. It returns an array normally and
    a `(mean, std)` pair when `return_std=True`, which is scikit-learn's own
    convention; spelling that as a union would break every call site that
    unpacks two values.
    """

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:  # noqa: N803
        ...

    def sample_y(
        self,
        X: np.ndarray,  # noqa: N803
        n_samples: int = 1,
        random_state: int = 0,
    ) -> np.ndarray: ...

    @property
    def kernel_(self) -> Any: ...


class _PriorMeanGP:
    """A fitted GP whose posterior mean is shifted by a deterministic prior.

    A stock `GaussianProcessRegressor` assumes the function is zero-mean away
    from the data (after `normalize_y`), which is why a four-point campaign
    recommends the middle of the widest gap: with no structure to extrapolate,
    the only thing that varies across the space is how little we know. Fitting
    the GP to `observed - physics(x)` and adding `physics(x)` back on prediction
    replaces that flat default with a real curve, so the acquisition function is
    reasoning about the material rather than about spacing.

    The prior is deterministic, so it moves the mean and leaves the standard
    deviation alone — the uncertainty still comes from the data, and a wrong
    prior therefore shows up as a large residual the GP has to absorb rather
    than as false confidence.

    This wraps rather than subclasses because every consumer in this module
    reaches for exactly three things — `predict`, `sample_y` and `kernel_`. A
    wrapper keeps the acquisition, the L-BFGS-B polish, the posterior surface
    and the regret bound working unchanged, which is the whole point: the prior
    must not become a special case threaded through a dozen call sites.
    """

    __slots__ = ("_gp", "_offsets")

    def __init__(
        self, gp: GaussianProcessRegressor, offsets: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        self._gp = gp
        self._offsets = offsets

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:  # noqa: N803
        shift = self._offsets(X)
        if return_std:
            mean, std = self._gp.predict(X, return_std=True)
            return mean + shift, std
        return self._gp.predict(X) + shift

    def sample_y(
        self,
        X: np.ndarray,  # noqa: N803
        n_samples: int = 1,
        random_state: int = 0,
    ) -> np.ndarray:
        """Joint posterior draws, each shifted by the prior.

        `_prob_within_epsilon` takes the regret of the incumbent across sample
        paths. Shifting every path by the same deterministic curve is *not* a
        no-op there: the incumbent and the running maximum sit at different
        points, so the offset does not cancel out of `max(f) - f(x_best)`.
        """
        draws = self._gp.sample_y(X, n_samples=n_samples, random_state=random_state)
        return np.asarray(draws + self._offsets(X)[:, None], dtype=float)

    @property
    def kernel_(self) -> Any:
        return self._gp.kernel_


def _fit_surrogate(
    x_norm: np.ndarray,
    y_int: np.ndarray,
    *,
    prior_mean: Callable[[np.ndarray], np.ndarray] | None,
    lows: np.ndarray,
    x_scales: np.ndarray,
    sign: float,
    transform: str,
    ravel_input: bool,
    noise_std: float,
    length_scale: float | None,
    seed: int,
    noise_scale: np.ndarray | None,
    n_dims: int = 1,
    ls_bounds: tuple[float, float] = _LS_BOUNDS,
    kernel: str = _DEFAULT_KERNEL,
) -> tuple[_Surrogate, np.ndarray]:
    """Fit the GP — on residuals when a physical prior is supplied.

    Returns the surrogate and the values it was actually fitted to. The second
    return matters: the leave-one-out reliability check has to run against the
    same target the model was trained on, or it is grading a different model
    than the one that made the recommendation.

    Both entry points need identical behaviour here, and duplicating it once
    already meant the residual and the prior could drift apart between the 1-D
    and d-D paths. `ravel_input` is the only real difference — `optimize()`
    hands the caller's prior a flat array because its whole contract is a single
    variable, while `optimize_nd` hands over (m, d).

    We deliberately bound the length-scale to keep a handful of points from
    overfitting; sklearn then warns when the optimizer sits on that bound.
    That is expected, not a problem, so it is suppressed locally.
    """
    offsets_at = (
        _prior_offsets_at(
            prior_mean,
            lows=lows,
            x_scales=x_scales,
            sign=sign,
            transform=transform,
            ravel_input=ravel_input,
        )
        if prior_mean is not None
        else None
    )
    y_fit = y_int if offsets_at is None else y_int - offsets_at(x_norm)

    gp_core = _build_gp(
        y_fit,
        noise_std,
        length_scale,
        seed,
        noise_scale=noise_scale,
        n_dims=n_dims,
        ls_bounds=ls_bounds,
        kernel=kernel,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp_core.fit(x_norm, y_fit)

    gp = gp_core if offsets_at is None else _PriorMeanGP(gp_core, offsets_at)
    return gp, y_fit


def _scales_per_axis(scales: tuple[float, ...], n_dims: int) -> tuple[float, ...]:
    """One length-scale per axis, whatever kernel produced them.

    An isotropic fit returns a single shared value. Reporting it once while the
    ARD path reports d values would make `BoConfigND.length_scales` mean
    different things depending on a flag, so the shared value is repeated.
    """
    if len(scales) == 1 and n_dims > 1:
        return scales * n_dims
    return scales


def _xi_absolute(xi: float, y_int: np.ndarray, noise_std: float) -> float:
    """Convert the relative exploration sweetener into the target's own units.

    `xi` is a *fraction of the observed spread*, not a value in the target's
    units. It has to be, because the same absolute number cannot be right for
    two targets measured on different scales: zT runs about 0.03-0.07, a power
    factor runs to several hundred microwatts per metre per kelvin squared. A
    fixed `xi = 0.01` is a fifth of the entire zT signal — larger than any real
    improvement, so `mu - f_best - xi` is negative everywhere and Expected
    Improvement collapses to zero. On the power-factor scale the same constant
    is invisible and EI turns purely greedy. Exploration silently depending on
    the units the researcher happened to record in is a bug, not a tuning
    choice, and it is why EI never functioned on this lab's primary target.

    The scale is the observed standard deviation, which makes `xi` mean exactly
    what it would in an implementation that standardises `y` before computing
    EI — the usual convention, and consistent with the GP here already fitting
    with `normalize_y=True`.

    The floor matters for the sparse campaigns this tool exists for. When every
    measurement sits within the measurement noise the observed spread is not a
    meaningful scale — it is scatter — so the noise level takes over as the
    unit of "an improvement worth having". Without that, a degenerate first
    round would drive `xi` to zero and make EI perfectly greedy at exactly the
    moment there is least reason to trust the surrogate.
    """
    scale = max(float(np.std(y_int)), float(noise_std))
    return float(xi) * scale


def _expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float
) -> np.ndarray:
    """Expected improvement over `f_best` (maximization).

    `xi` here is absolute, in the fit space's units — call `_xi_absolute` on
    the caller's relative value first.
    """
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


def _fill_distance(x_mat: np.ndarray) -> float:
    """Radius of the largest unsampled hole in the search box.

    `x_mat` is (n, d) in normalized coordinates, where every axis spans
    [0, _SPAN_UNITS] by construction. Returns the largest distance from any
    point of that box to its nearest observation.

    Exact in one dimension — the answer is half the widest interior gap, or the
    distance from an end of the box to the nearest point, whichever is larger.
    Estimated from a Sobol probe set above that, since the exact value is the
    radius of the largest empty ball and is not worth computing exactly for a
    grade. The probe seed is fixed rather than taken from the caller: this is a
    property of where the samples sit, and it should not shift because someone
    changed the RNG seed of the fit.

    Deliberately *not* scaled by the fitted length-scales. Dividing by the
    length-scale would look more principled, but it would let a model that
    over-smooths — the exact failure this grade exists to catch — report a small
    fill distance and earn a better grade for being more wrong. The geometry has
    to be measured independently of the model being judged.
    """
    pts = np.asarray(x_mat, dtype=float)
    pts = pts.reshape(pts.shape[0], -1)
    if pts.shape[0] == 0:
        return float(_SPAN_UNITS)
    if pts.shape[1] == 1:
        s = np.sort(pts[:, 0])
        gaps = np.diff(s)
        widest_interior = float(gaps.max()) / 2.0 if gaps.size else 0.0
        return float(max(widest_interior, s[0] - 0.0, _SPAN_UNITS - s[-1], 0.0))

    d = pts.shape[1]
    probe = (
        qmc.Sobol(d=d, scramble=True, seed=0).random_base2(int(np.ceil(np.log2(_FILL_PROBE))))
        * _SPAN_UNITS
    )
    tree = cKDTree(pts)
    nearest, _ = tree.query(probe, k=1)
    return float(np.max(nearest))


def _assess_reliability(
    x_norm: np.ndarray,
    y: np.ndarray,
    *,
    noise_std: float,
    seed: int,
    ls_bounds: tuple[float, float] = _LS_BOUNDS,
    kernel: str = _DEFAULT_KERNEL,
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
    # Accept (n,) or (n, d); the reshape is a no-op for the 1-D callers, which
    # previously did `.reshape(-1, 1)` at each use site.
    x_mat = np.asarray(x_norm, dtype=float)
    x_mat = x_mat.reshape(x_mat.shape[0], -1)
    n, d = x_mat.shape
    inside = 0
    for i in range(n):
        mask = np.arange(n) != i
        gp = _build_gp(y[mask], noise_std, None, seed, n_dims=d, ls_bounds=ls_bounds, kernel=kernel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(x_mat[mask], y[mask])
        mu, sd = gp.predict(x_mat[i : i + 1], return_std=True)
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

    # Coverage of the search box, independent of how many points there are.
    # Downgrade only, never upgrade: a well-filled box does not make a model
    # trustworthy, but a badly-filled one does make it untrustworthy, which is
    # the same asymmetry the leave-one-out gate above uses.
    fill = _fill_distance(x_mat)
    fill_limit = _FILL_CALIBRATED if level == "calibrated" else _FILL_INDICATIVE
    if fill > _FILL_INDICATIVE * (1 + _FILL_TOL) and level != "exploratory":
        level = "exploratory"
        note = (
            f"Exploratory: {n} points over {d} "
            f"{'axis' if d == 1 else 'axes'} still leave an unsampled gap of "
            f"radius {fill:.2f} (limit {_FILL_INDICATIVE:.2f}, in units where each "
            f"search range spans {_SPAN_UNITS:g}). The count is not the problem — "
            f"the points do not cover the space."
        )
    elif fill > _FILL_CALIBRATED * (1 + _FILL_TOL) and level == "calibrated":
        level = "indicative"
        note = (
            f"Indicative: {n} points would grade calibrated on count, but the "
            f"largest unsampled gap has radius {fill:.2f} (limit "
            f"{_FILL_CALIBRATED:.2f}) — well sampled in places, not everywhere. "
            f"Leave-one-out: {inside}/{n} inside the 95% band."
        )

    return ReliabilityReport(
        level=level,
        n_observations=n,
        loo_inside=inside,
        loo_total=n,
        loo_coverage=round(coverage, 3),
        note=note,
        n_dims=d,
        fill_distance=round(fill, 4),
        fill_limit=round(fill_limit, 4),
    )


def _validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float],
    *,
    direction: str,
    kernel: str,
    explore_policy: str,
) -> None:
    """Reject argument combinations the optimizer cannot act on.

    Split out of `optimize` so the fitting path reads as one continuous
    argument, rather than opening with a page of guards.

    Raises:
        OptimizationError: On mismatched shapes, too few points, or an
            unrecognised direction, kernel or exploration policy.
    """
    if x.shape != y.shape:
        raise OptimizationError(f"x and y length mismatch: {x.shape} vs {y.shape}")
    if x.size < _MIN_POINTS:
        raise OptimizationError(
            f"Need at least {_MIN_POINTS} measured points to optimize; got {x.size}"
        )
    if direction not in _DIRECTIONS:
        raise OptimizationError(f"direction must be one of {_DIRECTIONS}; got {direction!r}")
    if kernel not in _KERNELS:
        raise OptimizationError(f"kernel must be one of {_KERNELS}; got {kernel!r}")
    if explore_policy not in _EXPLORE_POLICIES:
        raise OptimizationError(
            f"explore_policy must be one of {_EXPLORE_POLICIES}; got {explore_policy!r}"
        )
    lo, hi = bounds
    if not hi > lo:
        raise OptimizationError(f"bounds must have high > low; got {bounds}")


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
    prior_mean: Callable[[np.ndarray], np.ndarray] | None = None,
    length_scale: float | None = None,
    rel_noise: float = _REL_NOISE,
    measured_noise: float | None = None,
    point_noise: np.ndarray | None = None,
    xi: float = _XI,
    grid_size: int = _GRID_SIZE,
    seed: int = 0,
    objective_aggregation: str = "peak",
    created_at: datetime | None = None,
    with_reliability: bool = True,
    unreliable: np.ndarray | None = None,
    epsilon: float | None = None,
    delta: float = _DEFAULT_DELTA,
    kernel: str = _DEFAULT_KERNEL,
    explore_policy: str = _DEFAULT_EXPLORE_POLICY,
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
        prior_mean: Optional physical model of the target. Called with the
            parameter values in their **original units** (a 1-D array) and must
            return the predicted property, also in original units — the
            transform and the maximize/minimize flip are applied here, not by
            the caller. The GP then fits `observed - prior_mean(x)`. See
            `optimize_nd` for the full rationale; the short version is that a
            zero-mean GP has nothing to extrapolate along, which is why sparse
            campaigns drift to the middle of the widest gap. Must be strictly
            positive when `y_transform="log"`.
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
        point_noise: Optional per-observation standard deviations, shape (n,),
            in the target's own units. Where `measured_noise` says how
            repeatable the technique is, this says how well *each* value is
            known — a fitted slope's standard error, a propagated uncertainty,
            the spread of a quantity across defensible modelling choices. The
            GP becomes heteroscedastic: precise points pull the surface, vague
            ones are held loosely.

            This is the quantitative form of the `unreliable` flag, and the two
            compose. A value that is both imprecise and physically implausible
            earns both penalties.

            Its scalar summary (the median) fills in for `measured_noise` when
            that is not supplied, since the exploration sweetener, the
            convergence floor and the epsilon statement all need one number.

            One caveat worth knowing at the call site: a standard error fitted
            from very few points is itself uncertain and tends to read low, so
            three-point fits hand the surrogate more confidence than they have
            earned. `analysis.thermovoltage.slope` documents the size of that
            effect for the case it produces.
        xi: Exploration sweetener in EI, as a **fraction of the observed
            spread** (see `_xi_absolute`) — not a value in the target's
            units. An absolute constant cannot serve targets measured on
            different scales, and got EI wrong on zT-scale objectives.
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
        kernel: Stationary factor of the covariance, one of `_KERNELS`.
            "rbf" (default) is the shipped behaviour. "matern52" is Matern
            with nu = 5/2, which is rougher: its sample paths are twice
            differentiable rather than infinitely so. That matters because an
            RBF assumes the property varies smoothly everywhere, and the
            features an audit tool exists to find — a phase boundary, a
            solubility limit, a percolation threshold — are exactly the ones
            an over-smooth kernel interpolates away. nu = 5/2 is also the
            least smooth choice that keeps the GP-UCB regret guarantee, which
            needs nu > 2.
        explore_policy: What to recommend when the improvement signal is
            exhausted but the data is still exploratory, one of
            `_EXPLORE_POLICIES`. "max_std" (default) is the shipped
            behaviour: recommend the point of greatest posterior sd, i.e. the
            largest unmeasured gap. "ei" removes the fallback and keeps the
            max-EI pick. "ucb" replaces it with `argmax(mean + 2 sd)`, which
            explores only where the model also thinks the value could be high.
            This governs the recommendation only — whether convergence is
            declared stays gated on the reliability tier regardless.

    Returns:
        An `OptimizationResult` with the posterior, the recommendation,
        the convergence verdict, and the frozen `BoConfig`.

    Raises:
        OptimizationError: Fewer than 3 points, mismatched shapes, or a
            degenerate (zero-width) bound range.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    _validate_inputs(
        x, y, bounds, direction=direction, kernel=kernel, explore_policy=explore_policy
    )
    lo, hi = bounds

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
    noise = _noise_model(
        y,
        y_work,
        rel_noise=rel_noise,
        measured_noise=measured_noise,
        point_noise=point_noise,
        unreliable=unreliable,
        transform=transform,
    )
    noise_std, noise_scale = noise.std, noise.scale

    # Minimization is exact negation in the (possibly log) fit space.
    sign = 1.0 if direction == "maximize" else -1.0
    y_int = sign * y_work

    x_col = x_norm.reshape(-1, 1)
    gp, y_fit = _fit_surrogate(
        x_col,
        y_int,
        prior_mean=prior_mean,
        lows=np.array([lo], dtype=float),
        x_scales=np.array([x_scale], dtype=float),
        sign=sign,
        transform=transform,
        ravel_input=True,
        noise_std=noise_std,
        length_scale=length_scale,
        seed=seed,
        noise_scale=noise_scale,
        kernel=kernel,
    )

    grid = np.linspace(lo, hi, grid_size)
    grid_norm = (grid - lo) / x_scale
    mean_int, std = gp.predict(grid_norm.reshape(-1, 1), return_std=True)

    best_i = int(np.argmax(y_int))
    f_best_int = float(y_int[best_i])
    best_x = float(x[best_i])
    best_label = f"{input_name} = {best_x:.4g}"
    # `xi` arrives as a fraction of the observed spread; EI works in units.
    xi_abs = _xi_absolute(xi, y_int, noise_std)
    ei = _expected_improvement(mean_int, std, f_best_int, xi_abs)

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
        kernel=_kernel_label(kernel),
        x_scale=float(x_scale),
        length_scale=(
            float(length_scale) if length_scale is not None else _fitted_length_scale(gp)
        ),
        length_scale_fitted=length_scale is None,
        length_scale_bounds=_LS_BOUNDS,
        xi=xi,
        rel_noise=rel_noise,
        noise_std=noise_std,
        xi_absolute=xi_abs,
        point_noise_used=noise.per_point,
        point_noise_scale=(
            tuple(float(v) for v in noise.scale)
            if noise.per_point and noise.scale is not None
            else None
        ),
        n_observations=int(x.size),
        grid_size=grid_size,
        seed=seed,
        created_at=created_at if created_at is not None else datetime.now(UTC),
    )

    reliability: ReliabilityReport | None = None
    if with_reliability:
        # Residuals when a prior is in play — see the same call in `optimize_nd`.
        reliability = _assess_reliability(
            x_norm,
            y_fit,
            noise_std=noise_std,
            seed=seed,
            kernel=kernel,
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
    #
    # `explore_policy` selects *where* to point when that fires. "max_std" is
    # the shipped behaviour — pure uncertainty sampling, which four of the
    # reviewed papers measure as the weakest available policy when used as a
    # whole strategy (Borg, Rohr, Srinivas, Shields). "ei" removes the fallback
    # entirely; "ucb" replaces it with optimism. Which is right is an empirical
    # question, which is why all three exist rather than one being assumed.
    # Note the convergence gate is deliberately NOT policy-dependent: a flat EI
    # on exploratory data still must not be reported as "optimum found".
    is_exploratory = reliability is not None and reliability.level == "exploratory"
    converged = signal_exhausted and not is_exploratory
    explore = signal_exhausted and is_exploratory and explore_policy != "ei"
    if not explore:
        rec_i = ei_i
    elif explore_policy == "ucb":
        rec_i = int(np.argmax(mean_int + _UCB_LAMBDA * std))
    else:
        rec_i = int(np.argmax(std))
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
        stopping=_stopping_verdict(
            probability=prob_within,
            epsilon=eps,
            delta=delta,
            signal_exhausted=signal_exhausted,
            reliability=reliability,
            best_label=best_label,
        ),
        n_unreliable=noise.n_unreliable,
        noise_measured=noise.measured,
    )


# ---------------------------------------------------------------------------
# Multi-dimensional search.
#
# `optimize()` above stays exactly as it was: it is 1-D, it is what produced
# every frozen pre-registration on disk, and those records must keep replaying
# to the same numbers. The functions below add d >= 1 alongside it rather than
# underneath it, sharing every helper that was already dimension-agnostic.
#
# Two things genuinely change with dimension. The kernel becomes ARD, so the
# model reports one length-scale per axis and can say which variable the
# property responds to. And the acquisition can no longer be maximised on a
# dense grid: a 200-point line becomes 200^d, so candidates come from a
# scrambled Sobol sequence, which fills the box far more evenly than random
# sampling at the same budget.
# ---------------------------------------------------------------------------

_ND_CANDIDATES = 2048  # Sobol points the acquisition is maximised over
_ND_MIN_EXTRA = 2  # a GP over d axes needs at least d + this many points
# Restarts for the continuous acquisition refinement. The surface is
# multi-modal, so one descent finds the nearest peak, not the best one.
_POLISH_STARTS = 4
_N_MATRIX_DIMS = 2  # observations arrive as a 2-D (n, d) matrix
# Side of the optional regular grid the posterior is reported on in 2-D. 48 is
# a compromise: fine enough that a contour reads smoothly, small enough that
# 48*48 predictions cost about the same as the Sobol candidate set already does.
_SURFACE_SIZE = 48


# ---------------------------------------------------------------------------
# Categorical axes
# ---------------------------------------------------------------------------
# A synthesis parameter reaches this engine as a float and nothing else:
# `SynthesisParams` is `dict[str, dict[str, float]]` all the way down. So a knob
# with no natural ordering — etching atmosphere Air / Ar / N2 — can only be
# entered by encoding it, and 0/1/2 is what people write. The GP then treats it
# as any other number, interpolates, and recommends "gas 0.55": a recipe nobody
# can make, reported with the same confidence as a real one.
#
# Floats carry no intent, so the guard has two halves that do different jobs.
# `axis_kinds` lets a caller SAY an axis is categorical, and that is refused
# outright. `_encoded_axis_warning` catches the undeclared case — the one that
# actually happened — and says so on the result. Detection is only ever a
# suspicion, because "anneal 1, 2, 3 h" has exactly the shape of "gas 0, 1, 2",
# so it warns and never blocks. Declaring blocks; guessing does not.
AXIS_CONTINUOUS = "continuous"
AXIS_CATEGORICAL = "categorical"
_AXIS_KINDS = (AXIS_CONTINUOUS, AXIS_CATEGORICAL)

# Levels few enough, whole, and one apart: the shape of an encoded category.
# One level is a constant, not an axis; the cap keeps a swept integer variable
# (0..9 percent, say) from being mistaken for a handful of names.
_ENCODED_MIN_LEVELS = 2
_ENCODED_MAX_LEVELS = 6
_ENCODED_LEVEL_STEP = 1.0
# How far off a level the recommendation must land before the suspicion is
# worth raising, as a fraction of the level spacing (which the test above pins
# at one). On a genuinely continuous axis a value between two measured ones is
# the ordinary answer and there is nothing to say; the warning fires only when
# acting on it would mean synthesising something between two categories.
_LEVEL_TOL_FRAC = 0.05


def _validate_axis_kinds(axis_kinds: Sequence[str] | None, names: tuple[str, ...]) -> None:
    """Reject any axis the caller has declared categorical, and unknown kinds.

    Raises rather than warns: the caller has stated the axis has no ordering, so
    every value this engine could return between two levels is meaningless, and
    returning one anyway is how a nonsense recipe acquires a confidence interval.
    """
    if axis_kinds is None:
        return
    kinds = tuple(str(k) for k in axis_kinds)
    if len(kinds) != len(names):
        raise OptimizationError(f"axis_kinds has {len(kinds)} entries for {len(names)} axes")
    unknown = sorted({k for k in kinds if k not in _AXIS_KINDS})
    if unknown:
        raise OptimizationError(f"axis_kinds must each be one of {_AXIS_KINDS}; got {unknown}")
    categorical = [n for n, k in zip(names, kinds, strict=True) if k == AXIS_CATEGORICAL]
    if categorical:
        listed = ", ".join(repr(n) for n in categorical)
        raise OptimizationError(
            f"Cannot optimize over categorical axes ({listed}): a Gaussian process "
            f"interpolates between numbers, so it would recommend a value between "
            f"two levels, which is not a recipe. Run one campaign per level and "
            f"compare them, or hold the axis fixed and optimize the continuous "
            f"knobs within it."
        )


def _encoded_axis_warning(col: np.ndarray, rec: float, name: str) -> str | None:
    """Warn when `col` looks like an encoded category and `rec` falls off-level.

    Both halves are needed. A column of a few unit-spaced whole numbers is
    *suspicious*, not wrong. What makes it actionable is the recommendation
    landing between those values, because that is the number the user would
    otherwise go to the bench and try to make.
    """
    levels = np.unique(col)
    if not _ENCODED_MIN_LEVELS <= levels.size <= _ENCODED_MAX_LEVELS:
        return None
    if not np.all(levels == np.round(levels)):
        return None
    if not np.allclose(np.diff(levels), _ENCODED_LEVEL_STEP):
        return None
    if float(np.min(np.abs(levels - rec))) <= _LEVEL_TOL_FRAC * _ENCODED_LEVEL_STEP:
        return None  # landed on a level; acting on it changes nothing
    shown = ", ".join(f"{v:g}" for v in levels)
    return (
        f"Axis {name!r} takes only the whole values {shown}, which is what an "
        f"encoded category (a gas, a substrate, a precursor) looks like — and its "
        f"recommended value, {rec:g}, falls between two of them. If those numbers "
        f"stand for names, this axis cannot be optimized: run one campaign per "
        f"level instead. If they are real quantities, nothing is wrong here."
    )


@dataclass(frozen=True, slots=True)
class RecommendationND:
    """The next experiment to run, in d dimensions.

    Identical in meaning to `Recommendation`; `x` is a point rather than a
    scalar. The two interval fields carry the same distinction — `ci95` is the
    model's own uncertainty, `predictive_interval_95` is what a new measurement
    at this point should fall inside, and is the band to test calibration
    against.
    """

    x: tuple[float, ...]
    predicted_mean: float
    ci95: float
    predictive_sd: float
    ci95_predictive: float
    predictive_interval_95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class BoConfigND:
    """Frozen record of a d-dimensional run.

    Deliberately a separate type from `BoConfig`: that one is persisted inside
    existing pre-registration files, and widening it would change how those
    deserialise. `length_scales` is the ARD result and the scientifically
    interesting field — a length-scale pinned at its upper bound is the model
    reporting that the axis does nothing.
    """

    objective: str
    direction: str
    y_transform: str
    objective_aggregation: str
    input_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    kernel: str
    acquisition: str  # "sobol" or "sobol+lbfgsb"
    n_dims: int
    x_scales: tuple[float, ...]  # raw units per normalized unit, per axis
    length_scales: tuple[float, ...]  # fitted ARD length-scales (normalized)
    length_scale_bounds: tuple[float, float]
    xi: float  # a FRACTION of the observed spread — see `_xi_absolute`
    rel_noise: float
    noise_std: float
    n_observations: int
    n_candidates: int
    seed: int
    created_at: datetime
    xi_absolute: float | None = None  # `xi` in objective units, as applied
    # True when per-observation standard deviations were supplied. Without it
    # two runs could carry byte-identical configs and still have produced
    # different recommendations, because a heteroscedastic fit weighs the same
    # points differently. A frozen record that cannot tell those apart is not a
    # record of how the answer was produced.
    point_noise_used: bool = False


@dataclass(frozen=True, slots=True)
class SurfaceND:
    """The fitted posterior on a regular 2-D grid.

    The Sobol candidate set is the right thing to *optimise* over and the wrong
    thing to *draw*: contouring scattered points needs a triangulation, which a
    plotting front-end should not have to carry. This is the same posterior
    resampled onto a lattice, so a heat map is a direct array read.

    Rows are indexed by the second axis and columns by the first, i.e.
    `mean[j][i]` is the value at `(axis_x[i], axis_y[j])`.
    """

    axis_names: tuple[str, str]
    axis_x: tuple[float, ...]
    axis_y: tuple[float, ...]
    mean: tuple[tuple[float, ...], ...]
    sd: tuple[tuple[float, ...], ...]
    ei: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class OptimizationResultND:
    """A d-dimensional optimization round.

    `candidates` is the Sobol set the acquisition was maximised over, with the
    posterior evaluated on it, so a caller can contour a 2-D surface or slice a
    higher-dimensional one without refitting anything. `surface` is the same
    posterior on a regular lattice, present only when asked for and only in 2-D.
    """

    input_names: tuple[str, ...]
    target_name: str
    candidates: tuple[tuple[float, ...], ...]
    cand_mean: tuple[float, ...]
    cand_lower: tuple[float, ...]
    cand_upper: tuple[float, ...]
    cand_ei: tuple[float, ...]
    observed_x: tuple[tuple[float, ...], ...]
    observed_y: tuple[float, ...]
    best_x: tuple[float, ...]
    best_y: float
    recommendation: RecommendationND
    max_ei: float
    noise_threshold: float
    converged: bool
    config: BoConfigND
    reliability: ReliabilityReport | None = None
    epsilon: float = 0.0
    delta: float = _DEFAULT_DELTA
    prob_within_epsilon: float = 0.0
    epsilon_delta_met: bool = False
    # The headline: one action, with the reasoning behind it. The four
    # fields above are its inputs and stay for callers that want them.
    stopping: StoppingVerdict | None = None
    n_unreliable: int = 0
    noise_measured: bool = False
    surface: SurfaceND | None = None
    # Axes whose observed values look like an encoded category, when the
    # recommendation for them landed between two levels. Advisory: the engine
    # cannot tell 'gas 0, 1, 2' from 'anneal 1, 2, 3 h' — see
    # `_encoded_axis_warning`. An axis the caller DECLARES categorical via
    # `axis_kinds` never gets here; it raises instead.
    axis_warnings: tuple[str, ...] = ()


def _posterior_surface(
    gp: GaussianProcessRegressor,
    box: np.ndarray,
    x_scales: np.ndarray,
    names: tuple[str, ...],
    size: int,
    *,
    sign: float,
    transform: str,
    y_min: float | None,
    y_max: float | None,
    f_best: float,
    xi: float,
) -> SurfaceND:
    """Evaluate the fitted posterior on a `size` x `size` lattice.

    Only meaningful in two dimensions, which is the only case the caller asks
    for it. The mean is returned in the target's own physical units — through
    the same inverse transform and clamp the recommendation goes through — so a
    colour bar reads in the units the researcher measured, not in the internal
    maximization frame.
    """
    ax = np.linspace(box[0, 0], box[0, 1], size)
    ay = np.linspace(box[1, 0], box[1, 1], size)
    gx, gy = np.meshgrid(ax, ay, indexing="xy")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    mu_int, sd = gp.predict((pts - box[:, 0]) / x_scales, return_std=True)
    ei = _expected_improvement(mu_int, sd, f_best, xi)
    mean_phys, _lo, _hi, _ = _physical_band(sign * mu_int, sd, transform, y_min, y_max)

    def _rows(v: np.ndarray) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(c) for c in row) for row in v.reshape(size, size))

    return SurfaceND(
        axis_names=(names[0], names[1]),
        axis_x=tuple(float(v) for v in ax),
        axis_y=tuple(float(v) for v in ay),
        mean=_rows(mean_phys),
        sd=_rows(sd),
        ei=_rows(ei),
    )


def _sobol_candidates(bounds: np.ndarray, n_candidates: int, seed: int) -> np.ndarray:
    """A scrambled Sobol fill of the search box, shape (m, d).

    Sobol rather than a grid because a grid costs `points**d`, and rather than
    uniform random because Sobol's discrepancy is far lower at the same budget
    — the acquisition maximum is found more reliably for the same compute.
    Drawn in powers of two, which is where the sequence's balance properties
    actually hold.
    """
    d = bounds.shape[0]
    m = max(4, int(np.ceil(np.log2(max(n_candidates, 2)))))
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    unit = engine.random_base2(m)
    # scipy ships no type information, so the draw arrives as Any and infects
    # the arithmetic; the asarray pins the declared return type back down.
    return np.asarray(bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0]), dtype=float)


def _prepare_nd_inputs(
    x: np.ndarray,
    y: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    input_names: Sequence[str],
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate and shape the multi-dimensional inputs.

    Returns `(x_obs, y, box, names)` with `x_obs` as (n, d) and `box` as
    (d, 2). Every failure is an `OptimizationError` naming the offending
    argument, because a silent broadcast between a (n,) and a (n, 1) is
    exactly the kind of mistake that produces a plausible wrong answer.
    """
    x_obs = np.asarray(x, dtype=float)
    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(-1, 1)
    if x_obs.ndim != _N_MATRIX_DIMS:
        raise OptimizationError(f"x must be 2-D (n, d); got shape {x_obs.shape}")
    y = np.asarray(y, dtype=float).ravel()
    n, d = x_obs.shape

    names = tuple(str(v) for v in input_names)
    if len(names) != d:
        raise OptimizationError(f"input_names has {len(names)} entries for {d} axes")
    if y.shape[0] != n:
        raise OptimizationError(f"x has {n} rows but y has {y.shape[0]} values")
    if direction not in _DIRECTIONS:
        raise OptimizationError(f"direction must be one of {_DIRECTIONS}; got {direction!r}")

    need = max(_MIN_POINTS, d + _ND_MIN_EXTRA)
    if n < need:
        raise OptimizationError(
            f"Need at least {need} measured points to optimize over {d} axes; got {n}"
        )
    if not np.all(np.isfinite(x_obs)) or not np.all(np.isfinite(y)):
        raise OptimizationError("x and y must be finite (no NaN or Inf)")

    box = np.asarray([(float(lo), float(hi)) for lo, hi in bounds], dtype=float)
    if box.shape != (d, _N_MATRIX_DIMS):
        raise OptimizationError(f"bounds must hold {d} (low, high) pairs; got {box.shape}")
    if not np.all(box[:, 1] > box[:, 0]):
        bad = [i for i in range(d) if box[i, 1] <= box[i, 0]]
        raise OptimizationError(f"bounds must have high > low; axes {bad} do not")
    return x_obs, y, box, names


def _polish_acquisition(
    objective: Callable[[np.ndarray], float],
    starts: np.ndarray,
    n_dims: int,
) -> tuple[np.ndarray | None, float]:
    """Refine an acquisition maximum off the candidate grid, with L-BFGS-B.

    The Sobol set locates the right basin; it cannot place a point inside it
    more precisely than its own spacing. In two dimensions 2048 points resolve
    about a fortieth of each axis, so the recommendation carries a quantisation
    error of roughly that size — real, and avoidable, because the surrogate is
    a closed-form function that can be optimised continuously between the
    candidates.

    Several starts, because the acquisition surface is multi-modal and a single
    descent finds the nearest peak rather than the best one. Gradients are
    finite-differenced: scikit-learn does not expose a derivative of the
    posterior, and at these dimensions the extra evaluations are far cheaper
    than the fit that preceded them.

    Returns `(point, value)` in normalized coordinates, or `(None, -inf)` if
    every start failed. The caller keeps its grid answer unless this beats it,
    so a failed or unhelpful polish can never make the recommendation worse.
    """
    box = [(0.0, _SPAN_UNITS)] * n_dims
    best_x: np.ndarray | None = None
    best_v = -np.inf
    for x0 in np.atleast_2d(starts):
        try:
            res = minimize(
                lambda p: -objective(np.asarray(p, dtype=float)),
                x0=np.clip(x0, 0.0, _SPAN_UNITS),
                method="L-BFGS-B",
                bounds=box,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(res.x)):
            continue
        value = -float(res.fun)
        if value > best_v:
            best_v, best_x = value, np.clip(res.x, 0.0, _SPAN_UNITS)
    return best_x, best_v


def _pick_point(
    gp: GaussianProcessRegressor,
    cand_norm: np.ndarray,
    driver: np.ndarray,
    rec_i: int,
    *,
    explore: bool,
    f_best: float,
    xi: float,
    n_dims: int,
    polish: bool,
) -> tuple[np.ndarray, str]:
    """Choose the next point, refining off the grid when that helps.

    `driver` is whichever quantity selected `rec_i` — Expected Improvement
    normally, posterior spread when the data is too sparse to exploit. The same
    quantity is what gets refined, so the polish never optimises something
    other than the criterion that made the choice.
    """
    if not polish:
        return cand_norm[rec_i], "sobol"

    def score(p: np.ndarray) -> float:
        mu, sd = gp.predict(p.reshape(1, -1), return_std=True)
        if explore:
            return float(sd[0])
        return float(_expected_improvement(mu, sd, f_best, xi)[0])

    top = np.argsort(driver)[-_POLISH_STARTS:][::-1]
    refined, value = _polish_acquisition(score, cand_norm[top], n_dims)
    if refined is not None and value > float(driver[rec_i]):
        return refined, "sobol+lbfgsb"
    return cand_norm[rec_i], "sobol"


def _recommend_nd(
    point: np.ndarray,
    mean_at: float,
    std_at: float,
    *,
    sign: float,
    noise_std: float,
    transform: str,
    y_min: float | None,
    y_max: float | None,
) -> RecommendationND:
    """The d-dimensional twin of `_recommend`.

    Same arithmetic, but takes the recommended point and the posterior there
    directly rather than an index into a candidate array — the point may have
    been refined off the grid and so need not be one of the candidates.
    """
    rec_sigma = float(std_at)
    predictive_sd = float(np.sqrt(rec_sigma**2 + noise_std**2))
    pm_work = float(sign * mean_at)
    lo_pred = _to_phys(pm_work - _CI95 * predictive_sd, transform, y_min, y_max)
    hi_pred = _to_phys(pm_work + _CI95 * predictive_sd, transform, y_min, y_max)
    lo_model = _to_phys(pm_work - _CI95 * rec_sigma, transform, y_min, y_max)
    hi_model = _to_phys(pm_work + _CI95 * rec_sigma, transform, y_min, y_max)
    return RecommendationND(
        x=tuple(float(v) for v in np.asarray(point, dtype=float).ravel()),
        predicted_mean=_to_phys(pm_work, transform, y_min, y_max),
        ci95=(hi_model - lo_model) / 2.0,
        predictive_sd=predictive_sd,
        ci95_predictive=(hi_pred - lo_pred) / 2.0,
        predictive_interval_95=(lo_pred, hi_pred),
    )


def optimize_nd(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bounds: Sequence[tuple[float, float]],
    input_names: Sequence[str],
    axis_kinds: Sequence[str] | None = None,
    target_name: str,
    direction: str = "maximize",
    y_transform: str = _IDENTITY,
    y_min: float | None = None,
    y_max: float | None = None,
    prior_mean: Callable[[np.ndarray], np.ndarray] | None = None,
    rel_noise: float = _REL_NOISE,
    measured_noise: float | None = None,
    point_noise: np.ndarray | None = None,
    xi: float = _XI,
    length_scale_bounds: tuple[float, float] = _LS_BOUNDS_ND,
    n_candidates: int = _ND_CANDIDATES,
    polish: bool = True,
    isotropic: bool = False,
    prob_max_points: int = _PROB_MAX_POINTS,
    surface_size: int = 0,
    seed: int = 0,
    objective_aggregation: str = "peak",
    created_at: datetime | None = None,
    with_reliability: bool = True,
    unreliable: np.ndarray | None = None,
    epsilon: float | None = None,
    delta: float = _DEFAULT_DELTA,
    kernel: str = _DEFAULT_KERNEL,
) -> OptimizationResultND:
    """Run one round of Bayesian optimization over d parameters.

    The d-dimensional sibling of `optimize()`. Every physics behaviour carries
    over unchanged — the log-space fit, the physical clamp, the down-weighting
    of observations the physics checks flagged, the noise-floor stopping gate,
    the leave-one-out self-check and the reliability-aware exploration
    fallback all work per-point and never depended on dimension.

    Args:
        x: Observed parameters, shape (n, d). Each column is normalized on its
            own range, so axes in wildly different units (percent and kelvin)
            are treated even-handedly.
        y: Observed property values, shape (n,).
        bounds: One (low, high) pair per input axis.
        input_names: One label per axis; length fixes d.
        axis_kinds: Optional "continuous" / "categorical" per axis. Declaring
            an axis categorical raises: a GP interpolates, so it would answer
            between two levels. Left None (the default) the engine assumes
            every axis is continuous and reports a suspicion on
            `axis_warnings` instead — it cannot read intent out of floats.
        target_name: Label of the property being optimized.
        direction: "maximize" (default) or "minimize".
        y_transform: "identity" (default) or "log", as in `optimize()`.
        y_min: Lower physical bound for the target, or None.
        y_max: Upper physical bound for the target, or None.
        prior_mean: Optional physical model of the target. Called with an
            (m, d) array of parameter values in their **original units** and
            must return m predicted property values, also in original units —
            the transform and the maximize/minimize flip are applied here, not
            by the caller. The GP then models `observed - prior_mean(x)`, so
            the physics carries the trend and the surrogate only corrects it.
            Where a plain GP reverts to a flat mean away from the data — the
            behaviour that makes a four-point campaign recommend the middle of
            the widest gap — this extrapolates along the physics instead. A
            wrong prior is not silently absorbed: it shows up as a large
            residual, and the leave-one-out coverage check is run against that
            residual for exactly that reason. Must be strictly positive when
            `y_transform="log"`.
        rel_noise: Assumed relative measurement noise, used when
            `measured_noise` is absent.
        measured_noise: Observed repeatability in the target's own units;
            beats `rel_noise` when supplied.
        point_noise: Optional per-observation standard deviations, shape (n,),
            in the target's own units — how well each individual value is
            known, rather than how repeatable the technique is. Makes the GP
            heteroscedastic; composes with `unreliable`; its median stands in
            for `measured_noise` when that is absent. See `optimize()` for the
            full rationale and the caveat about standard errors fitted from
            very few points.
        xi: Exploration sweetener in EI, as a **fraction of the observed
            spread** (see `_xi_absolute`) — not a value in the target's
            units. An absolute constant cannot serve targets measured on
            different scales, and got EI wrong on zT-scale objectives.
        length_scale_bounds: Range the ARD length-scales are fitted within, in
            normalized units where each search range spans `_SPAN_UNITS`.
            Defaults to `_LS_BOUNDS_ND`, whose floor of 0.2 was chosen by
            sweeping both benchmarks over eight seeds — see the table there.
            This is deliberately lower than the one-variable `_LS_BOUNDS`: the
            evidence for it is multi-dimensional, and applying it to 1-D
            measurably degrades interval calibration.
        n_candidates: Sobol points the acquisition is maximised over. Rounded
            up to the next power of two.
        polish: refine the chosen point continuously with L-BFGS-B instead of
            leaving it on the Sobol grid (default). The grid answer is kept
            unless the refinement genuinely beats it, so this can only help;
            set False to reproduce a pure grid search.
        isotropic: force one shared length-scale across all axes instead of the
            ARD default. Exists as the *control* condition for the anisotropy
            study — an isotropic kernel cannot represent a process window that
            is tighter in one knob than another, and measuring how much that
            costs requires being able to switch it off. Not a sensible
            production setting: it throws away the per-axis sensitivity that is
            the most useful thing a multi-variable run reports.
        prob_max_points: how many candidates the epsilon-delta regret bound is
            estimated over. Joint posterior sampling is cubic in this, so the
            full candidate set is far more than the estimate needs; the default
            keeps the bias within half the estimator's own Monte-Carlo error at
            a fraction of the cost. Raise it only to check that trade-off.
        surface_size: side of a regular lattice to also report the posterior
            on, for contouring. Ignored unless d == 2. Zero (default) skips it,
            so nothing that only wants a recommendation pays for the extra
            `surface_size**2` predictions.
        seed: RNG seed for the GP restarts and the Sobol scramble.
        objective_aggregation: How each sample's `y` was reduced.
        created_at: Timestamp for the frozen config; defaults to now (UTC).
        with_reliability: Run the leave-one-out self-check (default True).
        unreliable: Per-observation flags from the physics checks; flagged
            points are down-weighted rather than dropped.
        epsilon: Tolerance for the probabilistic regret bound.
        delta: Risk level for that bound.
        kernel: Stationary kernel family, "rbf" or "matern52". Defaults to
            `_DEFAULT_KERNEL`; see the measurement recorded beside it.

    Returns:
        An `OptimizationResultND`.

    Raises:
        OptimizationError: on shape mismatches, too few points for the
            dimension, a degenerate bound range, or an axis declared
            categorical in `axis_kinds`.
    """
    x_obs, y, box, names = _prepare_nd_inputs(x, y, bounds, input_names, direction)
    _validate_axis_kinds(axis_kinds, names)
    n, d = x_obs.shape

    # Per-axis normalization: each search span maps to _SPAN_UNITS, so one set
    # of length-scale bounds is meaningful for every axis whatever its units.
    x_scales = (box[:, 1] - box[:, 0]) / _SPAN_UNITS
    x_norm = (x_obs - box[:, 0]) / x_scales

    transform = y_transform if y_transform in _TRANSFORMS else _IDENTITY
    if transform == _LOG and bool(np.any(y <= 0)):
        transform = _IDENTITY
    y_work = _forward(y, transform)
    noise = _noise_model(
        y,
        y_work,
        rel_noise=rel_noise,
        measured_noise=measured_noise,
        point_noise=point_noise,
        unreliable=unreliable,
        transform=transform,
    )
    noise_std, noise_scale = noise.std, noise.scale

    sign = 1.0 if direction == "maximize" else -1.0
    y_int = sign * y_work

    # `_build_gp` keys ARD off n_dims, so asking for 1 gives the scalar kernel
    # regardless of how many columns X has — which is exactly the isotropic arm.
    gp, y_fit = _fit_surrogate(
        x_norm,
        y_int,
        prior_mean=prior_mean,
        lows=box[:, 0],
        x_scales=x_scales,
        sign=sign,
        transform=transform,
        ravel_input=False,
        noise_std=noise_std,
        length_scale=None,
        seed=seed,
        noise_scale=noise_scale,
        n_dims=1 if isotropic else d,
        ls_bounds=length_scale_bounds,
        kernel=kernel,
    )

    cand = _sobol_candidates(box, n_candidates, seed)
    cand_norm = (cand - box[:, 0]) / x_scales
    mean_int, std = gp.predict(cand_norm, return_std=True)

    best_i = int(np.argmax(y_int))
    best_label = ", ".join(
        f"{name} = {value:.4g}" for name, value in zip(input_names, x_obs[best_i], strict=True)
    )
    # `xi` arrives as a fraction of the observed spread; EI works in units.
    xi_abs = _xi_absolute(xi, y_int, noise_std)
    ei = _expected_improvement(mean_int, std, float(y_int[best_i]), xi_abs)

    mean_work = sign * mean_int
    cand_mean, cand_lower, cand_upper, _ = _physical_band(mean_work, std, transform, y_min, y_max)

    ei_i = int(np.argmax(ei))
    max_ei = float(ei[ei_i])
    signal_exhausted = max_ei < noise_std

    eps = float(epsilon) if epsilon is not None else noise_std
    prob_within = _prob_within_epsilon(
        gp, cand_norm, x_norm[best_i], eps, seed, max_points=prob_max_points
    )

    reliability: ReliabilityReport | None = None
    if with_reliability:
        # Residuals, not raw values, when a prior is in play. The leave-one-out
        # check asks whether the held-out point falls inside its own interval;
        # the prior shifts the prediction and the truth by the same fixed
        # amount, so coverage is identical either way — but only the residual
        # version is measuring the model that actually made the recommendation.
        reliability = _assess_reliability(
            x_norm,
            y_fit,
            noise_std=noise_std,
            seed=seed,
            ls_bounds=length_scale_bounds,
            kernel=kernel,
        )

    # Same reliability-aware rule as the 1-D path: a flat acquisition over
    # data too sparse to trust means "go and look where you have not looked",
    # not "we are done".
    is_exploratory = reliability is not None and reliability.level == "exploratory"
    converged = signal_exhausted and not is_exploratory
    explore = signal_exhausted and is_exploratory
    rec_i = int(np.argmax(std)) if explore else ei_i

    # Refine off the candidate grid. Whichever quantity is driving the pick —
    # Expected Improvement normally, posterior spread when exploring — is the
    # one that gets refined, so the polish never optimises something other than
    # the criterion that chose the point.
    rec_norm, acquisition = _pick_point(
        gp,
        cand_norm,
        std if explore else ei,
        rec_i,
        explore=explore,
        f_best=float(y_int[best_i]),
        xi=xi_abs,
        n_dims=d,
        polish=polish,
    )

    surface = None
    if surface_size > 0 and d == _N_MATRIX_DIMS:
        surface = _posterior_surface(
            gp,
            box,
            x_scales,
            names,
            int(surface_size),
            sign=sign,
            transform=transform,
            y_min=y_min,
            y_max=y_max,
            f_best=float(y_int[best_i]),
            xi=xi_abs,
        )

    rec_mu, rec_sd = gp.predict(rec_norm.reshape(1, -1), return_std=True)
    recommendation = _recommend_nd(
        box[:, 0] + rec_norm * x_scales,
        float(rec_mu[0]),
        float(rec_sd[0]),
        sign=sign,
        noise_std=noise_std,
        transform=transform,
        y_min=y_min,
        y_max=y_max,
    )

    axis_warnings = tuple(
        warning
        for i, axis_name in enumerate(names)
        if (warning := _encoded_axis_warning(x_obs[:, i], recommendation.x[i], axis_name))
    )

    config = BoConfigND(
        objective=target_name,
        direction=direction,
        y_transform=transform,
        objective_aggregation=objective_aggregation,
        input_names=names,
        bounds=tuple((float(a), float(b)) for a, b in box),
        kernel=_kernel_label(kernel, ard=d > 1 and not isotropic),
        acquisition=acquisition,
        n_dims=d,
        x_scales=tuple(float(v) for v in x_scales),
        length_scales=_scales_per_axis(_fitted_length_scales(gp), d),
        length_scale_bounds=length_scale_bounds,
        xi=xi,
        rel_noise=rel_noise,
        noise_std=noise_std,
        xi_absolute=xi_abs,
        point_noise_used=noise.per_point,
        n_observations=n,
        n_candidates=int(cand.shape[0]),
        seed=seed,
        created_at=created_at if created_at is not None else datetime.now(UTC),
    )

    return OptimizationResultND(
        input_names=names,
        target_name=target_name,
        candidates=tuple(tuple(float(v) for v in row) for row in cand),
        cand_mean=tuple(cand_mean.tolist()),
        cand_lower=tuple(cand_lower.tolist()),
        cand_upper=tuple(cand_upper.tolist()),
        cand_ei=tuple(ei.tolist()),
        observed_x=tuple(tuple(float(v) for v in row) for row in x_obs),
        observed_y=tuple(y.tolist()),
        best_x=tuple(float(v) for v in x_obs[best_i]),
        best_y=float(y[best_i]),
        recommendation=recommendation,
        max_ei=max_ei,
        noise_threshold=noise_std,
        converged=converged,
        config=config,
        reliability=reliability,
        epsilon=eps,
        delta=delta,
        prob_within_epsilon=prob_within,
        epsilon_delta_met=prob_within >= 1.0 - delta,
        stopping=_stopping_verdict(
            probability=prob_within,
            epsilon=eps,
            delta=delta,
            signal_exhausted=signal_exhausted,
            reliability=reliability,
            best_label=best_label,
        ),
        n_unreliable=noise.n_unreliable,
        noise_measured=noise.measured,
        surface=surface,
        axis_warnings=axis_warnings,
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
