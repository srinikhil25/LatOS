"""Acquisition functions, and the incumbent problem that motivates them.

Expected improvement is the shipped default and assumes noiseless observations.
This project's regime is noisy *and* small-budget, which is the worst case for
that assumption, and the reason is the incumbent rather than the formula.

EI measures improvement over `f_best`. The engine sets `f_best = max(y)` over
the *noisy* readings, and the maximum of n noisy draws is biased upward: with
scatter `s` and eight observations the expected overshoot is roughly `1.4 * s`,
because the run that happened to read high becomes the thing every future
candidate is asked to beat. The consequence is not random, it is directional —
EI is computed against a target nothing can actually reach, so it shrinks
everywhere, and it shrinks most in the places where the model is uncertain.
The optimizer becomes timid exactly when it should be looking around.

Every alternative here is a different answer to that one problem:

  ei          The shipped baseline. `f_best = max(y)`.
  ei_plugin   The same formula against `f_best = max(mu(x_observed))`. The
              posterior mean has already shrunk the lucky reading back toward
              its neighbours, so the incumbent is what the model believes
              rather than what the voltmeter happened to say. One line of
              difference from `ei`, and the cheapest available fix.
  aei         Augmented EI (Huang, Allen, Notz & Zeng 2006). `ei_plugin` scaled
              by `1 - sigma_n / sqrt(sigma(x)^2 + sigma_n^2)`, which is near 1
              where the model is uncertain and near 0 where the remaining
              uncertainty is just measurement noise. Declines to spend a sample
              on a point whose answer is already as known as the instrument
              allows.
  kg          Knowledge gradient (Frazier, Powell & Dayanik 2009), computed
              exactly on the grid. Asks the question the others do not: how
              much would one observation here improve the best value I would
              *report*, rather than how much would it improve on my luckiest
              reading. Alone among these it is willing to sample somewhere it
              knows is bad, when doing so would sharpen the answer elsewhere.
  mes         Max-value entropy search (Wang & Jegelka 2017), Gumbel
              approximation. Scores the information a sample carries about the
              *value* of the optimum rather than its location.
  ucb         `mu + 2 sigma`. No incumbent at all, so the bias cannot enter. The
              cost is that it has no notion of being done.
  pi          Probability of improvement. Included because it is the other
              textbook choice and because it should be the most damaged by an
              inflated incumbent, which makes it a useful negative control.

What this module does not decide
--------------------------------
Nothing here touches the stopping rule. `engine.optimize` computes EI
unconditionally and tests `max(ei) < noise_std` for convergence, because that
threshold is calibrated in EI's units and against the measurement noise; the
acquisitions above are on different scales (`kg` and `mes` are not even in the
objective's units), so a shared threshold would mean something different in
every arm. Selecting an acquisition changes *where the next point goes* and
nothing else. Any comparison between arms is therefore a comparison of one
thing, which is the only way the answer means anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = [
    "ACQUISITIONS",
    "DEFAULT_ACQUISITION",
    "AcquisitionInputs",
    "acquire",
    "expected_improvement",
]

# In the order they are argued for in the module docstring.
ACQUISITIONS: tuple[str, ...] = ("ei", "ei_plugin", "aei", "kg", "mes", "ucb", "pi")

# Unchanged behaviour: `ei` is what has always shipped, and this module exists
# to find out whether that should change, not to assume it.
DEFAULT_ACQUISITION = "ei"

# Posterior sd can reach zero at an observed point with no noise, and every
# formula here divides by it.
_SIGMA_FLOOR = 1e-9

# `mu + _UCB_LAMBDA * sigma`. Two sigma, matching the exploration fallback in
# `engine`, so the two uses of UCB in this codebase cannot drift apart.
_UCB_LAMBDA = 2.0

# Knowledge gradient costs O(candidates * grid) with a sort inside, against
# O(grid) for every other acquisition here. Evaluating it on every one of a
# 1001-point grid would make it the only arm whose runtime shows up in a
# rehearsal, so candidates are strided to about this many and the winner is
# reported at its full-grid index. The inner maximum still runs over the whole
# grid, so only the *choice* is coarsened, never the value. This is a real
# property of the method rather than a shortcut: KG is expensive, and a
# comparison that hid the cost would be flattering it.
_KG_MAX_CANDIDATES = 128

# Guard on the bracket search for the optimum's distribution in `mes`: if the
# upper end has not reached the 75th percentile by this width, the posterior is
# degenerate and doubling further would only spin.
_MES_BRACKET_LIMIT = 1e6

# Monte Carlo draws of the optimum's value for `mes`. The Gumbel fit is the
# approximation that matters; beyond a hundred or so draws the sampling error is
# far below it.
_MES_SAMPLES = 100


@dataclass(frozen=True, slots=True)
class AcquisitionInputs:
    """Everything any acquisition here needs, gathered once by the caller.

    Assembled in one place so that adding an arm cannot change the call site,
    and so no arm can quietly read a quantity the others were not offered.

    Attributes:
        mu: Posterior mean on the candidate grid, in fit space (already flipped
            for direction, so larger is always better).
        sigma: Posterior sd on the same grid.
        cov: Posterior covariance over the grid, or None. Only `kg` needs it,
            and it is the one O(grid^2) object in the engine, so it is computed
            lazily and its absence disables `kg` rather than silently changing
            what `kg` means.
        f_best_observed: `max(y)` over the noisy readings — the biased
            incumbent, kept because `ei` is defined against it.
        f_best_plugin: `max(mu(x_observed))` — the model's own best estimate of
            what has already been achieved.
        noise_std: Measurement sd in fit space.
        xi: Improvement margin, absolute, in fit space.
    """

    mu: np.ndarray
    sigma: np.ndarray
    f_best_observed: float
    f_best_plugin: float
    noise_std: float
    xi: float
    cov: np.ndarray | None = None


def _finite_sigma(sigma: np.ndarray) -> np.ndarray:
    """Posterior sd floored at `_SIGMA_FLOOR`, non-finite entries included.

    `np.maximum(nan, floor)` is NaN, so a plain floor lets an ill-conditioned
    fit's NaN straight through — and `argmax` ranks NaN above every number, so
    the grid point the surrogate could not describe at all becomes the
    recommendation. Flooring it instead treats that point as known at its mean:
    it can still win, but only on the mean, never on the NaN.
    """
    s = np.asarray(sigma, dtype=float)
    return np.where(np.isfinite(s), np.maximum(s, _SIGMA_FLOOR), _SIGMA_FLOOR)


def _never_pick_non_finite(scores: np.ndarray) -> np.ndarray:
    """Scores with every non-finite entry placed below every finite one.

    The last line of defence for the same failure: a NaN mean survives
    `_finite_sigma` and poisons whichever arm reads it. A surface with no finite
    score at all carries no preference, and is returned flat so the caller's
    tie handling applies rather than an arbitrary index.
    """
    scores = np.asarray(scores, dtype=float)
    bad = ~np.isfinite(scores)
    if not bad.any():
        return scores
    finite = scores[~bad]
    if finite.size == 0:
        return np.zeros_like(scores)
    return np.where(bad, float(np.min(finite)) - 1.0, scores)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float) -> np.ndarray:
    """Expected improvement over `f_best` (maximization).

    `xi` is absolute, in the fit space's units. Never NaN: a point whose
    posterior is not finite is scored as no expected improvement.
    """
    sigma = _finite_sigma(sigma)
    improvement = np.asarray(mu, dtype=float) - f_best - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    # EI is non-negative by construction, so zero — "nothing to gain here" — is
    # the floor that keeps the plotted surface honest, where the generic
    # below-everything floor would draw a negative expectation.
    return np.asarray(np.where(np.isfinite(ei), ei, 0.0), dtype=float)


def acquire(name: str, inputs: AcquisitionInputs) -> np.ndarray:
    """Score every grid point under the named acquisition. Larger is better.

    Returns an array the same length as `inputs.mu`. The scores are comparable
    within an arm and meaningless across arms — `kg` is in units of improvement
    in the reported optimum, `mes` in nats — so callers may take the argmax and
    must not compare magnitudes or test them against a threshold.
    """
    if name not in ACQUISITIONS:
        raise ValueError(f"acquisition must be one of {ACQUISITIONS}; got {name!r}")
    return _never_pick_non_finite(_DISPATCH[name](inputs))


def _ei(inputs: AcquisitionInputs) -> np.ndarray:
    return expected_improvement(inputs.mu, inputs.sigma, inputs.f_best_observed, inputs.xi)


def _ei_plugin(inputs: AcquisitionInputs) -> np.ndarray:
    return expected_improvement(inputs.mu, inputs.sigma, inputs.f_best_plugin, inputs.xi)


def _aei(inputs: AcquisitionInputs) -> np.ndarray:
    """Augmented EI.

    EI against the plug-in incumbent, discounted by how much of the remaining
    uncertainty is irreducible measurement noise rather than model ignorance.
    """
    sigma = _finite_sigma(inputs.sigma)
    noise = float(inputs.noise_std)
    ei = expected_improvement(inputs.mu, sigma, inputs.f_best_plugin, inputs.xi)
    # -> 1 where the model is uncertain, -> 0 where only the instrument is.
    penalty = 1.0 - noise / np.sqrt(sigma**2 + noise**2)
    return np.asarray(ei * penalty, dtype=float)


def _pi(inputs: AcquisitionInputs) -> np.ndarray:
    sigma = _finite_sigma(inputs.sigma)
    z = (np.asarray(inputs.mu, dtype=float) - inputs.f_best_observed - inputs.xi) / sigma
    return np.asarray(norm.cdf(z), dtype=float)


def _ucb(inputs: AcquisitionInputs) -> np.ndarray:
    return np.asarray(inputs.mu + _UCB_LAMBDA * _finite_sigma(inputs.sigma), dtype=float)


def _mes(inputs: AcquisitionInputs) -> np.ndarray:
    """Max-value entropy search, with Wang & Jegelka's Gumbel approximation.

    The mutual information between a sample at x and the optimum's value has a
    closed form once the optimum's value is sampled. Sampling it exactly needs
    posterior draws over the whole grid; the Gumbel approximation instead treats
    the grid maxima as independent, giving
    `P(max <= y) = prod_i Phi((y - mu_i) / sigma_i)`, and fits a Gumbel to that
    by matching three quantiles. Independence is wrong — neighbouring grid
    points are strongly correlated — but it is wrong in a way that widens the
    sampled distribution of the optimum, which is the safe direction here.
    """
    mu = np.asarray(inputs.mu, dtype=float)
    sigma = _finite_sigma(inputs.sigma)

    def log_cdf_max(y: float) -> float:
        return float(np.sum(norm.logcdf((y - mu) / sigma)))

    # Bracket the median of the max, then bisect for the three quartiles.
    lo = float(np.max(mu))
    hi = float(np.max(mu + 5.0 * sigma))
    if not hi > lo:
        hi = lo + max(float(inputs.noise_std), _SIGMA_FLOOR)
    while log_cdf_max(hi) < np.log(0.75) and hi - lo < _MES_BRACKET_LIMIT:
        hi += hi - lo

    def quantile(p: float) -> float:
        target, a, b = np.log(p), lo, hi
        for _ in range(50):
            mid = 0.5 * (a + b)
            if log_cdf_max(mid) < target:
                a = mid
            else:
                b = mid
        return 0.5 * (a + b)

    q25, q50, q75 = quantile(0.25), quantile(0.50), quantile(0.75)
    # Gumbel: y = loc - scale * log(-log p). The 25/75 spread fixes the scale,
    # the median fixes the location.
    denom = np.log(np.log(4.0)) - np.log(np.log(4.0 / 3.0))
    scale = (q75 - q25) / denom if denom > 0 else 0.0
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.mean(sigma)), _SIGMA_FLOOR)
    loc = q50 + scale * np.log(np.log(2.0))

    rng = np.random.default_rng(0)  # fixed: this is a property of the posterior
    y_star = loc - scale * np.log(-np.log(rng.uniform(size=_MES_SAMPLES)))
    # Every sampled optimum must be at least as good as the current best mean,
    # or the information term is being asked about an impossible world.
    y_star = np.maximum(y_star, float(np.max(mu)) + _SIGMA_FLOOR)

    gamma = (y_star[None, :] - mu[:, None]) / sigma[:, None]
    log_cdf = norm.logcdf(gamma)
    info = gamma * np.exp(norm.logpdf(gamma) - log_cdf) / 2.0 - log_cdf
    return np.asarray(np.mean(info, axis=1), dtype=float)


def _kg(inputs: AcquisitionInputs) -> np.ndarray:
    """Knowledge gradient on the grid: the expected gain in `max_x mu(x)`.

    One observation at candidate c moves the whole posterior mean along a single
    direction, because the update is rank one:

        mu_new(x) = mu(x) + sigma_tilde(x, c) * Z,    Z ~ N(0, 1)
        sigma_tilde(x, c) = cov(x, c) / sqrt(cov(c, c) + noise^2)

    so the post-observation best reported value is the maximum of a family of
    straight lines in one standard normal variable, whose expectation is exact
    and needs no sampling. KG is that expectation minus the present best mean,
    and it is zero only when no observation anywhere could change the answer.

    Returns zeros when the posterior covariance was not supplied, which the
    caller must treat as "unavailable" rather than "nothing to gain".
    """
    if inputs.cov is None:
        return np.zeros_like(np.asarray(inputs.mu, dtype=float))

    mu = np.asarray(inputs.mu, dtype=float)
    cov = np.asarray(inputs.cov, dtype=float)
    # A covariance on a different grid from the mean would still slice and still
    # produce numbers, and the numbers would be meaningless. Said plainly here
    # because the failure otherwise surfaces as an opaque sort error deep inside
    # the hull routine.
    if cov.shape != (mu.size, mu.size):
        raise ValueError(f"cov must be ({mu.size}, {mu.size}) to match mu; got {cov.shape}")
    noise_var = float(inputs.noise_std) ** 2
    current_best = float(np.max(mu))

    n = mu.size
    stride = max(1, int(np.ceil(n / _KG_MAX_CANDIDATES)))
    candidates = np.arange(0, n, stride)

    scores = np.zeros(n, dtype=float)
    for c in candidates:
        denom = float(cov[c, c]) + noise_var
        if denom <= _SIGMA_FLOOR:
            continue
        sigma_tilde = cov[:, c] / np.sqrt(denom)
        scores[c] = _expected_max_affine(mu, sigma_tilde) - current_best
    # A tie across every candidate is not a decision, and must not be dressed
    # up as one. It happens for a specific and common reason: when the surrogate
    # over-smooths — the fitted length-scale pinned at its ceiling, which is
    # exactly what a prior mean provokes — `cov[:, c]` is nearly constant in i,
    # so `max_i(mu_i + s * Z) = max(mu) + s * Z` and the expectation is `max(mu)`
    # for every candidate. KG is then identically zero.
    #
    # Reported as no preference at all, the same signal as a missing covariance.
    # The alternative is worse than useless: with the sentinel scheme below,
    # `argmax` over an all-tied surface returns the first scored index, and the
    # caller synthesises the left-hand end of the search range while believing a
    # decision procedure chose it.
    evaluated = scores[candidates] if candidates.size else scores
    if evaluated.size and float(np.ptp(evaluated)) == 0.0:
        return np.zeros(n, dtype=float)
    # Strided candidates leave the gaps at zero, which would read as "no gain
    # here" rather than "not asked". Interpolating would invent a value KG never
    # computed, so the unevaluated points keep a score below every evaluated one
    # and the argmax can only ever land on a point that was actually scored.
    if candidates.size:
        floor = float(np.min(scores[candidates]))
        mask = np.ones(n, dtype=bool)
        mask[candidates] = False
        scores[mask] = floor - 1.0
    return scores


def _expected_max_affine(a: np.ndarray, b: np.ndarray) -> float:
    """E[max_i (a_i + b_i Z)] for Z standard normal, exactly.

    The lines that are nowhere the maximum are dropped first (Frazier, Powell &
    Dayanik 2009, Algorithm 1: an upper convex hull over the lines, computed with
    a stack). What survives is a piecewise-linear convex function of Z whose
    segment boundaries are the crossing points, so the expectation is a sum of
    integrals of `a + b z` against the normal density, using the identity
    `integral of z phi(z) dz = -phi(z)`.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    # Sort by slope; among equal slopes only the largest intercept can ever win.
    order = np.lexsort((a, b))
    a, b = a[order], b[order]
    last_of_run = np.ones(a.size, dtype=bool)
    last_of_run[:-1] = b[:-1] != b[1:]
    a, b = a[last_of_run], b[last_of_run]
    if a.size == 1:
        return float(a[0])

    # Upper hull. `cross[k]` is the value of Z at which line `keep[k]` takes over
    # from `keep[k-1]`; a line whose takeover point is not to the right of the
    # previous one is dominated everywhere and is removed.
    keep = [0]
    cross = [-np.inf]
    for j in range(1, a.size):
        while True:
            k = keep[-1]
            c = (a[k] - a[j]) / (b[j] - b[k])
            if len(keep) > 1 and c <= cross[-1]:
                keep.pop()
                cross.pop()
                continue
            keep.append(j)
            cross.append(c)
            break

    idx = np.asarray(keep, dtype=int)
    lo = np.asarray(cross, dtype=float)  # left edge of each surviving segment
    hi = np.append(lo[1:], np.inf)  # right edge
    a_k, b_k = a[idx], b[idx]

    # E[max] = sum_k  a_k * (Phi(hi) - Phi(lo)) + b_k * (phi(lo) - phi(hi))
    cdf_hi = np.where(np.isposinf(hi), 1.0, norm.cdf(hi))
    cdf_lo = np.where(np.isneginf(lo), 0.0, norm.cdf(lo))
    pdf_hi = np.where(np.isposinf(hi), 0.0, norm.pdf(hi))
    pdf_lo = np.where(np.isneginf(lo), 0.0, norm.pdf(lo))
    return float(np.sum(a_k * (cdf_hi - cdf_lo) + b_k * (pdf_lo - pdf_hi)))


_DISPATCH = {
    "ei": _ei,
    "ei_plugin": _ei_plugin,
    "aei": _aei,
    "kg": _kg,
    "mes": _mes,
    "ucb": _ucb,
    "pi": _pi,
}
