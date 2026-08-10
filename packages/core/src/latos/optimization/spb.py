"""Single-parabolic-band (SPB) model — physics for thermoelectric optimization.

The SPB model (one parabolic band, acoustic-phonon scattering) is the standard
analytic description of a thermoelectric's transport. Its central, non-obvious
prediction is that zT is **maximized at an intermediate carrier concentration**
— too few carriers starve the conductivity, too many collapse the Seebeck
coefficient. That interior optimum is exactly what the optimizer's "reach a
value" objective targets, and it is *physics*, not a fitted curve.

This module gives Latos a physical prior. The key quantities, as functions of
the **reduced Fermi level** η = E_F / k_B T:

* Seebeck coefficient      S(η)  = (k_B/e) · [2·F₁/F₀ − η]
* Lorenz number            L(η)  = (k_B/e)² · [3·F₂/F₀ − (2·F₁/F₀)²]
* figure of merit          zT(η, β) = s² / (L_red + 1/(β·F₀))

where Fⱼ(η) are the Fermi–Dirac integrals, s and L_red are S and L in units of
(k_B/e) and (k_B/e)², and β is the dimensionless material **quality factor**
(β = (k_B/e)²·σ_E0·T / κ_L). A larger β (better weighted mobility, lower lattice
thermal conductivity) is a better thermoelectric. The optimal reduced Fermi
level η*(β) is found numerically; it *decreases* as β grows, so ultralow-κ
materials (SnSe-like) peak at a lower carrier concentration and a higher Seebeck
coefficient, while lattice-κ-dominated materials peak nearer η ≈ 0. As a sanity
anchor, β ≈ 0.4 gives peak zT ≈ 1.0 at |S| ≈ 240 µV/K — the Bi₂Te₃ class.

Because S(η) is monotonic in η, a *measured* Seebeck coefficient fixes η without
needing the carrier concentration — which lets Latos apply this physics on the
reliable Seebeck axis even when the Hall data is noise (see the Hall reliability
check). See `fit_quality_factor` and `optimal_seebeck`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize

__all__ = [
    "K_B_OVER_E_UV_K",
    "SpbGuidance",
    "SpbPrior",
    "fit_quality_factor",
    "guidance",
    "lorenz_reduced",
    "make_spb_prior",
    "optimal_eta",
    "optimal_seebeck",
    "reduced_seebeck",
    "seebeck_uv_k",
    "zt",
]

# Boltzmann constant over the elementary charge, in µV/K. Sets the natural
# scale of the Seebeck coefficient (S = this × a dimensionless reduced value).
K_B_OVER_E_UV_K = 86.17333262

# The optimal reduced Fermi level lives in a modest window; searching it is
# cheap and well-conditioned.
_ETA_LO, _ETA_HI = -5.0, 15.0

# |Seebeck| tolerance (µV/K) for calling a sample "at" its zT optimum, rather
# than over/under-doped — keeps the verdict from flipping on measurement noise.
_SEEBECK_AT_OPTIMUM_TOL_UV_K = 5.0


def fermi_integral(j: float, eta: float) -> float:
    """Fermi–Dirac integral Fⱼ(η) = ∫₀^∞ xʲ / (1 + exp(x − η)) dx.

    Evaluated numerically; the integrand decays exponentially so a finite upper
    limit past the Fermi level is exact to machine precision.
    """
    upper = max(40.0, eta + 40.0)

    def integrand(x: float) -> float:
        # exp overflow-safe: for large (x-η) the term is ~exp(-(x-η)).
        # math.pow keeps the result typed float (x**j widens to Any in typeshed).
        return math.pow(x, j) / (1.0 + math.exp(min(x - eta, 700.0)))

    value: float = integrate.quad(integrand, 0.0, upper, limit=200)[0]
    return value


def reduced_seebeck(eta: float) -> float:
    """Seebeck coefficient in units of (k_B/e): s = 2·F₁/F₀ − η (r = −1/2)."""
    f0 = fermi_integral(0.0, eta)
    f1 = fermi_integral(1.0, eta)
    return 2.0 * f1 / f0 - eta


def lorenz_reduced(eta: float) -> float:
    """Lorenz number in units of (k_B/e)²: 3·F₂/F₀ − (2·F₁/F₀)² (r = −1/2)."""
    f0 = fermi_integral(0.0, eta)
    f1 = fermi_integral(1.0, eta)
    f2 = fermi_integral(2.0, eta)
    return 3.0 * f2 / f0 - (2.0 * f1 / f0) ** 2


def seebeck_uv_k(eta: float) -> float:
    """Absolute Seebeck coefficient at reduced Fermi level η, in µV/K."""
    return K_B_OVER_E_UV_K * reduced_seebeck(eta)


def zt(eta: float, beta: float) -> float:
    """SPB figure of merit at reduced Fermi level η and quality factor β."""
    f0 = fermi_integral(0.0, eta)
    s = reduced_seebeck(eta)
    lorenz = lorenz_reduced(eta)
    return s * s / (lorenz + 1.0 / (beta * f0))


def optimal_eta(beta: float) -> float:
    """Reduced Fermi level η* that maximizes zT for a given quality factor β."""
    res = optimize.minimize_scalar(
        lambda e: -zt(e, beta), bounds=(_ETA_LO, _ETA_HI), method="bounded"
    )
    return float(res.x)


def eta_from_seebeck(seebeck_abs_uv_k: float) -> float:
    """Recover the reduced Fermi level from a measured |Seebeck| (µV/K).

    S(η) decreases monotonically as η rises, so a measured Seebeck pins η
    without the carrier concentration — the crux of applying SPB physics on the
    reliable Seebeck axis.
    """
    target = abs(seebeck_abs_uv_k) / K_B_OVER_E_UV_K

    def diff(eta: float) -> float:
        return reduced_seebeck(eta) - target

    # s spans ~[0, large]; bracket wide and solve.
    return float(optimize.brentq(diff, _ETA_LO, 60.0))


def fit_quality_factor(seebeck_abs_uv_k: float, zt_measured: float) -> float:
    """Quality factor β implied by one (|Seebeck|, zT) measurement.

    Inverts zT(η, β) at the η fixed by the measured Seebeck — a one-point
    physical calibration of the material's quality factor.
    """
    eta = eta_from_seebeck(seebeck_abs_uv_k)
    f0 = fermi_integral(0.0, eta)
    s = reduced_seebeck(eta)
    lorenz = lorenz_reduced(eta)
    # zt = s² / (L + 1/(β F₀))  ⇒  β = 1 / (F₀ (s²/zt − L)).
    denom = f0 * (s * s / zt_measured - lorenz)
    if denom <= 0:
        raise ValueError("measured zT exceeds the SPB ceiling for this Seebeck")
    return 1.0 / denom


def optimal_seebeck(beta: float) -> float:
    """The |Seebeck| (µV/K) at peak zT for quality factor β.

    The physics-informed target for a "reach-a-value" Seebeck optimization: the
    carrier level (hence Seebeck) that maximizes zT for this material.
    """
    return abs(seebeck_uv_k(optimal_eta(beta)))


@dataclass(frozen=True)
class SpbGuidance:
    """Physics-informed read on where a material sits versus its zT optimum.

    Built from one reliable ``(|Seebeck|, zT)`` point. ``applicable`` is False
    when the single-parabolic-band model cannot describe the measurement (the
    measured zT exceeds the SPB ceiling at that Seebeck) — itself a meaningful
    signal of multi-band transport or a Seebeck-data problem, spelled out in
    ``note``.
    """

    applicable: bool
    note: str
    measured_seebeck_uv_k: float
    measured_zt: float
    beta: float | None = None
    optimal_seebeck_uv_k: float | None = None
    zt_ceiling: float | None = None
    direction: str | None = None  # "increase_seebeck" | "decrease_seebeck" | "at_optimum"


def guidance(seebeck_abs_uv_k: float, zt_measured: float) -> SpbGuidance:
    """Interpret a measured (|Seebeck|, zT) against single-parabolic-band physics.

    Returns where the material sits relative to its zT optimum and which way to
    move the carrier concentration (via the Seebeck coefficient) to improve it —
    or, when the point lies above the SPB ceiling, an ``applicable=False``
    verdict that flags multi-band behaviour / a data issue instead of inventing
    a number. All decisions run on the *Seebeck* axis, which stays reliable when
    the Hall carrier concentration is noise.
    """
    s_meas = abs(seebeck_abs_uv_k)
    try:
        beta = fit_quality_factor(s_meas, zt_measured)
    except ValueError:
        eta = eta_from_seebeck(s_meas)
        s = reduced_seebeck(eta)
        ceiling = s * s / lorenz_reduced(eta)  # β → ∞ limit at this η
        return SpbGuidance(
            applicable=False,
            note=(
                f"Measured zT {zt_measured:.2f} exceeds the single-band ceiling "
                f"({ceiling:.2f}) at |S| = {s_meas:.0f} µV/K. A single parabolic "
                "band cannot reach this zT at such a low Seebeck — expect "
                "multi-band transport, or check the Seebeck data/units."
            ),
            measured_seebeck_uv_k=s_meas,
            measured_zt=zt_measured,
            zt_ceiling=float(ceiling),
        )

    s_opt = optimal_seebeck(beta)
    # Higher |S| ⇔ lower carrier concentration. Tolerance keeps "at optimum"
    # from firing on measurement noise.
    if abs(s_meas - s_opt) <= _SEEBECK_AT_OPTIMUM_TOL_UV_K:
        direction, verb = "at_optimum", "already near its zT optimum"
    elif s_meas < s_opt:
        direction, verb = (
            "increase_seebeck",
            (
                f"under-doped for peak zT — raise |S| toward {s_opt:.0f} µV/K "
                "(lower the carrier concentration)"
            ),
        )
    else:
        direction, verb = (
            "decrease_seebeck",
            (
                f"over-doped for peak zT — lower |S| toward {s_opt:.0f} µV/K "
                "(raise the carrier concentration)"
            ),
        )
    return SpbGuidance(
        applicable=True,
        note=f"Quality factor β ≈ {beta:.2f}; the material is {verb}.",
        measured_seebeck_uv_k=s_meas,
        measured_zt=zt_measured,
        beta=beta,
        optimal_seebeck_uv_k=s_opt,
        direction=direction,
    )


# Resolution of the tabulated zT(η) curve. `zt` costs three numerical
# integrations per call, and the optimizer evaluates its prior on thousands of
# candidates and again inside every L-BFGS-B step — enough to dominate the run.
# zT(η) is smooth and one-dimensional, so tabulating it once and interpolating
# is exact to well below the measurement noise at a fraction of the cost.
_ETA_TABLE_POINTS = 401

# Two points fix a line through eta(x). One fixes only a level, and a prior with
# an invented slope is worse than no prior at all.
_MIN_SAMPLES_FOR_TREND = 2


@dataclass(frozen=True)
class SpbPrior:
    """A callable SPB prior mean over a synthesis parameter, with its workings.

    Callable so it can be handed straight to `optimize(prior_mean=...)`, and a
    dataclass so the fit that produced it stays inspectable — a prior that
    silently shapes a recommendation without saying what it assumed is exactly
    the sort of thing this tool exists to prevent.
    """

    beta: float  # quality factor, assumed constant across the series
    eta_intercept: float  # η(x) = intercept + slope · x
    eta_slope: float
    n_used: int  # samples the fit actually rested on
    n_excluded: int  # samples above the SPB ceiling, dropped
    note: str
    axis: int  # which input column carries the doping knob
    _eta_grid: tuple[float, ...]
    _zt_grid: tuple[float, ...]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Predicted zT at knob values `x`, shape (m,) or (m, d)."""
        points = np.asarray(x, dtype=float)
        knob = points if points.ndim == 1 else points[:, self.axis]
        eta = self.eta_intercept + self.eta_slope * knob
        # Clamped at the table edges on purpose: beyond them the single-band
        # picture has stopped describing the material anyway, and a flat
        # continuation is a more honest extrapolation than a fitted tail.
        return np.interp(eta, self._eta_grid, self._zt_grid)


def make_spb_prior(
    x_obs: np.ndarray,
    seebeck_abs_uv_k: np.ndarray,
    zt_obs: np.ndarray,
    *,
    axis: int = 0,
) -> SpbPrior:
    """Build an SPB prior mean for a doping-like knob, from measured samples.

    The engine's `prior_mean` wants zT as a function of the *synthesis knob*.
    SPB gives zT as a function of the *reduced Fermi level* η. `eta_from_seebeck`
    is what joins them: because S(η) is monotonic, a measured Seebeck pins η
    without needing the carrier concentration — which matters here because Hall
    data is frequently the least trustworthy number in the set.

    So each sample yields an (x, η) pair and a β, and the prior is
    `zT(η(x), β)`. What that buys is the interior optimum: SPB says zT peaks at
    an intermediate carrier concentration, so unlike a zero-mean GP the prior
    knows the far edge of the doping range is bad *before* anyone makes a sample
    there.

    **Two assumptions, and only the first is physics.**

    1. Single parabolic band with acoustic-phonon scattering. Standard, and
       samples that violate it are detected rather than absorbed: a measured zT
       above the SPB ceiling makes `fit_quality_factor` raise, and those points
       are excluded and counted in `n_excluded`. A high count is a real signal —
       multi-band transport, or a problem with the Seebeck data.
    2. **η is linear in the knob, and β is constant across the series.** This is
       *not* physics, it is the cheapest defensible bridge. β depends on
       weighted mobility and lattice thermal conductivity, both of which doping
       genuinely changes, and η actually tracks the logarithm of carrier
       concentration. With a handful of samples neither refinement is
       identifiable, so they are deliberately not attempted. Treat the prior as
       a shape, not a prediction.

    Args:
        x_obs: Observed knob values, shape (n,) or (n, d).
        seebeck_abs_uv_k: |Seebeck| per sample in µV/K. Sign is ignored; the
            model is symmetric in carrier type.
        zt_obs: Measured zT per sample.
        axis: Which column of a multi-column `x_obs` carries the knob.

    Returns:
        An `SpbPrior`, callable as `prior(x) -> zT`.

    Raises:
        ValueError: On mismatched lengths, or when fewer than two samples
            survive the SPB ceiling check — one point fixes a level but cannot
            fix a trend, and a prior with an invented slope is worse than none.
    """
    knob_all = np.asarray(x_obs, dtype=float)
    knob = knob_all if knob_all.ndim == 1 else knob_all[:, axis]
    seebeck = np.abs(np.asarray(seebeck_abs_uv_k, dtype=float))
    zt_measured = np.asarray(zt_obs, dtype=float)

    if not (knob.shape == seebeck.shape == zt_measured.shape):
        raise ValueError(
            f"x_obs, seebeck and zt must align: got {knob.shape}, "
            f"{seebeck.shape}, {zt_measured.shape}"
        )

    etas: list[float] = []
    betas: list[float] = []
    kept: list[float] = []
    excluded = 0
    for k, s, z in zip(knob, seebeck, zt_measured, strict=True):
        if not (np.isfinite(s) and np.isfinite(z)) or s <= 0 or z <= 0:
            excluded += 1
            continue
        try:
            betas.append(fit_quality_factor(float(s), float(z)))
        except ValueError:
            # Above the single-band ceiling — see assumption 1.
            excluded += 1
            continue
        etas.append(eta_from_seebeck(float(s)))
        kept.append(float(k))

    if len(kept) < _MIN_SAMPLES_FOR_TREND:
        raise ValueError(
            f"Need at least two samples that the SPB model can describe; "
            f"{len(kept)} of {knob.size} survived. A single point fixes the "
            "level but not the trend."
        )

    # Median rather than mean: one sample sitting just under the ceiling can
    # return a huge β, and with four points a mean would follow it.
    beta = float(np.median(betas))
    slope, intercept = np.polyfit(np.asarray(kept), np.asarray(etas), 1)

    eta_grid = np.linspace(_ETA_LO, _ETA_HI, _ETA_TABLE_POINTS)
    zt_grid = np.array([zt(float(e), beta) for e in eta_grid], dtype=float)

    peak_eta = float(eta_grid[int(np.argmax(zt_grid))])
    peak_knob = (peak_eta - intercept) / slope if slope != 0 else float("nan")
    note = (
        f"β ≈ {beta:.3f} from {len(kept)} samples"
        + (f" ({excluded} excluded above the SPB ceiling)" if excluded else "")
        + f"; η = {intercept:.2f} + {slope:.3f}·x, peak zT at η ≈ {peak_eta:.2f}"
        + (f", i.e. x ≈ {peak_knob:.2f}" if np.isfinite(peak_knob) else "")
    )

    return SpbPrior(
        beta=beta,
        eta_intercept=float(intercept),
        eta_slope=float(slope),
        n_used=len(kept),
        n_excluded=excluded,
        note=note,
        axis=axis,
        _eta_grid=tuple(float(v) for v in eta_grid),
        _zt_grid=tuple(float(v) for v in zt_grid),
    )
