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

from scipy import integrate, optimize

__all__ = [
    "K_B_OVER_E_UV_K",
    "SpbGuidance",
    "fit_quality_factor",
    "guidance",
    "lorenz_reduced",
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
