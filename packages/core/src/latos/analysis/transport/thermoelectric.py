# ruff: noqa: N806, RUF001
# N806: thermoelectric code uses single uppercase letters (S, T) by convention.
# RUF001: provenance strings use scientific notation (ρ, σ, ×, –) on purpose —
# they are shown to materials scientists, who read them as physics, not typos.
"""Thermoelectric figure-of-merit (zT) kernel — *owns the derivation*.

zT is never measured directly. It is assembled from three separately
measured transport properties, on two different instruments, on two
different temperature grids:

    zT(T) = S(T)² · σ(T) · T / κ(T) = (S² / ρ) · T / κ

- S  Seebeck coefficient        (Linseis R&S, µV/K)
- ρ  electrical resistivity     (Linseis R&S, µΩ·m)   σ = 1/ρ
- κ  thermal conductivity       (LFA, W/m·K)

This module does exactly what the researcher does by hand — unit
normalisation, temperature-grid harmonisation, then the formula — but
records every step so the result is reproducible and auditable. It was
validated against Dhivya's hand calculation (reproduces peak zT ≈ 0.97).

The result is *not* trusted blindly: a physical-plausibility check flags
any zT outside [0, Z_MAX] so a unit slip can never silently feed the
optimizer (the exact failure mode the provenance work is built to catch).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["TransportError", "ZtResult", "compute_zt"]

# Unit conversions to SI.
_UOHM_M_TO_OHM_M = 1e-6  # µΩ·m → Ω·m
_UV_K_TO_V_K = 1e-6  # µV/K → V/K
_W_TO_UW = 1e6  # W/m·K² → µW/m·K² (display unit for power factor)

# Physical-plausibility bound on zT. Real bulk thermoelectrics sit well
# under this; a value above it almost always means a unit error upstream.
_ZT_MAX_PLAUSIBLE = 4.0


class TransportError(ValueError):
    """Raised when the transport inputs are empty or inconsistent."""


@dataclass(frozen=True, slots=True)
class ZtResult:
    """Derived thermoelectric properties on a common temperature grid.

    Arrays are aligned to `temperature_k` (the LFA grid). `provenance`
    lists the transformation steps in order; `warnings` carries
    plausibility flags for the reviewer.
    """

    temperature_k: NDArray[np.float64]
    seebeck_v_k: NDArray[np.float64]
    resistivity_ohm_m: NDArray[np.float64]
    conductivity_s_m: NDArray[np.float64]
    thermal_conductivity_w_mk: NDArray[np.float64]
    power_factor_uw_mk2: NDArray[np.float64]
    zt: NDArray[np.float64]
    peak_zt: float
    peak_zt_temperature_k: float
    provenance: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _as_sorted(
    t: NDArray[np.float64], *cols: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """Sort `t` ascending and apply the same order to every column."""
    order = np.argsort(t)
    return (t[order], *(c[order] for c in cols))


def compute_zt(
    *,
    rs_temperature_k: NDArray[np.float64],
    resistivity_uohm_m: NDArray[np.float64],
    seebeck_uv_k: NDArray[np.float64],
    lfa_temperature_k: NDArray[np.float64],
    thermal_conductivity_w_mk: NDArray[np.float64],
) -> ZtResult:
    """Derive zT(T) from raw R&S + LFA arrays, on the LFA temperature grid.

    The LFA grid is chosen as the common grid: κ is the limiting (and
    usually regularly-spaced) measurement. S and ρ are linearly
    interpolated onto it. Inputs are taken in their native units
    (µΩ·m, µV/K, W/m·K, K) and converted to SI internally.

    Raises:
        TransportError: if any array is empty or the R&S arrays differ
            in length.
    """
    rs_t = np.asarray(rs_temperature_k, dtype=np.float64)
    rho = np.asarray(resistivity_uohm_m, dtype=np.float64)
    S_uv = np.asarray(seebeck_uv_k, dtype=np.float64)
    lfa_t = np.asarray(lfa_temperature_k, dtype=np.float64)
    kappa = np.asarray(thermal_conductivity_w_mk, dtype=np.float64)

    if rs_t.size == 0 or lfa_t.size == 0:
        raise TransportError("Empty transport input arrays.")
    if not (rs_t.size == rho.size == S_uv.size):
        raise TransportError(
            f"R&S arrays differ in length: T={rs_t.size}, rho={rho.size}, S={S_uv.size}."
        )
    if lfa_t.size != kappa.size:
        raise TransportError(
            f"LFA arrays differ in length: T={lfa_t.size}, kappa={kappa.size}."
        )

    provenance: list[str] = []
    warnings: list[str] = []

    # 1. Unit normalisation to SI.
    rho_si = rho * _UOHM_M_TO_OHM_M
    S_si = S_uv * _UV_K_TO_V_K
    provenance.append("Converted resistivity µΩ·m → Ω·m (×1e-6)")
    provenance.append("Converted Seebeck µV/K → V/K (×1e-6)")
    provenance.append("Thermal conductivity already in W/m·K")

    # 2. Common grid = LFA grid; interpolate S, ρ onto it.
    rs_t_s, rho_s, S_s = _as_sorted(rs_t, rho_si, S_si)
    lfa_t_s, kappa_s = _as_sorted(lfa_t, kappa)
    S_grid = np.interp(lfa_t_s, rs_t_s, S_s)
    rho_grid = np.interp(lfa_t_s, rs_t_s, rho_s)
    provenance.append(
        f"Interpolated S and ρ onto the LFA grid "
        f"({lfa_t_s[0]:.0f}–{lfa_t_s[-1]:.0f} K, {lfa_t_s.size} pts), linear"
    )

    # Flag clamped extrapolation where the LFA grid runs past the R&S range.
    below = int(np.sum(lfa_t_s < rs_t_s[0]))
    above = int(np.sum(lfa_t_s > rs_t_s[-1]))
    if below or above:
        warnings.append(
            f"R&S data covers {rs_t_s[0]:.0f}–{rs_t_s[-1]:.0f} K; "
            f"{below + above} LFA grid point(s) outside this range use clamped "
            f"(extrapolated) S/ρ values."
        )

    # 3. The formula.
    sigma = 1.0 / rho_grid
    power_factor_si = S_grid**2 * sigma  # W/m·K²
    zt = power_factor_si * lfa_t_s / kappa_s
    provenance.append("σ = 1/ρ ; PF = S²·σ ; zT = PF·T/κ")

    # 4. Physical-plausibility gate.
    if np.any(zt < 0):
        warnings.append("Some zT values are negative — check input signs/units.")
    over = float(np.nanmax(zt)) if zt.size else 0.0
    if over > _ZT_MAX_PLAUSIBLE:
        warnings.append(
            f"Peak zT = {over:.2f} exceeds the plausible bound "
            f"({_ZT_MAX_PLAUSIBLE}); a unit error upstream is likely."
        )

    peak_i = int(np.nanargmax(zt))
    return ZtResult(
        temperature_k=lfa_t_s,
        seebeck_v_k=S_grid,
        resistivity_ohm_m=rho_grid,
        conductivity_s_m=sigma,
        thermal_conductivity_w_mk=kappa_s,
        power_factor_uw_mk2=power_factor_si * _W_TO_UW,
        zt=zt,
        peak_zt=float(zt[peak_i]),
        peak_zt_temperature_k=float(lfa_t_s[peak_i]),
        provenance=provenance,
        warnings=warnings,
    )
