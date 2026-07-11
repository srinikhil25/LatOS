"""Transport summary analyzer — per-measurement facts for R&S and LFA runs.

The zT kernel (`thermoelectric.py`) works at the *sample* level: it
needs both an R&S and an LFA measurement to assemble zT(T). This
analyzer works at the *measurement* level, so each thermoelectric file
shows meaningful analysis on its own:

- **Resistivity/Seebeck run** → carrier type from the Seebeck sign,
  S and ρ at the temperature extremes, and the power-factor curve
  PF(T) = S²/ρ with its peak.
- **LFA run** → the thermal-conductivity range and where it bottoms
  out (low κ at high T is what a thermoelectric wants).

Same plausibility-first policy as the rest of the analysis layer:
flag suspicious values, never silently correct them.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from latos.analysis.base_analyzer import AnalyzerInputs, AnalyzerOutput, BaseAnalyzer
from latos.core.enums import Severity, Technique
from latos.core.models import Measurement, ValidationIssue, utc_now

__all__ = ["TransportSummaryAnalyzer"]

# Plausibility bound on total thermal conductivity for bulk solids
# (diamond ~2000 W/m·K is the ceiling; thermoelectrics sit ~0.5–10).
_KAPPA_MAX_W_MK = 500.0


class TransportSummaryAnalyzer(BaseAnalyzer):
    """Headline numbers for a single R&S or LFA measurement."""

    name: ClassVar[str] = "transport-summary"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.THERMOELECTRIC,)
    default_params: ClassVar[dict[str, Any]] = {}

    def accepts(self, measurement: Measurement) -> bool:
        """Accept any thermoelectric measurement with a source file."""
        return len(measurement.files) > 0

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        """Dispatch on the arrays present: R&S run or LFA run."""
        arrays = inputs.arrays
        t = arrays.get("temperature_k")
        if t is None or np.asarray(t).size == 0:
            return _error("Missing temperature_k array — cannot summarize transport run.")
        t = np.asarray(t, dtype=np.float64)

        if "resistivity_uohm_m" in arrays and "seebeck_uv_k" in arrays:
            return _summarize_rs(
                t,
                np.asarray(arrays["resistivity_uohm_m"], dtype=np.float64),
                np.asarray(arrays["seebeck_uv_k"], dtype=np.float64),
            )
        if "thermal_conductivity" in arrays:
            return _summarize_lfa(
                t, np.asarray(arrays["thermal_conductivity"], dtype=np.float64),
            )
        return _error(
            "Unrecognized thermoelectric arrays — expected an R&S run "
            "(resistivity + Seebeck) or an LFA run (thermal conductivity).",
        )


def _summarize_rs(
    t: np.ndarray, rho_uohm_m: np.ndarray, s_uv_k: np.ndarray,
) -> AnalyzerOutput:
    """Seebeck sign, extremes, and the power-factor curve for an R&S run."""
    issues: list[ValidationIssue] = []

    if np.any(rho_uohm_m <= 0):
        issues.append(_warn(
            "resistivity",
            "Some resistivity values are ≤ 0 — power factor is not physical there.",
        ))
    sign = np.sign(s_uv_k[np.abs(s_uv_k) > 0])
    if sign.size and not (np.all(sign > 0) or np.all(sign < 0)):
        issues.append(_warn(
            "seebeck",
            "Seebeck changes sign across the temperature range — bipolar/mixed "
            "conduction, or a data problem worth checking.",
        ))

    # PF = S²·σ = S² / ρ.  S: µV/K → V/K; ρ: µΩ·m → Ω·m; PF → µW/(m·K²).
    with np.errstate(divide="ignore", invalid="ignore"):
        pf_w_mk2 = np.where(
            rho_uohm_m > 0,
            (s_uv_k * 1e-6) ** 2 / (rho_uohm_m * 1e-6),
            np.nan,
        )
    pf_uw_mk2 = pf_w_mk2 * 1e6

    i_smax = int(np.argmax(np.abs(s_uv_k)))
    carrier = "p-type (holes)" if float(np.median(s_uv_k)) > 0 else "n-type (electrons)"
    outputs: dict[str, Any] = {
        "kind": "resistivity + Seebeck",
        "temperature_range_k": [round(float(t.min()), 1), round(float(t.max()), 1)],
        "carrier_type_from_seebeck": carrier,
        "seebeck_max_uv_k": round(float(s_uv_k[i_smax]), 1),
        "seebeck_max_at_k": round(float(t[i_smax]), 1),
        "resistivity_range_uohm_m": [
            round(float(np.min(rho_uohm_m)), 4), round(float(np.max(rho_uohm_m)), 4),
        ],
    }
    if np.any(np.isfinite(pf_uw_mk2)):
        i_pf = int(np.nanargmax(pf_uw_mk2))
        outputs["peak_power_factor_uw_mk2"] = round(float(pf_uw_mk2[i_pf]), 1)
        outputs["peak_power_factor_at_k"] = round(float(t[i_pf]), 1)

    return AnalyzerOutput(
        outputs=outputs,
        derived_arrays={"temperature_k": t, "power_factor_uw_mk2": pf_uw_mk2},
        issues=tuple(issues),
    )


def _summarize_lfa(t: np.ndarray, kappa: np.ndarray) -> AnalyzerOutput:
    """Thermal-conductivity range for an LFA run, with plausibility flags."""
    issues: list[ValidationIssue] = []
    if np.any(kappa <= 0):
        issues.append(_warn(
            "thermal_conductivity",
            "Some thermal-conductivity values are ≤ 0 — check the LFA export.",
        ))
    if float(np.nanmax(kappa)) > _KAPPA_MAX_W_MK:
        issues.append(_warn(
            "thermal_conductivity",
            f"κ exceeds {_KAPPA_MAX_W_MK:.0f} W/(m·K) — implausible for a bulk "
            "thermoelectric; a unit error upstream is likely.",
        ))

    i_min = int(np.nanargmin(kappa))
    outputs: dict[str, Any] = {
        "kind": "LFA (thermal conductivity)",
        "temperature_range_k": [round(float(t.min()), 1), round(float(t.max()), 1)],
        "kappa_range_w_mk": [
            round(float(np.nanmin(kappa)), 3), round(float(np.nanmax(kappa)), 3),
        ],
        "kappa_min_at_k": round(float(t[i_min]), 1),
    }
    return AnalyzerOutput(outputs=outputs, derived_arrays={}, issues=tuple(issues))


def _warn(field: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        field=field, severity=Severity.WARNING, message=message, detected_at=utc_now(),
    )


def _error(message: str) -> AnalyzerOutput:
    return AnalyzerOutput(
        outputs={},
        derived_arrays={},
        issues=(
            ValidationIssue(
                field="analyze", severity=Severity.ERROR,
                message=message, detected_at=utc_now(),
            ),
        ),
    )
