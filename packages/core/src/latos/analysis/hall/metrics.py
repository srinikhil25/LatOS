"""Hall metrics analyzer — carrier type, concentration, mobility + a physics check.

A single-temperature Hall export carries no arrays; the parser distils
its scalar columns into `Measurement.features` (carrier concentration,
mobility, resistivity, conductivity, Hall coefficient, sheet
resistance). This analyzer turns those into interpreted outputs:

- **Carrier type** from the sign of the Hall coefficient (negative →
  electrons / n-type, positive → holes / p-type), falling back to the
  sign of the exported bulk carrier concentration.
- The headline transport numbers, restated with units.
- An **internal-consistency check**: conductivity recomputed from
  σ = q·n·μ must agree with the measured σ. A large mismatch means a
  unit slip or a bad contact somewhere upstream — exactly the kind of
  quiet error Latos exists to flag.

Plausibility gates follow the same philosophy as the zT kernel: flag,
never silently correct.
"""

from __future__ import annotations

from typing import Any, ClassVar

from latos.analysis.base_analyzer import AnalyzerInputs, AnalyzerOutput, BaseAnalyzer
from latos.core.enums import Severity, Technique
from latos.core.models import Measurement, ValidationIssue, utc_now

__all__ = ["HallMetricsAnalyzer"]

# Elementary charge (C). σ [S/cm] = q · n [cm⁻³] · μ [cm²/(V·s)].
_Q_COULOMB = 1.602176634e-19

# Relative σ mismatch beyond which we flag internal inconsistency.
# 25% absorbs normal rounding in instrument exports; a unit slip is
# orders of magnitude.
_SIGMA_MISMATCH_FRAC = 0.25

# Physical-plausibility bounds for bulk semiconductors at ~300 K.
_N_MIN_CM3, _N_MAX_CM3 = 1e10, 1e23
_MU_MAX_CM2_VS = 1e5

# The two van der Pauw cross-configurations should agree; when their
# magnitudes differ by more than this factor the Hall signal is rough.
_CROSS_RATIO_WARN = 3.0


class HallMetricsAnalyzer(BaseAnalyzer):
    """Interpret a single-point Hall export: carrier type + reliability.

    Beyond restating the headline metrics, two data-quality checks run:

    * **Cross-configuration reliability** — the export's AC and BD van
      der Pauw diagonals each yield a Hall coefficient. Opposite signs
      mean the Hall voltage is at the noise floor: the averaged R_H —
      and the carrier type / concentration / mobility derived from it —
      is not trustworthy (conductivity remains valid).
    * **Cross-technique carrier type** — when the caller provides the
      sample's Seebeck sign (from the R&S measurement), the two
      independent carrier-type determinations are compared. Agreement
      is corroboration; disagreement is flagged, and when the Hall
      signal is unreliable the Seebeck determination is the one to trust.
    """

    name: ClassVar[str] = "hall-metrics"
    # 1.1.0: cross-configuration reliability + Seebeck cross-check.
    version: ClassVar[str] = "1.1.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.HALL,)
    default_params: ClassVar[dict[str, Any]] = {
        # Sign of the sample's Seebeck coefficient (+1 p-type, -1 n-type),
        # injected by the caller when an R&S measurement exists. None = skip.
        "seebeck_sign": None,
    }

    def accepts(self, measurement: Measurement) -> bool:
        """Accept Hall measurements whose parser extracted scalar features."""
        return bool(measurement.features)

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        """Restate the Hall scalars with interpretation + quality checks."""
        f = inputs.measurement.features
        if all(
            f.get(k) is None
            for k in ("carrier_concentration_cm3", "mobility_cm2_vs", "hall_coefficient_cm3_c")
        ):
            return _error("Hall export carried no usable carrier metrics.")

        issues: list[ValidationIssue] = []
        outputs = _restate_metrics(f, issues)
        hall_reliable = _cross_configuration_check(f, outputs, issues)
        _consistency_check(f, outputs, issues)
        _cross_technique_check(
            f,
            inputs.params.get("seebeck_sign"),
            outputs,
            issues,
            hall_reliable=hall_reliable,
        )
        return AnalyzerOutput(outputs=outputs, derived_arrays={}, issues=tuple(issues))


def _restate_metrics(
    f: dict[str, float],
    issues: list[ValidationIssue],
) -> dict[str, Any]:
    """Interpreted outputs from the exported scalars, with plausibility flags."""
    outputs: dict[str, Any] = {}
    n_signed = f.get("carrier_concentration_cm3")
    mobility = f.get("mobility_cm2_vs")
    hall_coeff = f.get("hall_coefficient_cm3_c")

    # Carrier type: Hall-coefficient sign is authoritative; the
    # exported bulk concentration carries the same sign convention.
    sign_source = hall_coeff if hall_coeff is not None else n_signed
    if sign_source:
        outputs["carrier_type"] = "n-type (electrons)" if sign_source < 0 else "p-type (holes)"

    if n_signed is not None:
        n_abs = abs(n_signed)
        outputs["carrier_concentration_cm3"] = f"{n_abs:.3e}"
        if not _N_MIN_CM3 <= n_abs <= _N_MAX_CM3:
            issues.append(
                _warn(
                    "carrier_concentration",
                    f"Carrier concentration {n_abs:.2e} cm⁻³ is outside the "
                    f"plausible bulk range [{_N_MIN_CM3:.0e}, {_N_MAX_CM3:.0e}] — "
                    "check units or contact quality.",
                )
            )
    if mobility is not None:
        outputs["mobility_cm2_vs"] = round(mobility, 3)
        if not 0 < mobility <= _MU_MAX_CM2_VS:
            issues.append(
                _warn(
                    "mobility",
                    f"Mobility {mobility:.3g} cm²/(V·s) is outside the plausible "
                    f"range (0, {_MU_MAX_CM2_VS:.0e}] — check units.",
                )
            )
    for key, out_key in (
        ("resistivity_ohm_cm", "resistivity_ohm_cm"),
        ("conductivity_s_cm", "conductivity_s_cm"),
        ("hall_coefficient_cm3_c", "hall_coefficient_cm3_c"),
        ("sheet_resistance_ohm_sq", "sheet_resistance_ohm_sq"),
    ):
        value = f.get(key)
        if value is not None:
            outputs[out_key] = f"{value:.4e}"
    return outputs


def _cross_configuration_check(
    f: dict[str, float],
    outputs: dict[str, Any],
    issues: list[ValidationIssue],
) -> bool:
    """Do the AC and BD van der Pauw diagonals tell the same story?

    Returns False when the two cross Hall coefficients disagree in sign —
    the averaged R_H is then noise, and every quantity derived from it
    (carrier type, concentration, mobility) is marked unreliable. Missing
    cross data (older exports) skips the check and counts as reliable.
    """
    ac = f.get("hall_ac_cross_cm3_c")
    bd = f.get("hall_bd_cross_cm3_c")
    if not ac or not bd:
        return True

    if ac * bd < 0:
        outputs["carrier_type_reliability"] = "unreliable — cross-configurations disagree in sign"
        if "carrier_type" in outputs:
            outputs["carrier_type"] = f"{outputs['carrier_type']} — UNRELIABLE"
        issues.append(
            _warn(
                "hall_reliability",
                f"The two Hall cross-configurations disagree in sign "
                f"(AC {ac:+.3g}, BD {bd:+.3g} cm³/C) — the Hall voltage is at the noise "
                "floor. Carrier type, concentration and mobility are unreliable; "
                "conductivity and resistivity remain valid.",
            )
        )
        return False

    ratio = max(abs(ac), abs(bd)) / min(abs(ac), abs(bd))
    outputs["hall_cross_ratio"] = round(ratio, 1)
    if ratio > _CROSS_RATIO_WARN:
        outputs["carrier_type_reliability"] = (
            f"questionable — cross-configurations differ by {ratio:.0f}x"
        )
        issues.append(
            _warn(
                "hall_reliability",
                f"The two Hall cross-configurations agree in sign but differ by "
                f"{ratio:.0f}x (AC {ac:+.3g}, BD {bd:+.3g} cm³/C) — treat the carrier "
                "concentration and mobility as rough estimates.",
            )
        )
    else:
        outputs["carrier_type_reliability"] = "good — cross-configurations agree"
    return True


def _cross_technique_check(
    f: dict[str, float],
    seebeck_sign: object,
    outputs: dict[str, Any],
    issues: list[ValidationIssue],
    *,
    hall_reliable: bool,
) -> None:
    """Compare the Hall carrier type with the Seebeck determination.

    `seebeck_sign` is +1/-1 when the caller derived it from the sample's
    R&S measurement; anything else skips the check.
    """
    if not isinstance(seebeck_sign, (int, float)) or seebeck_sign == 0:
        return
    hall_sign = f.get("hall_coefficient_cm3_c") or f.get("carrier_concentration_cm3")
    if not hall_sign:
        return

    hall_p = hall_sign > 0
    seebeck_p = seebeck_sign > 0
    seebeck_type = "p-type (holes)" if seebeck_p else "n-type (electrons)"
    outputs["carrier_type_from_seebeck"] = seebeck_type
    if hall_p == seebeck_p:
        issues.append(
            ValidationIssue(
                field="carrier_type",
                severity=Severity.INFO,
                message=(
                    f"Cross-technique agreement: Hall and Seebeck independently "
                    f"indicate {seebeck_type}."
                ),
                detected_at=utc_now(),
            )
        )
        return

    hall_type = "p-type" if hall_p else "n-type"
    trust = (
        " Given the cross-configuration disagreement above, the Seebeck "
        "determination is the trustworthy one."
        if not hall_reliable
        else " Check both measurements — in a single-carrier material these must agree."
    )
    issues.append(
        _warn(
            "carrier_type",
            f"Cross-technique disagreement: Hall indicates {hall_type} but the "
            f"Seebeck sign indicates {seebeck_type.split(' ', 1)[0]}.{trust}",
        )
    )


def _consistency_check(
    f: dict[str, float],
    outputs: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    """Conductivity recomputed from q·n·μ must agree with the measured one."""
    n_signed = f.get("carrier_concentration_cm3")
    mobility = f.get("mobility_cm2_vs")
    resistivity = f.get("resistivity_ohm_cm")
    sigma_meas = f.get("conductivity_s_cm")
    if sigma_meas is None and resistivity:
        sigma_meas = 1.0 / resistivity
    if n_signed is None or mobility is None or not sigma_meas:
        return

    sigma_calc = _Q_COULOMB * abs(n_signed) * abs(mobility)
    outputs["conductivity_from_n_mu_s_cm"] = f"{sigma_calc:.4e}"
    rel = abs(sigma_calc - abs(sigma_meas)) / abs(sigma_meas)
    outputs["consistency_deviation_pct"] = round(rel * 100, 1)
    if rel > _SIGMA_MISMATCH_FRAC:
        issues.append(
            _warn(
                "consistency",
                f"Conductivity from q·n·μ ({sigma_calc:.3e} S/cm) disagrees with the "
                f"measured value ({abs(sigma_meas):.3e} S/cm) by {rel * 100:.0f}% — "
                "possible unit slip or unreliable Hall fit.",
            )
        )
    else:
        issues.append(
            ValidationIssue(
                field="consistency",
                severity=Severity.INFO,
                message=(
                    f"Internally consistent: conductivity from q·n·μ reproduces the "
                    f"measured value within {rel * 100:.0f}%."
                ),
                detected_at=utc_now(),
            )
        )


def _warn(field: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        field=field,
        severity=Severity.WARNING,
        message=message,
        detected_at=utc_now(),
    )


def _error(message: str) -> AnalyzerOutput:
    return AnalyzerOutput(
        outputs={},
        derived_arrays={},
        issues=(
            ValidationIssue(
                field="analyze",
                severity=Severity.ERROR,
                message=message,
                detected_at=utc_now(),
            ),
        ),
    )
