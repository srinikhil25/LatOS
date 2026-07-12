"""Physical-properties registry: the physics each measured quantity must obey.

Materials properties are not interchangeable numbers: mobility and thermal
conductivity are strictly positive and span orders of magnitude; the Seebeck
coefficient and the Hall coefficient carry a physically meaningful sign; the
band gap and zT live in a bounded window. This module records those facts once,
so both the analysis plausibility gates and the optimizer's physical
constraints read the same physics rather than scattering magic constants.

Two flags drive the optimizer's behaviour (see `latos.optimization.engine`):

* ``sign`` — ``"positive"`` (a prediction below zero is unphysical) or
  ``"any"`` (sign is meaningful, e.g. n-type Seebeck is negative).
* ``log_natural`` — the quantity varies over orders of magnitude with no upper
  bound (mobility, conductivity, resistivity, …). The surrogate fits it in log
  space, which both guarantees a positive prediction and matches how the
  quantity actually behaves (multiplicative, not additive, noise).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PhysicalProperty", "lookup"]


@dataclass(frozen=True, slots=True)
class PhysicalProperty:
    """The physical domain of one measurable/optimizable property.

    ``min_value`` / ``max_value`` are plausibility bounds (None = unbounded on
    that side). ``sign`` and ``log_natural`` steer the optimizer.
    """

    name: str
    unit: str
    sign: str  # "positive" | "any"
    log_natural: bool
    min_value: float | None = None
    max_value: float | None = None

    @property
    def positive(self) -> bool:
        """True when a value below zero is physically impossible."""
        return self.sign == "positive"


# Canonical registry. Bounds mirror the analyzers' plausibility gates
# (hall.metrics, transport.summary/thermoelectric, uv_drs.tauc) so there is
# one physics, not two.
_PROPERTIES: dict[str, PhysicalProperty] = {
    "zT (derived)": PhysicalProperty("zT", "dimensionless", "positive", False, 0.0, 4.0),
    "zt": PhysicalProperty("zT", "dimensionless", "positive", False, 0.0, 4.0),
    "power_factor": PhysicalProperty("power factor", "µW/m·K²", "positive", False, 0.0, None),
    "power_factor_uw_mk2": PhysicalProperty(
        "power factor", "µW/m·K²", "positive", False, 0.0, None
    ),
    "thermal_conductivity": PhysicalProperty(
        "thermal conductivity", "W/m·K", "positive", False, 0.0, 500.0
    ),
    "band_gap_ev": PhysicalProperty("band gap", "eV", "positive", False, 0.0, 10.0),
    # Sign-bearing quantities — a negative value is physically meaningful.
    "seebeck_uv_k": PhysicalProperty("Seebeck coefficient", "µV/K", "any", False),
    "seebeck_uvk": PhysicalProperty("Seebeck coefficient", "µV/K", "any", False),
    "hall_coefficient_cm3_c": PhysicalProperty("Hall coefficient", "cm³/C", "any", False),
    "carrier_concentration_cm3": PhysicalProperty(
        # Stored with the carrier-type sign convention (n-type negative), so
        # not strictly positive and not safe to log; bounds are on magnitude.
        "carrier concentration",
        "cm⁻³",
        "any",
        False,
        1e10,
        1e23,
    ),
    # Order-of-magnitude, strictly-positive transport quantities → log space.
    "mobility_cm2_vs": PhysicalProperty("mobility", "cm²/V·s", "positive", True, 0.0, 1e5),
    "conductivity_s_cm": PhysicalProperty("conductivity", "S/cm", "positive", True, 0.0, None),
    "resistivity_ohm_cm": PhysicalProperty("resistivity", "Ω·cm", "positive", True, 0.0, None),
    "sheet_resistance_ohm_sq": PhysicalProperty(
        "sheet resistance", "Ω/□", "positive", True, 0.0, None
    ),
}

# Property labels can arrive decorated: "zT (derived) @ 600 K", or a target-mode
# distance "|zT (derived) - 1.0|". The base lookup strips the temperature suffix
# and returns None for distances (they have no single-property physics).
_TEMP_SUFFIX = " @ "


def lookup(name: str) -> PhysicalProperty | None:
    """Physics for a property label, or None if it has none (e.g. a distance)."""
    base = name.split(_TEMP_SUFFIX, 1)[0].strip()
    return _PROPERTIES.get(base)
