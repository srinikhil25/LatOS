"""Assemble the (X, y) optimization dataset from a project.

Bridges the BO inputs and the measured properties into the table the
optimization engine consumes:

* **X (input variable)** — a synthesis parameter entered by the
  researcher (doping %, etching time, …) *or* a measured per-sample
  feature (e.g. the Hall carrier concentration). Measured inputs matter
  when a sample set shares no synthesis knob: heterogeneous samples
  still land on a common physical axis such as carrier concentration.
* **y (target)** — the peak of a measured array property, the
  Latos-derived zT (optionally at a chosen temperature rather than the
  peak), or a measured scalar feature.

Samples missing either side are reported as skipped, with a reason, so
the UI can tell the user exactly what to fill in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from latos.analysis.transport import TransportError
from latos.server.transport_data import sample_zt

if TYPE_CHECKING:
    from latos.core.models import Project, Sample
    from latos.ingestion.array_store import ArrayStore
    from latos.server.synthesis_store import SynthesisParams

__all__ = [
    "DERIVED_ZT",
    "DatasetRow",
    "InputVariable",
    "SkippedSample",
    "build_dataset",
    "list_input_variables",
    "list_target_properties",
    "peak_target",
]

# Special target option: zT that Latos derives in-app from a sample's
# Resistivity/Seebeck + LFA measurements (provenance-tracked), as opposed
# to a pre-computed `zt` column read from a file. Preferred when available.
DERIVED_ZT = "zT (derived)"

# Columns that are genuine optimization *objectives*, as opposed to
# independent axes (temperature, 2θ, wavelength) or raw / intermediate
# quantities (resistivity, diffusivity, intensity). Only these populate
# the target dropdown, so the user never sees "maximize wavelength".
_OBJECTIVE_PROPERTIES: frozenset[str] = frozenset(
    {
        "zt",
        "power_factor",
        "seebeck_uv_k",
        "seebeck_uvk",
        "band_gap_ev",
        # A minimization objective (κ low is what a thermoelectric wants);
        # meaningful now that the engine supports direction.
        "thermal_conductivity",
    }
)

# Measured per-sample features (Measurement.features) that make sense as
# optimization targets. Curated for the same reason as above.
_OBJECTIVE_FEATURES: frozenset[str] = frozenset(
    {
        "carrier_concentration_cm3",
        "mobility_cm2_vs",
        "conductivity_s_cm",
    }
)

# Diagnostic features that are never a meaningful optimization axis
# (e.g. the two van der Pauw cross-configuration Hall coefficients —
# they exist to judge Hall reliability, not to be optimized over).
_NON_AXIS_FEATURES: frozenset[str] = frozenset(
    {
        "hall_ac_cross_cm3_c",
        "hall_bd_cross_cm3_c",
    }
)


def derived_zt_peak(sample: Sample, store: ArrayStore) -> float | None:
    """Peak of the Latos-derived zT(T), or None if the sample can't derive it."""
    try:
        return sample_zt(sample, store.load).peak_zt
    except TransportError:
        return None


def derived_zt_at(sample: Sample, store: ArrayStore, t_kelvin: float) -> float | None:
    """Latos-derived zT interpolated at `t_kelvin`, or None.

    Devices run at one temperature, not at the peak — so zT *at the
    operating temperature* is often the honest objective. Returns None
    when the sample can't derive zT or the requested temperature lies
    outside the measured range (extrapolating zT would be fiction).
    """
    try:
        zt = sample_zt(sample, store.load)
    except TransportError:
        return None
    t = np.asarray(zt.temperature_k, dtype=float)
    if not float(t.min()) <= t_kelvin <= float(t.max()):
        return None
    return float(np.interp(t_kelvin, t, np.asarray(zt.zt, dtype=float)))


def feature_value(sample: Sample, name: str) -> float | None:
    """First measured feature `name` found on this sample's measurements."""
    for measurement in sample.measurements:
        value = measurement.features.get(name)
        if value is not None:
            return float(value)
    return None


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """One usable (input, target) point for optimization."""

    sample_id: str
    sample_name: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class SkippedSample:
    """A sample left out of the dataset, with why."""

    sample_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class InputVariable:
    """One available BO input axis and its per-sample values.

    `source` distinguishes researcher-entered synthesis parameters
    (editable in the UI) from measured features (read-only — they came
    from an instrument, e.g. the Hall carrier concentration).
    """

    name: str
    source: str  # "synthesis" | "measured"
    values: dict[str, float]  # sample_id -> value


def peak_target(sample: Sample, store: ArrayStore, prop: str) -> float | None:
    """Peak value of property `prop` across a sample's measurements.

    The headline a thermoelectric paper reports is the *peak* of a
    property over its temperature sweep, so we take the max. Returns
    None if no measurement of this sample carries the property.
    """
    best: float | None = None
    for measurement in sample.measurements:
        column = store.load(measurement.id).get(prop)
        if column is not None and len(column) > 0:
            value = float(np.nanmax(column))
            best = value if best is None else max(best, value)
    return best


def list_target_properties(project: Project, store: ArrayStore) -> list[str]:
    """Distinct target names available across the project.

    Array properties (curated), measured features (curated), and the
    derived zT when any sample can compute it. Drives the target
    dropdown; loads arrays (cached), so it's a one-time cost when the
    Optimize screen opens.
    """
    names: set[str] = set()
    feature_names: set[str] = set()
    can_derive = False
    for sample in project.samples:
        for measurement in sample.measurements:
            names.update(store.load(measurement.id).keys())
            feature_names.update(measurement.features.keys())
        if not can_derive and derived_zt_peak(sample, store) is not None:
            can_derive = True
    ordered = sorted(n for n in names if n in _OBJECTIVE_PROPERTIES)
    ordered += sorted(n for n in feature_names if n in _OBJECTIVE_FEATURES)
    # Surface the provenance-tracked derived zT first when any sample
    # has the R&S + LFA pair to compute it.
    return [DERIVED_ZT, *ordered] if can_derive else ordered


def list_input_variables(project: Project, params: SynthesisParams) -> list[InputVariable]:
    """Every available BO input axis with its per-sample values.

    Synthesis parameters first (the researcher's own knobs), then
    measured features (a common physical axis when samples share no
    recipe knob — e.g. optimizing zT against the Hall carrier
    concentration, the classic Ioffe analysis).
    """
    synthesis: dict[str, dict[str, float]] = {}
    for sample_id, rec in params.items():
        for name, value in rec.items():
            synthesis.setdefault(name, {})[sample_id] = float(value)

    measured: dict[str, dict[str, float]] = {}
    for sample in project.samples:
        for measurement in sample.measurements:
            for name, value in measurement.features.items():
                measured.setdefault(name, {}).setdefault(sample.id, float(value))

    out = [
        InputVariable(name=n, source="synthesis", values=v) for n, v in sorted(synthesis.items())
    ]
    out += [
        InputVariable(name=n, source="measured", values=v)
        for n, v in sorted(measured.items())
        # A researcher-entered value wins the name; diagnostics are not axes.
        if n not in synthesis and n not in _NON_AXIS_FEATURES
    ]
    return out


def _resolve_x(sample: Sample, params: SynthesisParams, input_variable: str) -> float | None:
    """Input value for a sample: synthesis parameter first, then features."""
    x = params.get(sample.id, {}).get(input_variable)
    if x is not None:
        return float(x)
    return feature_value(sample, input_variable)


def _resolve_y(
    sample: Sample,
    store: ArrayStore,
    target_property: str,
    at_temperature_k: float | None,
) -> tuple[float | None, str]:
    """Target value for a sample, plus the reason to report if missing."""
    if target_property == DERIVED_ZT:
        if at_temperature_k is not None:
            return (
                derived_zt_at(sample, store, at_temperature_k),
                f"cannot derive zT at {at_temperature_k:g} K "
                "(missing R&S + LFA pair, or temperature outside the measured range)",
            )
        return (
            derived_zt_peak(sample, store),
            "no Resistivity/Seebeck + LFA pair to derive zT",
        )
    y = peak_target(sample, store, target_property)
    if y is None:
        y = feature_value(sample, target_property)
    return y, f"no '{target_property}' data"


def build_dataset(
    project: Project,
    store: ArrayStore,
    params: SynthesisParams,
    input_variable: str,
    target_property: str,
    *,
    at_temperature_k: float | None = None,
) -> tuple[list[DatasetRow], list[SkippedSample]]:
    """Assemble (x, y) rows; report samples missing either side.

    `input_variable` may name a synthesis parameter or a measured
    feature; `at_temperature_k` applies only to the derived-zT target
    (zT at that temperature instead of the peak).
    """
    rows: list[DatasetRow] = []
    skipped: list[SkippedSample] = []
    for sample in project.samples:
        x = _resolve_x(sample, params, input_variable)
        y, missing_reason = _resolve_y(sample, store, target_property, at_temperature_k)
        if x is None:
            skipped.append(SkippedSample(sample.canonical_name, f"no '{input_variable}' value"))
        elif y is None:
            skipped.append(SkippedSample(sample.canonical_name, missing_reason))
        else:
            rows.append(
                DatasetRow(
                    sample_id=sample.id,
                    sample_name=sample.canonical_name,
                    x=float(x),
                    y=float(y),
                )
            )
    return rows, skipped
