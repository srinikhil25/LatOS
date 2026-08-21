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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from latos.analysis.hall import cross_config_reliability
from latos.analysis.transport import TransportError
from latos.core.enums import Technique
from latos.server.transport_data import sample_zt

if TYPE_CHECKING:
    from latos.core.models import Project, Sample
    from latos.ingestion.array_store import ArrayStore
    from latos.server.synthesis_store import SynthesisParams

__all__ = [
    "DERIVED_ZT",
    "DatasetRow",
    "InputVariable",
    "QualityFlag",
    "SkippedSample",
    "build_dataset",
    "list_input_variables",
    "list_target_properties",
    "peak_target",
    "quality_flags",
]

# Hall-derived features that inherit the Hall measurement's reliability:
# each is computed from the (possibly noise-floor) Hall coefficient, so an
# unreliable Hall makes them untrustworthy. Conductivity, resistivity and
# sheet resistance are NOT here — they stay valid even when R_H is noise.
_HALL_CARRIER_FEATURES: frozenset[str] = frozenset(
    {"carrier_concentration_cm3", "mobility_cm2_vs", "hall_coefficient_cm3_c"}
)

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
        # Mechanical shock (drop test): the transmitted peak force is the
        # objective — minimized for better absorption.
        "peak_force_n",
    }
)

# Diagnostic features that are never a meaningful optimization axis
# (e.g. the two van der Pauw cross-configuration Hall coefficients —
# they exist to judge Hall reliability, not to be optimized over).
_NON_AXIS_FEATURES: frozenset[str] = frozenset(
    {
        "hall_ac_cross_cm3_c",
        "hall_bd_cross_cm3_c",
        # Shock-test outcomes/diagnostics: the transmitted force is the
        # objective and its scatter/raw voltage are diagnostics — the
        # optimization axis is the composition (particle loading), not these.
        "peak_force_n",
        "peak_voltage_v",
        "peak_force_sd_n",
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


# Targets whose parser also records the scatter of repeat measurements on the
# same sample. That scatter IS the measurement noise, so where it exists it
# should be used instead of an assumed percentage. Mapped explicitly rather
# than guessed from the name, so this only ever claims what actually exists.
_TARGET_SD_FEATURE: dict[str, str] = {
    "peak_force_n": "peak_force_sd_n",
}

# Fewer samples than this and the pooled scatter is one sample's spread, which
# is not a repeatability estimate for the campaign.
_MIN_SAMPLES_FOR_NOISE = 2


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """One usable (input, target) point for optimization."""

    sample_id: str
    sample_name: str
    x: float
    y: float
    # Scatter of the repeat measurements behind `y`, in the target's units, or
    # None when the technique records a single value per sample.
    y_sd: float | None = None


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


@dataclass(frozen=True, slots=True)
class QualityFlag:
    """A point in the optimization dataset whose value is untrustworthy.

    Raised when a target or axis is a Hall-derived carrier metric from a
    Hall measurement flagged unreliable (cross-configurations disagree), or
    when the value is physically impossible (e.g. negative mobility). The
    optimizer still runs — the researcher is warned, not blocked — so the
    reliability gate and this flag together say *how much* and *why* to
    distrust the recommendation.
    """

    sample_name: str
    variable: str  # the target or input-variable name involved
    value: float
    reason: str


def _hall_features(sample: Sample) -> Mapping[str, float] | None:
    """The features of the sample's Hall measurement, or None."""
    for measurement in sample.measurements:
        if measurement.technique is Technique.HALL:
            return measurement.features
    return None


def _flag_reason(sample: Sample, variable: str, value: float) -> str | None:
    """Why this (variable, value) is a data-quality concern, or None.

    Combines the Hall cross-configuration verdict (for carrier metrics) with
    a physical-plausibility check (mobility cannot be negative).
    """
    reasons: list[str] = []
    if variable in _HALL_CARRIER_FEATURES:
        features = _hall_features(sample)
        if features is not None:
            level, reason = cross_config_reliability(features)
            if level == "unreliable" and reason:
                reasons.append(reason)
    if variable == "mobility_cm2_vs" and value < 0:
        reasons.append("mobility cannot be negative")
    return "; ".join(reasons) if reasons else None


def _zt_wf_reason(sample: Sample, store: ArrayStore) -> str | None:
    """Wiedemann–Franz flag for a sample's derived zT, or None.

    A derived zT built from a σ/κ pair that violates Wiedemann–Franz
    (measured κ below the electronic floor L·σ·T) is physically unreliable
    no matter how clean it looks — so a zT optimization target should warn.
    """
    try:
        zt = sample_zt(sample, store.load)
    except TransportError:
        return None
    if any("Wiedemann" in w for w in zt.warnings):
        return (
            "derived zT is physically unreliable: measured thermal conductivity is "
            "below the Wiedemann-Franz electronic floor (L * conductivity * T)"
        )
    return None


def quality_flags(
    project: Project,
    rows: list[DatasetRow],
    input_variable: str,
    target_property: str,
    store: ArrayStore,
) -> list[QualityFlag]:
    """Flag dataset points whose target or axis value can't be trusted.

    Checks each row's target (y) and input variable (x) against the Hall
    reliability of the same sample and against physical plausibility, and —
    for a derived-zT target — against Wiedemann–Franz consistency of the
    underlying transport. Returns one flag per (sample, variable) concern,
    so the UI can warn before the researcher acts on the recommendation.
    """
    by_id = {s.id: s for s in project.samples}
    flags: list[QualityFlag] = []
    for row in rows:
        sample = by_id.get(row.sample_id)
        if sample is None:
            continue
        for variable, value in ((target_property, row.y), (input_variable, row.x)):
            reason = _flag_reason(sample, variable, value)
            if variable == target_property and target_property == DERIVED_ZT:
                wf = _zt_wf_reason(sample, store)
                reason = "; ".join(r for r in (reason, wf) if r) or None
            if reason:
                flags.append(
                    QualityFlag(
                        sample_name=row.sample_name,
                        variable=variable,
                        value=value,
                        reason=reason,
                    )
                )
    return flags


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


def axis_values(project: Project, params: SynthesisParams, input_variable: str) -> dict[str, float]:
    """Per-sample value of one input axis, keyed by sample id.

    `build_dataset` resolves the *first* axis of a run while also resolving the
    target and skipping unusable samples. A multi-axis run needs the remaining
    axes for the samples that survived that pass, and needs them resolved the
    same way — synthesis parameter first, then measured feature — so a second
    axis means exactly what a first axis would have meant. Samples with no value
    are absent rather than filled in; the caller drops them.
    """
    out: dict[str, float] = {}
    for sample in project.samples:
        value = _resolve_x(sample, params, input_variable)
        if value is not None:
            out[sample.id] = value
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
            sd_feature = _TARGET_SD_FEATURE.get(target_property)
            rows.append(
                DatasetRow(
                    sample_id=sample.id,
                    sample_name=sample.canonical_name,
                    x=float(x),
                    y=float(y),
                    y_sd=feature_value(sample, sd_feature) if sd_feature else None,
                )
            )
    return rows, skipped


def measured_noise(rows: Sequence[DatasetRow]) -> float | None:
    """Pooled repeatability of a single measurement, or None if unrecorded.

    Each row may carry the scatter of its own repeats. Pooling them (the root
    mean of the variances) estimates how far one measurement moves for reasons
    that have nothing to do with the composition, which is exactly the floor an
    improvement has to clear.

    Returns None unless at least two samples carry a usable scatter: a floor
    inferred from a single sample is not a repeatability estimate, and falling
    back to the assumed relative noise is the honest outcome.
    """
    variances = [float(r.y_sd) ** 2 for r in rows if r.y_sd is not None and r.y_sd > 0]
    if len(variances) < _MIN_SAMPLES_FOR_NOISE:
        return None
    return float(np.sqrt(sum(variances) / len(variances)))
