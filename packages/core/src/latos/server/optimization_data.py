"""Assemble the (X, y) optimization dataset from a project.

Bridges the stored synthesis parameters (X, the BO input) and the
measured properties (y, extracted from each sample's arrays) into the
table the optimization engine consumes. Samples missing either side are
reported as skipped, with a reason, so the UI can tell the user exactly
what to fill in.
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
    "SkippedSample",
    "build_dataset",
    "list_target_properties",
    "peak_target",
]

# Special target option: zT that Latos derives in-app from a sample's
# Resistivity/Seebeck + LFA measurements (provenance-tracked), as opposed
# to a pre-computed `zt` column read from a file. Preferred when available.
DERIVED_ZT = "zT (derived)"

# Columns that are genuine optimization *objectives* (things you maximize),
# as opposed to independent axes (temperature, 2θ, wavelength) or raw /
# intermediate quantities (resistivity, diffusivity, intensity). Only these
# populate the target dropdown, so the user never sees "maximize wavelength".
_OBJECTIVE_PROPERTIES: frozenset[str] = frozenset(
    {
        "zt",
        "power_factor",
        "seebeck_uv_k",
        "seebeck_uvk",
        "band_gap_ev",
    }
)


def derived_zt_peak(sample: Sample, store: ArrayStore) -> float | None:
    """Peak of the Latos-derived zT(T), or None if the sample can't derive it."""
    try:
        return sample_zt(sample, store.load).peak_zt
    except TransportError:
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
    """Distinct numeric property names available across the project.

    Drives the target dropdown. Loads arrays (cached), so it's a
    one-time cost when the Optimize screen opens.
    """
    names: set[str] = set()
    can_derive = False
    for sample in project.samples:
        for measurement in sample.measurements:
            names.update(store.load(measurement.id).keys())
        if not can_derive and derived_zt_peak(sample, store) is not None:
            can_derive = True
    # Keep only real objectives, so the dropdown never offers an axis or
    # a raw component ("maximize wavelength" / "maximize resistivity").
    ordered = sorted(n for n in names if n in _OBJECTIVE_PROPERTIES)
    # Surface the provenance-tracked derived zT first when any sample
    # has the R&S + LFA pair to compute it.
    return [DERIVED_ZT, *ordered] if can_derive else ordered


def build_dataset(
    project: Project,
    store: ArrayStore,
    params: SynthesisParams,
    input_variable: str,
    target_property: str,
) -> tuple[list[DatasetRow], list[SkippedSample]]:
    """Assemble (x, y) rows; report samples missing either side."""
    rows: list[DatasetRow] = []
    skipped: list[SkippedSample] = []
    for sample in project.samples:
        x = params.get(sample.id, {}).get(input_variable)
        if target_property == DERIVED_ZT:
            y = derived_zt_peak(sample, store)
            missing_reason = "no Resistivity/Seebeck + LFA pair to derive zT"
        else:
            y = peak_target(sample, store, target_property)
            missing_reason = f"no '{target_property}' data"
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
