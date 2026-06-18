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

if TYPE_CHECKING:
    from latos.core.models import Project, Sample
    from latos.ingestion.array_store import ArrayStore
    from latos.server.synthesis_store import SynthesisParams

__all__ = [
    "DatasetRow",
    "SkippedSample",
    "build_dataset",
    "list_target_properties",
    "peak_target",
]


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
    for sample in project.samples:
        for measurement in sample.measurements:
            names.update(store.load(measurement.id).keys())
    return sorted(names)


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
        y = peak_target(sample, store, target_property)
        if x is None:
            skipped.append(SkippedSample(sample.canonical_name, f"no '{input_variable}' value"))
        elif y is None:
            skipped.append(SkippedSample(sample.canonical_name, f"no '{target_property}' data"))
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
