"""Has the campaign settled? Measured across pre-registrations, not within one.

Every stopping signal in `engine.py` looks at a single fit: expected
improvement against the noise floor, the leave-one-out self-check, the
probability that the incumbent is within epsilon of the optimum. All of them
ask the model about itself, so all of them fail together when the model is
wrong.

This module asks a different question, from outside the model: **have the
recommendations stopped moving?** Ishiyama et al. (NPG Asia Mater. 16, 17,
2024) used exactly this signal, the distance between successive proposed
conditions, as their evidence that the optimization had converged; it fell as
their campaign progressed.

It is harder to fool than an in-model criterion, because it is a fact about
the record on disk. Each frozen pre-registration already stores the
recommendation it committed to, so the whole diagnostic reads data Latos has
been writing all along, and no run can retune it after the fact.

Drift is reported as a fraction of the search span, which makes it comparable
across variables measured in different units (wt%, degrees C, cm^-3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

__all__ = ["CampaignDrift", "DriftStep", "recommendation_drift"]


class Freeze(Protocol):
    """What this module needs from a frozen record.

    Structural rather than a `PreregEntry` import: drift is a property of the
    committed history, not of the class that happens to carry it, and keeping
    it structural leaves `validate.py` free to change around it.

    Declared read-only (properties, not bare attributes) so a frozen dataclass
    satisfies it — which `PreregEntry` is, and should be.
    """

    @property
    def created_at(self) -> str: ...

    @property
    def input_variable(self) -> str: ...

    @property
    def property_name(self) -> str: ...

    @property
    def direction(self) -> str: ...

    @property
    def recommended_x(self) -> float: ...

    @property
    def search_bounds(self) -> tuple[float, float] | None: ...


# Below this fraction of the search span, two successive recommendations are
# close enough to count as pointing at the same place.
_SETTLED_FRACTION = 0.05

# Two freezes is the minimum that can show movement at all.
_MIN_FREEZES_FOR_DRIFT = 2


@dataclass(frozen=True, slots=True)
class DriftStep:
    """How far the recommendation moved between two consecutive freezes."""

    from_created_at: str
    to_created_at: str
    from_x: float
    to_x: float
    distance: float  # absolute move, in the input variable's own units
    fraction_of_span: float  # the same move, relative to the search range


@dataclass(frozen=True, slots=True)
class CampaignDrift:
    """The movement of the recommendation across one objective's freezes.

    `settled` answers the question a researcher actually asks: is the tool
    still changing its mind? It is None when there is only one freeze, because
    a single point cannot show movement. Reporting "settled" off one freeze
    would be the same premature confidence this module exists to catch.
    """

    input_variable: str
    property_name: str
    direction: str
    n_freezes: int
    steps: tuple[DriftStep, ...]
    search_span: float | None  # None when no record carried usable bounds
    latest_fraction: float | None  # most recent move, relative to the span
    settled: bool | None
    note: str


def _span_from_entries(entries: Sequence[Freeze]) -> float | None:
    """Search span taken from the freezes themselves, newest usable one wins.

    The bounds are part of the frozen record, so this is the span the campaign
    actually committed to searching. Returns None when no record carries it
    (older records predate the field) rather than inventing one.
    """
    for entry in reversed(entries):
        bounds = entry.search_bounds
        if bounds is None:
            continue
        low, high = (float(v) for v in bounds)
        if high > low:
            return high - low
    return None


def _note(
    n_freezes: int,
    latest: float | None,
    settled: bool | None,
    settled_fraction: float,
) -> str:
    """One sentence a researcher can act on, in the style of ReliabilityReport."""
    if n_freezes < _MIN_FREEZES_FOR_DRIFT:
        return (
            "Only one freeze so far. Drift needs two recommendations to compare, "
            "so there is nothing to read yet."
        )
    if latest is None:
        return (
            f"{n_freezes} freezes, but none recorded its search bounds, so the "
            "movement cannot be scaled to the search range."
        )
    pct = latest * 100.0
    bar = settled_fraction * 100.0
    if settled:
        return (
            f"The last two recommendations differ by {pct:.1f}% of the search "
            f"range (under the {bar:.0f}% mark). The campaign is pointing at the "
            "same place twice, which is independent of what the model says about "
            "itself."
        )
    return (
        f"The last two recommendations differ by {pct:.1f}% of the search range "
        f"(over the {bar:.0f}% mark). The campaign is still moving, so treat a "
        "convergence claim from the model with caution."
    )


def recommendation_drift(
    entries: Sequence[Freeze],
    *,
    search_span: float | None = None,
    settled_fraction: float = _SETTLED_FRACTION,
) -> list[CampaignDrift]:
    """Group frozen pre-registrations by objective and measure the movement.

    Args:
        entries: `PreregEntry` objects, or anything matching `Freeze`. Order
            does not matter; they are sorted by `created_at` here.
        search_span: Override for the range the moves are scaled against.
            Normally left None, in which case the span comes from the frozen
            records themselves.
        settled_fraction: Moves below this fraction of the span count as
            "pointing at the same place".

    Returns:
        One `CampaignDrift` per (input variable, property, direction), each
        with the steps between consecutive freezes, oldest first.
    """
    grouped: dict[tuple[str, str, str], list[Freeze]] = {}
    for entry in entries:
        key = (entry.input_variable, entry.property_name, entry.direction)
        grouped.setdefault(key, []).append(entry)

    out: list[CampaignDrift] = []
    for (variable, prop, direction), group in grouped.items():
        ordered = sorted(group, key=lambda e: e.created_at)
        span = search_span if search_span is not None else _span_from_entries(ordered)

        steps: list[DriftStep] = []
        for previous, current in pairwise(ordered):
            distance = abs(float(current.recommended_x) - float(previous.recommended_x))
            # Without a span the distance is still true, just not comparable
            # across variables — so the fraction stays 0.0 and `latest_fraction`
            # below reports None rather than a number that means nothing.
            fraction = distance / span if span else 0.0
            steps.append(
                DriftStep(
                    from_created_at=previous.created_at,
                    to_created_at=current.created_at,
                    from_x=float(previous.recommended_x),
                    to_x=float(current.recommended_x),
                    distance=distance,
                    fraction_of_span=fraction,
                )
            )

        latest = steps[-1].fraction_of_span if (steps and span) else None
        settled = None if latest is None else latest <= settled_fraction
        out.append(
            CampaignDrift(
                input_variable=variable,
                property_name=prop,
                direction=direction,
                n_freezes=len(ordered),
                steps=tuple(steps),
                search_span=span,
                latest_fraction=latest,
                settled=settled,
                note=_note(len(ordered), latest, settled, settled_fraction),
            )
        )
    out.sort(key=lambda d: (d.property_name, d.input_variable))
    return out
