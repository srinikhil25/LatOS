"""A measured value, and what its uncertainty is an uncertainty *of*.

Two numbers with the same name are routinely confused. One says how well a
single specimen's value was determined by the measurements made on it. The
other says how well that value estimates the response at the condition the
specimen was made under, which also includes everything that differs between
one specimen and the next. They are not interchangeable, and a surrogate fed the
first while believing it is the second is over-confident by an unknown amount.
So every uncertainty here carries its scope, and nothing downstream may use one
without reading it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["MeasurementValue", "UncertaintyScope"]


class UncertaintyScope(StrEnum):
    """What a stated uncertainty covers.

    SPECIMEN is a within-specimen component: how well this one specimen's value
    is determined by the repeated or fitted measurements made on it in one
    session. It leaves out specimen-to-specimen variation, so as a statement
    about the condition it is always a lower bound.

    CONDITION is the uncertainty of the value as an estimate of the response at
    its condition, specimen-to-specimen variation included.
    """

    SPECIMEN = "specimen"
    CONDITION = "condition"


@dataclass(frozen=True, slots=True)
class MeasurementValue:
    """One response value of one specimen, as its producer reported it.

    Attributes:
        value: The value, in `unit`. Signed and unprojected: nothing here has
            taken a magnitude or a distance to a target.
        unit: The declared unit, as an exact string.
        uncertainty: The 1-sigma standard uncertainty of `value`, already
            divided by any coverage factor. Never an expanded value. None when
            no estimate exists.
        uncertainty_scope: What `uncertainty` covers. Required exactly when
            `uncertainty` is given.
        provenance: The producer's statement of where the value came from, and
            for a SPECIMEN-scope uncertainty, the basis it rests on.
    """

    value: float
    unit: str
    uncertainty: float | None = None
    uncertainty_scope: UncertaintyScope | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"MeasurementValue.value must be finite; got {self.value!r}")
        if not self.unit:
            raise ValueError("MeasurementValue.unit cannot be empty; use '1' for dimensionless")
        if self.uncertainty is None:
            if self.uncertainty_scope is not None:
                raise ValueError("uncertainty_scope is set but there is no uncertainty")
            return
        if not math.isfinite(self.uncertainty) or self.uncertainty < 0:
            raise ValueError(
                f"MeasurementValue.uncertainty must be finite and non-negative; "
                f"got {self.uncertainty!r}"
            )
        if self.uncertainty_scope is None:
            raise ValueError("an uncertainty needs an uncertainty_scope saying what it covers")
