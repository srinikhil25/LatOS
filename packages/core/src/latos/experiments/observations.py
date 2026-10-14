"""What a record table becomes on its way to the optimizer.

Three stages, each a frozen type:

* an `Observation` is one specimen that was read successfully and passed the
  definition's checks, with every raw value still on it;
* a `ReducedObservation` is one condition, the specimens at it reduced to a
  signed value and an uncertainty that carries a label saying what it is;
* an `ObservationSet` holds both, along with everything that was excluded and
  why.

Nothing is erased on the way. An excluded row keeps its whole source record,
and a reduced condition keeps the ids of the specimens it came from.

Two different things keep a row out of the fit, and they are kept apart. A row
can be *rejected*: a value is missing or was refused, or a control is out of
tolerance. Or a row can be a complete specimen with no reason to leave it out
except that it lies *outside the declared design bounds*. The second says something
about where the search was declared to look and nothing about the specimen, so
that row keeps its full observation and is labelled for what it is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from latos.core.enums import Severity
from latos.core.models import ValidationIssue
from latos.experiments.records import Record, Scalar
from latos.experiments.values import MeasurementValue

__all__ = [
    "EXCLUDED_OUTSIDE_BOUNDS",
    "EXCLUDED_REJECTED",
    "EXCLUSION_GLOSSARY",
    "ConditionIssue",
    "ExcludedObservation",
    "Observation",
    "ObservationSet",
    "ReducedObservation",
]

EXCLUDED_REJECTED = "rejected"
EXCLUDED_OUTSIDE_BOUNDS = "outside_design_bounds"

EXCLUSION_GLOSSARY: Mapping[str, str] = {
    EXCLUDED_REJECTED: (
        "The row could not be used: a value was missing or refused, or a control was out "
        "of tolerance."
    ),
    EXCLUDED_OUTSIDE_BOUNDS: (
        "An accepted specimen: complete, and it passed the same checks as every other "
        "accepted specimen, its recorded controls included. Its only reason to be left out "
        "of the fit is that its design value lies outside the declared design bounds. It is "
        "kept as evidence and is not part of the fit."
    ),
}


@dataclass(frozen=True, slots=True)
class Observation:
    """One specimen that entered the model.

    Attributes:
        sample_id: The specimen's id.
        inputs: The recorded value of every DESIGN and ENVIRONMENT variable,
            and of every CONTROL that names a column. Actual values, never
            nominal ones.
        metadata: The recorded value of every METADATA variable.
        responses: Every declared response the row had a value for.
        issues: The producer's issues for the row, plus any found while
            building it.
        producer_detail: The producer's opaque detail for the row, carried
            unread to the frozen record.
    """

    sample_id: str
    inputs: Mapping[str, Scalar]
    metadata: Mapping[str, Scalar]
    responses: Mapping[str, MeasurementValue]
    issues: tuple[ValidationIssue, ...] = ()
    producer_detail: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ExcludedObservation:
    """A row that did not enter the model, with every reason why.

    Attributes:
        record: The whole source record, so nothing about the row is lost.
        sample_id: The specimen's id, or None when the row had none.
        reasons: Every reason for the exclusion, not only the first.
        category: A label from `EXCLUSION_GLOSSARY`: rejected, or a complete
            specimen outside the design bounds.
        observation: For a specimen outside the design bounds, the observation
            it would have been, with its values and labelled uncertainties.
            None for a rejected row.
    """

    record: Record
    sample_id: str | None
    reasons: tuple[str, ...]
    category: str = EXCLUDED_REJECTED
    observation: Observation | None = None


@dataclass(frozen=True, slots=True)
class ConditionIssue:
    """Something worth saying about a reduced condition.

    Kept separate from `ValidationIssue`, which carries a timestamp: a reduction
    has to give the same result every time it is run.

    Attributes:
        code: A stable identifier, such as ``sign_instability``.
        severity: How serious it is.
        message: A sentence for a person.
    """

    code: str
    severity: Severity
    message: str


@dataclass(frozen=True, slots=True)
class ReducedObservation:
    """One condition: the specimens made at it, reduced.

    Attributes:
        x: The DESIGN values in declared order; for a numeric axis, the mean
            over the specimens.
        y_raw: The mean of the specimens' **signed** objective-response values.
            No projection has been applied.
        sigma: The 1-sigma uncertainty of `y_raw`, or None when no estimate
            exists. What it covers is stated by `sigma_source`.
        sigma_source: A label from `reduce.SIGMA_GLOSSARY`.
        n_specimens: How many specimens were reduced.
        sample_ids: Their ids.
        issues: Anything worth saying about the condition.
    """

    x: tuple[float | str, ...]
    y_raw: float
    sigma: float | None
    sigma_source: str
    n_specimens: int
    sample_ids: tuple[str, ...]
    issues: tuple[ConditionIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservationSet:
    """Everything known about a campaign's data under one definition.

    Attributes:
        definition_hash: The identity hash of the definition these were built
            and reduced under.
        observations: Every specimen that entered, ordered by design values.
        excluded: Every row that did not, with reasons.
        reduced: One entry per condition.
        messages: Notes for the report and the record.
        pooled_sd: The pooled specimen-to-specimen standard deviation, or None
            when no condition was replicated.
        pooled_df: The degrees of freedom behind it.
    """

    definition_hash: str
    observations: tuple[Observation, ...]
    excluded: tuple[ExcludedObservation, ...]
    reduced: tuple[ReducedObservation, ...]
    messages: tuple[str, ...]
    pooled_sd: float | None
    pooled_df: int

    def outside_bounds(self) -> tuple[ExcludedObservation, ...]:
        """The complete specimens kept out of the fit by the design bounds alone."""
        return tuple(row for row in self.excluded if row.category == EXCLUDED_OUTSIDE_BOUNDS)

    def rejected(self) -> tuple[ExcludedObservation, ...]:
        """The rows that could not be used."""
        return tuple(row for row in self.excluded if row.category == EXCLUDED_REJECTED)
