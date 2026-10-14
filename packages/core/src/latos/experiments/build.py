"""Turn a producer's record table into observations, under one definition.

Pure, and it does no I/O. Each row either becomes an `Observation` or is
excluded with **every** reason that applies, not only the first: a row with a
missing value and a control out of tolerance has two problems, and reporting one
of them costs a second trip to the bench to find the other.

An excluded row is never dropped. It keeps its whole source record, so the
frozen record can show what was left out and why.

A row whose only reason is a design value outside its bounds is a complete
specimen that the search was declared not to cover. It is kept out of the fit
like any other excluded row, but it is labelled as such and carries the
observation it would have been. The bounds say where to look; they are not a
judgment of the specimen.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from latos.core.enums import Severity
from latos.core.models import ValidationIssue, utc_now
from latos.experiments.definition import (
    ExperimentDefinition,
    ResponseDefinition,
    VariableDefinition,
    VariableKind,
    VariableRole,
)
from latos.experiments.observations import (
    EXCLUDED_OUTSIDE_BOUNDS,
    ExcludedObservation,
    Observation,
)
from latos.experiments.records import Record, RecordTable, Scalar
from latos.experiments.validation import TableMismatch
from latos.experiments.values import MeasurementValue

__all__ = ["build_observations"]

_NUMERIC_KINDS = (VariableKind.CONTINUOUS, VariableKind.INTEGER)

# A deviation exactly at the tolerance must pass whichever way the last bit of
# a subtraction fell. This is representation error, not a widened tolerance.
_TOLERANCE_REL = 1e-9
_TOLERANCE_ABS = 1e-12


def build_observations(
    definition: ExperimentDefinition, table: RecordTable
) -> tuple[tuple[Observation, ...], tuple[ExcludedObservation, ...]]:
    """Check every row of `table` against `definition`.

    Args:
        definition: The experiment the rows are read under.
        table: One row per specimen, from the producer the definition names.

    Returns:
        `(observations, excluded)`.

    Raises:
        TableMismatch: If the table came from another producer contract, or
            lacks a declared column. Every missing column is listed.
    """
    _check_table(definition, table)
    sample_ids = [_sample_id(definition, row) for row in table.rows]
    counts = Counter(sid for sid in sample_ids if sid is not None)

    observations: list[Observation] = []
    excluded: list[ExcludedObservation] = []
    for row, sample_id in zip(table.rows, sample_ids, strict=True):
        reasons: list[str] = []
        issues: list[ValidationIssue] = list(row.issues)
        if sample_id is None:
            reasons.append("no sample id")
        elif counts[sample_id] > 1:
            reasons.append(f"sample id {sample_id!r} appears on {counts[sample_id]} rows")
        reasons += _objective_reasons(definition, row)
        design_reasons, n_outside = _design_reasons(definition, row)
        reasons += design_reasons
        control_reasons, control_issues = _control_checks(definition, row)
        reasons += control_reasons
        issues += control_issues

        # When every reason is a design value outside its bounds, the row is a
        # complete specimen that passed every check. It is kept whole.
        only_outside = n_outside > 0 and len(reasons) == n_outside
        if sample_id is None or (reasons and not only_outside):
            excluded.append(
                ExcludedObservation(record=row, sample_id=sample_id, reasons=tuple(reasons))
            )
            continue

        responses, response_issues = _responses(definition, row)
        issues += response_issues
        observation = Observation(
            sample_id=sample_id,
            inputs=_recorded(definition, row, metadata=False),
            metadata=_recorded(definition, row, metadata=True),
            responses=responses,
            issues=tuple(issues),
            producer_detail=row.producer_detail,
        )
        if only_outside:
            excluded.append(
                ExcludedObservation(
                    record=row,
                    sample_id=sample_id,
                    reasons=tuple(reasons),
                    category=EXCLUDED_OUTSIDE_BOUNDS,
                    observation=observation,
                )
            )
        else:
            observations.append(observation)
    return tuple(observations), tuple(excluded)


def _check_table(definition: ExperimentDefinition, table: RecordTable) -> None:
    if table.producer != definition.producer:
        raise TableMismatch(
            f"The table was produced under contract {table.producer!r}, but the definition "
            f"reads {definition.producer!r}. A different contract means a different "
            "transformation from raw measurement to value, so the rows are not comparable."
        )
    needed = [definition.sample_key]
    needed += [v.column for v in definition.variables if v.column is not None]
    for response in definition.responses:
        needed.append(response.column)
        if response.uncertainty_column is not None:
            needed.append(response.uncertainty_column)
    missing = sorted(set(needed) - set(table.columns))
    if missing:
        raise TableMismatch("The table lacks declared column(s): " + ", ".join(missing))


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _real(value: Any) -> float | None:
    """`value` as a finite float, or None when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sample_id(definition: ExperimentDefinition, row: Record) -> str | None:
    value = row.values.get(definition.sample_key)
    return None if _blank(value) else str(value).strip()


def _issue(field: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        field=field, severity=Severity.WARNING, message=message, detected_at=utc_now()
    )


def _outside(value: float, low: float | None, high: float | None) -> bool:
    return (low is not None and value < low) or (high is not None and value > high)


def _range_text(low: float | None, high: float | None) -> str:
    return f"[{'-inf' if low is None else f'{low:g}'}, {'inf' if high is None else f'{high:g}'}]"


def _objective_reasons(definition: ExperimentDefinition, row: Record) -> list[str]:
    response = definition.objective_response()
    value = _real(row.values.get(response.column))
    if value is None:
        # The producer's own explanation, when it gave one, is the reason.
        errors = [i.message for i in row.issues if i.severity is Severity.ERROR]
        return errors or [f"no value for {response.name}"]
    low, high = response.valid_range
    if _outside(value, low, high):
        return [f"{response.name} = {value:g} is outside its valid range {_range_text(low, high)}"]
    return []


def _design_reasons(definition: ExperimentDefinition, row: Record) -> tuple[list[str], int]:
    """Why the row's design values keep it out, and how many are bounds alone.

    Returns:
        The reasons in declared order, and the number of them that say only
        that a value lies outside its bounds.
    """
    reasons: list[str] = []
    n_outside = 0
    for variable in definition.design_variables():
        raw = row.values.get(variable.column or "")
        if variable.kind not in _NUMERIC_KINDS:
            if _blank(raw):
                reasons.append(f"no value for design variable {variable.name}")
            continue
        value = _real(raw)
        if value is None:
            reasons.append(f"no value for design variable {variable.name}")
            continue
        # The optimizer does not check that observations lie inside the bounds
        # it is given, so this is the only place that does.
        if variable.bounds is not None and _outside(value, variable.bounds[0], variable.bounds[1]):
            low, high = variable.bounds
            reasons.append(f"{variable.name} = {value:g} is outside its bounds [{low:g}, {high:g}]")
            n_outside += 1
    return reasons, n_outside


def _control_checks(
    definition: ExperimentDefinition, row: Record
) -> tuple[list[str], list[ValidationIssue]]:
    """Compare every recorded control with what the definition expects.

    A control out of tolerance excludes the specimen: it observed a slightly
    different function, which is bias and not noise, so widening its error bar
    would misrepresent it. A control that was simply not recorded does not
    exclude; treating a blank more harshly than a control that is never
    recorded at all would punish recording.
    """
    reasons: list[str] = []
    issues: list[ValidationIssue] = []
    for variable in definition.variables:
        if variable.role is not VariableRole.CONTROL or variable.column is None:
            continue
        raw = row.values.get(variable.column)
        if _blank(raw):
            issues.append(
                _issue(
                    f"control:{variable.name}",
                    f"control {variable.name} not recorded for this specimen",
                )
            )
            continue
        reason = _control_violation(variable, raw)
        if reason is not None:
            reasons.append(reason)
    return reasons, issues


def _control_violation(variable: VariableDefinition, raw: Scalar) -> str | None:
    expected = variable.expected_value
    if variable.kind is not VariableKind.CONTINUOUS:
        if raw == expected:
            return None
        return f"control {variable.name}: recorded {raw!r}, expected {expected!r}"

    actual = _real(raw)
    if actual is None:
        return f"control {variable.name}: recorded {raw!r} is not a number"
    target = float(expected)  # type: ignore[arg-type]  # validated numeric for this kind
    tolerance = float(variable.tolerance or 0.0)
    deviation = abs(actual - target)
    within = deviation <= tolerance or math.isclose(
        deviation, tolerance, rel_tol=_TOLERANCE_REL, abs_tol=_TOLERANCE_ABS
    )
    if within:
        return None
    # No tolerance is a required value, not a measurement claimed to be exact.
    wanted = f"exactly {target:g}" if tolerance == 0 else f"{target:g} ± {tolerance:g}"
    return f"control {variable.name}: recorded {actual:g} {variable.unit}, expected {wanted}"


def _responses(
    definition: ExperimentDefinition, row: Record
) -> tuple[dict[str, MeasurementValue], list[ValidationIssue]]:
    values: dict[str, MeasurementValue] = {}
    issues: list[ValidationIssue] = []
    for response in definition.responses:
        value = _real(row.values.get(response.column))
        if value is None:
            # A missing non-objective response is simply absent; the objective
            # was checked before this is reached.
            continue
        low, high = response.valid_range
        if response.name != definition.objective.response and _outside(value, low, high):
            issues.append(
                _issue(
                    response.name,
                    f"{response.name} = {value:g} is outside its valid range "
                    f"{_range_text(low, high)}",
                )
            )
        uncertainty, issue = _uncertainty(response, row)
        if issue is not None:
            issues.append(issue)
        values[response.name] = MeasurementValue(
            value=value,
            unit=response.unit,
            uncertainty=uncertainty,
            uncertainty_scope=response.uncertainty_scope if uncertainty is not None else None,
            provenance=_provenance(response, row, with_uncertainty=uncertainty is not None),
        )
    return values, issues


def _provenance(response: ResponseDefinition, row: Record, *, with_uncertainty: bool) -> str:
    """Where a value came from and, when it has one, what its uncertainty is.

    A value carries one statement. What the producer said about the uncertainty
    is joined to it here, because a number offered as an uncertainty without its
    basis is the number most likely to be misread downstream.
    """
    stated = row.provenance.get(response.column, "")
    if not with_uncertainty or response.uncertainty_column is None:
        return stated
    basis = row.provenance.get(response.uncertainty_column, "")
    if not basis:
        return stated
    return f"{stated}; uncertainty: {basis}" if stated else f"uncertainty: {basis}"


def _uncertainty(
    response: ResponseDefinition, row: Record
) -> tuple[float | None, ValidationIssue | None]:
    """The 1-sigma uncertainty of a response, after the coverage factor."""
    if response.uncertainty_column is None:
        return None, None
    raw = row.values.get(response.uncertainty_column)
    if raw is None:
        return None, None
    value = _real(raw)
    if value is None or value < 0:
        return None, _issue(
            response.uncertainty_column,
            f"the uncertainty of {response.name} ({raw!r}) is not a finite, non-negative "
            "number, so it was treated as absent",
        )
    return value / response.coverage_factor, None


def _recorded(
    definition: ExperimentDefinition, row: Record, *, metadata: bool
) -> dict[str, Scalar]:
    """The recorded values of the metadata variables, or of all the others."""
    out: dict[str, Scalar] = {}
    for variable in definition.variables:
        if variable.column is None or (variable.role is VariableRole.METADATA) != metadata:
            continue
        raw = row.values.get(variable.column)
        number = _real(raw) if variable.kind in _NUMERIC_KINDS else None
        out[variable.name] = number if number is not None else raw
    return out
