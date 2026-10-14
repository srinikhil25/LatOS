"""The rules a definition must satisfy, and the errors the pipeline raises.

`validate_definition` reports **every** violated rule in one error. A
definition is written by a person, usually with several things wrong at once,
and being told about them one run at a time is how people stop reading the
messages.

A field set where it cannot be used is refused rather than ignored. Bounds on a
control or a tolerance on a categorical variable would otherwise be silently
dropped, leaving a declaration that says something the pipeline does not do.
"""

from __future__ import annotations

import math
import re
from typing import Any

from latos.experiments.definition import (
    ExperimentDefinition,
    Goal,
    ObjectiveDefinition,
    ResponseDefinition,
    VariableDefinition,
    VariableKind,
    VariableRole,
)
from latos.experiments.records import PRODUCER_CONTRACT_RE

__all__ = [
    "DefinitionError",
    "InsufficientConditions",
    "TableMismatch",
    "UnsupportedDesignSpace",
    "definition_warnings",
    "validate_definition",
]

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Kinds whose values are numbers, and so can carry bounds and a precision.
_NUMERIC_KINDS = (VariableKind.CONTINUOUS, VariableKind.INTEGER)
# Kinds whose values are labels: no unit but "1", and compared exactly.
_LABEL_KINDS = (VariableKind.CATEGORICAL, VariableKind.ORDINAL)

_DIMENSIONLESS = "1"
_BOUND_COUNT = 2


class DefinitionError(ValueError):
    """An experiment definition breaks one or more rules.

    Attributes:
        problems: Every violated rule, one sentence each.
    """

    def __init__(self, problems: tuple[str, ...]) -> None:
        self.problems = problems
        listed = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"The experiment definition is not valid:\n{listed}")


class TableMismatch(ValueError):  # noqa: N818 — the agreed public name
    """A record table does not fit the definition it was offered to."""


class UnsupportedDesignSpace(ValueError):  # noqa: N818 — the agreed public name
    """The definition is well formed but its design space cannot be optimized."""


class InsufficientConditions(ValueError):  # noqa: N818 — the agreed public name
    """There are too few distinct conditions to fit a surrogate.

    Attributes:
        have: The number of distinct conditions available.
        need: The minimum the backend requires.
    """

    def __init__(self, have: int, need: int) -> None:
        self.have = have
        self.need = need
        super().__init__(
            f"{have} distinct condition(s) available; a surrogate needs at least {need}."
        )


def _is_real(value: Any) -> bool:
    """True for a finite int or float. A bool is not a number here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_definition(definition: ExperimentDefinition) -> None:
    """Check every rule, and raise once with all the violations.

    Raises:
        DefinitionError: Listing every violated rule.
    """
    problems: list[str] = []
    problems += _experiment_problems(definition)
    for variable in definition.variables:
        problems += _variable_problems(variable)
    for response in definition.responses:
        problems += _response_problems(response)
    problems += _objective_problems(definition.objective, definition.responses)
    if problems:
        raise DefinitionError(tuple(problems))


def definition_warnings(definition: ExperimentDefinition) -> tuple[str, ...]:
    """Non-fatal remarks about a valid definition. Frozen into every record."""
    warnings: list[str] = []
    objective = definition.objective
    response = definition.objective_response()
    low = response.valid_range[0]
    if objective.magnitude and low is not None and low >= 0:
        warnings.append(
            f"The objective takes the magnitude of {response.name!r}, whose valid range "
            f"starts at {low:g}: |y| of a non-negative response is y, so the flag does nothing."
        )
    return tuple(warnings)


def _experiment_problems(definition: ExperimentDefinition) -> list[str]:
    problems: list[str] = []
    if not _NAME_RE.match(definition.name):
        problems.append(
            f"name {definition.name!r} must be lower-case letters, digits and hyphens, "
            "starting with a letter and not ending with a hyphen"
        )
    if not _VERSION_RE.match(definition.version):
        problems.append(f"version {definition.version!r} must look like 1.0.0")
    if not definition.title:
        problems.append("title cannot be empty")
    if not PRODUCER_CONTRACT_RE.match(definition.producer):
        problems.append(
            f"producer {definition.producer!r} must be a contract identifier of the form "
            "'<name>/<revision>', for example 'example-reader/1'"
        )
    if not definition.sample_key:
        problems.append("sample_key cannot be empty")
    if definition.specimens_per_condition < 1:
        problems.append("specimens_per_condition must be at least 1")

    if not definition.variables:
        problems.append("at least one variable is required")
    elif not any(v.role is VariableRole.DESIGN for v in definition.variables):
        problems.append("at least one variable must have the DESIGN role")
    if not definition.responses:
        problems.append("at least one response is required")

    names = [v.name for v in definition.variables] + [r.name for r in definition.responses]
    repeated = sorted({name for name in names if name and names.count(name) > 1})
    if repeated:
        problems.append(
            "names must be unique across variables and responses; repeated: " + ", ".join(repeated)
        )
    return problems


def _variable_problems(variable: VariableDefinition) -> list[str]:
    label = f"variable {variable.name!r}"
    problems: list[str] = []
    if not variable.name:
        problems.append("a variable has an empty name")
    if not variable.unit:
        problems.append(f"{label}: unit cannot be empty; use '1' when dimensionless")
    elif variable.kind in _LABEL_KINDS and variable.unit != _DIMENSIONLESS:
        problems.append(f"{label}: a {variable.kind.value} variable must use unit '1'")

    if variable.role is not VariableRole.CONTROL and not variable.column:
        problems.append(f"{label}: a {variable.role.value} variable needs a column")

    if variable.role is VariableRole.DESIGN:
        problems += _design_problems(variable, label)
    else:
        if variable.bounds is not None:
            problems.append(f"{label}: bounds apply only to DESIGN variables")
        if variable.precision is not None:
            problems.append(f"{label}: precision applies only to DESIGN variables")

    if variable.role is VariableRole.CONTROL:
        problems += _control_problems(variable, label)
    else:
        if variable.expected_value is not None:
            problems.append(f"{label}: expected_value applies only to CONTROL variables")
        if variable.tolerance is not None:
            problems.append(f"{label}: tolerance applies only to CONTROL variables")
    return problems


def _design_problems(variable: VariableDefinition, label: str) -> list[str]:
    problems: list[str] = []
    if variable.kind not in _NUMERIC_KINDS:
        # A label axis can be described and recorded, never searched, so there
        # is no range or closeness to declare for it.
        if variable.bounds is not None:
            problems.append(f"{label}: bounds need a numeric kind")
        if variable.precision is not None:
            problems.append(f"{label}: precision needs a numeric kind")
        return problems

    bounds = variable.bounds
    if bounds is None:
        problems.append(f"{label}: a DESIGN variable needs bounds")
    elif len(bounds) != _BOUND_COUNT or not all(_is_real(b) for b in bounds):
        problems.append(f"{label}: bounds must be two finite numbers")
    elif not bounds[0] < bounds[1]:
        problems.append(f"{label}: bounds must have low < high; got {bounds}")
    precision = variable.precision
    if precision is not None and not (_is_real(precision) and precision > 0):
        problems.append(f"{label}: precision must be a finite number above zero")
    return problems


def _control_problems(variable: VariableDefinition, label: str) -> list[str]:
    problems: list[str] = []
    expected = variable.expected_value
    if expected is None:
        problems.append(f"{label}: a CONTROL variable needs an expected_value")
    elif variable.kind is VariableKind.CONTINUOUS:
        if not _is_real(expected):
            problems.append(f"{label}: expected_value of a continuous control must be a number")
    elif variable.kind is VariableKind.INTEGER:
        if not isinstance(expected, int) or isinstance(expected, bool):
            problems.append(f"{label}: expected_value of an integer control must be an integer")
    elif not isinstance(expected, str) or not expected:
        problems.append(
            f"{label}: expected_value of a {variable.kind.value} control must be a non-empty string"
        )

    tolerance = variable.tolerance
    if variable.kind is VariableKind.CONTINUOUS:
        if tolerance is None:
            problems.append(f"{label}: a continuous control needs a tolerance")
        elif not (_is_real(tolerance) and tolerance >= 0):
            problems.append(f"{label}: tolerance must be a finite number, zero or above")
    elif tolerance is not None:
        problems.append(
            f"{label}: tolerance applies only to continuous controls; a "
            f"{variable.kind.value} control is compared exactly"
        )
    return problems


def _response_problems(response: ResponseDefinition) -> list[str]:
    label = f"response {response.name!r}"
    problems: list[str] = []
    if not response.name:
        problems.append("a response has an empty name")
    if not response.unit:
        problems.append(f"{label}: unit cannot be empty; use '1' when dimensionless")
    if not response.column:
        problems.append(f"{label}: column cannot be empty")

    has_column = response.uncertainty_column is not None
    if has_column and not response.uncertainty_column:
        problems.append(f"{label}: uncertainty_column cannot be an empty string")
    if has_column and response.uncertainty_scope is None:
        problems.append(f"{label}: an uncertainty_column needs an uncertainty_scope")
    if not has_column and response.uncertainty_scope is not None:
        problems.append(f"{label}: uncertainty_scope is set but there is no uncertainty_column")

    factor = response.coverage_factor
    if not (_is_real(factor) and factor > 0):
        problems.append(f"{label}: coverage_factor must be a finite number above zero")
    elif factor != 1.0 and not has_column:
        problems.append(f"{label}: coverage_factor other than 1 needs an uncertainty_column")

    valid = response.valid_range
    if len(valid) != _BOUND_COUNT or not all(v is None or _is_real(v) for v in valid):
        problems.append(f"{label}: valid_range must be (low, high), each a number or None")
    else:
        low, high = valid
        if low is not None and high is not None and not low < high:
            problems.append(f"{label}: valid_range must have low < high; got {valid}")
    return problems


def _objective_problems(
    objective: ObjectiveDefinition, responses: tuple[ResponseDefinition, ...]
) -> list[str]:
    problems: list[str] = []
    if objective.response not in {r.name for r in responses}:
        problems.append(f"objective.response {objective.response!r} is not a declared response")
    if objective.goal is Goal.TARGET:
        if objective.target is None or not _is_real(objective.target):
            problems.append("a target objective needs a finite target value")
        if objective.magnitude:
            # |(|y|) - t| or ||y| - |t||: nobody should have to guess which.
            problems.append("magnitude cannot be combined with a target objective")
    elif objective.target is not None:
        problems.append("target applies only when the goal is 'target'")
    return problems
