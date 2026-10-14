"""The declared description of an experiment.

An `ExperimentDefinition` says what is varied, what is held fixed, what is
merely recorded, what is measured, and what is being optimized. It is a
declaration and nothing else: it holds no data and performs no computation.
Everything downstream reads it instead of inferring intent from column names.

Every field has a consumer, and a field set on a role or kind that cannot use
it is refused rather than ignored. A declaration nothing reads is how a
definition comes to describe an experiment other than the one that was run.

The types are frozen. Validation happens once, when the `ExperimentDefinition`
is built, and reports every violated rule together.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from latos.experiments.values import UncertaintyScope

__all__ = [
    "ExperimentDefinition",
    "FitSpace",
    "Goal",
    "ObjectiveDefinition",
    "ResponseDefinition",
    "VariableDefinition",
    "VariableKind",
    "VariableRole",
]


class VariableRole(StrEnum):
    """What a variable is to the experiment.

    DESIGN is varied on purpose and becomes an axis of the search. CONTROL is
    meant to be held at one value, and a specimen that departs from it is
    excluded. ENVIRONMENT varies, is measured, and is not set. METADATA is
    bookkeeping.
    """

    DESIGN = "design"
    CONTROL = "control"
    ENVIRONMENT = "environment"
    METADATA = "metadata"


class VariableKind(StrEnum):
    """The kind of value a variable takes. Only CONTINUOUS can be optimized."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


class Goal(StrEnum):
    """What the objective asks of its response."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"


class FitSpace(StrEnum):
    """The space the optimized quantity is fitted in."""

    IDENTITY = "identity"
    LOG = "log"


@dataclass(frozen=True, slots=True, kw_only=True)
class VariableDefinition:
    """One variable of the experiment.

    Attributes:
        name: The variable's name. Part of identity.
        role: DESIGN, CONTROL, ENVIRONMENT or METADATA.
        kind: CONTINUOUS, INTEGER, CATEGORICAL or ORDINAL.
        unit: The declared unit; ``"1"`` when dimensionless. CATEGORICAL and
            ORDINAL variables must use ``"1"``.
        column: The table column holding the recorded value. Required except
            for a CONTROL, where None means "declared, not verified".
        bounds: The search range. Numeric DESIGN variables only, and required
            there.
        precision: How close two values must be to count as the same condition.
            Numeric DESIGN variables only. None means exact equality.
        expected_value: What a CONTROL is meant to be held at.
        tolerance: How far a CONTINUOUS control may depart from
            `expected_value`.
        display_name: A label for reports. Not part of identity.
    """

    name: str
    role: VariableRole
    kind: VariableKind
    unit: str
    column: str | None = None
    bounds: tuple[float, float] | None = None
    precision: float | None = None
    expected_value: float | int | str | None = None
    tolerance: float | None = None
    display_name: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseDefinition:
    """One measured response of the experiment.

    Attributes:
        name: The response's name. Part of identity, and the stem of the label
            the optimizer reports under.
        unit: The declared unit, compared as an exact string.
        column: The table column holding the value.
        uncertainty_column: The column holding its uncertainty, if any.
        uncertainty_scope: What that uncertainty covers. Required exactly when
            `uncertainty_column` is set.
        coverage_factor: The factor the uncertainty column was expanded by; the
            column is divided by it so every stored uncertainty is 1-sigma.
        valid_range: Physically admissible (low, high); either end may be None.
        display_name: A label for reports. Not part of identity.
    """

    name: str
    unit: str
    column: str
    uncertainty_column: str | None = None
    uncertainty_scope: UncertaintyScope | None = None
    coverage_factor: float = 1.0
    valid_range: tuple[float | None, float | None] = (None, None)
    display_name: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectiveDefinition:
    """What is optimized, and how the response is projected onto it.

    Attributes:
        response: The name of the declared response to optimize.
        goal: Maximize, minimize, or hit a target.
        target: The value to hit. Required exactly when `goal` is TARGET.
        magnitude: Optimize the absolute value of the response. The projection
            is applied once, after specimens have been reduced on their signed
            values; no producer and no reducer ever takes a magnitude.
        fit_space: The space the optimized quantity is fitted in.
    """

    response: str
    goal: Goal
    target: float | None = None
    magnitude: bool = False
    fit_space: FitSpace = FitSpace.IDENTITY


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentDefinition:
    """The whole declaration of one experiment.

    Attributes:
        name: The handle a definition hash is bound to. Lower-case words joined
            by hyphens.
        version: ``major.minor.patch``. Bumped by the scientist, never
            automatically.
        title: A human-readable title.
        description: Free text.
        producer: The contract identifier of the producer whose tables this
            definition reads, as ``<name>/<revision>``. Part of identity: when
            the producer's scientific transformation changes, its revision
            changes, and this definition no longer accepts its tables.
        sample_key: The column that identifies a specimen.
        variables: Every variable, DESIGN ones in the order of the search axes.
        responses: Every declared response.
        objective: What is optimized.
        specimens_per_condition: How many independently prepared specimens one
            future observation will average. The predictive interval is stated
            for that many.
    """

    name: str
    version: str
    title: str
    description: str = ""
    producer: str
    sample_key: str
    variables: tuple[VariableDefinition, ...]
    responses: tuple[ResponseDefinition, ...]
    objective: ObjectiveDefinition
    specimens_per_condition: int = 1

    def __post_init__(self) -> None:
        # Imported here because `validation` needs the types defined above.
        from latos.experiments.validation import validate_definition  # noqa: PLC0415

        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "responses", tuple(self.responses))
        validate_definition(self)

    def design_variables(self) -> tuple[VariableDefinition, ...]:
        """The DESIGN variables, in declared order: the axes of the search."""
        return tuple(v for v in self.variables if v.role is VariableRole.DESIGN)

    def response(self, name: str) -> ResponseDefinition:
        """The declared response called `name`.

        Raises:
            KeyError: If no response has that name.
        """
        for response in self.responses:
            if response.name == name:
                return response
        raise KeyError(name)

    def objective_response(self) -> ResponseDefinition:
        """The response the objective optimizes."""
        return self.response(self.objective.response)
