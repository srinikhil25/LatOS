"""What makes two definitions "the same experiment", as one hash.

The hash covers every declared field that can change which observations enter
the model, what the optimizer is handed, or how an outcome is judged. Fields
that only change how something is displayed are left out, so correcting a title
never looks like a scientific change.

Canonical form ``latos.experiment.v1``:

* Only identity fields are included.
* A field at its absent value (None, an empty tuple, or ``(None, None)``) is
  omitted, so adding an optional field later never changes the hash of a
  definition that does not use it.
* DESIGN variables appear in declared order, because that order is the order of
  the search axes. Every other variable, and every response, is sorted by name.
* Enums appear as their string values and tuples as lists. Numbers in
  float-typed fields are converted to float first, so ``(0, 1)`` and
  ``(0.0, 1.0)`` hash alike.
* The bytes hashed are the canonical JSON of that structure.

`IDENTITY_FIELDS` and `DISPLAY_FIELDS` partition every field of every definition
type. A test fails the build when a field is in neither set or in both, so a new
field cannot be added without deciding which it is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from latos.experiments._canonical import canonical_json, sha256_hex
from latos.experiments.definition import (
    ExperimentDefinition,
    FitSpace,
    Goal,
    ObjectiveDefinition,
    ResponseDefinition,
    VariableDefinition,
    VariableKind,
    VariableRole,
)
from latos.experiments.values import UncertaintyScope

__all__ = [
    "CANONICAL_FORM",
    "DISPLAY_FIELDS",
    "IDENTITY_FIELDS",
    "canonical_definition",
    "definition_from_dict",
    "definition_sha256",
    "definition_to_dict",
]

CANONICAL_FORM = "latos.experiment.v1"

IDENTITY_FIELDS: Mapping[type, frozenset[str]] = {
    ExperimentDefinition: frozenset(
        {
            "producer",
            "sample_key",
            "variables",
            "responses",
            "objective",
            "specimens_per_condition",
        }
    ),
    VariableDefinition: frozenset(
        {
            "name",
            "role",
            "kind",
            "unit",
            "column",
            "bounds",
            "precision",
            "expected_value",
            "tolerance",
        }
    ),
    ResponseDefinition: frozenset(
        {
            "name",
            "unit",
            "column",
            "uncertainty_column",
            "uncertainty_scope",
            "coverage_factor",
            "valid_range",
        }
    ),
    ObjectiveDefinition: frozenset({"response", "goal", "target", "magnitude", "fit_space"}),
}

# `name` and `version` are the handle a hash is bound to, so they are display:
# including them would make a rename look like a scientific change.
DISPLAY_FIELDS: Mapping[type, frozenset[str]] = {
    ExperimentDefinition: frozenset({"name", "version", "title", "description"}),
    VariableDefinition: frozenset({"display_name"}),
    ResponseDefinition: frozenset({"display_name"}),
    ObjectiveDefinition: frozenset(),
}


def _expected(variable: VariableDefinition) -> float | int | str | None:
    """A control's expected value in the type its kind compares in."""
    value = variable.expected_value
    if value is None or isinstance(value, str):
        return value
    if variable.kind is VariableKind.CONTINUOUS:
        return float(value)
    return int(value)


def _variable_identity(variable: VariableDefinition) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": variable.name,
        "role": variable.role.value,
        "kind": variable.kind.value,
        "unit": variable.unit,
    }
    if variable.column is not None:
        out["column"] = variable.column
    if variable.bounds is not None:
        out["bounds"] = [float(variable.bounds[0]), float(variable.bounds[1])]
    if variable.precision is not None:
        out["precision"] = float(variable.precision)
    if variable.expected_value is not None:
        out["expected_value"] = _expected(variable)
    if variable.tolerance is not None:
        out["tolerance"] = float(variable.tolerance)
    return out


def _response_identity(response: ResponseDefinition) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": response.name,
        "unit": response.unit,
        "column": response.column,
        "coverage_factor": float(response.coverage_factor),
    }
    if response.uncertainty_column is not None:
        out["uncertainty_column"] = response.uncertainty_column
    if response.uncertainty_scope is not None:
        out["uncertainty_scope"] = response.uncertainty_scope.value
    low, high = response.valid_range
    if low is not None or high is not None:
        out["valid_range"] = [
            None if low is None else float(low),
            None if high is None else float(high),
        ]
    return out


def _objective_identity(objective: ObjectiveDefinition) -> dict[str, Any]:
    out: dict[str, Any] = {
        "response": objective.response,
        "goal": objective.goal.value,
        "magnitude": bool(objective.magnitude),
        "fit_space": objective.fit_space.value,
    }
    if objective.target is not None:
        out["target"] = float(objective.target)
    return out


def canonical_definition(definition: ExperimentDefinition) -> dict[str, Any]:
    """The identity fields of `definition`, in canonical structure."""
    design = [v for v in definition.variables if v.role is VariableRole.DESIGN]
    others = sorted(
        (v for v in definition.variables if v.role is not VariableRole.DESIGN),
        key=lambda v: v.name,
    )
    return {
        "producer": definition.producer,
        "sample_key": definition.sample_key,
        "specimens_per_condition": int(definition.specimens_per_condition),
        "design_variables": [_variable_identity(v) for v in design],
        "other_variables": [_variable_identity(v) for v in others],
        "responses": [
            _response_identity(r) for r in sorted(definition.responses, key=lambda r: r.name)
        ],
        "objective": _objective_identity(definition.objective),
    }


def definition_sha256(definition: ExperimentDefinition) -> str:
    """SHA-256 of the canonical form: the identity of the experiment."""
    return sha256_hex(canonical_json(canonical_definition(definition)))


def definition_to_dict(definition: ExperimentDefinition) -> dict[str, Any]:
    """The whole definition, identity and display fields, as JSON-safe data.

    This is what a frozen record stores, so the record can be read, and its
    hash recomputed, without the code that declared the experiment.
    """
    return {
        "name": definition.name,
        "version": definition.version,
        "title": definition.title,
        "description": definition.description,
        "producer": definition.producer,
        "sample_key": definition.sample_key,
        "specimens_per_condition": definition.specimens_per_condition,
        "variables": [
            {
                "name": v.name,
                "role": v.role.value,
                "kind": v.kind.value,
                "unit": v.unit,
                "column": v.column,
                "bounds": None if v.bounds is None else list(v.bounds),
                "precision": v.precision,
                "expected_value": v.expected_value,
                "tolerance": v.tolerance,
                "display_name": v.display_name,
            }
            for v in definition.variables
        ],
        "responses": [
            {
                "name": r.name,
                "unit": r.unit,
                "column": r.column,
                "uncertainty_column": r.uncertainty_column,
                "uncertainty_scope": (
                    None if r.uncertainty_scope is None else r.uncertainty_scope.value
                ),
                "coverage_factor": r.coverage_factor,
                "valid_range": list(r.valid_range),
                "display_name": r.display_name,
            }
            for r in definition.responses
        ],
        "objective": {
            "response": definition.objective.response,
            "goal": definition.objective.goal.value,
            "target": definition.objective.target,
            "magnitude": definition.objective.magnitude,
            "fit_space": definition.objective.fit_space.value,
        },
    }


def definition_from_dict(data: Mapping[str, Any]) -> ExperimentDefinition:
    """Rebuild a definition from `definition_to_dict` output.

    Raises:
        DefinitionError: If the stored definition breaks a rule.
        KeyError: If a required field is missing.
    """
    variables = tuple(
        VariableDefinition(
            name=v["name"],
            role=VariableRole(v["role"]),
            kind=VariableKind(v["kind"]),
            unit=v["unit"],
            column=v.get("column"),
            bounds=None if v.get("bounds") is None else (v["bounds"][0], v["bounds"][1]),
            precision=v.get("precision"),
            expected_value=v.get("expected_value"),
            tolerance=v.get("tolerance"),
            display_name=v.get("display_name", ""),
        )
        for v in data["variables"]
    )
    responses = tuple(
        ResponseDefinition(
            name=r["name"],
            unit=r["unit"],
            column=r["column"],
            uncertainty_column=r.get("uncertainty_column"),
            uncertainty_scope=(
                None
                if r.get("uncertainty_scope") is None
                else UncertaintyScope(r["uncertainty_scope"])
            ),
            coverage_factor=r.get("coverage_factor", 1.0),
            valid_range=(r["valid_range"][0], r["valid_range"][1]),
            display_name=r.get("display_name", ""),
        )
        for r in data["responses"]
    )
    objective = data["objective"]
    return ExperimentDefinition(
        name=data["name"],
        version=data["version"],
        title=data["title"],
        description=data.get("description", ""),
        producer=data["producer"],
        sample_key=data["sample_key"],
        variables=variables,
        responses=responses,
        objective=ObjectiveDefinition(
            response=objective["response"],
            goal=Goal(objective["goal"]),
            target=objective.get("target"),
            magnitude=bool(objective.get("magnitude", False)),
            fit_space=FitSpace(objective.get("fit_space", FitSpace.IDENTITY.value)),
        ),
        specimens_per_condition=int(data.get("specimens_per_condition", 1)),
    )
