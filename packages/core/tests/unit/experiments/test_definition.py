"""The rules an experiment definition must satisfy."""

from __future__ import annotations

from dataclasses import replace

import pytest

from latos.experiments import (
    DefinitionError,
    Goal,
    ObjectiveDefinition,
    ResponseDefinition,
    VariableKind,
    VariableRole,
    definition_warnings,
)
from tests.unit.experiments._examples import (
    ATMOSPHERE,
    BAKE,
    HUMIDITY,
    PASSES,
    SIGNAL,
    X,
    make_definition,
)


def _problems(**kwargs: object) -> tuple[str, ...]:
    with pytest.raises(DefinitionError) as caught:
        make_definition(**kwargs)
    return caught.value.problems


def test_the_example_definition_is_valid() -> None:
    definition = make_definition()
    assert [v.name for v in definition.design_variables()] == ["additive_fraction"]
    assert definition.objective_response().name == "signal"


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        ({"name": "Bad Name"}, "name"),
        ({"name": "trailing-"}, "name"),
        ({"version": "1.0"}, "version"),
        ({"title": ""}, "title"),
        ({"sample_key": ""}, "sample_key"),
        ({"specimens_per_condition": 0}, "specimens_per_condition"),
        ({"producer": "no-revision"}, "contract identifier"),
        ({"producer": "Reader/1"}, "contract identifier"),
        ({"producer": "reader/0"}, "contract identifier"),
    ],
)
def test_experiment_level_rules(fields: dict[str, object], fragment: str) -> None:
    assert any(fragment in problem for problem in _problems(**fields))


def test_a_definition_needs_a_design_variable_and_a_response() -> None:
    assert any("DESIGN" in p for p in _problems(variables=(BAKE,)))
    assert any("at least one variable" in p for p in _problems(variables=()))
    assert any("at least one response" in p for p in _problems(responses=()))


def test_names_are_unique_across_variables_and_responses() -> None:
    clash = replace(SIGNAL, name="additive_fraction")
    objective = ObjectiveDefinition(response="additive_fraction", goal=Goal.MAXIMIZE)
    assert any("unique" in p for p in _problems(responses=(clash,), objective=objective))


@pytest.mark.parametrize(
    ("variable", "fragment"),
    [
        (replace(X, unit=""), "unit cannot be empty"),
        (replace(X, column=None), "needs a column"),
        (replace(X, bounds=None), "needs bounds"),
        (replace(X, bounds=(1.0, 1.0)), "low < high"),
        (replace(X, bounds=(0.0, float("inf"))), "two finite numbers"),
        (replace(X, precision=0.0), "precision"),
        (replace(X, expected_value=0.5), "expected_value applies only to CONTROL"),
        (replace(X, tolerance=0.1), "tolerance applies only to CONTROL"),
        (replace(BAKE, bounds=(0.0, 1.0)), "bounds apply only to DESIGN"),
        (replace(BAKE, precision=0.1), "precision applies only to DESIGN"),
        (replace(BAKE, expected_value=None), "needs an expected_value"),
        (replace(BAKE, expected_value="sixty"), "must be a number"),
        (replace(BAKE, tolerance=None), "needs a tolerance"),
        (replace(BAKE, tolerance=-1.0), "zero or above"),
        (replace(ATMOSPHERE, tolerance=0.1), "compared exactly"),
        (replace(ATMOSPHERE, unit="bar"), "must use unit '1'"),
        (replace(ATMOSPHERE, expected_value=3), "non-empty string"),
        (replace(PASSES, expected_value=3.5), "must be an integer"),
        (replace(PASSES, expected_value=True), "must be an integer"),
        (replace(HUMIDITY, column=None), "needs a column"),
        (replace(HUMIDITY, expected_value=40.0), "expected_value applies only to CONTROL"),
    ],
)
def test_variable_rules(variable: object, fragment: str) -> None:
    others = tuple(v for v in (X, BAKE) if v.name != variable.name)
    problems = _problems(variables=(*others, variable))
    assert any(fragment in problem for problem in problems), problems


def test_a_label_design_axis_takes_no_bounds_or_precision() -> None:
    labelled = replace(X, kind=VariableKind.CATEGORICAL, bounds=(0.0, 1.0), precision=0.1)
    problems = _problems(variables=(labelled,))
    assert any("bounds need a numeric kind" in p for p in problems)
    assert any("precision needs a numeric kind" in p for p in problems)


def test_a_categorical_control_is_accepted() -> None:
    definition = make_definition(variables=(X, ATMOSPHERE))
    control = definition.variables[1]
    assert control.role is VariableRole.CONTROL
    assert control.kind is VariableKind.CATEGORICAL


@pytest.mark.parametrize(
    ("response", "fragment"),
    [
        (replace(SIGNAL, unit=""), "unit cannot be empty"),
        (replace(SIGNAL, column=""), "column cannot be empty"),
        (replace(SIGNAL, uncertainty_scope=None), "needs an uncertainty_scope"),
        (replace(SIGNAL, uncertainty_column=None), "no uncertainty_column"),
        (replace(SIGNAL, coverage_factor=0.0), "coverage_factor"),
        (replace(SIGNAL, valid_range=(2.0, 1.0)), "low < high"),
        (
            ResponseDefinition(name="signal", unit="a.u.", column="signal", coverage_factor=2.0),
            "needs an uncertainty_column",
        ),
    ],
)
def test_response_rules(response: ResponseDefinition, fragment: str) -> None:
    problems = _problems(responses=(response,))
    assert any(fragment in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("objective", "fragment"),
    [
        (ObjectiveDefinition(response="absent", goal=Goal.MAXIMIZE), "not a declared response"),
        (ObjectiveDefinition(response="signal", goal=Goal.TARGET), "needs a finite target"),
        (
            ObjectiveDefinition(response="signal", goal=Goal.TARGET, target=1.5, magnitude=True),
            "magnitude cannot be combined",
        ),
        (
            ObjectiveDefinition(response="signal", goal=Goal.MAXIMIZE, target=1.5),
            "target applies only",
        ),
    ],
)
def test_objective_rules(objective: ObjectiveDefinition, fragment: str) -> None:
    problems = _problems(objective=objective)
    assert any(fragment in problem for problem in problems), problems


def test_every_violation_is_reported_together() -> None:
    problems = _problems(
        title="",
        version="one",
        variables=(replace(X, bounds=None), replace(BAKE, tolerance=None)),
    )
    assert len(problems) == 4
    message = str(DefinitionError(problems))
    assert all(problem in message for problem in problems)


def test_magnitude_of_a_non_negative_response_is_warned_about_not_refused() -> None:
    non_negative = replace(SIGNAL, valid_range=(0.0, None))
    assert definition_warnings(make_definition()) == ()
    warnings = definition_warnings(make_definition(responses=(non_negative,)))
    assert len(warnings) == 1
    assert "non-negative" in warnings[0]
