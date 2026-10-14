"""What does, and does not, change an experiment's identity."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from latos.experiments import (
    ExperimentDefinition,
    FitSpace,
    Goal,
    ObjectiveDefinition,
    ResponseDefinition,
    UncertaintyScope,
    VariableDefinition,
    canonical_definition,
    definition_from_dict,
    definition_sha256,
    definition_to_dict,
)
from latos.experiments.identity import DISPLAY_FIELDS, IDENTITY_FIELDS
from tests.unit.experiments._examples import (
    ATMOSPHERE,
    BAKE,
    HUMIDITY,
    PASSES,
    SIDE,
    SIGNAL,
    X,
    make_definition,
    make_nd_definition,
)

BASE = make_definition()
BASE_HASH = definition_sha256(BASE)


@pytest.mark.parametrize(
    "kind",
    [ExperimentDefinition, VariableDefinition, ResponseDefinition, ObjectiveDefinition],
)
def test_every_field_is_identity_or_display_and_never_both(kind: type) -> None:
    declared = {f.name for f in fields(kind)}
    identity, display = IDENTITY_FIELDS[kind], DISPLAY_FIELDS[kind]
    assert identity | display == declared
    assert not identity & display


def _with_variable(changed: VariableDefinition) -> ExperimentDefinition:
    variables = tuple(changed if v.name == changed.name else v for v in BASE.variables)
    return replace(BASE, variables=variables)


def _with_response(changed: ResponseDefinition) -> ExperimentDefinition:
    responses = tuple(changed if r.name == changed.name else r for r in BASE.responses)
    return replace(BASE, responses=responses)


IDENTITY_CHANGES = [
    replace(BASE, producer="bench-table/2"),
    replace(BASE, producer="other-table/1"),
    replace(BASE, sample_key="specimen"),
    replace(BASE, specimens_per_condition=2),
    _with_variable(replace(X, unit="%")),
    _with_variable(replace(X, column="fraction")),
    _with_variable(replace(X, bounds=(0.0, 0.9))),
    _with_variable(replace(X, precision=0.02)),
    _with_variable(replace(X, precision=None)),
    _with_variable(replace(BAKE, expected_value=61.0)),
    _with_variable(replace(BAKE, tolerance=5.0)),
    _with_variable(replace(ATMOSPHERE, expected_value="N2")),
    _with_variable(replace(PASSES, expected_value=4)),
    _with_variable(replace(HUMIDITY, unit="g/m3")),
    _with_response(replace(SIGNAL, unit="mV")),
    _with_response(replace(SIGNAL, column="signal_b")),
    _with_response(replace(SIGNAL, uncertainty_column="signal_sd")),
    _with_response(replace(SIGNAL, uncertainty_scope=UncertaintyScope.CONDITION)),
    _with_response(replace(SIGNAL, coverage_factor=2.0)),
    _with_response(replace(SIGNAL, valid_range=(-10.0, 10.0))),
    replace(BASE, objective=ObjectiveDefinition(response="signal", goal=Goal.MAXIMIZE)),
    replace(
        BASE,
        objective=ObjectiveDefinition(response="signal", goal=Goal.MINIMIZE, magnitude=True),
    ),
    replace(
        BASE,
        objective=ObjectiveDefinition(
            response="signal", goal=Goal.MAXIMIZE, magnitude=True, fit_space=FitSpace.LOG
        ),
    ),
    replace(BASE, objective=ObjectiveDefinition(response="signal", goal=Goal.TARGET, target=1.5)),
    replace(BASE, objective=ObjectiveDefinition(response="side_signal", goal=Goal.MAXIMIZE)),
]


@pytest.mark.parametrize("changed", IDENTITY_CHANGES)
def test_an_identity_field_changes_the_hash(changed: ExperimentDefinition) -> None:
    assert definition_sha256(changed) != BASE_HASH


def test_every_identity_change_gives_a_different_hash() -> None:
    hashes = {definition_sha256(changed) for changed in IDENTITY_CHANGES}
    assert len(hashes) == len(IDENTITY_CHANGES)


@pytest.mark.parametrize(
    "changed",
    [
        replace(BASE, name="renamed-campaign"),
        replace(BASE, version="9.9.9"),
        replace(BASE, title="A corrected title"),
        replace(BASE, description="More words."),
        _with_variable(replace(X, display_name="Additive fraction")),
        _with_response(replace(SIGNAL, display_name="Signal")),
    ],
)
def test_a_display_field_does_not_change_the_hash(changed: ExperimentDefinition) -> None:
    assert definition_sha256(changed) == BASE_HASH


def test_absent_optional_fields_are_omitted_from_the_canonical_form() -> None:
    canonical = canonical_definition(BASE)
    design = canonical["design_variables"][0]
    assert "expected_value" not in design
    assert "tolerance" not in design
    side = next(r for r in canonical["responses"] if r["name"] == "side_signal")
    assert "uncertainty_column" not in side
    signal = next(r for r in canonical["responses"] if r["name"] == "signal")
    assert "valid_range" not in signal
    assert "target" not in canonical["objective"]
    assert set(canonical) == {
        "producer",
        "sample_key",
        "specimens_per_condition",
        "design_variables",
        "other_variables",
        "responses",
        "objective",
    }


def test_integers_and_floats_in_float_fields_hash_alike() -> None:
    as_ints = _with_variable(replace(X, bounds=(0, 1)))
    as_ints = replace(
        as_ints,
        variables=tuple(
            replace(v, expected_value=60, tolerance=2) if v.name == "bake_time_min" else v
            for v in as_ints.variables
        ),
    )
    assert definition_sha256(as_ints) == BASE_HASH


def test_design_order_is_identity_and_other_order_is_not() -> None:
    two = make_nd_definition(2)
    swapped = replace(two, variables=tuple(reversed(two.variables)))
    assert definition_sha256(swapped) != definition_sha256(two)

    reordered = replace(
        BASE,
        variables=(X, *reversed(BASE.variables[1:])),
        responses=(SIDE, SIGNAL),
    )
    assert definition_sha256(reordered) == BASE_HASH


def test_a_definition_round_trips_through_its_stored_form() -> None:
    for definition in (BASE, make_nd_definition(3), IDENTITY_CHANGES[-2]):
        rebuilt = definition_from_dict(definition_to_dict(definition))
        assert rebuilt == definition
        assert definition_sha256(rebuilt) == definition_sha256(definition)
