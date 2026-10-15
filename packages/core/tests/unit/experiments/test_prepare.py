"""Projecting the objective and labelling what the optimizer is handed."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from latos.experiments import (
    FitSpace,
    Goal,
    ObjectiveDefinition,
    UnsupportedDesignSpace,
    VariableKind,
    prepare_optimization,
)
from latos.experiments.prepare import (
    AGGREGATION_IDENTITY,
    AGGREGATION_MAGNITUDE,
    AGGREGATION_TARGET,
    PREDICTION_NOISE_DECLARED,
    PREDICTION_NOISE_GLOSSARY,
    PREDICTION_NOISE_LOWER_BOUND,
    PREDICTION_NOISE_NOT_PASSED,
    PREDICTION_NOISE_PARTLY_ASSUMED,
    PREDICTION_NOISE_POOLED,
    PREDICTION_NOISE_UNDETERMINED,
    noise_source_from_sigma,
    project,
)
from latos.experiments.reduce import SIGMA_DECLARED, SIGMA_LOWER_BOUND, SIGMA_UNDETERMINED
from tests.unit.experiments._examples import (
    SIDE,
    SIGNAL,
    X,
    make_definition,
    make_row,
    reduced_set,
)

MIXED = (
    make_row("a", 0.0, -2.5, 0.4),
    make_row("b", 0.25, 2.0, 0.01),
    make_row("c", 0.5, 2.25, 0.08),
)


def _objective(**fields: object) -> ObjectiveDefinition:
    settings: dict[str, object] = {"response": "signal", "goal": Goal.MAXIMIZE}
    settings.update(fields)
    return ObjectiveDefinition(**settings)


@pytest.mark.parametrize(
    ("objective", "y", "direction", "label", "aggregation"),
    [
        (_objective(), [-2.5, 2.0, 2.25], "maximize", "signal", AGGREGATION_IDENTITY),
        (
            _objective(goal=Goal.MINIMIZE),
            [-2.5, 2.0, 2.25],
            "minimize",
            "signal",
            AGGREGATION_IDENTITY,
        ),
        (
            _objective(magnitude=True),
            [2.5, 2.0, 2.25],
            "maximize",
            "abs_signal",
            AGGREGATION_MAGNITUDE,
        ),
        (
            _objective(goal=Goal.MINIMIZE, magnitude=True),
            [2.5, 2.0, 2.25],
            "minimize",
            "abs_signal",
            AGGREGATION_MAGNITUDE,
        ),
        (
            _objective(goal=Goal.TARGET, target=1.5),
            [4.0, 0.5, 0.75],
            "minimize",
            "|signal - 1.5|",
            AGGREGATION_TARGET,
        ),
    ],
)
def test_the_objective_is_projected_once_and_y_raw_is_untouched(
    objective: ObjectiveDefinition,
    y: list[float],
    direction: str,
    label: str,
    aggregation: str,
) -> None:
    definition = make_definition(objective=objective)
    observation_set = reduced_set(definition, *MIXED)
    before = [c.y_raw for c in observation_set.reduced]
    prepared = prepare_optimization(definition, observation_set)

    assert prepared.y.tolist() == y
    assert prepared.direction == direction
    assert prepared.target_name == label
    assert prepared.objective_aggregation == aggregation
    # The signed values come through element-identical, and the reducer's
    # output was not modified on the way.
    assert prepared.y_raw.tolist() == before == [-2.5, 2.0, 2.25]
    assert [c.y_raw for c in observation_set.reduced] == before
    # y is a pure function of y_raw and the declared objective.
    recomputed = project(
        prepared.y_raw,
        goal=objective.goal.value,
        magnitude=objective.magnitude,
        target=objective.target,
    )
    assert recomputed.tolist() == prepared.y.tolist()


def test_the_prepared_arrays_cannot_be_written_to() -> None:
    definition = make_definition()
    prepared = prepare_optimization(definition, reduced_set(definition, *MIXED))
    for array in (prepared.x, prepared.y, prepared.y_raw, prepared.sigma):
        assert array is not None
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0.0


def test_the_design_matrix_bounds_and_names_follow_the_definition() -> None:
    definition = make_definition(objective=_objective(magnitude=True, fit_space=FitSpace.LOG))
    prepared = prepare_optimization(definition, reduced_set(definition, *MIXED))
    assert prepared.x.tolist() == [[0.0], [0.25], [0.5]]
    assert prepared.bounds == ((0.0, 1.0),)
    assert prepared.input_names == ("additive_fraction",)
    assert prepared.n_dims == 1
    assert prepared.fit_space == "log"


def test_an_unknown_sigma_gets_the_largest_known_one() -> None:
    definition = make_definition()
    prepared = prepare_optimization(
        definition,
        reduced_set(
            definition,
            make_row("a", 0.1, 1.0, 0.1),
            make_row("b", 0.4, 2.0),
            make_row("c", 0.7, 3.0, 0.3),
        ),
    )
    assert prepared.sigma is not None
    assert prepared.sigma.tolist() == [0.1, 0.3, 0.3]
    assert prepared.sigma_sources == (
        "lower_bound_specimen",
        "undetermined",
        "lower_bound_specimen",
    )
    assert prepared.prediction_noise == pytest.approx(0.3)  # the median handed over
    # The substitute is assumed, and it is in the median: the noise is not
    # called estimated from measurements.
    assert prepared.prediction_noise_source == PREDICTION_NOISE_PARTLY_ASSUMED
    meaning = PREDICTION_NOISE_GLOSSARY[PREDICTION_NOISE_PARTLY_ASSUMED]
    assert meaning.startswith("Partly assumed")
    assert "an assumption, not a measurement" in meaning


def test_with_no_sigma_anywhere_every_condition_gets_the_spread_of_y() -> None:
    definition = make_definition()
    rows = (make_row("a", 0.1, 1.0), make_row("b", 0.4, 2.0), make_row("c", 0.7, 6.0))
    prepared = prepare_optimization(definition, reduced_set(definition, *rows))
    spread = float(np.std([1.0, 2.0, 6.0]))
    assert prepared.sigma is not None
    assert prepared.sigma.tolist() == pytest.approx([spread] * 3)
    assert prepared.prediction_noise == pytest.approx(spread)
    assert prepared.prediction_noise_source == PREDICTION_NOISE_UNDETERMINED


def test_identical_values_with_no_sigma_fall_back_to_one() -> None:
    definition = make_definition()
    rows = (make_row("a", 0.1, 2.0), make_row("b", 0.4, 2.0), make_row("c", 0.7, 2.0))
    prepared = prepare_optimization(definition, reduced_set(definition, *rows))
    assert prepared.sigma is not None
    assert prepared.sigma.tolist() == [1.0, 1.0, 1.0]


def test_vanishing_uncertainties_are_not_passed_at_all() -> None:
    definition = make_definition()
    rows = (
        make_row("a", 0.1, 1.0, 1e-12),
        make_row("b", 0.4, 2.0, 1e-12),
        make_row("c", 0.7, 3.0, 1e-12),
    )
    prepared = prepare_optimization(definition, reduced_set(definition, *rows))
    assert prepared.sigma is None
    assert prepared.prediction_noise is None
    assert prepared.prediction_noise_source == PREDICTION_NOISE_NOT_PASSED
    assert any("vanishingly small" in message for message in prepared.messages)


@pytest.mark.parametrize("specimens_per_condition", [1, 4])
def test_the_pooled_spread_sets_the_prediction_noise(specimens_per_condition: int) -> None:
    definition = make_definition(specimens_per_condition=specimens_per_condition)
    rows = (
        make_row("a1", 0.1, 1.0, 0.01),
        make_row("a2", 0.1, 1.2, 0.01),
        make_row("b", 0.4, 2.0, 0.01),
        make_row("c", 0.7, 3.0, 0.01),
    )
    prepared = prepare_optimization(definition, reduced_set(definition, *rows))
    pooled = math.sqrt(0.02)
    assert prepared.prediction_noise == pytest.approx(pooled / math.sqrt(specimens_per_condition))
    assert prepared.prediction_noise_source == PREDICTION_NOISE_POOLED
    assert prepared.sigma is not None
    assert prepared.sigma.tolist() == pytest.approx([pooled / math.sqrt(2), pooled, pooled])


def test_a_lower_bound_prediction_noise_scales_with_specimens_per_condition() -> None:
    definition = make_definition(specimens_per_condition=4)
    prepared = prepare_optimization(definition, reduced_set(definition, *MIXED))
    assert prepared.prediction_noise == pytest.approx(0.08 / 2)
    assert prepared.prediction_noise_source == PREDICTION_NOISE_LOWER_BOUND
    meaning = PREDICTION_NOISE_GLOSSARY[PREDICTION_NOISE_LOWER_BOUND]
    # Measured, but only a part of the budget; the interval may be too narrow,
    # which is not said to be certain.
    assert meaning.startswith("Estimated from measured observations")
    assert "not a complete uncertainty budget" in meaning
    assert "may understate" in meaning
    assert "too narrow" not in meaning


def test_each_prediction_noise_says_whether_it_was_estimated_or_assumed() -> None:
    glossary = PREDICTION_NOISE_GLOSSARY
    assert glossary[PREDICTION_NOISE_POOLED].startswith("Estimated from measured observations")
    assert glossary[PREDICTION_NOISE_UNDETERMINED].startswith("Assumed, not estimated")
    assert glossary[PREDICTION_NOISE_PARTLY_ASSUMED].startswith("Partly assumed")
    assert glossary[PREDICTION_NOISE_NOT_PASSED].startswith("None passed")


@pytest.mark.parametrize(
    ("labels", "source"),
    [
        ([SIGMA_LOWER_BOUND, SIGMA_LOWER_BOUND], PREDICTION_NOISE_LOWER_BOUND),
        ([SIGMA_LOWER_BOUND, SIGMA_UNDETERMINED], PREDICTION_NOISE_PARTLY_ASSUMED),
        ([SIGMA_DECLARED, SIGMA_UNDETERMINED], PREDICTION_NOISE_PARTLY_ASSUMED),
        ([SIGMA_LOWER_BOUND, SIGMA_DECLARED], PREDICTION_NOISE_DECLARED),
        ([SIGMA_DECLARED, SIGMA_DECLARED], PREDICTION_NOISE_DECLARED),
        ([SIGMA_UNDETERMINED, SIGMA_UNDETERMINED], PREDICTION_NOISE_UNDETERMINED),
        ([], PREDICTION_NOISE_UNDETERMINED),
    ],
)
def test_an_unpooled_noise_is_measured_only_when_every_condition_had_one(
    labels: list[str], source: str
) -> None:
    # Measured only when every condition's spread was estimated from its
    # specimens; a declared spread is known but was not measured here.
    assert noise_source_from_sigma(labels) == source


def test_only_an_unprojected_objective_inherits_the_valid_range() -> None:
    ranged = (replace(SIGNAL, valid_range=(-5.0, 10.0)), SIDE)
    plain = make_definition(responses=ranged, objective=_objective())
    prepared = prepare_optimization(plain, reduced_set(plain, *MIXED))
    assert (prepared.y_min, prepared.y_max) == (-5.0, 10.0)

    for objective in (_objective(magnitude=True), _objective(goal=Goal.TARGET, target=1.0)):
        projected = make_definition(responses=ranged, objective=objective)
        prepared = prepare_optimization(projected, reduced_set(projected, *MIXED))
        assert (prepared.y_min, prepared.y_max) == (None, None)


@pytest.mark.parametrize(
    "kind", [VariableKind.INTEGER, VariableKind.CATEGORICAL, VariableKind.ORDINAL]
)
def test_a_non_continuous_design_axis_is_refused_by_name(kind: VariableKind) -> None:
    numeric = kind is VariableKind.INTEGER
    axis = replace(
        X,
        kind=kind,
        bounds=(0.0, 4.0) if numeric else None,
        precision=None,
    )
    definition = make_definition(variables=(axis,))
    levels = [1, 2, 3] if numeric else ["a", "b", "c"]
    observation_set = reduced_set(
        definition, *(make_row(f"s{i}", level, 1.0 + i) for i, level in enumerate(levels))
    )
    assert len(observation_set.reduced) == 3  # describable and recordable
    with pytest.raises(UnsupportedDesignSpace, match="additive_fraction") as caught:
        prepare_optimization(definition, observation_set)
    assert kind.value in str(caught.value)


def test_a_magnitude_objective_given_both_signs_says_so() -> None:
    definition = make_definition()
    prepared = prepare_optimization(definition, reduced_set(definition, *MIXED))
    (note,) = prepared.messages
    assert "changes sign" in note
    assert "negative at additive_fraction = 0;" in note
    assert "positive at additive_fraction = 0.25; additive_fraction = 0.5." in note
    assert "Nothing was discarded or altered" in note
    # A warning only: the signed data and their magnitudes are both intact.
    assert prepared.y_raw.tolist() == [-2.5, 2.0, 2.25]
    assert prepared.y.tolist() == [2.5, 2.0, 2.25]


def test_no_sign_note_for_one_sign_or_for_a_signed_objective() -> None:
    definition = make_definition()
    one_sign = (make_row("a", 0.1, 1.0, 0.1), make_row("b", 0.4, 2.0, 0.1))
    assert prepare_optimization(definition, reduced_set(definition, *one_sign)).messages == ()

    signed = make_definition(objective=_objective())
    assert prepare_optimization(signed, reduced_set(signed, *MIXED)).messages == ()


def test_an_observation_set_from_another_definition_is_refused() -> None:
    definition = make_definition()
    other = make_definition(specimens_per_condition=2)
    with pytest.raises(ValueError, match="different definition"):
        prepare_optimization(other, reduced_set(definition, *MIXED))


def test_no_conditions_prepare_to_empty_arrays() -> None:
    definition = make_definition()
    prepared = prepare_optimization(definition, reduced_set(definition))
    assert prepared.x.shape == (0, 1)
    assert prepared.sigma is None
