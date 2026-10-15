"""Reducing specimens to conditions, on signed values, with labelled sigma."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from latos.core.enums import Severity
from latos.experiments import (
    UncertaintyScope,
    VariableKind,
    build_observations,
    definition_sha256,
    reduce_observations,
)
from latos.experiments.reduce import (
    MIN_POOLED_DF,
    SIGMA_DECLARED,
    SIGMA_EMPIRICAL,
    SIGMA_GLOSSARY,
    SIGMA_LOWER_BOUND,
    SIGMA_POOLED,
    SIGMA_UNDETERMINED,
)
from tests.unit.experiments._examples import (
    SIDE,
    SIGNAL,
    X,
    make_definition,
    make_nd_definition,
    make_nd_table,
    make_row,
    reduced_set,
)

DEFINITION = make_definition()


def test_each_specimen_made_once_is_its_own_condition_in_design_order() -> None:
    result = reduced_set(
        DEFINITION,
        make_row("c", 0.75, 2.2, 0.03),
        make_row("a", 0.0, -2.6, 0.4),
        make_row("b", 0.25, 2.0, 0.01),
    )
    assert result.definition_hash == definition_sha256(DEFINITION)
    assert [obs.sample_id for obs in result.observations] == ["a", "b", "c"]
    assert [c.x for c in result.reduced] == [(0.0,), (0.25,), (0.75,)]
    assert [c.y_raw for c in result.reduced] == [-2.6, 2.0, 2.2]  # signed
    assert [c.n_specimens for c in result.reduced] == [1, 1, 1]
    assert result.pooled_sd is None
    assert result.pooled_df == 0
    assert result.messages == ()


def test_a_specimen_scope_uncertainty_is_a_labelled_lower_bound() -> None:
    result = reduced_set(DEFINITION, make_row("a", 0.25, 2.0, 0.01))
    (condition,) = result.reduced
    assert condition.sigma == 0.01
    assert condition.sigma_source == SIGMA_LOWER_BOUND
    meaning = SIGMA_GLOSSARY[SIGMA_LOWER_BOUND]
    assert "Specimen-to-specimen variation is not in it" in meaning
    assert "not a complete uncertainty budget" in meaning


def test_a_condition_scope_uncertainty_is_labelled_as_declared() -> None:
    declared = make_definition(
        responses=(replace(SIGNAL, uncertainty_scope=UncertaintyScope.CONDITION), SIDE)
    )
    (condition,) = reduced_set(declared, make_row("a", 0.25, 2.0, 0.2)).reduced
    assert (condition.sigma, condition.sigma_source) == (0.2, SIGMA_DECLARED)


def test_no_uncertainty_is_labelled_undetermined() -> None:
    (condition,) = reduced_set(DEFINITION, make_row("a", 0.25, 2.0)).reduced
    assert (condition.sigma, condition.sigma_source) == (None, SIGMA_UNDETERMINED)


def test_replicates_are_reduced_by_their_signed_mean_and_pooled() -> None:
    result = reduced_set(
        DEFINITION,
        make_row("a1", 0.300, 1.0, 0.01),
        make_row("a2", 0.302, 1.2, 0.01),
        make_row("b", 0.500, 2.0, 0.01),
    )
    replicated, single = result.reduced
    assert replicated.sample_ids == ("a1", "a2")
    assert replicated.x == (pytest.approx(0.301),)
    assert replicated.y_raw == pytest.approx(1.1)

    pooled = math.sqrt(0.02)
    assert result.pooled_sd == pytest.approx(pooled)
    assert result.pooled_df == 1
    # The between-specimen spread replaces the declared within-specimen value
    # everywhere; it is never added to it.
    assert replicated.sigma == pytest.approx(pooled / math.sqrt(2))
    assert replicated.sigma_source == SIGMA_EMPIRICAL
    assert single.sigma == pytest.approx(pooled)
    assert single.sigma_source == SIGMA_POOLED

    scatter, collapse = result.messages
    assert "pooled standard deviation 0.1414 a.u. on 1 degree(s) of freedom" in scatter
    # Below MIN_POOLED_DF: the chi-square 95% interval for a spread on 1 d.o.f.
    assert "runs from about 0.45 to 32 times it" in scatter
    assert "It is a specimen-to-specimen spread." in scatter
    assert collapse == (
        "3 sample(s) collapsed onto 2 design point(s); "
        "1 were replicates of a condition already present."
    )


def test_the_low_df_caveat_goes_once_enough_degrees_of_freedom_exist() -> None:
    rows = [
        make_row(f"s{i}{j}", x, y + 0.1 * j, 0.01)
        for i, (x, y) in enumerate([(0.1, 1.0), (0.4, 2.0), (0.7, 3.0)])
        for j in range(2)
    ]
    result = reduced_set(DEFINITION, *rows)
    assert result.pooled_df == MIN_POOLED_DF
    assert "95% confidence interval" not in result.messages[0]


def test_grouping_chains_along_the_axis() -> None:
    # Each neighbour is within 0.005; the ends are 0.008 apart.
    result = reduced_set(
        DEFINITION,
        make_row("a", 0.300, 1.0),
        make_row("b", 0.304, 1.1),
        make_row("c", 0.308, 1.2),
        make_row("d", 0.320, 5.0),
    )
    assert [c.sample_ids for c in result.reduced] == [("a", "b", "c"), ("d",)]


def test_without_a_precision_only_equal_values_group() -> None:
    exact = make_definition(variables=(replace(X, precision=None),))
    result = reduced_set(
        exact, make_row("a", 0.30, 1.0), make_row("b", 0.30, 1.2), make_row("c", 0.3001, 1.4)
    )
    assert [c.sample_ids for c in result.reduced] == [("a", "b"), ("c",)]


def test_conditions_group_on_every_axis_in_two_dimensions() -> None:
    definition = make_nd_definition(2)
    table = make_nd_table(
        2,
        [((1.00, 5.0), 10.0), ((1.05, 5.05), 12.0), ((1.05, 6.0), 20.0), ((3.0, 5.0), 30.0)],
    )
    observations, excluded = build_observations(definition, table)
    result = reduce_observations(definition, observations, excluded)
    assert [c.sample_ids for c in result.reduced] == [("s0", "s1"), ("s2",), ("s3",)]
    assert result.reduced[0].x == (pytest.approx(1.025), pytest.approx(5.025))
    assert result.reduced[0].y_raw == pytest.approx(11.0)


def test_opposite_signs_at_one_condition_are_kept_and_reported() -> None:
    result = reduced_set(DEFINITION, make_row("p", 0.30, 1.0), make_row("n", 0.30, -0.4))
    (condition,) = result.reduced
    # |mean|, not mean of magnitudes: 0.3, never 0.7.
    assert condition.y_raw == pytest.approx(0.3)
    (issue,) = condition.issues
    assert issue.code == "sign_instability"
    assert issue.severity is Severity.WARNING
    assert "p = +1" in issue.message
    assert "n = -0.4" in issue.message
    # Pooled from the signed values, so the disagreement shows as spread.
    assert result.pooled_sd == pytest.approx(math.sqrt(0.98))
    # Every raw value is still on its observation.
    assert [obs.responses["signal"].value for obs in result.observations] == [1.0, -0.4]


@pytest.mark.parametrize("values", [(0.5, 0.7), (-0.5, -0.7), (0.0, 0.5), (0.0, -0.5)])
def test_agreeing_signs_and_zeros_raise_no_issue(values: tuple[float, float]) -> None:
    result = reduced_set(DEFINITION, make_row("a", 0.30, values[0]), make_row("b", 0.30, values[1]))
    assert result.reduced[0].issues == ()


def test_identical_replicates_are_discarded_as_an_estimate() -> None:
    result = reduced_set(DEFINITION, make_row("a", 0.30, 1.0, 0.01), make_row("b", 0.30, 1.0, 0.01))
    assert result.pooled_sd is None
    assert result.pooled_df == 0
    assert "computed rather than measured" in result.messages[0]
    (condition,) = result.reduced
    assert (condition.sigma, condition.sigma_source) == (None, SIGMA_UNDETERMINED)


def test_excluded_rows_travel_through_unchanged() -> None:
    result = reduced_set(DEFINITION, make_row("a", 0.3, 1.0), make_row("bad", 0.3, None))
    assert [e.sample_id for e in result.excluded] == ["bad"]
    assert result.reduced[0].n_specimens == 1


def test_a_label_design_axis_groups_by_exact_label() -> None:
    labelled = make_definition(
        variables=(replace(X, kind=VariableKind.ORDINAL, bounds=None, precision=None),)
    )
    result = reduced_set(
        labelled,
        make_row("a", "low", 1.0),
        make_row("b", "low", 1.4),
        make_row("c", "high", 3.0),
    )
    assert [(c.x, c.sample_ids) for c in result.reduced] == [
        (("high",), ("c",)),
        (("low",), ("a", "b")),
    ]


def test_nothing_is_reduced_when_there_are_no_observations() -> None:
    result = reduced_set(DEFINITION)
    assert result.reduced == ()
    assert result.observations == ()


# ─── Outside the design bounds ──────────────────────────────────────
def test_specimens_outside_the_bounds_are_named_and_never_reduced() -> None:
    narrow = make_definition(variables=(replace(X, bounds=(0.2, 1.0)),))
    result = reduced_set(
        narrow,
        make_row("far", 0.0, -2.6, 0.4),
        make_row("a", 0.3, 1.0, 0.1),
        make_row("bad", 0.5, None),
        make_row("edge", 0.2, 1.5, 0.1),
    )
    # The two inside are conditions; the one outside is in neither list of them.
    assert [c.sample_ids for c in result.reduced] == [("edge",), ("a",)]
    assert [o.sample_id for o in result.observations] == ["edge", "a"]

    (outside,) = result.outside_bounds()
    (rejected,) = result.rejected()
    assert (outside.sample_id, rejected.sample_id) == ("far", "bad")
    assert result.excluded == (outside, rejected)
    (message,) = result.messages
    assert message == (
        "1 specimen(s) passed the same checks as every other accepted specimen and lie "
        "outside the design bounds: far (additive_fraction = 0; signal = -2.6 a.u.). They are "
        "kept in the record and are not in the fit; the bounds alone keep them out."
    )


def test_a_specimen_outside_the_bounds_never_joins_a_condition_inside_them() -> None:
    # 0.199 is within the precision of 0.2, and outside the bounds all the same.
    narrow = make_definition(variables=(replace(X, bounds=(0.2, 1.0), precision=0.005),))
    result = reduced_set(narrow, make_row("in", 0.2, 1.0, 0.1), make_row("out", 0.199, 9.0, 0.1))
    (condition,) = result.reduced
    assert condition.sample_ids == ("in",)
    assert condition.y_raw == 1.0
    assert [row.sample_id for row in result.outside_bounds()] == ["out"]


def test_nothing_is_said_about_the_bounds_when_nothing_lies_outside_them() -> None:
    result = reduced_set(DEFINITION, make_row("a", 0.3, 1.0, 0.1), make_row("bad", 0.3, None))
    assert result.outside_bounds() == ()
    assert [row.sample_id for row in result.rejected()] == ["bad"]
    assert not any("outside the design bounds" in message for message in result.messages)
