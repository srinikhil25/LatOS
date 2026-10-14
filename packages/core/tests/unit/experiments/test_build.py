"""Checking a record table against a definition, row by row."""

from __future__ import annotations

from dataclasses import replace

import pytest

from latos.core.enums import Severity
from latos.core.models import ValidationIssue, utc_now
from latos.experiments import (
    EXCLUDED_OUTSIDE_BOUNDS,
    EXCLUDED_REJECTED,
    RecordTable,
    TableMismatch,
    UncertaintyScope,
    build_observations,
)
from tests.unit.experiments._examples import (
    BAKE,
    COLUMNS,
    SIDE,
    SIGNAL,
    VARIABLES,
    X,
    make_definition,
    make_row,
    make_table,
)

DEFINITION = make_definition()


def _excluded_reasons(*rows: object) -> tuple[str, ...]:
    observations, excluded = build_observations(DEFINITION, make_table(*rows))
    assert not observations
    assert len(excluded) == 1
    return excluded[0].reasons


def test_one_row_becomes_one_observation_with_its_values() -> None:
    detail = {"repeats": [1.9, 2.0, 2.1], "note": "opaque"}
    table = make_table(make_row("s1", 0.25, -2.0, 0.1, detail=detail))
    observations, excluded = build_observations(DEFINITION, table)

    assert not excluded
    (obs,) = observations
    assert obs.sample_id == "s1"
    assert obs.inputs == {
        "additive_fraction": 0.25,
        "bake_time_min": 60.0,
        "atmosphere": "Ar",
        "passes": 3.0,
        "humidity_pct": 40.0,
    }
    assert obs.metadata == {"batch": "b1"}
    signal = obs.responses["signal"]
    assert signal.value == -2.0  # signed, untouched
    assert signal.unit == "a.u."
    assert signal.uncertainty == 0.1
    assert signal.uncertainty_scope is UncertaintyScope.SPECIMEN
    assert signal.provenance == "bench-table/1: invented value"
    # Carried, not copied and not read.
    assert obs.producer_detail is detail


def test_what_the_producer_said_about_an_uncertainty_travels_with_the_value() -> None:
    stated = {"signal": "mean of three repeats", "signal_u": "repeat scatter; a lower bound"}
    with_basis = replace(make_row("s1", 0.25, 2.0, 0.1), provenance=stated)
    no_uncertainty = replace(make_row("s2", 0.5, 2.0), provenance=stated)
    basis_only = replace(make_row("s3", 0.75, 2.0, 0.1), provenance={"signal_u": "repeat scatter"})
    observations, _ = build_observations(
        DEFINITION, make_table(with_basis, no_uncertainty, basis_only)
    )
    assert [obs.responses["signal"].provenance for obs in observations] == [
        "mean of three repeats; uncertainty: repeat scatter; a lower bound",
        # No uncertainty was reported, so nothing is said about one.
        "mean of three repeats",
        "uncertainty: repeat scatter",
    ]


def test_a_table_from_another_producer_contract_is_refused() -> None:
    table = make_table(make_row("s1", 0.25, 2.0), producer="bench-table/2")
    with pytest.raises(TableMismatch, match="bench-table/2"):
        build_observations(DEFINITION, table)


def test_every_missing_column_is_listed() -> None:
    kept = tuple(c for c in COLUMNS if c not in {"signal_u", "humidity_pct", "atmosphere"})
    row = make_row("s1", 0.25, 2.0)
    table = RecordTable(
        producer="bench-table/1",
        columns=kept,
        rows=(replace(row, values={k: v for k, v in row.values.items() if k in kept}),),
    )
    with pytest.raises(TableMismatch) as caught:
        build_observations(DEFINITION, table)
    assert "atmosphere, humidity_pct, signal_u" in str(caught.value)


def test_a_control_within_tolerance_is_included_with_its_actual_value() -> None:
    (obs,), excluded = build_observations(
        DEFINITION, make_table(make_row("s1", 0.25, 2.0, bake_time_min=61.5))
    )
    assert not excluded
    assert obs.inputs["bake_time_min"] == 61.5


def test_a_control_exactly_at_its_tolerance_passes() -> None:
    # 62.0 - 60.0 is exactly the tolerance; so is a sum that lands a bit off it.
    for value in (62.0, 58.0, 60.0 + (0.1 + 0.2) * 20 / 3):
        observations, _ = build_observations(
            DEFINITION, make_table(make_row("s1", 0.25, 2.0, bake_time_min=value))
        )
        assert len(observations) == 1, value


def test_a_continuous_control_outside_tolerance_excludes_the_specimen() -> None:
    (reason,) = _excluded_reasons(make_row("s1", 0.25, 2.0, bake_time_min=63.0))
    assert reason == "control bake_time_min: recorded 63 min, expected 60 ± 2"


def test_categorical_and_integer_controls_compare_exactly() -> None:
    (reason,) = _excluded_reasons(make_row("s1", 0.25, 2.0, atmosphere="N2"))
    assert reason == "control atmosphere: recorded 'N2', expected 'Ar'"
    (reason,) = _excluded_reasons(make_row("s1", 0.25, 2.0, passes=4))
    assert reason == "control passes: recorded 4, expected 3"


def test_a_blank_control_is_flagged_and_kept() -> None:
    (obs,), excluded = build_observations(
        DEFINITION, make_table(make_row("s1", 0.25, 2.0, bake_time_min=None, atmosphere=" "))
    )
    assert not excluded
    flagged = {issue.field: issue for issue in obs.issues}
    assert set(flagged) == {"control:bake_time_min", "control:atmosphere"}
    assert flagged["control:bake_time_min"].severity is Severity.WARNING
    assert "not recorded for this specimen" in flagged["control:bake_time_min"].message


def test_a_declared_only_control_is_not_checked() -> None:
    # `glovebox` names no column, so no row can violate it.
    (obs,), _ = build_observations(DEFINITION, make_table(make_row("s1", 0.25, 2.0)))
    assert "glovebox" not in obs.inputs
    assert not obs.issues


def test_design_bounds_are_inclusive_and_enforced() -> None:
    table = make_table(make_row("lo", 0.0, 1.0), make_row("hi", 1.0, 1.0))
    observations, excluded = build_observations(DEFINITION, table)
    assert [obs.sample_id for obs in observations] == ["lo", "hi"]
    assert not excluded

    (reason,) = _excluded_reasons(make_row("s1", 1.2, 2.0))
    assert reason == "additive_fraction = 1.2 is outside its bounds [0, 1]"
    (reason,) = _excluded_reasons(make_row("s1", None, 2.0))
    assert reason == "no value for design variable additive_fraction"


def test_a_missing_objective_value_excludes_with_the_producers_own_reason() -> None:
    (reason,) = _excluded_reasons(make_row("s1", 0.25, None))
    assert reason == "no value for signal"

    said = ValidationIssue(
        field="signal",
        severity=Severity.ERROR,
        message="the trace ended early",
        detected_at=utc_now(),
    )
    (reason,) = _excluded_reasons(make_row("s1", 0.25, None, issues=(said,)))
    assert reason == "the trace ended early"

    (reason,) = _excluded_reasons(make_row("s1", 0.25, float("nan")))
    assert reason == "no value for signal"


def test_the_valid_range_excludes_for_the_objective_and_only_flags_otherwise() -> None:
    ranged = make_definition(responses=(replace(SIGNAL, valid_range=(-5.0, 5.0)), SIDE))
    observations, excluded = build_observations(ranged, make_table(make_row("s1", 0.25, 9.0)))
    assert not observations
    assert excluded[0].reasons == ("signal = 9 is outside its valid range [-5, 5]",)

    # `side_signal` is declared non-negative and is not the objective.
    (obs,), excluded = build_observations(
        ranged, make_table(make_row("s1", 0.25, 2.0, side_signal=-1.0))
    )
    assert not excluded
    assert obs.responses["side_signal"].value == -1.0
    assert [issue.field for issue in obs.issues] == ["side_signal"]


def test_a_missing_side_response_is_simply_absent() -> None:
    (obs,), _ = build_observations(
        DEFINITION, make_table(make_row("s1", 0.25, 2.0, side_signal=None))
    )
    assert set(obs.responses) == {"signal"}


def test_the_uncertainty_column_is_divided_by_the_coverage_factor() -> None:
    expanded = make_definition(responses=(replace(SIGNAL, coverage_factor=2.0), SIDE))
    (obs,), _ = build_observations(expanded, make_table(make_row("s1", 0.25, 2.0, 0.3)))
    assert obs.responses["signal"].uncertainty == pytest.approx(0.15)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), "wide"])
def test_an_unusable_uncertainty_becomes_absent_with_an_issue(bad: object) -> None:
    (obs,), _ = build_observations(DEFINITION, make_table(make_row("s1", 0.25, 2.0, bad)))
    signal = obs.responses["signal"]
    assert signal.uncertainty is None
    assert signal.uncertainty_scope is None
    assert [issue.field for issue in obs.issues] == ["signal_u"]


def test_a_repeated_id_excludes_every_row_that_carries_it() -> None:
    table = make_table(
        make_row("s1", 0.25, 2.0), make_row("s1", 0.50, 2.1), make_row("s2", 0.75, 2.2)
    )
    observations, excluded = build_observations(DEFINITION, table)
    assert [obs.sample_id for obs in observations] == ["s2"]
    assert [e.reasons for e in excluded] == [("sample id 's1' appears on 2 rows",)] * 2


def test_a_row_without_an_id_is_excluded() -> None:
    (reason,) = _excluded_reasons(make_row(None, 0.25, 2.0))
    assert reason == "no sample id"


def test_every_reason_is_collected_and_the_source_record_is_kept() -> None:
    row = make_row("s1", 1.5, None, bake_time_min=70.0, atmosphere="air")
    observations, (excluded,) = build_observations(DEFINITION, make_table(row))
    assert not observations
    assert excluded.reasons == (
        "no value for signal",
        "additive_fraction = 1.5 is outside its bounds [0, 1]",
        "control bake_time_min: recorded 70 min, expected 60 ± 2",
        "control atmosphere: recorded 'air', expected 'Ar'",
    )
    assert excluded.record is row
    assert excluded.sample_id == "s1"


def test_a_label_design_axis_can_be_built() -> None:
    from latos.experiments import VariableKind

    labelled = make_definition(
        variables=(replace(X, kind=VariableKind.CATEGORICAL, bounds=None, precision=None),)
    )
    table = make_table(make_row("s1", "route-a", 2.0), make_row("s2", None, 2.0))
    (obs,), (excluded,) = build_observations(labelled, table)
    assert obs.inputs["additive_fraction"] == "route-a"
    assert excluded.reasons == ("no value for design variable additive_fraction",)


# ─── Outside the design bounds ──────────────────────────────────────
def test_a_complete_specimen_outside_the_bounds_is_kept_whole_and_labelled() -> None:
    detail = {"repeats": [-2.1, -2.0, -1.9]}
    table = make_table(make_row("far", 1.2, -2.0, 0.3, detail=detail))
    observations, (row,) = build_observations(DEFINITION, table)

    assert observations == ()  # it is not one of the fitted points
    assert row.category == EXCLUDED_OUTSIDE_BOUNDS
    assert row.reasons == ("additive_fraction = 1.2 is outside its bounds [0, 1]",)
    assert row.record is table.rows[0]
    # Everything an observation carries, because it passed every check.
    kept = row.observation
    assert kept is not None
    assert kept.sample_id == "far"
    assert kept.inputs["additive_fraction"] == 1.2
    assert kept.inputs["bake_time_min"] == 60.0
    assert kept.metadata == {"batch": "b1"}
    signal = kept.responses["signal"]
    assert (signal.value, signal.uncertainty) == (-2.0, 0.3)  # signed, with its uncertainty
    assert signal.uncertainty_scope is UncertaintyScope.SPECIMEN
    assert kept.producer_detail == detail


def test_a_warning_on_a_specimen_outside_the_bounds_travels_with_it() -> None:
    note = ValidationIssue(
        field="signal",
        severity=Severity.WARNING,
        message="drifted during the session",
        detected_at=utc_now(),
    )
    _, (row,) = build_observations(
        DEFINITION, make_table(make_row("far", -0.1, 2.0, 0.1, issues=(note,)))
    )
    # The bounds are the whole reason. A warning is carried, and is not one.
    assert row.category == EXCLUDED_OUTSIDE_BOUNDS
    assert row.reasons == ("additive_fraction = -0.1 is outside its bounds [0, 1]",)
    assert row.observation is not None
    assert row.observation.issues == (note,)


def test_a_specimen_outside_the_bounds_that_also_fails_a_check_is_rejected() -> None:
    _, (row,) = build_observations(
        DEFINITION, make_table(make_row("far", 1.2, 2.0, bake_time_min=70.0))
    )
    assert row.category == EXCLUDED_REJECTED
    assert row.observation is None
    assert row.reasons == (
        "additive_fraction = 1.2 is outside its bounds [0, 1]",
        "control bake_time_min: recorded 70 min, expected 60 ± 2",
    )


def test_every_other_exclusion_is_a_rejection() -> None:
    rows = (
        make_row("unread", 0.25, None),
        make_row("late", 0.25, 2.0, bake_time_min=70.0),
        make_row(None, 0.25, 2.0),
        make_row("blank", None, 2.0),
    )
    observations, excluded = build_observations(DEFINITION, make_table(*rows))
    assert observations == ()
    assert [(row.category, row.observation) for row in excluded] == [(EXCLUDED_REJECTED, None)] * 4


def test_a_row_outside_the_bounds_with_no_id_is_rejected() -> None:
    _, (row,) = build_observations(DEFINITION, make_table(make_row(None, 1.2, 2.0)))
    assert (row.category, row.observation) == (EXCLUDED_REJECTED, None)


def test_a_control_with_no_tolerance_is_a_required_value() -> None:
    exact = make_definition(
        variables=tuple(replace(v, tolerance=0.0) if v is BAKE else v for v in VARIABLES)
    )
    observations, excluded = build_observations(exact, make_table(make_row("s1", 0.25, 2.0)))
    assert len(observations) == 1
    assert not excluded

    _, (row,) = build_observations(exact, make_table(make_row("s1", 0.25, 2.0, bake_time_min=60.5)))
    # Worded as a requirement, not as a quantity known with no uncertainty.
    assert row.reasons == ("control bake_time_min: recorded 60.5 min, expected exactly 60",)
