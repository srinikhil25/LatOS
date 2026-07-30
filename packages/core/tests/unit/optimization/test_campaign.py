"""Tests for campaign drift (`latos.optimization.campaign`).

The point of the module is a convergence signal that does NOT come from the
model, so the tests exercise it purely from frozen-record fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from latos.optimization.campaign import recommendation_drift


@dataclass(frozen=True)
class _Entry:
    """The subset of PreregEntry that drift reads."""

    created_at: str
    recommended_x: float
    input_variable: str = "doping_pct"
    property_name: str = "zt"
    direction: str = "maximize"
    search_bounds: tuple[float, float] | None = (0.0, 5.0)


class TestGrouping:
    def test_no_entries_is_empty(self):
        assert recommendation_drift([]) == []

    def test_separate_objectives_do_not_mix(self):
        drifts = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 1.0),
                _Entry("2026-07-02T00:00:00", 1.1),
                _Entry("2026-07-01T00:00:00", 400.0, input_variable="anneal_c"),
                _Entry("2026-07-02T00:00:00", 401.0, input_variable="anneal_c"),
            ]
        )
        assert len(drifts) == 2
        assert {d.input_variable for d in drifts} == {"doping_pct", "anneal_c"}
        assert all(d.n_freezes == 2 for d in drifts)

    def test_direction_separates_campaigns(self):
        # Minimizing and maximizing the same property are different searches.
        drifts = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 1.0, direction="maximize"),
                _Entry("2026-07-02T00:00:00", 4.0, direction="minimize"),
            ]
        )
        assert len(drifts) == 2

    def test_entries_are_ordered_by_time_not_input_order(self):
        drifts = recommendation_drift(
            [
                _Entry("2026-07-03T00:00:00", 3.0),
                _Entry("2026-07-01T00:00:00", 1.0),
                _Entry("2026-07-02T00:00:00", 2.0),
            ]
        )
        steps = drifts[0].steps
        assert [s.from_x for s in steps] == [1.0, 2.0]
        assert [s.to_x for s in steps] == [2.0, 3.0]


class TestSettled:
    def test_single_freeze_is_unknown_not_settled(self):
        # A single point cannot show movement. Reporting "settled" here would
        # be exactly the premature confidence the module exists to catch.
        (d,) = recommendation_drift([_Entry("2026-07-01T00:00:00", 3.0)])
        assert d.n_freezes == 1
        assert d.settled is None
        assert d.latest_fraction is None
        assert d.steps == ()
        assert "one freeze" in d.note.lower()

    def test_small_move_is_settled(self):
        # 0.1 on a span of 5.0 is 2%, under the 5% mark.
        (d,) = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 3.0),
                _Entry("2026-07-02T00:00:00", 3.1),
            ]
        )
        assert d.settled is True
        assert d.latest_fraction == pytest.approx(0.1 / 5.0)

    def test_large_move_is_not_settled(self):
        # 2.0 on a span of 5.0 is 40%.
        (d,) = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 1.0),
                _Entry("2026-07-02T00:00:00", 3.0),
            ]
        )
        assert d.settled is False
        assert d.latest_fraction == 2.0 / 5.0
        assert "still moving" in d.note

    def test_only_the_latest_move_decides(self):
        # Settling late still counts as settled: an early jump is history.
        (d,) = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 1.0),
                _Entry("2026-07-02T00:00:00", 4.0),
                _Entry("2026-07-03T00:00:00", 4.05),
            ]
        )
        assert len(d.steps) == 2
        assert d.settled is True

    def test_threshold_is_configurable_and_inclusive(self):
        entries = [
            _Entry("2026-07-01T00:00:00", 3.0),
            _Entry("2026-07-02T00:00:00", 3.5),  # 10% of the span
        ]
        assert recommendation_drift(entries, settled_fraction=0.10)[0].settled is True
        assert recommendation_drift(entries, settled_fraction=0.09)[0].settled is False


class TestSpan:
    def test_span_comes_from_the_frozen_bounds(self):
        (d,) = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 1.0, search_bounds=(0.0, 10.0)),
                _Entry("2026-07-02T00:00:00", 2.0, search_bounds=(0.0, 10.0)),
            ]
        )
        assert d.search_span == 10.0
        assert d.latest_fraction == 0.1

    def test_missing_bounds_leaves_the_fraction_unknown(self):
        # The distance is still true, but scaling it against a span nobody
        # recorded would be inventing a number.
        (d,) = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 1.0, search_bounds=None),
                _Entry("2026-07-02T00:00:00", 2.0, search_bounds=None),
            ]
        )
        assert d.search_span is None
        assert d.latest_fraction is None
        assert d.settled is None
        assert d.steps[0].distance == 1.0
        assert "search bounds" in d.note

    def test_degenerate_bounds_are_not_used_as_a_span(self):
        # A zero-width range would divide by zero; fall through to unknown.
        (d,) = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 1.0, search_bounds=(2.0, 2.0)),
                _Entry("2026-07-02T00:00:00", 2.0, search_bounds=(2.0, 2.0)),
            ]
        )
        assert d.search_span is None
        assert d.settled is None

    def test_explicit_override_wins(self):
        (d,) = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 1.0, search_bounds=(0.0, 5.0)),
                _Entry("2026-07-02T00:00:00", 2.0, search_bounds=(0.0, 5.0)),
            ],
            search_span=100.0,
        )
        assert d.search_span == 100.0
        assert d.latest_fraction == 0.01


class TestSteps:
    def test_distance_is_absolute(self):
        # Moving down is still movement.
        (d,) = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 4.0),
                _Entry("2026-07-02T00:00:00", 1.0),
            ]
        )
        assert d.steps[0].distance == 3.0

    def test_step_carries_both_timestamps(self):
        (d,) = recommendation_drift(
            [
                _Entry("2026-07-01T00:00:00", 1.0),
                _Entry("2026-07-02T00:00:00", 2.0),
            ]
        )
        step = d.steps[0]
        assert step.from_created_at == "2026-07-01T00:00:00"
        assert step.to_created_at == "2026-07-02T00:00:00"
