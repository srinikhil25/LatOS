"""Tests for the dimension-aware half of the reliability grade (MV3).

The count tiers answer "how much data is there". Fill distance answers "does it
cover the space", which is the question counting cannot answer once there is
more than one axis. These tests pin three properties:

  * in one dimension with evenly spread points the geometric rule and the count
    rule agree, so nothing that used to grade indicative or calibrated is
    silently demoted;
  * clustered points are caught however many of them there are;
  * the gate only ever downgrades.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization import optimize, optimize_nd
from latos.optimization.engine import (
    _FILL_CALIBRATED,
    _FILL_INDICATIVE,
    _SPAN_UNITS,
    _fill_distance,
)


def _hump(x: np.ndarray) -> np.ndarray:
    """A smooth, well-behaved target so the LOO gate never fires by accident."""
    return 1.0 - 0.04 * (x - 2.5) ** 2


class TestFillDistanceItself:
    def test_evenly_spaced_1d_is_half_the_gap(self):
        for n in (5, 10, 25, 40):
            pts = np.linspace(0.0, _SPAN_UNITS, n).reshape(-1, 1)
            assert _fill_distance(pts) == pytest.approx(_SPAN_UNITS / (2 * (n - 1)), rel=1e-9)

    def test_limits_are_the_count_tiers_restated(self):
        """The whole design rests on this: the geometric limits are the count
        thresholds expressed as a distance, so 1-D behaviour cannot drift."""
        ten = np.linspace(0.0, _SPAN_UNITS, 10).reshape(-1, 1)
        twenty_five = np.linspace(0.0, _SPAN_UNITS, 25).reshape(-1, 1)
        assert _fill_distance(ten) == pytest.approx(_FILL_INDICATIVE, rel=1e-9)
        assert _fill_distance(twenty_five) == pytest.approx(_FILL_CALIBRATED, rel=1e-9)

    def test_an_unsampled_end_of_the_box_counts(self):
        """Points crowded at one end leave the far end unexplored, and the
        measure has to notice — a gap at the edge is still a gap."""
        pts = np.linspace(0.0, 1.0, 30).reshape(-1, 1)
        assert _fill_distance(pts) == pytest.approx(_SPAN_UNITS - 1.0, rel=1e-9)

    def test_clustering_beats_counting(self):
        many_clustered = np.linspace(1.9, 2.1, 200).reshape(-1, 1)
        few_spread = np.linspace(0.0, _SPAN_UNITS, 12).reshape(-1, 1)
        assert _fill_distance(many_clustered) > _fill_distance(few_spread)

    def test_two_dimensions_need_far_more_points(self):
        """A line of points inside a plane leaves most of the plane empty."""
        axis = np.linspace(0.0, _SPAN_UNITS, 25)
        line = np.column_stack([axis, np.full_like(axis, _SPAN_UNITS / 2)])
        grid = np.array(
            [
                [a, b]
                for a in np.linspace(0, _SPAN_UNITS, 15)
                for b in np.linspace(0, _SPAN_UNITS, 15)
            ]
        )
        assert _fill_distance(line) > _FILL_INDICATIVE
        assert _fill_distance(grid) < _fill_distance(line)


class TestOneDimensionalTiersUnchanged:
    """Evenly spread 1-D data must grade exactly as it did before MV3."""

    def _run(self, n: int):
        x = np.linspace(0.0, 5.0, n)
        return optimize(x, _hump(x), bounds=(0.0, 5.0), input_name="doping_pct", target_name="zt")

    def test_fifteen_points_still_indicative(self):
        assert self._run(15).reliability.level == "indicative"

    def test_thirty_points_still_calibrated(self):
        assert self._run(30).reliability.level == "calibrated"

    def test_exactly_ten_points_is_not_demoted_by_a_rounding_step(self):
        """Ten evenly spaced points sit precisely on the limit they define.
        Without the boundary tolerance a float ULP decides the tier."""
        assert self._run(10).reliability.level != "exploratory"

    def test_exactly_twenty_five_points_is_not_demoted(self):
        assert self._run(25).reliability.level == "calibrated"


class TestTheGateOnlyDowngrades:
    def test_clustered_points_lose_their_count_tier(self):
        """Thirty points crammed into a tenth of the range would grade
        calibrated on count alone. They cover nothing, so they must not."""
        x = np.linspace(2.4, 2.6, 30)
        r = optimize(x, _hump(x), bounds=(0.0, 5.0), input_name="doping_pct", target_name="zt")
        assert r.reliability.n_observations == 30
        assert r.reliability.level == "exploratory"
        assert r.reliability.fill_distance > r.reliability.fill_limit
        assert "do not cover the space" in r.reliability.note

    def test_sparse_data_is_never_promoted_by_good_coverage(self):
        """Four well-spread points fill the box respectably for their number,
        but four points is still four points."""
        x = np.linspace(0.0, 5.0, 4)
        r = optimize(x, _hump(x), bounds=(0.0, 5.0), input_name="doping_pct", target_name="zt")
        assert r.reliability.level == "exploratory"

    def test_report_carries_the_measured_gap_and_its_limit(self):
        x = np.linspace(0.0, 5.0, 15)
        rel = optimize(
            x, _hump(x), bounds=(0.0, 5.0), input_name="doping_pct", target_name="zt"
        ).reliability
        assert rel.fill_distance > 0.0
        # The report rounds to 4 dp for display, so compare at that precision.
        assert rel.fill_limit == pytest.approx(_FILL_INDICATIVE, abs=1e-4)


class TestTwoDimensionalGrading:
    def test_a_sparse_axis_downgrades_a_large_point_count(self):
        """The research_1 shape: many temperatures, few compositions. Counting
        says calibrated; the composition axis is barely sampled."""
        dop = np.array([0.0, 1.0, 3.0, 5.0])
        temp = np.linspace(300.0, 600.0, 13)
        x = np.array([[a, b] for a in dop for b in temp])
        y = np.exp(-((x[:, 0] - 3.0) ** 2) / 2.0) * (x[:, 1] / 600.0)
        r = optimize_nd(
            x,
            y,
            bounds=[(0.0, 5.0), (300.0, 600.0)],
            input_names=("doping", "temp"),
            target_name="zt",
        )
        assert r.reliability.n_observations == 52
        assert r.reliability.level == "exploratory"
        assert r.reliability.n_dims == 2
        assert r.reliability.fill_distance > _FILL_INDICATIVE

    def test_a_well_filled_plane_keeps_a_better_grade(self):
        e = np.linspace(0.0, 5.0, 16)
        f = np.linspace(300.0, 600.0, 16)
        x = np.array([[a, b] for a in e for b in f])
        y = np.exp(-((x[:, 0] - 3.0) ** 2) / 2.0) * (x[:, 1] / 600.0)
        r = optimize_nd(
            x,
            y,
            bounds=[(0.0, 5.0), (300.0, 600.0)],
            input_names=("doping", "temp"),
            target_name="zt",
        )
        sparse_dop = np.array([0.0, 1.0, 3.0, 5.0])
        sparse = np.array([[a, b] for a in sparse_dop for b in np.linspace(300.0, 600.0, 13)])
        assert r.reliability.fill_distance < _fill_distance(
            (sparse - [0.0, 300.0]) / [5.0 / _SPAN_UNITS, 300.0 / _SPAN_UNITS]
        )
