"""Thermal-cycle windows, the per-cycle fit, and the reduction to one specimen."""

from __future__ import annotations

import math

import numpy as np
import pytest

from latos.analysis.thermovoltage.cycles import (
    AUTOMATIC_SETTINGS,
    CycleBoundaryError,
    CycleWindow,
    automatic_boundaries,
    declared_boundaries,
    event_boundaries,
    fit_cycle,
    reduce_cycles,
    windows_from_boundaries,
)
from latos.analysis.thermovoltage.slope import fit_seebeck_slope
from tests.unit._synthetic_trace import boundaries, series

SEVEN = boundaries()  # three cycles of 180 s on, 180 s off, from t = 10 s


# ─── Windows ────────────────────────────────────────────────────────
def test_seven_boundaries_are_three_cycles() -> None:
    windows = windows_from_boundaries(SEVEN)
    assert [(w.index, w.on_s, w.off_s, w.end_s) for w in windows] == [
        (1, 10.0, 190.0, 370.0),
        (2, 370.0, 550.0, 730.0),
        (3, 730.0, 910.0, 1090.0),
    ]
    assert all(w.on_duration_s == 180.0 and w.off_duration_s == 180.0 for w in windows)


def test_each_cycle_ends_where_the_next_begins() -> None:
    windows = windows_from_boundaries(boundaries((170.0, 185.0), (190.0, 175.0)))
    assert windows[0].end_s == windows[1].on_s
    assert (windows[0].on_duration_s, windows[0].off_duration_s) == (170.0, 190.0)
    assert (windows[1].on_duration_s, windows[1].off_duration_s) == (185.0, 175.0)


@pytest.mark.parametrize("count", [0, 1, 2, 4, 6, 8])
def test_a_count_that_cannot_be_whole_cycles_is_refused(count: int) -> None:
    with pytest.raises(CycleBoundaryError, match="must be odd and at least 3"):
        windows_from_boundaries([float(i) for i in range(count)])


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ([0.0, 10.0, 10.0], "10 s does not come after 10 s"),
        ([0.0, 20.0, 10.0], "10 s does not come after 20 s"),
        ([0.0, math.nan, 10.0], "not a finite time"),
    ],
)
def test_boundaries_must_be_finite_and_increasing(times: list[float], message: str) -> None:
    with pytest.raises(CycleBoundaryError, match=message):
        windows_from_boundaries(times)


def test_declared_times_are_interleaved_with_the_end_last() -> None:
    assert declared_boundaries([1, 5], [3, 7], 9) == [1.0, 3.0, 5.0, 7.0, 9.0]
    for on, off in (([1.0, 5.0], [3.0]), ([], [])):
        with pytest.raises(CycleBoundaryError, match="every cycle needs one of each"):
            declared_boundaries(on, off, 9.0)


# ─── Event marks ────────────────────────────────────────────────────
def test_marks_one_to_seven_are_three_cycles() -> None:
    times = event_boundaries([1, 2, 3, 4, 5, 6, 7], SEVEN)
    assert times == SEVEN
    assert len(windows_from_boundaries(times)) == 3


def test_marks_one_to_nine_are_four_cycles() -> None:
    nine = boundaries((180.0,) * 4, (180.0,) * 4)
    windows = windows_from_boundaries(event_boundaries(list(range(1, 10)), nine))
    assert [(w.on_s, w.end_s) for w in windows] == [
        (nine[0], nine[2]),
        (nine[2], nine[4]),
        (nine[4], nine[6]),
        (nine[6], nine[8]),
    ]


@pytest.mark.parametrize(
    ("numbers", "message"),
    [
        ([], "no event marks"),
        ([1, 2, 4, 5, 6, 7, 8], "numbered 1, 2, 4, 5, 6, 7, 8; expected 1 to 7 in order"),
        ([1, 2, 3, 3, 5, 6, 7], "missing, repeated or out of sequence"),
        ([1, 3, 2, 4, 5, 6, 7], "missing, repeated or out of sequence"),
        ([2, 3, 4], "expected 1 to 3 in order"),
        ([1, 2, 3, 4, 5, 6], "6 event mark.s. cannot describe whole cycles"),
        ([1, 2, 3, 4, 5, 6, 7, 8], "8 event mark.s. cannot describe whole cycles"),
        ([1], "1 event mark.s. cannot describe whole cycles"),
    ],
)
def test_a_wrong_mark_count_or_order_is_refused(numbers: list[int], message: str) -> None:
    with pytest.raises(CycleBoundaryError, match=message):
        event_boundaries(numbers, [10.0 * n for n in range(len(numbers))])


# ─── One cycle ──────────────────────────────────────────────────────
def test_the_slope_put_in_is_the_slope_got_back_cycle_by_cycle() -> None:
    time, cold, hot, voltage = series(SEVEN, seebeck_mv_k=[2.0, -1.5, 0.75], offset_mv=5.0)
    fits = [
        fit_cycle(time, cold, hot, voltage, window, polarity_sign=1)
        for window in windows_from_boundaries(SEVEN)
    ]
    # Exact only if no sample of a neighbouring cycle is inside the window:
    # each window is closed at its start and open at its end.
    assert [f.slope_mv_k for f in fits] == pytest.approx([2.0, -1.5, 0.75], abs=1e-12)
    assert [f.intercept_mv for f in fits] == pytest.approx([5.0] * 3, abs=1e-10)
    assert [f.n_samples for f in fits] == [360, 360, 360]
    assert all(f.problems == () for f in fits)
    assert all(f.delta_t_max_k == pytest.approx(15.0, abs=0.01) for f in fits)
    assert all(f.mean_temperature == pytest.approx(25.0) for f in fits)


def test_the_sign_follows_the_wiring_and_is_never_dropped() -> None:
    time, cold, hot, voltage = series(SEVEN, seebeck_mv_k=-2.0, offset_mv=5.0)
    window = windows_from_boundaries(SEVEN)[0]
    straight = fit_cycle(time, cold, hot, voltage, window, polarity_sign=1)
    swapped = fit_cycle(time, cold, hot, voltage, window, polarity_sign=-1)
    assert straight.slope_mv_k == pytest.approx(-2.0)  # negative stays negative
    assert swapped.slope_mv_k == pytest.approx(2.0)
    # The series is oriented, not the slope, so the intercept turns over too.
    assert (straight.intercept_mv, swapped.intercept_mv) == pytest.approx((5.0, -5.0))


def test_the_whole_cycle_fit_is_the_production_slope_fit_on_the_window() -> None:
    time, cold, hot, voltage = series(SEVEN, lag_mv=1.5)
    window = windows_from_boundaries(SEVEN)[1]
    fit = fit_cycle(time, cold, hot, voltage, window, polarity_sign=1)

    inside = (time >= window.on_s) & (time < window.end_s)
    expected = fit_seebeck_slope((hot - cold)[inside], voltage[inside])
    assert fit.slope_mv_k == expected.slope
    assert fit.intercept_mv == expected.intercept
    assert fit.slope_stderr_mv_k == expected.slope_stderr
    assert fit.r_squared == expected.r_squared


def test_the_branches_are_reported_beside_the_value_not_in_it() -> None:
    time, cold, hot, voltage = series(SEVEN, lag_mv=1.5)
    window = windows_from_boundaries(SEVEN)[0]
    fit = fit_cycle(time, cold, hot, voltage, window, polarity_sign=1)

    delta = hot - cold
    heating = (time >= window.on_s) & (time < window.off_s)
    cooling = (time >= window.off_s) & (time < window.end_s)
    assert fit.on_slope_mv_k == fit_seebeck_slope(delta[heating], voltage[heating]).slope
    assert fit.off_slope_mv_k == fit_seebeck_slope(delta[cooling], voltage[cooling]).slope
    # Each branch alone is the exact line; the loop between them is what moves
    # the whole-cycle slope, which is why it is the whole cycle that is fitted.
    assert fit.on_slope_mv_k == pytest.approx(2.0)
    assert fit.off_slope_mv_k == pytest.approx(2.0)
    assert fit.slope_mv_k != pytest.approx(2.0, abs=1e-3)
    assert fit.slope_stderr_mv_k > 0


def test_swapped_thermocouples_or_shifted_boundaries_are_reported() -> None:
    time, cold, hot, voltage = series(SEVEN)
    window = windows_from_boundaries(SEVEN)[0]

    swapped = fit_cycle(time, hot, cold, voltage, window, polarity_sign=1)
    assert len(swapped.problems) == 2
    assert "does not rise while the drive is on in cycle 1" in swapped.problems[0]
    assert "does not fall while the drive is off in cycle 1" in swapped.problems[1]

    # Half a period late: "on" now covers a cooling phase.
    late = CycleWindow(index=1, on_s=190.0, off_s=370.0, end_s=550.0)
    assert len(fit_cycle(time, cold, hot, voltage, late, polarity_sign=1).problems) == 2


def test_a_flat_temperature_difference_gives_no_slope() -> None:
    time = np.arange(0.0, 100.0)
    flat = np.full_like(time, 25.0)
    window = CycleWindow(index=1, on_s=10.0, off_s=40.0, end_s=70.0)
    fit = fit_cycle(time, flat, flat, np.ones_like(time), window, polarity_sign=1)
    assert any("does not vary in cycle 1" in problem for problem in fit.problems)


def test_a_window_outside_the_recording_is_refused() -> None:
    time, cold, hot, voltage = series(SEVEN, tail_s=10.0)
    beyond = CycleWindow(index=3, on_s=730.0, off_s=910.0, end_s=2000.0)
    with pytest.raises(CycleBoundaryError, match="outside the recording"):
        fit_cycle(time, cold, hot, voltage, beyond, polarity_sign=1)
    thin = CycleWindow(index=1, on_s=10.0, off_s=11.5, end_s=200.0)
    with pytest.raises(CycleBoundaryError, match="heating branch of cycle 1 holds 2 sample"):
        fit_cycle(time, cold, hot, voltage, thin, polarity_sign=1)


# ─── One specimen ───────────────────────────────────────────────────
def test_cycles_reduce_to_a_signed_mean_and_its_standard_error() -> None:
    reduction = reduce_cycles([2.0, 2.1, 2.3])
    sd = float(np.std([2.0, 2.1, 2.3], ddof=1))
    assert reduction.k == 3
    assert reduction.dof == 2
    assert reduction.mean == pytest.approx(6.4 / 3)
    assert reduction.sd == pytest.approx(sd)
    assert reduction.sem == pytest.approx(sd / math.sqrt(3))


def test_the_reduction_keeps_signs() -> None:
    assert reduce_cycles([-3.0, -2.5, -2.0]).mean == pytest.approx(-2.5)
    mixed = reduce_cycles([1.0, -1.0, 0.3])
    # The mean of the signed values, never the mean of their magnitudes.
    assert mixed.mean == pytest.approx(0.1)
    assert mixed.sd == pytest.approx(float(np.std([1.0, -1.0, 0.3], ddof=1)))


def test_one_cycle_has_no_repeatability_and_none_cannot_be_reduced() -> None:
    single = reduce_cycles([2.0])
    assert (single.k, single.mean, single.sd, single.sem, single.dof) == (1, 2.0, None, None, 0)
    with pytest.raises(ValueError, match="no cycles"):
        reduce_cycles([])
    with pytest.raises(ValueError, match="not finite"):
        reduce_cycles([2.0, math.nan])


# ─── The automatic fallback ─────────────────────────────────────────
def _measured_like(switches: list[float], *, interval_s: float = 0.2) -> tuple:
    """A trace with thermocouple noise, rounded to a tenth of a degree."""
    time, cold, hot, _ = series(switches, interval_s=interval_s, tail_s=12.0)
    rng = np.random.default_rng(3)
    cold = np.round(cold + rng.normal(0.0, 0.04, time.size), 1)
    hot = np.round(hot + rng.normal(0.0, 0.04, time.size), 1)
    return time, hot - cold


def test_the_fallback_finds_the_switches_of_a_noisy_trace() -> None:
    time, delta = _measured_like(SEVEN)
    found = automatic_boundaries(time, delta)
    assert found.end_is_recording_end is True
    assert len(found.boundaries) == 7
    # It runs late by design: the trace has to leave its trough to be seen.
    late = np.asarray(found.boundaries[:-1]) - np.asarray(SEVEN[:-1])
    assert np.all(late >= -1.0)
    assert np.all(late <= 4.0)
    # It cannot see where the last cycle ends, so it uses the end of the file.
    assert found.boundaries[-1] == time[-1]
    assert found.swing_k == pytest.approx(15.0, abs=0.5)


def test_a_blip_is_not_counted_as_a_switch() -> None:
    time, delta = _measured_like(SEVEN)
    blip = (time >= 250.0) & (time < 252.0)  # during the first cooling phase
    delta = delta + np.where(blip, 1.5, 0.0)
    assert len(automatic_boundaries(time, delta).boundaries) == 7


def test_an_unfinished_last_cycle_closes_the_one_before_it() -> None:
    time, delta = _measured_like(SEVEN)
    keep = time <= 800.0  # the third switch-on is at 730 s; no switch-off follows
    found = automatic_boundaries(time[keep], delta[keep])
    assert found.end_is_recording_end is False
    assert len(found.boundaries) == 5  # two whole cycles, closed by the third switch-on
    assert found.boundaries[-1] == pytest.approx(730.0, abs=4.0)


def test_the_fallback_refuses_what_it_cannot_segment() -> None:
    time = np.arange(0.0, 600.0, 0.2)
    with pytest.raises(CycleBoundaryError, match="swings by only"):
        automatic_boundaries(time, np.full_like(time, 0.3))

    full_time, delta = _measured_like(SEVEN)
    started = full_time >= 100.0  # the drive is already on
    with pytest.raises(CycleBoundaryError, match="already high"):
        automatic_boundaries(full_time[started], delta[started])


def test_the_fallback_settings_are_published_for_the_record() -> None:
    assert AUTOMATIC_SETTINGS == {
        "smoothing_window_s": 2.0,
        "swing_low_percentile": 5.0,
        "swing_high_percentile": 95.0,
        "switch_on_level": 0.75,
        "switch_off_level": 0.25,
        "onset_share_of_swing": 0.05,
        "minimum_swing_k": 1.0,
    }
