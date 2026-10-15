"""Thermal-cycle analysis of a logged trace: windows, per-cycle response, reduction.

A dynamic-cycle run switches the temperature drive on and off several times on
one mounted specimen while the logger records both electrode temperatures and
the cell voltage. This module holds the arithmetic for such a run, in three
parts that do not know about each other:

1. **Where each cycle begins and ends** — `windows_from_boundaries`, fed by
   `declared_boundaries`, `event_boundaries` or `automatic_boundaries`.
2. **The response of one cycle** — `fit_cycle`.
3. **What the cycles of one specimen reduce to** — `reduce_cycles`.

Everything here takes plain arrays and numbers. It reads no file and knows no
file format, so the same functions serve any logger.

The measurand
-------------
The response of a cycle is the **apparent cycle Seebeck coefficient**: the
ordinary least-squares slope of the oriented voltage against the temperature
difference, over every sample of the whole cycle, heating and cooling
together::

    V(t) = S_cycle · ΔT(t) + b,        t in [t_on, t_next_on)

where `V` is the recorded voltage times the wiring sign (+1 with the positive
lead on the cold electrode, -1 with it on the hot electrode), so that `V` is
the cold electrode's potential minus the hot one's, given the logger reads the
positive lead minus the negative lead. The fit has a free intercept `b`, no
time term and no correction for the lag between
the thermocouples and the cell. It is an operational, dynamic quantity. It is
**not** a steady-state Seebeck coefficient: the cell is never at rest during a
cycle, so the slope mixes the thermovoltage with whatever part of the ionic
response keeps up with the drive. Two runs are comparable only when their drive
programme is the same, which is why the programme is recorded and checked by
the caller rather than assumed here.

The slopes of the heating branch alone and of the cooling branch alone are
reported next to it as diagnostics. A large gap between them is hysteresis: the
voltage is not a function of ΔT alone. They never enter the response.

Windows
-------
A run is described by its switching times, in order::

    on_1, off_1, on_2, off_2, …, on_n, off_n, end

Cycle `k` is the half-open window ``[on_k, on_{k+1})``, and the last cycle is
``[on_n, end)``. Half-open so that a sample at a switch belongs to exactly one
cycle; a sample at `end` belongs to none. Its heating branch is
``[on_k, off_k)`` and its cooling branch ``[off_k, on_{k+1})``.

Where the switching times come from decides how far they can be trusted, and
the three sources are ranked:

* `BOUNDARY_DECLARED` — written down by the person who ran it. Always wins.
* `BOUNDARY_EVENT_MARKS` — the operator pressed the logger's mark key at each
  switch. The marks must be numbered 1, 2, 3, … in order, and there must be an
  odd number of them: two per cycle and one for the end.
* `BOUNDARY_AUTOMATIC` — read off the temperature trace. A fallback only.

Automatic segmentation, and why it is only a fallback
-----------------------------------------------------
With no marks and no declaration, the switching times can be estimated from
the temperature difference itself: it climbs when the drive is on and falls
when it is off. `automatic_boundaries` does it in two steps.

*Structure.* The trace is smoothed with a short moving mean, and its swing is
taken as the distance between its 5th and 95th percentiles. A switch-on is
counted when the smoothed trace climbs past 75 % of the swing, a switch-off
when it then falls past 25 %. The two levels are far apart, so a blip of a few
tenths of a kelvin cannot be counted as a switch.

*Timing.* Crossing a level happens well after the switch. So from each crossing
the trace is followed backwards to the point where it last sat within 5 % of
the swing of the preceding trough (for a switch-on) or crest (for a
switch-off). That point is the estimate.

The estimate runs late, by about a second on the traces it was checked against,
and it scatters by about a second around the operator's own marks. Two things
it cannot do at all. It cannot see the end of the last cycle, because nothing
happens to the temperature when a run is simply stopped: with an even number of
switches it closes the last cycle at the end of the recording, and with an odd
number the last switch-on closes the cycle before it, and the unfinished cycle
after that switch-on is left out. And it cannot tell a test cycle from a real
one. Every result it gives must therefore travel with a warning, and a
declaration or a set of marks replaces it as soon as either exists.

Reducing the cycles of one specimen
-----------------------------------
The cycles of one mounting are repeated measurements of **one specimen**. They
reduce to a single signed value::

    S_specimen = mean(S_1 … S_k)
    s          = sample standard deviation of S_1 … S_k    (k − 1 in the divisor)
    SEM        = s / sqrt(k),    with k − 1 degrees of freedom

Signs are kept throughout. No absolute value is taken of a cycle slope, so
cycles that disagree in sign pull the mean towards zero instead of hiding the
disagreement.

What the SEM is, and is not. It is a Type A estimate of how well the cycles of
one mounting, in one session, pin down that mounting's mean. It says nothing
about the next mounting of the same mixture: dosing, contact and the cloth
itself all change, and none of that varies between cycles. It is therefore
**one component** of the specimen's uncertainty, not a complete uncertainty
budget: variation between mountings and sessions, and the instrument, are not
quantified in it. It travels labelled as within-specimen only. The label's
identifier keeps the name `lower_bound_specimen` for compatibility, but the
SEM is not a rigorously established lower bound on the total experimental
uncertainty. It is itself an estimate: from few cycles, such as only three
(two degrees of freedom), it may underestimate the underlying standard error it
estimates. For independent, normally distributed cycles from one distribution
(one mean and one spread) it does so more than half the time for any number of
cycles, and about 63% of the time with three (1 - 1/e). The SEM also assumes
the cycles are independent and exchangeable, which has **not** been shown for
consecutive cycles on one cell; a drift from cycle to cycle makes it too small
or too large in ways three values cannot reveal.

The standard error of a single cycle's slope is reported as a diagnostic and is
**never** used as a measurement uncertainty. Its samples are a fraction of a
second apart and are unlikely to be independent, so the least-squares formula,
which assumes independent residuals, may understate the scatter of the slope.
By how much is not quantified here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from latos.analysis.thermovoltage.slope import fit_seebeck_slope

__all__ = [
    "AUTOMATIC_SETTINGS",
    "BOUNDARY_AUTOMATIC",
    "BOUNDARY_DECLARED",
    "BOUNDARY_EVENT_MARKS",
    "AutomaticBoundaries",
    "CycleBoundaryError",
    "CycleFit",
    "CycleReduction",
    "CycleWindow",
    "automatic_boundaries",
    "declared_boundaries",
    "event_boundaries",
    "fit_cycle",
    "reduce_cycles",
    "windows_from_boundaries",
]

# Where a run's switching times came from, most trusted first.
BOUNDARY_DECLARED = "declared"
BOUNDARY_EVENT_MARKS = "event_marks"
BOUNDARY_AUTOMATIC = "automatic"

# Two boundaries per cycle and one for the end of the run.
_MIN_BOUNDARIES = 3

# A branch needs enough samples for its own slope to mean anything.
_MIN_BRANCH_SAMPLES = 3

# The share of a branch, at each of its ends, averaged to decide whether the
# temperature difference rose or fell across it.
_DIRECTION_SHARE = 0.2

# Automatic segmentation. Recorded with every result that used it.
_AUTO_SMOOTH_S = 2.0
_AUTO_SWING_PERCENTILES = (5.0, 95.0)
_AUTO_RISE_LEVEL = 0.75
_AUTO_FALL_LEVEL = 0.25
_AUTO_ONSET_SHARE = 0.05
# Below this swing there is no cycle to find, only thermocouple noise.
_AUTO_MIN_SWING_K = 1.0

AUTOMATIC_SETTINGS: dict[str, float] = {
    "smoothing_window_s": _AUTO_SMOOTH_S,
    "swing_low_percentile": _AUTO_SWING_PERCENTILES[0],
    "swing_high_percentile": _AUTO_SWING_PERCENTILES[1],
    "switch_on_level": _AUTO_RISE_LEVEL,
    "switch_off_level": _AUTO_FALL_LEVEL,
    "onset_share_of_swing": _AUTO_ONSET_SHARE,
    "minimum_swing_k": _AUTO_MIN_SWING_K,
}


class CycleBoundaryError(ValueError):
    """The switching times offered do not describe a run of whole cycles."""


@dataclass(frozen=True, slots=True)
class CycleWindow:
    """One thermal cycle of a run.

    Attributes:
        index: Position in the run, counting from 1.
        on_s: When the drive was switched on: the start of the cycle.
        off_s: When it was switched off.
        end_s: When the cycle ended: the next switch-on, or the end of the run.
    """

    index: int
    on_s: float
    off_s: float
    end_s: float

    @property
    def on_duration_s(self) -> float:
        """How long the drive was on."""
        return self.off_s - self.on_s

    @property
    def off_duration_s(self) -> float:
        """How long the drive was off before the cycle ended."""
        return self.end_s - self.off_s


@dataclass(frozen=True, slots=True)
class AutomaticBoundaries:
    """Switching times estimated from the temperature trace.

    Attributes:
        boundaries: ``on_1, off_1, …, on_n, off_n, end``.
        end_is_recording_end: True when the last cycle was closed at the end of
            the recording because no later switch-on was found.
        swing_k: The swing of the smoothed trace the levels were set from.
    """

    boundaries: tuple[float, ...]
    end_is_recording_end: bool
    swing_k: float


@dataclass(frozen=True, slots=True)
class CycleFit:
    """The response of one cycle, with its diagnostics.

    Attributes:
        window: The cycle that was fitted.
        n_samples: How many samples fell in the window.
        slope_mv_k: The apparent cycle Seebeck coefficient, signed.
        intercept_mv: The fitted intercept.
        slope_stderr_mv_k: The least-squares standard error of the slope.
            Diagnostic only; never a measurement uncertainty.
        r_squared: Coefficient of determination of the whole-cycle fit.
        on_slope_mv_k: Slope of the heating branch alone. Diagnostic.
        off_slope_mv_k: Slope of the cooling branch alone. Diagnostic.
        delta_t_min_k: The smallest temperature difference in the window.
        delta_t_max_k: The largest.
        mean_temperature: The mean of the two electrode temperatures over the
            window, in the unit they were recorded in.
        problems: Why this window is not a valid cycle, if it is not. A fit
            with problems must not be used as a response.
    """

    window: CycleWindow
    n_samples: int
    slope_mv_k: float
    intercept_mv: float
    slope_stderr_mv_k: float
    r_squared: float
    on_slope_mv_k: float
    off_slope_mv_k: float
    delta_t_min_k: float
    delta_t_max_k: float
    mean_temperature: float
    problems: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CycleReduction:
    """The cycles of one specimen, reduced.

    Attributes:
        k: How many cycles were reduced.
        mean: Their signed mean.
        sd: Their sample standard deviation, or None with fewer than two.
        sem: `sd / sqrt(k)`, or None with fewer than two. A within-specimen,
            within-session Type A estimate of how well the cycles pin down
            their mean, one component of the uncertainty; see the module
            docstring.
        dof: Degrees of freedom behind `sd`: `k - 1`.
    """

    k: int
    mean: float
    sd: float | None
    sem: float | None
    dof: int


def windows_from_boundaries(boundaries: Sequence[float]) -> tuple[CycleWindow, ...]:
    """Turn ``on_1, off_1, …, on_n, off_n, end`` into one window per cycle.

    Raises:
        CycleBoundaryError: If there are too few boundaries, an even number of
            them, or they are not strictly increasing.
    """
    times = [float(b) for b in boundaries]
    count = len(times)
    if count < _MIN_BOUNDARIES or count % 2 == 0:
        raise CycleBoundaryError(
            f"{count} boundaries cannot describe whole cycles: each cycle needs a switch-on "
            "and a switch-off, and the run needs one end, so the count must be odd and at "
            f"least {_MIN_BOUNDARIES}"
        )
    if not all(math.isfinite(t) for t in times):
        raise CycleBoundaryError("a boundary is not a finite time")
    for earlier, later in pairwise(times):
        if not earlier < later:
            raise CycleBoundaryError(
                f"boundaries must increase; {later:g} s does not come after {earlier:g} s"
            )
    return tuple(
        CycleWindow(index=k + 1, on_s=times[2 * k], off_s=times[2 * k + 1], end_s=times[2 * k + 2])
        for k in range(count // 2)
    )


def declared_boundaries(on_s: Sequence[float], off_s: Sequence[float], end_s: float) -> list[float]:
    """Interleave declared switch-on and switch-off times, and append the end.

    Raises:
        CycleBoundaryError: If there is not exactly one switch-off for every
            switch-on.
    """
    if not on_s or len(on_s) != len(off_s):
        raise CycleBoundaryError(
            f"{len(on_s)} switch-on time(s) and {len(off_s)} switch-off time(s) were declared; "
            "every cycle needs one of each"
        )
    boundaries: list[float] = []
    for on, off in zip(on_s, off_s, strict=True):
        boundaries += [float(on), float(off)]
    boundaries.append(float(end_s))
    return boundaries


def event_boundaries(numbers: Sequence[int], times_s: Sequence[float]) -> list[float]:
    """The switching times given by a run's event marks.

    The marks must be numbered 1, 2, 3, … in the order they were pressed. Odd
    marks are switch-ons, even marks are switch-offs, and the last mark is the
    end of the run.

    Args:
        numbers: The number the logger wrote for each mark, in row order.
        times_s: When each mark was pressed, in the same order.

    Raises:
        CycleBoundaryError: If there are no marks, the numbers are not
            1, 2, 3, … in order, or the count cannot describe whole cycles.
    """
    if len(numbers) != len(times_s):
        raise ValueError("numbers and times_s must have the same length")
    if not numbers:
        raise CycleBoundaryError("the recording has no event marks")
    expected = list(range(1, len(numbers) + 1))
    if list(numbers) != expected:
        found = ", ".join(str(n) for n in numbers)
        raise CycleBoundaryError(
            f"the event marks are numbered {found}; expected 1 to {len(numbers)} in order, "
            "so a mark is missing, repeated or out of sequence"
        )
    count = len(numbers)
    if count < _MIN_BOUNDARIES or count % 2 == 0:
        raise CycleBoundaryError(
            f"{count} event mark(s) cannot describe whole cycles: each cycle needs a "
            "switch-on mark and a switch-off mark, and the run needs one end mark, so the "
            f"count must be odd and at least {_MIN_BOUNDARIES}"
        )
    return [float(t) for t in times_s]


def _moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Centred moving mean over an odd `width`, with the ends held."""
    if width <= 1:
        return values.astype(np.float64, copy=True)
    pad = width // 2
    padded = np.concatenate([np.full(pad, values[0]), values, np.full(pad, values[-1])])
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def automatic_boundaries(time_s: np.ndarray, delta_t_k: np.ndarray) -> AutomaticBoundaries:
    """Estimate a run's switching times from its temperature difference.

    A fallback for a recording with neither a declaration nor event marks. See
    the module docstring for the method and for what it cannot see.

    Args:
        time_s: Sample times, increasing.
        delta_t_k: Hot minus cold, one value per sample.

    Raises:
        CycleBoundaryError: If the trace does not swing enough to hold a cycle,
            starts with the drive already on, or holds no complete cycle.
    """
    time = np.asarray(time_s, dtype=np.float64)
    delta = np.asarray(delta_t_k, dtype=np.float64)
    interval = float(np.median(np.diff(time)))
    width = max(1, round(_AUTO_SMOOTH_S / interval))
    width += 1 - width % 2  # odd, so the mean is centred
    smoothed = _moving_mean(delta, width)

    low, high = (float(v) for v in np.percentile(smoothed, _AUTO_SWING_PERCENTILES))
    swing = high - low
    if swing < _AUTO_MIN_SWING_K:
        raise CycleBoundaryError(
            f"the temperature difference swings by only {swing:.2g} K, under the "
            f"{_AUTO_MIN_SWING_K:g} K needed to find a cycle in it"
        )
    rise_level = low + _AUTO_RISE_LEVEL * swing
    fall_level = low + _AUTO_FALL_LEVEL * swing
    onset = _AUTO_ONSET_SHARE * swing
    if smoothed[0] >= rise_level:
        raise CycleBoundaryError(
            "the recording starts with the temperature difference already high, so the "
            "first switch-on is not in it"
        )

    switches: list[float] = []
    driven = False
    since = 0  # index of the previous crossing
    for index, value in enumerate(smoothed):
        if not driven and value >= rise_level:
            # Back to where the trace last sat at the bottom of its trough.
            threshold = float(np.min(smoothed[since : index + 1])) + onset
            at = index
            while at > since and smoothed[at] > threshold:
                at -= 1
        elif driven and value <= fall_level:
            threshold = float(np.max(smoothed[since : index + 1])) - onset
            at = index
            while at > since and smoothed[at] < threshold:
                at -= 1
        else:
            continue
        switches.append(float(time[at]))
        driven = not driven
        since = index

    # An odd count ends on a switch-on with no switch-off after it: that
    # switch-on closes the cycle before it, and the unfinished one is left out.
    closed_by_recording = len(switches) % 2 == 0
    boundaries = [*switches, float(time[-1])] if closed_by_recording else switches
    if len(boundaries) < _MIN_BOUNDARIES:
        raise CycleBoundaryError(
            "no complete cycle was found in the temperature difference: "
            f"{len(switches)} switch(es) detected"
        )
    return AutomaticBoundaries(
        boundaries=tuple(boundaries),
        end_is_recording_end=closed_by_recording,
        swing_k=swing,
    )


def _direction(delta_t: np.ndarray) -> float:
    """Mean of the last share of a branch minus the mean of its first share."""
    span = max(1, round(delta_t.size * _DIRECTION_SHARE))
    return float(np.mean(delta_t[-span:]) - np.mean(delta_t[:span]))


def fit_cycle(
    time_s: np.ndarray,
    cold: np.ndarray,
    hot: np.ndarray,
    voltage_mv: np.ndarray,
    window: CycleWindow,
    *,
    polarity_sign: int,
) -> CycleFit:
    """Fit the whole-cycle slope of one window, with its branch diagnostics.

    The voltage is multiplied by `polarity_sign` before fitting, which is how
    the wiring reaches the arithmetic everywhere in this package: the series is
    oriented, not the fitted slope, so the intercept stays on the same sign.

    Args:
        time_s: Sample times.
        cold: Temperature of the cold electrode, one value per sample.
        hot: Temperature of the hot electrode, same unit.
        voltage_mv: Recorded cell voltage in millivolts, exactly as wired.
        window: The cycle to fit.
        polarity_sign: +1 when the positive lead sat on the cold electrode,
            -1 when it sat on the hot one. From `slope.resolve_polarity`.

    Returns:
        A `CycleFit`. Check `problems` before using it.

    Raises:
        CycleBoundaryError: If the window reaches outside the recording, or a
            branch holds too few samples to fit.
    """
    time = np.asarray(time_s, dtype=np.float64)
    if window.on_s < time[0] or window.end_s > time[-1]:
        raise CycleBoundaryError(
            f"cycle {window.index} runs from {window.on_s:g} s to {window.end_s:g} s, outside "
            f"the recording ({time[0]:g} s to {time[-1]:g} s)"
        )
    cold_t = np.asarray(cold, dtype=np.float64)
    hot_t = np.asarray(hot, dtype=np.float64)
    delta_t = hot_t - cold_t
    oriented = np.asarray(voltage_mv, dtype=np.float64) * polarity_sign

    heating = (time >= window.on_s) & (time < window.off_s)
    cooling = (time >= window.off_s) & (time < window.end_s)
    for name, branch in (("heating", heating), ("cooling", cooling)):
        held = int(np.count_nonzero(branch))
        if held < _MIN_BRANCH_SAMPLES:
            raise CycleBoundaryError(
                f"the {name} branch of cycle {window.index} holds {held} sample(s); "
                f"at least {_MIN_BRANCH_SAMPLES} are needed"
            )
    whole = heating | cooling

    fit = fit_seebeck_slope(delta_t[whole], oriented[whole])
    on_fit = fit_seebeck_slope(delta_t[heating], oriented[heating])
    off_fit = fit_seebeck_slope(delta_t[cooling], oriented[cooling])

    problems: list[str] = []
    if not _direction(delta_t[heating]) > 0:
        problems.append(
            f"the temperature difference does not rise while the drive is on in cycle "
            f"{window.index}, so either the boundaries or the hot and cold channels are wrong"
        )
    if not _direction(delta_t[cooling]) < 0:
        problems.append(
            f"the temperature difference does not fall while the drive is off in cycle "
            f"{window.index}, so either the boundaries or the hot and cold channels are wrong"
        )
    if not math.isfinite(fit.slope):
        problems.append(
            f"the temperature difference does not vary in cycle {window.index}, so no "
            "slope can be fitted"
        )

    return CycleFit(
        window=window,
        n_samples=int(np.count_nonzero(whole)),
        slope_mv_k=fit.slope,
        intercept_mv=fit.intercept,
        slope_stderr_mv_k=fit.slope_stderr,
        r_squared=fit.r_squared,
        on_slope_mv_k=on_fit.slope,
        off_slope_mv_k=off_fit.slope,
        delta_t_min_k=float(np.min(delta_t[whole])),
        delta_t_max_k=float(np.max(delta_t[whole])),
        mean_temperature=float(np.mean((cold_t[whole] + hot_t[whole]) / 2.0)),
        problems=tuple(problems),
    )


def reduce_cycles(values: Sequence[float]) -> CycleReduction:
    """Reduce the cycle responses of one specimen to a signed mean and its SEM.

    Args:
        values: The signed response of each accepted cycle.

    Returns:
        The reduction. `sd` and `sem` are None when there is one value: a
        single cycle says nothing about its own repeatability.

    Raises:
        ValueError: If there are no values, or one is not finite.
    """
    cycles = np.asarray(list(values), dtype=np.float64)
    if cycles.size == 0:
        raise ValueError("there are no cycles to reduce")
    if not np.all(np.isfinite(cycles)):
        raise ValueError("a cycle response is not finite")
    k = int(cycles.size)
    mean = float(np.mean(cycles))
    if k == 1:
        return CycleReduction(k=1, mean=mean, sd=None, sem=None, dof=0)
    sd = float(np.std(cycles, ddof=1))
    return CycleReduction(k=k, mean=mean, sd=sd, sem=sd / math.sqrt(k), dof=k - 1)
