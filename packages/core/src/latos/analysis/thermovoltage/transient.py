"""The time constant of an ionic cell, and whether its reading was taken at rest.

What this answers
-----------------
An ionic thermoelectric cell does not respond quickly. Voltage build-up is
reported to run from several hundred to several thousand seconds, limited by the
evolution of the temperature field rather than by any RC constant of the cell.
A reading taken before the cell has settled is a point on a rising edge, and
dividing it by ΔT gives a number smaller than the real coefficient — with no
indication in the record that anything was wrong.

Fitting the approach gives the two things a single reading cannot:

* **τ**, which is diagnostic of ion mobility and should vary smoothly with
  composition. A τ that jumps between neighbouring mixtures says the sample
  changed, not the physics. This is the `tau_fitted_s` column of
  `ite_workbook_template.py`, which until now had no producer.
* **whether the measurement window was long enough.** The test is the span of
  the window against τ: a run that stopped at one τ has reached 63 % of its
  final value, and at two τ, 86 %. The flag fires below three.

The model
---------
A single relaxation toward a limit::

    V(t) = V_inf + A · exp(-(t - t_start) / τ)

`A` carries the direction. For an ordinary build-up the measured voltage climbs
toward `V_inf` from below, so `A` is negative. A positive `A` means the trace is
falling toward its limit, which on a cell that was supposed to be charging is a
finding rather than a detail — see below.

One exponential, deliberately. Ionic cells are also described by a
two-relaxation form, and when one relaxation does not fit, the honest response
is to report the poor fit and say a second may be needed — not to quietly fit
five parameters to a noisy trace and present the result as if it were resolved.
So the fit stays at three parameters and `r_squared` is reported beside τ.

The loading signature
---------------------
A cell measured through too low an input resistance is discharged by its own
voltmeter. The trace then rises, turns over, and settles lower: thermodiffusion
driving ions in, and the meter draining them out, reach a balance below the
open-circuit voltage. A cell measured at true open circuit rises and stays.

So a peak followed by a decline, while ΔT is held constant, is visible evidence
of meter loading, and `decay_from_peak_fraction` reports it. This matters here
because the HIOKI 8430-20 in use has an input resistance of 1 MΩ against the
≥10 GΩ an i-TE cell wants, and the resulting error scales with cell impedance —
which tracks composition, the very axis being scanned. An error that varies
smoothly with the knob looks exactly like a result.

On the thresholds
-----------------
`_STEADY_STATE_TAU_MULTIPLE` comes from the measurement plan. The other two are
judgement calls with no measured basis yet, and are named and commented as such
rather than buried in a comparison. They should be revisited against real traces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import curve_fit

from latos.analysis.base_analyzer import AnalyzerInputs, AnalyzerOutput, BaseAnalyzer
from latos.core.enums import Severity, Technique
from latos.core.models import Measurement, ValidationIssue, utc_now

__all__ = ["ApproachFit", "ThermovoltageTransientAnalyzer", "fit_exponential_approach"]

# Three parameters, so a fit needs more than three points to say anything about
# residuals. Ten is the smallest window worth reporting a time constant from.
_MIN_POINTS_FOR_FIT = 10

# A window of three τ reaches 95 % of the final value. The measurement plan
# flags anything shorter as a reading taken on the rising edge.
_STEADY_STATE_TAU_MULTIPLE = 3.0

# Below this, one relaxation does not describe the trace and a second may be
# needed. JUDGEMENT CALL, not a measured threshold — no real i-TE trace has
# been fitted yet. Revisit once the campaign has traces on disk.
_POOR_FIT_R_SQUARED = 0.90

# A turnover is reported when the trace falls this far below its own extremum.
# JUDGEMENT CALL, as above: large enough to ignore sampling noise on a clean
# plateau, small enough to catch the loading signature early.
_DECAY_FLAG_FRACTION = 0.05

# A peak inside the final slice of the window is just the end of the trace, not
# a turnover, so the decay test ignores it.
_PEAK_TAIL_FRACTION = 0.05

# Fraction of the window averaged to seed `V_inf` before fitting.
_TAIL_FRACTION_FOR_GUESS = 0.10

# A difference needs two samples before it exists at all.
_MIN_POINTS_FOR_GRADIENT = 2

# How far the mean temperature difference must move across the candidate step
# before it counts as one. The instrument resolves thermocouples to 0.1 C and
# the smallest difference the measurement plan calls for is 2 K, so half a
# kelvin sits clear of the noise floor and well under the smallest real step.
_MIN_DETECTABLE_STEP_K = 0.5


@dataclass(frozen=True, slots=True)
class ApproachFit:
    """Parameters of a single-relaxation approach, with its residuals.

    Attributes:
        v_infinity_mv: The limit the trace is heading toward, in millivolts.
        amplitude_mv: `A`. Negative while rising toward the limit, positive
            while falling toward it.
        tau_s: Time constant, in seconds. NaN when the fit did not converge.
        tau_stderr_s: Standard error on τ, NaN when unavailable.
        r_squared: Coefficient of determination over the fitted window.
        n: Points in the window.
        t_start_s: First time in the window.
        t_end_s: Last time in the window.
        time_s: Window times, co-indexed with `fitted_mv` and `residual_mv`.
        fitted_mv: Model evaluated on `time_s`.
        residual_mv: Measured minus fitted.
    """

    v_infinity_mv: float
    amplitude_mv: float
    tau_s: float
    tau_stderr_s: float
    r_squared: float
    n: int
    t_start_s: float
    t_end_s: float
    time_s: np.ndarray
    fitted_mv: np.ndarray
    residual_mv: np.ndarray

    @property
    def span_s(self) -> float:
        """Length of the fitted window, in seconds."""
        return self.t_end_s - self.t_start_s

    @property
    def converged(self) -> bool:
        """True when the fit produced a finite, positive time constant."""
        return math.isfinite(self.tau_s) and self.tau_s > 0.0


def _model(t: np.ndarray, v_inf: float, amplitude: float, tau: float) -> np.ndarray:
    """Single relaxation toward `v_inf`, with `t` already offset to the window."""
    return v_inf + amplitude * np.exp(-t / tau)


def fit_exponential_approach(time_s: np.ndarray, voltage_mv: np.ndarray) -> ApproachFit:
    """Fit `V = V_inf + A·exp(-(t - t0)/τ)` by least squares.

    Seeded from the data rather than from constants: `V_inf` from the mean of
    the window's tail, `A` from how far the first point sits below it, and τ from
    a third of the window. A trace that has not turned over yet gives a τ larger
    than the window, which is a legitimate answer and the reason τ is not bounded
    above — the caller compares span against τ and flags it.

    Args:
        time_s: Window times, in seconds, ascending.
        voltage_mv: Measured voltage, in millivolts, same length.

    Returns:
        An `ApproachFit`. Check `converged` before trusting `tau_s`.
    """
    t = np.asarray(time_s, dtype=np.float64)
    v = np.asarray(voltage_mv, dtype=np.float64)
    t0 = float(t[0])
    offset = t - t0
    span = float(offset[-1])

    tail = max(1, round(t.size * _TAIL_FRACTION_FOR_GUESS))
    v_inf_guess = float(np.mean(v[-tail:]))
    amplitude_guess = float(v[0] - v_inf_guess)
    tau_guess = span / 3.0 if span > 0 else 1.0

    failed = ApproachFit(
        v_infinity_mv=math.nan,
        amplitude_mv=math.nan,
        tau_s=math.nan,
        tau_stderr_s=math.nan,
        r_squared=math.nan,
        n=t.size,
        t_start_s=t0,
        t_end_s=float(t[-1]),
        time_s=t,
        fitted_mv=np.full_like(v, math.nan),
        residual_mv=np.full_like(v, math.nan),
    )
    if span <= 0:
        return failed

    try:
        popt, pcov = curve_fit(
            _model,
            offset,
            v,
            p0=(v_inf_guess, amplitude_guess, tau_guess),
            # τ must stay positive; nothing else is constrained. The upper
            # bound is deliberately absent so an unsettled trace can report a
            # τ longer than the record, rather than being clipped into looking
            # as though it had settled.
            bounds=((-np.inf, -np.inf, 1e-9), (np.inf, np.inf, np.inf)),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return failed

    fitted = _model(offset, *popt)
    residual = v - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((v - float(np.mean(v))) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan

    tau_stderr = math.nan
    if np.all(np.isfinite(pcov)) and pcov[2, 2] >= 0:
        tau_stderr = float(math.sqrt(pcov[2, 2]))

    return ApproachFit(
        v_infinity_mv=float(popt[0]),
        amplitude_mv=float(popt[1]),
        tau_s=float(popt[2]),
        tau_stderr_s=tau_stderr,
        r_squared=r_squared,
        n=t.size,
        t_start_s=t0,
        t_end_s=float(t[-1]),
        time_s=t,
        fitted_mv=fitted,
        residual_mv=residual,
    )


class ThermovoltageTransientAnalyzer(BaseAnalyzer):
    """Time constant and steady-state verdict from a V(t) trace."""

    name: ClassVar[str] = "thermovoltage-transient"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.THERMOELECTRIC,)
    default_params: ClassVar[dict[str, Any]] = {
        # Window bounds in seconds. None means "work it out from the data":
        # the start from when ΔT was applied if temperatures are present, the
        # end from the last sample.
        "t_start_s": None,
        "t_end_s": None,
        "steady_state_tau_multiple": _STEADY_STATE_TAU_MULTIPLE,
    }

    def accepts(self, measurement: Measurement) -> bool:
        """Accept any thermoelectric measurement with a source file.

        Whether the arrays are actually present is decided in `analyze`, where a
        missing column becomes an issue a reviewer reads, rather than here,
        where it would drop the measurement from the run in silence.
        """
        return len(measurement.files) > 0

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        """Fit the approach over the chosen window and judge what it shows."""
        params = self.merge_params(inputs.params)
        series = _extract_series(inputs.arrays)
        if isinstance(series, str):
            return _error(series)
        time_s, voltage_mv = series

        window = _window(time_s, inputs.arrays, params)
        if isinstance(window, str):
            return _error(window)
        mask, start_source = window

        if int(np.count_nonzero(mask)) < _MIN_POINTS_FOR_FIT:
            return _error(
                f"Window holds {int(np.count_nonzero(mask))} finite points; need at "
                f"least {_MIN_POINTS_FOR_FIT} to fit three parameters and still have "
                "residuals to judge.",
            )

        fit = fit_exponential_approach(time_s[mask], voltage_mv[mask])
        if not fit.converged:
            return _error(
                "The approach fit did not converge on a positive time constant. The "
                "trace may be monotonic without curvature, or dominated by noise; "
                "inspect it before quoting a steady-state value.",
            )
        if not math.isfinite(fit.r_squared):
            # R-squared is undefined only when the window has no variance at all.
            # The optimiser will still "converge" on such a trace, with zero
            # amplitude and an arbitrary time constant, so it has to be caught
            # here rather than trusted.
            return _error(
                "The voltage does not vary across the window, so it carries no time "
                "constant. Check that the channel was connected and that the "
                "temperature difference was actually applied.",
            )

        turnover = _turnover(fit)
        multiple = float(params["steady_state_tau_multiple"])
        issues = list(_judge(fit, turnover, multiple))
        return AnalyzerOutput(
            outputs=_payload(fit, turnover, multiple, start_source),
            derived_arrays={
                "fit_time_s": fit.time_s,
                "fit_voltage_mv": fit.fitted_mv,
                "residual_mv": fit.residual_mv,
            },
            issues=tuple(issues),
        )


# ─── Window selection ───────────────────────────────────────────────
def _extract_series(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | str:
    """Get (time, voltage) from the arrays, or say what was missing."""
    if "time_s" not in arrays or "voltage_mv" not in arrays:
        return "Missing arrays — expected both time_s and voltage_mv."
    time_s = np.asarray(arrays["time_s"], dtype=np.float64).ravel()
    voltage_mv = np.asarray(arrays["voltage_mv"], dtype=np.float64).ravel()
    if time_s.size != voltage_mv.size:
        return (
            f"time_s and voltage_mv differ in length ({time_s.size} vs "
            f"{voltage_mv.size}); they must be sampled together."
        )
    if time_s.size == 0:
        return "No samples supplied."
    return time_s, voltage_mv


def _window(
    time_s: np.ndarray,
    arrays: dict[str, np.ndarray],
    params: dict[str, Any],
) -> tuple[np.ndarray, str] | str:
    """Build the boolean window mask, and say where its start came from."""
    voltage = np.asarray(arrays["voltage_mv"], dtype=np.float64).ravel()
    finite = np.isfinite(time_s) & np.isfinite(voltage)

    explicit_start = params.get("t_start_s")
    if explicit_start is not None:
        start, source = float(explicit_start), "explicit"
    else:
        detected = _delta_t_step_time(time_s, arrays)
        start = detected if detected is not None else float(np.min(time_s[finite]))
        source = "delta_t_step" if detected is not None else "trace_start"

    end_param = params.get("t_end_s")
    end = float(end_param) if end_param is not None else float(np.max(time_s[finite]))
    if end <= start:
        return f"Window end ({end:.4g} s) is not after its start ({start:.4g} s)."
    return (finite & (time_s >= start) & (time_s <= end), source)


def _delta_t_step_time(
    time_s: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> float | None:
    """When ΔT was applied, taken as the steepest rise in ΔT.

    Starting the fit where the heat was applied rather than where logging began
    keeps the pre-trigger baseline out of the exponential, which would otherwise
    pull τ long and `V_inf` low.

    Returns None when there is no temperature pair, or when nothing in it looks
    like an applied step, and the caller then falls back to the start of the
    trace. Reporting a detected step on a trace whose ΔT never moved would name
    a cause the data does not show.

    On a record covering several plateaus this finds the *largest* rise, not the
    first. Such a record should be cut on its event marks and each plateau fitted
    separately; `window_start_s` is reported so the choice is visible either way.
    """
    if "t_hot_c" not in arrays or "t_cold_c" not in arrays:
        return None
    hot = np.asarray(arrays["t_hot_c"], dtype=np.float64).ravel()
    cold = np.asarray(arrays["t_cold_c"], dtype=np.float64).ravel()
    if hot.size != time_s.size or cold.size != time_s.size or hot.size < _MIN_POINTS_FOR_GRADIENT:
        return None
    delta_t = hot - cold
    gradient = np.diff(delta_t)
    if not np.any(np.isfinite(gradient)):
        return None
    step = int(np.nanargmax(gradient))

    # The largest single jump is not evidence on its own: over a few thousand
    # samples, thermocouple noise alone produces an apparent jump of several
    # tenths of a kelvin, so any fixed per-sample threshold is eventually
    # cleared by chance. What separates an applied step from noise is that it
    # moves the *level* — ΔT is one value before and another after — whereas
    # noise leaves the two halves with the same mean.
    before = float(np.nanmean(delta_t[: step + 1]))
    after = float(np.nanmean(delta_t[step + 1 :]))
    if not math.isfinite(after - before) or after - before < _MIN_DETECTABLE_STEP_K:
        return None
    # The step index sits on the sample *before* the rise, so the window opens
    # at the following sample — the first one actually under the new ΔT.
    return float(time_s[min(step + 1, time_s.size - 1)])


# ─── Judgement and reporting ────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _Turnover:
    """Whether the trace peaked and fell back inside the window."""

    peak_mv: float
    final_mv: float
    peak_time_s: float
    decay_fraction: float
    turned_over: bool


def _turnover(fit: ApproachFit) -> _Turnover:
    """Measure any fall-back from the window's extremum.

    The extremum is taken on the magnitude, because an n-type cell runs negative
    and its "peak" is the most negative point. A peak inside the final slice of
    the window is the end of the trace rather than a turnover, so it does not
    count.
    """
    measured = fit.fitted_mv + fit.residual_mv
    magnitude = np.abs(measured)
    peak_index = int(np.nanargmax(magnitude))
    peak = float(measured[peak_index])
    final = float(measured[-1])

    peak_magnitude = float(magnitude[peak_index])
    decay = (peak_magnitude - abs(final)) / peak_magnitude if peak_magnitude > 0 else 0.0
    in_tail = peak_index >= int(measured.size * (1.0 - _PEAK_TAIL_FRACTION))
    return _Turnover(
        peak_mv=peak,
        final_mv=final,
        peak_time_s=float(fit.time_s[peak_index]),
        decay_fraction=float(decay),
        turned_over=bool(decay > _DECAY_FLAG_FRACTION and not in_tail),
    )


def _judge(fit: ApproachFit, turnover: _Turnover, multiple: float) -> list[ValidationIssue]:
    """Everything worth telling the reviewer about this fit."""
    issues: list[ValidationIssue] = []
    span_over_tau = fit.span_s / fit.tau_s

    if span_over_tau < multiple:
        reached = 1.0 - math.exp(-span_over_tau)
        issues.append(
            _issue(
                "steady_state",
                Severity.WARNING,
                f"The window spans {span_over_tau:.2f} tau "
                f"({fit.span_s:.0f} s against tau = {fit.tau_s:.0f} s), so the trace "
                f"reached about {reached:.0%} of its limit. A value read here is on "
                f"the rising edge, not the plateau. Hold the temperature difference "
                f"for at least {multiple:.0f} tau, which is "
                f"{multiple * fit.tau_s:.0f} s.",
            ),
        )

    if math.isfinite(fit.r_squared) and fit.r_squared < _POOR_FIT_R_SQUARED:
        issues.append(
            _issue(
                "r_squared",
                Severity.WARNING,
                f"One relaxation describes this trace poorly (R2 = {fit.r_squared:.3f}). "
                "A second relaxation may be needed; tau from this fit should not be "
                "quoted as the cell's time constant until that is checked.",
            ),
        )

    if turnover.turned_over:
        issues.append(
            _issue(
                "decay_from_peak_fraction",
                Severity.WARNING,
                f"The trace peaked at {turnover.peak_mv:.4g} mV "
                f"({turnover.peak_time_s:.0f} s) and fell to "
                f"{turnover.final_mv:.4g} mV, a drop of "
                f"{turnover.decay_fraction:.1%}. Under a held temperature difference "
                "a cell at open circuit does not fall back. Check the voltmeter's "
                "input resistance against the cell's, and whether the sample is "
                "drying out.",
            ),
        )
    return issues


def _payload(
    fit: ApproachFit,
    turnover: _Turnover,
    multiple: float,
    start_source: str,
) -> dict[str, Any]:
    """The scalar outputs, named as the workbook and the plan name them."""
    span_over_tau = fit.span_s / fit.tau_s
    return {
        "tau_s": fit.tau_s,
        "tau_stderr_s": fit.tau_stderr_s,
        "v_infinity_mv": fit.v_infinity_mv,
        "amplitude_mv": fit.amplitude_mv,
        "r_squared": fit.r_squared,
        "n_points_fitted": fit.n,
        "direction": "rising" if fit.amplitude_mv < 0 else "falling",
        "window_start_s": fit.t_start_s,
        "window_end_s": fit.t_end_s,
        "window_span_s": fit.span_s,
        "window_start_source": start_source,
        "span_over_tau": span_over_tau,
        "fraction_of_limit_reached": 1.0 - math.exp(-span_over_tau),
        "steady_state_reached": bool(span_over_tau >= multiple),
        "peak_mv": turnover.peak_mv,
        "final_mv": turnover.final_mv,
        "peak_time_s": turnover.peak_time_s,
        "decay_from_peak_fraction": turnover.decay_fraction,
        "turned_over": turnover.turned_over,
    }


def _error(message: str) -> AnalyzerOutput:
    """An output carrying nothing but the reason there is nothing."""
    return AnalyzerOutput(
        outputs={},
        derived_arrays={},
        issues=(_issue("data", Severity.ERROR, message),),
    )


def _issue(field: str, severity: Severity, message: str) -> ValidationIssue:
    """Build a `ValidationIssue` with the current timestamp."""
    return ValidationIssue(
        field=field,
        severity=severity,
        message=message,
        detected_at=utc_now(),
    )
