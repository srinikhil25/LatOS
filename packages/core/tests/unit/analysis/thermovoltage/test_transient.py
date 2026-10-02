"""Tests for the V(t) approach fit — the time constant and the steady-state verdict.

Most of what follows builds a trace with a *known* time constant and checks that
it comes back, because the analyzer's whole purpose is to say whether a reading
was taken at rest. The cases that matter most are the two refusals: a window too
short to have settled, and a trace that peaks and falls back, which is what a
cell being discharged by its own voltmeter looks like.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from latos.analysis.base_analyzer import AnalyzerInputs
from latos.analysis.registry import default_registry
from latos.analysis.thermovoltage.transient import (
    ThermovoltageTransientAnalyzer,
    fit_exponential_approach,
)
from latos.core.enums import Severity, Technique

TAU = 400.0
V_INF = 12.5


def _measure_stub(*, with_file: bool = True):
    """Same lightweight stub the slope and Hall tests use."""

    class _M:
        pass

    m = _M()
    m.files = (object(),) if with_file else ()
    return m


def _rise(tau: float = TAU, v_inf: float = V_INF, *, span: float = 3000.0, step: float = 1.0):
    t = np.arange(0.0, span, step)
    return t, v_inf * (1.0 - np.exp(-t / tau))


def _run(arrays, params=None):
    analyzer = ThermovoltageTransientAnalyzer()
    return analyzer.analyze(
        AnalyzerInputs(measurement=_measure_stub(), arrays=arrays, params=params or {}),
    )


def _fields(output, field):
    return [i for i in output.issues if i.field == field]


# ─── The fit kernel ─────────────────────────────────────────────────
class TestFitKernel:
    def test_recovers_tau_and_limit_from_a_clean_trace(self):
        t, v = _rise()
        fit = fit_exponential_approach(t, v)
        assert fit.tau_s == pytest.approx(TAU, rel=1e-4)
        assert fit.v_infinity_mv == pytest.approx(V_INF, rel=1e-4)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-6)
        assert fit.converged

    def test_amplitude_is_negative_while_rising(self):
        # A rising trace approaches its limit from below, so A < 0. The sign is
        # the only thing distinguishing a build-up from a decay.
        t, v = _rise()
        assert fit_exponential_approach(t, v).amplitude_mv < 0

    def test_amplitude_is_positive_while_falling(self):
        t = np.arange(0.0, 3000.0, 1.0)
        v = V_INF * np.exp(-t / TAU)
        assert fit_exponential_approach(t, v).amplitude_mv > 0

    def test_survives_noise(self):
        t, v = _rise()
        noisy = v + np.random.default_rng(0).normal(0.0, 0.05, t.size)
        fit = fit_exponential_approach(t, noisy)
        assert fit.tau_s == pytest.approx(TAU, rel=0.02)
        assert math.isfinite(fit.tau_stderr_s)

    def test_zero_span_does_not_raise(self):
        fit = fit_exponential_approach(np.zeros(5), np.ones(5))
        assert not fit.converged
        assert math.isnan(fit.tau_s)

    def test_span_is_the_window_length(self):
        t, v = _rise(span=1000.0)
        assert fit_exponential_approach(t, v).span_s == pytest.approx(999.0)


# ─── The steady-state verdict ───────────────────────────────────────
class TestSteadyState:
    def test_long_window_passes(self):
        t, v = _rise()
        out = _run({"time_s": t, "voltage_mv": v})
        assert out.outputs["steady_state_reached"] is True
        assert not _fields(out, "steady_state")

    def test_window_cut_at_one_tau_is_flagged(self):
        t, v = _rise(span=TAU + 1.0)
        out = _run({"time_s": t, "voltage_mv": v})
        assert out.outputs["steady_state_reached"] is False
        flagged = _fields(out, "steady_state")
        assert len(flagged) == 1
        assert flagged[0].severity is Severity.WARNING

    def test_fraction_reached_matches_the_exponential(self):
        # One tau reaches 1 - 1/e. Reporting this is the point: it turns "not
        # settled" into "you measured 63% of the real value".
        t, v = _rise(span=TAU + 1.0)
        out = _run({"time_s": t, "voltage_mv": v})
        assert out.outputs["fraction_of_limit_reached"] == pytest.approx(0.632, abs=0.01)

    def test_multiple_is_configurable(self):
        t, v = _rise(span=2 * TAU)
        strict = _run({"time_s": t, "voltage_mv": v}, {"steady_state_tau_multiple": 5.0})
        lenient = _run({"time_s": t, "voltage_mv": v}, {"steady_state_tau_multiple": 1.0})
        assert strict.outputs["steady_state_reached"] is False
        assert lenient.outputs["steady_state_reached"] is True


# ─── The loading signature ──────────────────────────────────────────
class TestTurnover:
    def test_a_plateau_does_not_count_as_a_turnover(self):
        t, v = _rise()
        out = _run({"time_s": t, "voltage_mv": v})
        assert out.outputs["turned_over"] is False
        assert not _fields(out, "decay_from_peak_fraction")

    def test_rise_then_fall_back_is_reported(self):
        # Charging against a slow drain: what a cell measured through too low an
        # input resistance does. It must not be reported as a settled value.
        t = np.arange(0.0, 3000.0, 1.0)
        v = V_INF * (1.0 - np.exp(-t / TAU)) * np.exp(-t / 2500.0)
        out = _run({"time_s": t, "voltage_mv": v})
        assert out.outputs["turned_over"] is True
        assert out.outputs["decay_from_peak_fraction"] > 0.1
        assert out.outputs["peak_time_s"] < t[-1]
        flagged = _fields(out, "decay_from_peak_fraction")
        assert len(flagged) == 1
        assert "input resistance" in flagged[0].message

    def test_turnover_is_detected_on_a_negative_going_trace(self):
        # An n-type cell runs negative, so its "peak" is the most negative
        # point. Taking the extremum on the raw value would miss this entirely.
        t = np.arange(0.0, 3000.0, 1.0)
        v = -V_INF * (1.0 - np.exp(-t / TAU)) * np.exp(-t / 2500.0)
        out = _run({"time_s": t, "voltage_mv": v})
        assert out.outputs["turned_over"] is True
        assert out.outputs["peak_mv"] < 0

    def test_poor_single_relaxation_fit_is_flagged(self):
        t = np.arange(0.0, 3000.0, 1.0)
        v = V_INF * (1.0 - np.exp(-t / TAU)) * np.exp(-t / 2500.0)
        out = _run({"time_s": t, "voltage_mv": v})
        # One exponential cannot describe two, and the analyzer says so rather
        # than silently fitting five parameters.
        assert _fields(out, "r_squared")


# ─── Window selection ───────────────────────────────────────────────
class TestWindow:
    def test_start_is_detected_from_the_delta_t_step(self):
        applied = 300.0
        t = np.arange(0.0, 2400.0, 1.0)
        v = np.where(t < applied, 0.0, V_INF * (1.0 - np.exp(-(t - applied) / TAU)))
        delta = np.where(t < applied, 0.0, 5.0)
        out = _run(
            {
                "time_s": t,
                "voltage_mv": v,
                "t_hot_c": 25.0 + delta,
                "t_cold_c": np.full_like(t, 25.0),
            },
        )
        assert out.outputs["window_start_source"] == "delta_t_step"
        assert out.outputs["window_start_s"] == pytest.approx(applied)
        # With the flat pre-trigger baseline excluded, tau comes back right.
        assert out.outputs["tau_s"] == pytest.approx(TAU, rel=0.01)

    def test_falls_back_to_the_trace_start_without_temperatures(self):
        t, v = _rise()
        out = _run({"time_s": t, "voltage_mv": v})
        assert out.outputs["window_start_source"] == "trace_start"
        assert out.outputs["window_start_s"] == pytest.approx(0.0)

    def test_a_constant_delta_t_is_not_reported_as_a_detected_step(self):
        # The temperatures are present but never move: logging started after the
        # heat was already on. Naming a step here would claim a cause the data
        # does not show.
        t, v = _rise()
        out = _run(
            {
                "time_s": t,
                "voltage_mv": v,
                "t_hot_c": np.full_like(t, 35.0),
                "t_cold_c": np.full_like(t, 25.0),
            },
        )
        assert out.outputs["window_start_source"] == "trace_start"
        assert out.outputs["window_start_s"] == pytest.approx(0.0)

    def test_thermocouple_noise_is_not_mistaken_for_a_step(self):
        t, v = _rise()
        jitter = np.random.default_rng(1).normal(0.0, 0.1, t.size)
        out = _run(
            {
                "time_s": t,
                "voltage_mv": v,
                "t_hot_c": 35.0 + jitter,
                "t_cold_c": np.full_like(t, 25.0),
            },
        )
        assert out.outputs["window_start_source"] == "trace_start"

    def test_explicit_bounds_win(self):
        t, v = _rise()
        out = _run({"time_s": t, "voltage_mv": v}, {"t_start_s": 500.0, "t_end_s": 2000.0})
        assert out.outputs["window_start_source"] == "explicit"
        assert out.outputs["window_start_s"] == pytest.approx(500.0)
        # The bound is inclusive and the trace has a sample exactly on it.
        assert out.outputs["window_end_s"] == pytest.approx(2000.0)

    def test_inverted_window_is_refused(self):
        t, v = _rise()
        out = _run({"time_s": t, "voltage_mv": v}, {"t_start_s": 2000.0, "t_end_s": 500.0})
        assert out.outputs == {}
        assert any(i.severity is Severity.ERROR for i in out.issues)


# ─── Refusals ───────────────────────────────────────────────────────
class TestRefusals:
    def test_missing_voltage_is_an_error(self):
        out = _run({"time_s": np.arange(20.0)})
        assert out.outputs == {}
        assert any("voltage_mv" in i.message for i in out.issues)

    def test_length_mismatch_is_an_error(self):
        out = _run({"time_s": np.arange(20.0), "voltage_mv": np.arange(10.0)})
        assert out.outputs == {}
        assert any("differ in length" in i.message for i in out.issues)

    def test_too_few_points_is_an_error(self):
        t, v = _rise(span=5.0)
        out = _run({"time_s": t, "voltage_mv": v})
        assert out.outputs == {}
        assert any(i.severity is Severity.ERROR for i in out.issues)

    def test_a_flat_trace_reports_no_time_constant(self):
        # The optimiser will happily "converge" on a constant trace with zero
        # amplitude and an arbitrary tau. Reporting that tau would be a fiction.
        t = np.arange(0.0, 1000.0, 1.0)
        out = _run({"time_s": t, "voltage_mv": np.zeros_like(t)})
        assert out.outputs == {}
        assert any("does not vary" in i.message for i in out.issues)

    def test_empty_arrays_are_an_error(self):
        out = _run({"time_s": np.asarray([]), "voltage_mv": np.asarray([])})
        assert out.outputs == {}
        assert any(i.severity is Severity.ERROR for i in out.issues)


# ─── Contract ───────────────────────────────────────────────────────
class TestContract:
    def test_metadata(self):
        assert ThermovoltageTransientAnalyzer.name == "thermovoltage-transient"
        assert ThermovoltageTransientAnalyzer.version == "1.0.0"
        assert ThermovoltageTransientAnalyzer.accepts_techniques == (Technique.THERMOELECTRIC,)

    def test_accepts_needs_a_file(self):
        analyzer = ThermovoltageTransientAnalyzer()
        assert analyzer.accepts(_measure_stub()) is True
        assert analyzer.accepts(_measure_stub(with_file=False)) is False

    def test_registered_in_the_default_registry(self):
        # The gap this closes is a declared output with no producer, so being
        # importable is not enough — it has to be dispatched to.
        assert "thermovoltage-transient" in default_registry()

    def test_derived_arrays_are_co_indexed(self):
        t, v = _rise()
        out = _run({"time_s": t, "voltage_mv": v})
        sizes = {name: arr.size for name, arr in out.derived_arrays.items()}
        assert set(sizes) == {"fit_time_s", "fit_voltage_mv", "residual_mv"}
        assert len(set(sizes.values())) == 1

    def test_residuals_are_measured_minus_fitted(self):
        t, v = _rise()
        out = _run({"time_s": t, "voltage_mv": v})
        rebuilt = out.derived_arrays["fit_voltage_mv"] + out.derived_arrays["residual_mv"]
        assert rebuilt == pytest.approx(v, abs=1e-9)

    def test_it_takes_the_trace_the_slope_analyzer_declines(self):
        """The registry docstring claims the two are complementary; check it.

        Three analyzers now accept THERMOELECTRIC. A trace held at one
        temperature difference has no slope to fit, and this is the analyzer
        that has something to say about it. If both ever claimed the same data
        the registry note above them would be wrong.
        """
        from latos.analysis.thermovoltage.slope import ThermovoltageSlopeAnalyzer

        t, v = _rise()
        held = {
            "time_s": t,
            "voltage_mv": v,
            "t_hot_c": np.full_like(t, 35.0),
            "t_cold_c": np.full_like(t, 25.0),
        }
        inputs = AnalyzerInputs(measurement=_measure_stub(), arrays=held)

        slope = ThermovoltageSlopeAnalyzer().analyze(inputs)
        assert slope.outputs == {}
        assert any(i.severity is Severity.ERROR for i in slope.issues)

        transient = ThermovoltageTransientAnalyzer().analyze(inputs)
        assert transient.outputs["tau_s"] == pytest.approx(TAU, rel=0.01)
