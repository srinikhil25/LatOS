"""Tests for the ΔV-versus-ΔT Seebeck fit.

The analyzer's whole reason to exist is that a single-point reading cannot tell
a Seebeck coefficient from an electrode offset. Most of what follows therefore
builds series with a *known* slope and a *known* intercept and checks that both
come back out, including the cases where the offset is large enough to have
ruined the single-point answer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from latos.analysis.base_analyzer import AnalyzerInputs
from latos.analysis.thermovoltage.slope import (
    ThermovoltageSlopeAnalyzer,
    fit_seebeck_slope,
)
from latos.core.enums import Severity, Technique


def _measure_stub(*, with_file: bool = True):
    """Same lightweight stub the Hall tests use.

    `analyze` reads only `inputs.arrays`, and `accepts` reads only `files`, so
    building a full `Measurement` here would test the model rather than the
    analyzer.
    """

    class _M:
        pass

    m = _M()
    m.files = (object(),) if with_file else ()
    return m


def _run(arrays: dict[str, np.ndarray]):
    analyzer = ThermovoltageSlopeAnalyzer()
    return analyzer.analyze(AnalyzerInputs(measurement=_measure_stub(), arrays=arrays))


def _series(slope: float, intercept: float, deltas: list[float], noise: float = 0.0):
    dt = np.asarray(deltas, dtype=float)
    dv = slope * dt + intercept
    if noise:
        dv = dv + np.random.default_rng(0).normal(0.0, noise, dt.size)
    return {"delta_t_k": dt, "delta_v_mv": dv}


class TestFitKernel:
    """The closed-form fit, checked against values it cannot get wrong."""

    def test_recovers_slope_and_intercept_exactly_without_noise(self):
        dt = np.array([2.0, 5.0, 10.0])
        fit = fit_seebeck_slope(dt, 3.5 * dt + 1.25)
        assert fit.slope == pytest.approx(3.5)
        assert fit.intercept == pytest.approx(1.25)
        assert fit.r_squared == pytest.approx(1.0)

    def test_two_points_give_a_slope_but_no_uncertainty(self):
        """An exact fit through two points is not a precise one.

        With two parameters and two observations there are no degrees of
        freedom left, so reporting a standard error of zero would claim
        certainty the data cannot support.
        """
        dt = np.array([2.0, 8.0])
        fit = fit_seebeck_slope(dt, 4.0 * dt + 2.0)
        assert fit.slope == pytest.approx(4.0)
        assert math.isnan(fit.slope_stderr)
        assert math.isnan(fit.intercept_stderr)

    def test_standard_error_scales_with_the_noise(self):
        """Doubling the scatter doubles the reported uncertainty. Exactly."""
        dt = np.linspace(2.0, 10.0, 7)
        scatter = np.random.default_rng(3).normal(0.0, 1.0, dt.size)
        quiet = fit_seebeck_slope(dt, 3.0 * dt + 0.05 * scatter)
        loud = fit_seebeck_slope(dt, 3.0 * dt + 0.10 * scatter)
        assert loud.slope_stderr == pytest.approx(2.0 * quiet.slope_stderr)

    def test_a_wider_delta_t_spread_tightens_the_slope(self):
        """Same scatter, wider lever arm, better-determined slope.

        This is the practical argument for not clustering every measurement at
        one convenient temperature difference.
        """
        scatter = np.random.default_rng(11).normal(0.0, 0.05, 7)
        narrow = np.linspace(5.0, 7.0, 7)
        wide = np.linspace(2.0, 10.0, 7)
        fit_narrow = fit_seebeck_slope(narrow, 3.0 * narrow + scatter)
        fit_wide = fit_seebeck_slope(wide, 3.0 * wide + scatter)
        assert fit_wide.slope_stderr < fit_narrow.slope_stderr

    def test_a_three_point_standard_error_understates_the_true_uncertainty(self):
        """A caveat worth having in the test suite, not just in a comment.

        With three points the residual variance carries one degree of freedom,
        and a chi-squared with one degree of freedom is heavily right-skewed:
        its median is 0.455 of its mean. So the reported standard error lands
        *below* the truth more often than not.

        This matters downstream. The per-point variance the optimizer will
        consume comes from exactly this number, so a three-ΔT campaign hands the
        surrogate an over-confident noise estimate — on precisely the sparse
        campaigns where over-confidence is most costly. Four or more ΔT values
        is the cheap fix.
        """
        dt = np.array([2.0, 6.0, 10.0])
        s_xx = float(np.sum((dt - dt.mean()) ** 2))
        true_stderr = 0.05 / math.sqrt(s_xx)

        observed = [
            fit_seebeck_slope(
                dt, 3.0 * dt + np.random.default_rng(seed).normal(0.0, 0.05, dt.size)
            ).slope_stderr
            for seed in range(60)
        ]
        assert float(np.median(observed)) < true_stderr

    def test_identical_delta_t_leaves_the_slope_undefined(self):
        fit = fit_seebeck_slope(np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0]))
        assert math.isnan(fit.slope)


class TestAnalyzer:
    """End to end, through the analyzer contract."""

    def test_clean_series_reports_the_coefficient_and_no_warnings(self):
        out = _run(_series(slope=2.4, intercept=0.0, deltas=[2.0, 5.0, 10.0]))
        assert out.outputs["seebeck_mv_k"] == pytest.approx(2.4)
        assert out.outputs["offset_mv"] == pytest.approx(0.0, abs=1e-9)
        assert out.outputs["n_points"] == 3
        assert out.issues == ()

    def test_electrode_offset_is_recovered_and_flagged(self):
        """The case the analyzer exists for.

        A 3 mV offset on a 2 mV/K coefficient means a single reading at
        ΔT = 5 K would report 2.6 mV/K, thirty percent high. The fit should
        recover both numbers and say so.
        """
        out = _run(_series(slope=2.0, intercept=3.0, deltas=[2.0, 5.0, 10.0]))
        assert out.outputs["seebeck_mv_k"] == pytest.approx(2.0)
        assert out.outputs["offset_mv"] == pytest.approx(3.0)
        # 3.0 mV against 2.0 mV/K * 10 K = 20 mV of signal.
        assert out.outputs["offset_fraction"] == pytest.approx(0.15)
        assert any("Electrode offset" in i.message for i in out.issues)

    def test_a_small_offset_does_not_raise_a_warning(self):
        out = _run(_series(slope=2.0, intercept=0.2, deltas=[2.0, 5.0, 10.0]))
        assert out.outputs["offset_fraction"] == pytest.approx(0.01)
        assert not any("Electrode offset" in i.message for i in out.issues)

    def test_two_points_warn_that_the_offset_cannot_be_separated(self):
        out = _run(_series(slope=2.0, intercept=1.0, deltas=[2.0, 10.0]))
        assert out.outputs["seebeck_mv_k"] == pytest.approx(2.0)
        assert out.outputs["seebeck_stderr_mv_k"] is None
        assert any("three or more" in i.message for i in out.issues)

    def test_curvature_is_caught_by_the_residual_pattern_not_by_r_squared(self):
        """The case that R² alone misses.

        A deliberately quadratic five-point series scores R² = 0.963, which
        passes for a good fit almost anywhere. The residuals tell the truth:
        positive at both ends, negative through the middle.
        """
        dt = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        out = _run({"delta_t_k": dt, "delta_v_mv": 0.05 * dt**2})
        assert out.outputs["r_squared"] == pytest.approx(0.963, abs=0.01)
        assert any("bow" in i.message for i in out.issues)

    def test_clean_linear_data_does_not_trigger_the_curvature_check(self):
        out = _run(_series(slope=2.0, intercept=0.5, deltas=[2.0, 4.0, 6.0, 8.0, 10.0]))
        assert not any("bow" in i.message for i in out.issues)

    def test_random_scatter_does_not_trigger_the_curvature_check(self):
        """Noise must not read as curvature, or the warning becomes worthless."""
        dt = np.linspace(2.0, 10.0, 9)
        false_alarms = 0
        for seed in range(30):
            rng = np.random.default_rng(seed)
            dv = 2.0 * dt + 0.5 + rng.normal(0.0, 0.3, dt.size)
            out = _run({"delta_t_k": dt, "delta_v_mv": dv})
            false_alarms += any("bow" in i.message for i in out.issues)
        assert false_alarms <= 3

    def test_ramp_arrays_are_reduced_to_pairs(self):
        """A continuous ramp stores hot and cold traces, not differences."""
        cold = np.full(6, 25.0)
        hot = np.array([27.0, 29.0, 31.0, 33.0, 35.0, 37.0])
        out = _run(
            {
                "t_hot_c": hot,
                "t_cold_c": cold,
                "voltage_mv": 1.5 * (hot - cold) + 0.4,
            }
        )
        assert out.outputs["seebeck_mv_k"] == pytest.approx(1.5)
        assert out.outputs["offset_mv"] == pytest.approx(0.4)
        assert out.outputs["n_points"] == 6

    def test_negative_seebeck_is_preserved(self):
        """Sign carries the carrier type and must survive the fit."""
        out = _run(_series(slope=-4.2, intercept=0.0, deltas=[2.0, 5.0, 10.0]))
        assert out.outputs["seebeck_mv_k"] == pytest.approx(-4.2)

    def test_non_finite_points_are_dropped_and_reported(self):
        out = _run(
            {
                "delta_t_k": np.array([2.0, 5.0, np.nan, 10.0]),
                "delta_v_mv": np.array([4.0, 10.0, 1.0, 20.0]),
            }
        )
        assert out.outputs["n_points"] == 3
        assert any("dropped" in i.message for i in out.issues)

    def test_derived_arrays_line_up_with_the_input(self):
        out = _run(_series(slope=2.0, intercept=1.0, deltas=[2.0, 5.0, 10.0]))
        assert set(out.derived_arrays) == {"fit_delta_t_k", "fit_delta_v_mv", "residual_mv"}
        assert all(v.size == 3 for v in out.derived_arrays.values())
        assert np.allclose(out.derived_arrays["residual_mv"], 0.0, atol=1e-9)


class TestRefusals:
    """Cases where producing a number would be worse than refusing."""

    def test_missing_arrays_error(self):
        out = _run({"temperature_k": np.array([300.0, 400.0])})
        assert out.outputs == {}
        assert out.issues[0].severity is Severity.ERROR
        assert "Missing arrays" in out.issues[0].message

    def test_single_point_errors(self):
        out = _run({"delta_t_k": np.array([5.0]), "delta_v_mv": np.array([10.0])})
        assert out.outputs == {}
        assert out.issues[0].severity is Severity.ERROR

    def test_all_delta_t_effectively_equal_errors(self):
        """Three readings at one ΔT is a replicate set, not a slope measurement."""
        out = _run(
            {
                "delta_t_k": np.array([5.0, 5.001, 4.999]),
                "delta_v_mv": np.array([10.0, 10.2, 9.9]),
            }
        )
        assert out.outputs == {}
        assert "spans only" in out.issues[0].message

    def test_mismatched_lengths_error(self):
        out = _run(
            {
                "delta_t_k": np.array([2.0, 5.0, 10.0]),
                "delta_v_mv": np.array([4.0, 10.0]),
            }
        )
        assert out.outputs == {}
        assert "differ in length" in out.issues[0].message


class TestContract:
    """The class-attribute contract `BaseAnalyzer` validates at import time."""

    def test_metadata(self):
        analyzer = ThermovoltageSlopeAnalyzer()
        assert analyzer.name == "thermovoltage-slope"
        assert analyzer.version == "1.0.0"
        assert analyzer.accepts_techniques == (Technique.THERMOELECTRIC,)

    def test_accepts_needs_a_source_file(self):
        analyzer = ThermovoltageSlopeAnalyzer()
        assert analyzer.accepts(_measure_stub()) is True
        assert analyzer.accepts(_measure_stub(with_file=False)) is False

    def test_outputs_are_json_safe(self):
        """NaN is not JSON, and it reads as a value. None says 'not determined'."""
        import json

        out = _run(_series(slope=2.0, intercept=1.0, deltas=[2.0, 10.0]))
        json.dumps(out.outputs)  # must not raise
        assert out.outputs["seebeck_stderr_mv_k"] is None
