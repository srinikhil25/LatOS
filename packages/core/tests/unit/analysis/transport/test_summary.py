"""Tests for `latos.analysis.transport.summary.TransportSummaryAnalyzer`."""

from __future__ import annotations

import numpy as np
import pytest

from latos.analysis.base_analyzer import AnalyzerInputs
from latos.analysis.transport.summary import TransportSummaryAnalyzer
from latos.core.enums import Severity, Technique


def _measure_stub():
    class _M:
        files = (object(),)

    return _M()


def _run(arrays):
    a = TransportSummaryAnalyzer()
    inputs = AnalyzerInputs(
        measurement=_measure_stub(), arrays=arrays, params=a.merge_params(None),
    )
    return a.analyze(inputs)


def _rs_arrays(seebeck_sign=+1.0):
    """A plausible p-type R&S sweep, 300→600 K."""
    t = np.linspace(300.0, 600.0, 13)
    rho = np.linspace(8.0, 15.0, 13)  # µΩ·m, metallic-ish rise
    s = seebeck_sign * np.linspace(120.0, 220.0, 13)  # µV/K
    return {"temperature_k": t, "resistivity_uohm_m": rho, "seebeck_uv_k": s}


class TestMetadata:
    def test_accepts_thermoelectric_only(self):
        assert TransportSummaryAnalyzer().accepts_techniques == (
            Technique.THERMOELECTRIC,
        )


class TestResistivitySeebeck:
    def test_p_type_from_positive_seebeck(self):
        out = _run(_rs_arrays(+1.0))
        assert out.outputs["carrier_type_from_seebeck"] == "p-type (holes)"

    def test_n_type_from_negative_seebeck(self):
        out = _run(_rs_arrays(-1.0))
        assert out.outputs["carrier_type_from_seebeck"] == "n-type (electrons)"

    def test_power_factor_value_and_units(self):
        # At 600 K: S = 220 µV/K, ρ = 15 µΩ·m
        # PF = (220e-6 V/K)² / (15e-6 Ω·m) = 3.227e-3 W/(m·K²) ≈ 3227 µW/(m·K²).
        out = _run(_rs_arrays())
        assert out.outputs["peak_power_factor_uw_mk2"] == pytest.approx(3226.7, rel=0.02)
        assert out.outputs["peak_power_factor_at_k"] == pytest.approx(600.0, abs=1)

    def test_power_factor_curve_derived(self):
        out = _run(_rs_arrays())
        assert "power_factor_uw_mk2" in out.derived_arrays
        assert out.derived_arrays["power_factor_uw_mk2"].shape == (13,)

    def test_seebeck_sign_change_flagged(self):
        arrays = _rs_arrays()
        arrays["seebeck_uv_k"] = np.linspace(-50.0, 150.0, 13)  # crosses zero
        out = _run(arrays)
        assert any(
            i.severity is Severity.WARNING and i.field == "seebeck" for i in out.issues
        )


class TestLfa:
    def test_kappa_range_and_min_location(self):
        t = np.linspace(300.0, 600.0, 7)
        kappa = np.array([1.2, 1.1, 1.0, 0.9, 0.85, 0.8, 0.82])
        out = _run({"temperature_k": t, "thermal_conductivity": kappa})
        assert out.outputs["kappa_range_w_mk"] == [0.8, 1.2]
        assert out.outputs["kappa_min_at_k"] == pytest.approx(550.0, abs=1)

    def test_implausible_kappa_flagged(self):
        t = np.linspace(300.0, 600.0, 5)
        kappa = np.array([1.0, 2.0, 3.0, 4.0, 900.0])  # unit slip
        out = _run({"temperature_k": t, "thermal_conductivity": kappa})
        assert any(
            i.severity is Severity.WARNING and "unit error" in i.message
            for i in out.issues
        )


class TestGuards:
    def test_unrecognized_arrays_error(self):
        out = _run({"temperature_k": np.array([300.0, 400.0]), "foo": np.array([1.0, 2.0])})
        assert out.outputs == {}
        assert any(i.severity is Severity.ERROR for i in out.issues)

    def test_missing_temperature_error(self):
        out = _run({"thermal_conductivity": np.array([1.0, 2.0])})
        assert any(i.severity is Severity.ERROR for i in out.issues)
