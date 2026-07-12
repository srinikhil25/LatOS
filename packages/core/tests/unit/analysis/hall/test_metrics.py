"""Tests for `latos.analysis.hall.metrics.HallMetricsAnalyzer`."""

from __future__ import annotations

from latos.analysis.base_analyzer import AnalyzerInputs
from latos.analysis.hall.metrics import HallMetricsAnalyzer
from latos.core.enums import Severity, Technique

# A self-consistent p-type export: σ = q·n·μ
#   n = 1e19 cm⁻³, μ = 50 cm²/Vs → σ = 1.602e-19 · 1e19 · 50 ≈ 80.1 S/cm.
_CONSISTENT = {
    "carrier_concentration_cm3": 1.0e19,
    "mobility_cm2_vs": 50.0,
    "conductivity_s_cm": 80.11,
    "resistivity_ohm_cm": 1.0 / 80.11,
    "hall_coefficient_cm3_c": +0.624,  # positive → holes
}


def _measure_stub(features: dict[str, float]):
    class _M:
        pass

    m = _M()
    m.features = features
    m.files = (object(),)
    return m


def _run(features: dict[str, float]):
    a = HallMetricsAnalyzer()
    inputs = AnalyzerInputs(
        measurement=_measure_stub(features),
        arrays={},
        params=a.merge_params(None),
    )
    return a.analyze(inputs)


def _run_with_params(features: dict[str, float], params):
    a = HallMetricsAnalyzer()
    inputs = AnalyzerInputs(
        measurement=_measure_stub(features), arrays={}, params=a.merge_params(params)
    )
    return a.analyze(inputs)


class TestMetadata:
    def test_accepts_hall_only(self):
        assert HallMetricsAnalyzer().accepts_techniques == (Technique.HALL,)

    def test_accepts_requires_features(self):
        a = HallMetricsAnalyzer()
        assert a.accepts(_measure_stub(_CONSISTENT))
        assert not a.accepts(_measure_stub({}))


class TestInterpretation:
    def test_p_type_from_positive_hall_coefficient(self):
        out = _run(_CONSISTENT)
        assert out.outputs["carrier_type"] == "p-type (holes)"

    def test_n_type_from_negative_hall_coefficient(self):
        f = dict(_CONSISTENT)
        f["hall_coefficient_cm3_c"] = -0.624
        f["carrier_concentration_cm3"] = -1.0e19
        out = _run(f)
        assert out.outputs["carrier_type"] == "n-type (electrons)"

    def test_concentration_reported_unsigned(self):
        f = dict(_CONSISTENT)
        f["carrier_concentration_cm3"] = -1.0e19
        out = _run(f)
        assert out.outputs["carrier_concentration_cm3"] == "1.000e+19"


class TestConsistency:
    def test_consistent_export_gets_info(self):
        out = _run(_CONSISTENT)
        assert out.outputs["consistency_deviation_pct"] < 5
        assert any(
            i.severity is Severity.INFO and "consistent" in i.message.lower() for i in out.issues
        )

    def test_unit_slip_flagged(self):
        f = dict(_CONSISTENT)
        f["conductivity_s_cm"] = 80.11e3  # three orders off — classic unit slip
        out = _run(f)
        assert any(i.severity is Severity.WARNING and i.field == "consistency" for i in out.issues)

    def test_implausible_concentration_flagged(self):
        f = dict(_CONSISTENT)
        f["carrier_concentration_cm3"] = 1.0e25
        out = _run(f)
        assert any(i.field == "carrier_concentration" for i in out.issues)


class TestGuards:
    def test_empty_features_error(self):
        out = _run({"something_else": 1.0})
        assert out.outputs == {}
        assert any(i.severity is Severity.ERROR for i in out.issues)


# ─── Cross-configuration reliability + Seebeck cross-check (1.1.0) ──


def _run_with(features: dict[str, float], seebeck_sign=None):
    a = HallMetricsAnalyzer()
    inputs = AnalyzerInputs(
        measurement=_measure_stub(features),
        arrays={},
        params=a.merge_params({"seebeck_sign": seebeck_sign}),
    )
    return a.analyze(inputs)


class TestCrossConfiguration:
    def test_sign_disagreement_marks_unreliable(self):
        f = dict(_CONSISTENT)
        f["hall_ac_cross_cm3_c"] = +0.084
        f["hall_bd_cross_cm3_c"] = -0.418
        out = _run_with(f)
        assert "UNRELIABLE" in out.outputs["carrier_type"]
        assert "disagree in sign" in out.outputs["carrier_type_reliability"]
        assert any(
            i.severity is Severity.WARNING and i.field == "hall_reliability" for i in out.issues
        )

    def test_large_ratio_is_questionable(self):
        f = dict(_CONSISTENT)
        f["hall_ac_cross_cm3_c"] = 0.025
        f["hall_bd_cross_cm3_c"] = 0.180  # same sign, 7.2x apart
        out = _run_with(f)
        assert "questionable" in out.outputs["carrier_type_reliability"]
        assert "UNRELIABLE" not in out.outputs["carrier_type"]

    def test_agreeing_configs_are_good(self):
        f = dict(_CONSISTENT)
        f["hall_ac_cross_cm3_c"] = 0.60
        f["hall_bd_cross_cm3_c"] = 0.85
        out = _run_with(f)
        assert out.outputs["carrier_type_reliability"].startswith("good")
        assert not any(i.field == "hall_reliability" for i in out.issues)

    def test_missing_cross_data_skips_check(self):
        out = _run_with(dict(_CONSISTENT))
        assert "carrier_type_reliability" not in out.outputs


class TestSeebeckCrossCheck:
    def test_agreement_reports_info(self):
        # _CONSISTENT is p-type (positive Hall coefficient); Seebeck also p.
        out = _run_with(dict(_CONSISTENT), seebeck_sign=+1.0)
        assert out.outputs["carrier_type_from_seebeck"] == "p-type (holes)"
        assert any(
            i.severity is Severity.INFO and "agreement" in i.message.lower() for i in out.issues
        )

    def test_disagreement_with_unreliable_hall_trusts_seebeck(self):
        f = dict(_CONSISTENT)
        f["hall_coefficient_cm3_c"] = -0.167  # Hall says n-type
        f["carrier_concentration_cm3"] = -3.7e19
        f["hall_ac_cross_cm3_c"] = +0.084  # ...but its own configs disagree
        f["hall_bd_cross_cm3_c"] = -0.418
        out = _run_with(f, seebeck_sign=+1.0)  # Seebeck says p-type
        warning = next(i for i in out.issues if i.field == "carrier_type")
        assert warning.severity is Severity.WARNING
        assert "Seebeck determination is the trustworthy one" in warning.message

    def test_disagreement_with_reliable_hall_says_check_both(self):
        f = dict(_CONSISTENT)
        f["hall_coefficient_cm3_c"] = -0.5
        f["carrier_concentration_cm3"] = -1e19
        f["hall_ac_cross_cm3_c"] = -0.45  # configs agree: Hall is confident
        f["hall_bd_cross_cm3_c"] = -0.55
        out = _run_with(f, seebeck_sign=+1.0)
        warning = next(i for i in out.issues if i.field == "carrier_type")
        assert "Check both measurements" in warning.message

    def test_no_seebeck_sign_skips_check(self):
        out = _run_with(dict(_CONSISTENT), seebeck_sign=None)
        assert "carrier_type_from_seebeck" not in out.outputs


# ─── Reusable cross-config reliability function (DQ1) ───────────────


class TestCrossConfigReliability:
    def test_sign_disagreement_unreliable(self):
        from latos.analysis.hall.metrics import cross_config_reliability

        level, reason = cross_config_reliability(
            {"hall_ac_cross_cm3_c": 0.084, "hall_bd_cross_cm3_c": -0.418}
        )
        assert level == "unreliable"
        assert reason and "disagree in sign" in reason

    def test_large_ratio_questionable(self):
        from latos.analysis.hall.metrics import cross_config_reliability

        level, _ = cross_config_reliability(
            {"hall_ac_cross_cm3_c": 0.18, "hall_bd_cross_cm3_c": 0.025}
        )
        assert level == "questionable"

    def test_agreement_good(self):
        from latos.analysis.hall.metrics import cross_config_reliability

        level, reason = cross_config_reliability(
            {"hall_ac_cross_cm3_c": 0.6, "hall_bd_cross_cm3_c": 0.85}
        )
        assert level == "good" and reason is None

    def test_missing_cross_data_unknown(self):
        from latos.analysis.hall.metrics import cross_config_reliability

        assert cross_config_reliability({})[0] == "unknown"


# ─── Conductivity cross-check vs R&S (1.2.0) ────────────────────────


class TestConductivityCrossCheck:
    def test_large_disagreement_flagged(self):
        f = dict(_CONSISTENT)
        f["conductivity_s_cm"] = 2930.0
        out = _run_with_params(f, {"rs_conductivity_s_cm": 66667.0})  # ~23x
        assert out.outputs["conductivity_cross_ratio"] > 3
        assert any(
            i.severity is Severity.WARNING and i.field == "conductivity_cross" for i in out.issues
        )

    def test_agreement_reports_info(self):
        f = dict(_CONSISTENT)
        f["conductivity_s_cm"] = 2930.0
        out = _run_with_params(f, {"rs_conductivity_s_cm": 3200.0})
        assert any(
            i.severity is Severity.INFO and i.field == "conductivity_cross" for i in out.issues
        )

    def test_no_rs_conductivity_skips(self):
        out = _run_with_params(dict(_CONSISTENT), {})
        assert "conductivity_cross_ratio" not in out.outputs
