"""Tests for the physical-properties registry (`latos.core.physics`)."""

from __future__ import annotations

from latos.core.physics import lookup


def test_log_natural_positive_transport():
    for name in ("mobility_cm2_vs", "conductivity_s_cm", "resistivity_ohm_cm"):
        p = lookup(name)
        assert p is not None and p.log_natural and p.positive


def test_sign_bearing_not_positive_not_log():
    for name in ("seebeck_uv_k", "hall_coefficient_cm3_c", "carrier_concentration_cm3"):
        p = lookup(name)
        assert p is not None and not p.positive and not p.log_natural


def test_bounded_positive_linear():
    z = lookup("zT (derived)")
    assert z is not None and z.positive and not z.log_natural
    assert z.min_value == 0.0 and z.max_value == 4.0


def test_temperature_suffix_stripped():
    assert lookup("zT (derived) @ 600 K") is lookup("zT (derived)")


def test_target_mode_distance_has_no_physics():
    assert lookup("|zT (derived) - 1.0|") is None


def test_unknown_returns_none():
    assert lookup("etching_time_h") is None
