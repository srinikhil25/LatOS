"""Tests for the physical-properties registry (`latos.core.physics`)."""

from __future__ import annotations

from latos.core.physics import lookup, lookup_axis


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


# ─── Mechanical shock: a transmitted force cannot be negative ────────────
def test_peak_force_is_strictly_positive():
    p = lookup("peak_force_n")
    assert p is not None and p.positive and not p.log_natural
    assert p.min_value == 0.0 and p.max_value is None


def test_peak_voltage_is_strictly_positive():
    p = lookup("peak_voltage_v")
    assert p is not None and p.positive and p.min_value == 0.0


# ─── Input-axis physics: the packing limit ──────────────────────────────
def test_particle_volume_axis_is_capped_at_close_packing():
    ax = lookup_axis("particle_vol_pct")
    assert ax is not None
    assert ax.max_value == 64.0  # random close packing of spheres
    assert ax.min_value == 40.0


def test_axis_clamp_leaves_a_feasible_request_untouched():
    ax = lookup_axis("particle_vol_pct")
    assert ax is not None
    assert ax.clamp(43.8, 58.7) == (43.8, 58.7, False)


def test_axis_clamp_cuts_an_unpreparable_request_back():
    ax = lookup_axis("particle_vol_pct")
    assert ax is not None
    lo, hi, clamped = ax.clamp(20.0, 80.0)
    assert (lo, hi, clamped) == (40.0, 64.0, True)


def test_axis_without_known_physics_returns_none():
    # wt% has no universal ceiling: it depends on the particle/liquid densities.
    assert lookup_axis("particle_wt_pct") is None
    assert lookup_axis("doping_pct") is None
