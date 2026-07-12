# ruff: noqa: N806
# N806: `S` (Seebeck) follows the thermoelectric single-letter convention.
"""Tests for the thermoelectric zT kernel."""

from __future__ import annotations

import numpy as np
import pytest

from latos.analysis.transport import TransportError, compute_zt


def _cs_like():
    """Synthetic CS-like inputs (native units) with a known-ish zT scale."""
    rs_t = np.array([316.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0])
    rho = np.array([0.1244, 0.147, 0.176, 0.205, 0.235, 0.262, 0.285])  # µΩ·m
    seebeck = np.array([7.85, 10.5, 14.0, 18.0, 22.0, 27.0, 32.0])  # µV/K
    lfa_t = np.array([300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0])
    kappa = np.array([5.15, 4.95, 4.74, 4.50, 4.30, 4.15, 4.08])  # W/m·K
    return rs_t, rho, seebeck, lfa_t, kappa


class TestComputeZt:
    def test_basic_shape_and_units(self):
        rs_t, rho, S, lfa_t, kappa = _cs_like()
        r = compute_zt(
            rs_temperature_k=rs_t,
            resistivity_uohm_m=rho,
            seebeck_uv_k=S,
            lfa_temperature_k=lfa_t,
            thermal_conductivity_w_mk=kappa,
        )
        # Output is on the LFA grid.
        assert r.temperature_k.tolist() == lfa_t.tolist()
        # SI conversions applied.
        assert r.resistivity_ohm_m[1] == pytest.approx(0.147 * 1e-6, rel=1e-6)
        assert r.seebeck_v_k[1] == pytest.approx(10.5 * 1e-6, rel=1e-6)
        # zT is physical and modest.
        assert 0.0 < r.peak_zt < 1.5
        assert r.peak_zt_temperature_k in lfa_t.tolist()
        assert "zT = PF·T/κ" in " ".join(r.provenance)

    def test_zt_formula_matches_manual(self):
        # One point where R&S and LFA grids coincide (no interpolation):
        # zT = (S²/ρ)·T/κ.
        rs_t = np.array([400.0])
        rho = np.array([0.2])  # µΩ·m → 2e-7 Ω·m
        S = np.array([20.0])  # µV/K → 2e-5 V/K
        lfa_t = np.array([400.0])
        kappa = np.array([2.0])
        r = compute_zt(
            rs_temperature_k=rs_t,
            resistivity_uohm_m=rho,
            seebeck_uv_k=S,
            lfa_temperature_k=lfa_t,
            thermal_conductivity_w_mk=kappa,
        )
        expected = ((2e-5) ** 2 / 2e-7) * 400.0 / 2.0
        assert r.zt[0] == pytest.approx(expected, rel=1e-9)

    def test_extrapolation_warning(self):
        # LFA grid starts at 300 K but R&S only from 316 K → clamp warning.
        rs_t, rho, S, lfa_t, kappa = _cs_like()
        r = compute_zt(
            rs_temperature_k=rs_t,
            resistivity_uohm_m=rho,
            seebeck_uv_k=S,
            lfa_temperature_k=lfa_t,
            thermal_conductivity_w_mk=kappa,
        )
        assert any("clamped" in w for w in r.warnings)

    def test_plausibility_flags_unit_error(self):
        # Resistivity 10× too small (a µΩ·m vs 0.1×µΩ·m slip) → zT explodes.
        rs_t = np.array([400.0, 500.0])
        rho = np.array([0.02, 0.025])  # 10× too small
        S = np.array([200.0, 220.0])
        lfa_t = np.array([400.0, 500.0])
        kappa = np.array([1.0, 1.0])
        r = compute_zt(
            rs_temperature_k=rs_t,
            resistivity_uohm_m=rho,
            seebeck_uv_k=S,
            lfa_temperature_k=lfa_t,
            thermal_conductivity_w_mk=kappa,
        )
        assert any("exceeds the plausible bound" in w for w in r.warnings)


class TestWiedemannFranz:
    def test_consistent_material_no_wf_warning(self):
        # Semiconductor-like: ρ ~ 10 µΩ·m, κ ~ 1.5 W/m·K. The electronic
        # floor L·σ·T (~0.6 W/m·K) sits well below κ → no violation.
        rs_t = np.array([400.0, 500.0])
        rho = np.array([10.0, 11.0])  # µΩ·m
        S = np.array([180.0, 200.0])
        lfa_t = np.array([400.0, 500.0])
        kappa = np.array([1.5, 1.4])
        r = compute_zt(
            rs_temperature_k=rs_t,
            resistivity_uohm_m=rho,
            seebeck_uv_k=S,
            lfa_temperature_k=lfa_t,
            thermal_conductivity_w_mk=kappa,
        )
        assert not any("Wiedemann" in w for w in r.warnings)

    def test_metallic_sigma_with_low_kappa_flags_wf(self):
        # Metallic ρ ~ 0.3 µΩ·m (σ ~ 3e6 S/m) with κ ~ 1.5 W/m·K: the
        # electronic floor is ~30 W/m·K, so κ is impossibly low. zT stays
        # under the plausible bound, so ONLY the WF check should fire.
        rs_t = np.array([500.0, 600.0])
        rho = np.array([0.28, 0.30])  # µΩ·m
        S = np.array([26.0, 30.0])
        lfa_t = np.array([500.0, 600.0])
        kappa = np.array([1.4, 1.5])
        r = compute_zt(
            rs_temperature_k=rs_t,
            resistivity_uohm_m=rho,
            seebeck_uv_k=S,
            lfa_temperature_k=lfa_t,
            thermal_conductivity_w_mk=kappa,
        )
        wf = [w for w in r.warnings if "Wiedemann" in w]
        assert wf, "expected a Wiedemann-Franz violation"
        assert "electronic floor" in wf[0]
        assert r.peak_zt < 4.0  # plausibility gate did not also fire
        assert not any("exceeds the plausible bound" in w for w in r.warnings)

    def test_nan_kappa_does_not_crash_or_flag(self):
        # A NaN-κ point (bad LFA data) must not raise or spuriously trigger
        # the WF check; the consistent point stays unflagged.
        rs_t = np.array([400.0, 500.0])
        rho = np.array([10.0, 11.0])
        S = np.array([180.0, 200.0])
        lfa_t = np.array([400.0, 500.0])
        kappa = np.array([1.5, np.nan])  # second point missing
        r = compute_zt(
            rs_temperature_k=rs_t,
            resistivity_uohm_m=rho,
            seebeck_uv_k=S,
            lfa_temperature_k=lfa_t,
            thermal_conductivity_w_mk=kappa,
        )
        assert not any("Wiedemann" in w for w in r.warnings)


class TestGuards:
    def test_empty_raises(self):
        with pytest.raises(TransportError):
            compute_zt(
                rs_temperature_k=np.array([]),
                resistivity_uohm_m=np.array([]),
                seebeck_uv_k=np.array([]),
                lfa_temperature_k=np.array([300.0]),
                thermal_conductivity_w_mk=np.array([5.0]),
            )

    def test_mismatched_rs_lengths_raise(self):
        with pytest.raises(TransportError):
            compute_zt(
                rs_temperature_k=np.array([300.0, 400.0]),
                resistivity_uohm_m=np.array([0.1]),
                seebeck_uv_k=np.array([7.0, 8.0]),
                lfa_temperature_k=np.array([300.0]),
                thermal_conductivity_w_mk=np.array([5.0]),
            )
