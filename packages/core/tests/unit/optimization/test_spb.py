"""Tests for the single-parabolic-band physics model (`latos.optimization.spb`).

Validated against analytic Fermi-integral values and the known qualitative
physics (interior zT optimum, monotone Seebeck, Bi2Te3-class anchor).
"""

from __future__ import annotations

import math

import pytest

from latos.optimization import spb


class TestFermiIntegrals:
    def test_known_values_at_eta_zero(self):
        assert spb.fermi_integral(0.0, 0.0) == pytest.approx(math.log(2), abs=1e-4)
        assert spb.fermi_integral(1.0, 0.0) == pytest.approx(math.pi**2 / 12, abs=1e-4)
        assert spb.fermi_integral(2.0, 0.0) == pytest.approx(1.5 * 1.2020569, abs=1e-3)

    def test_nondegenerate_limit(self):
        # F_j(eta) -> Gamma(j+1) * exp(eta) as eta -> -inf.
        eta = -6.0
        assert spb.fermi_integral(0.0, eta) == pytest.approx(math.exp(eta), rel=0.02)
        assert spb.fermi_integral(1.0, eta) == pytest.approx(math.exp(eta), rel=0.02)


class TestSeebeck:
    def test_monotonically_decreases_with_eta(self):
        etas = [-2.0, 0.0, 2.0, 5.0, 10.0]
        s = [spb.seebeck_uv_k(e) for e in etas]
        assert all(s[i] > s[i + 1] for i in range(len(s) - 1))

    def test_scale_is_kb_over_e(self):
        # S(η) is (k_B/e) times the dimensionless reduced Seebeck.
        expected = spb.K_B_OVER_E_UV_K * spb.reduced_seebeck(0.0)
        assert spb.seebeck_uv_k(0.0) == pytest.approx(expected)


class TestOptimum:
    def test_zt_has_interior_maximum(self):
        beta = 0.4
        eta_star = spb.optimal_eta(beta)
        z = spb.zt(eta_star, beta)
        # Interior: strictly better than points well to either side.
        assert z > spb.zt(eta_star - 2.0, beta)
        assert z > spb.zt(eta_star + 2.0, beta)
        assert -5.0 < eta_star < 15.0

    def test_bi2te3_class_anchor(self):
        # beta ~ 0.4 is the Bi2Te3 class: peak zT ~ 1 at |S| ~ 240 uV/K.
        beta = 0.4
        assert spb.zt(spb.optimal_eta(beta), beta) == pytest.approx(1.0, abs=0.15)
        assert spb.optimal_seebeck(beta) == pytest.approx(240.0, abs=25)

    def test_better_material_optimizes_at_higher_seebeck(self):
        # Lower lattice kappa (higher beta) -> optimum at lower n / higher S (SnSe-like).
        assert spb.optimal_seebeck(1.0) > spb.optimal_seebeck(0.4) > spb.optimal_seebeck(0.1)


class TestInversion:
    def test_eta_from_seebeck_round_trips(self):
        for eta in (-1.0, 0.5, 3.0):
            s = spb.seebeck_uv_k(eta)
            assert spb.eta_from_seebeck(s) == pytest.approx(eta, abs=1e-3)

    def test_fit_quality_factor_round_trips(self):
        beta_true = 0.5
        eta = spb.optimal_eta(beta_true)
        s = spb.seebeck_uv_k(eta)
        z = spb.zt(eta, beta_true)
        assert spb.fit_quality_factor(s, z) == pytest.approx(beta_true, rel=1e-3)

    def test_fit_rejects_zt_above_spb_ceiling(self):
        # A very degenerate (low |S|) point cannot support a high zT under SPB:
        # this is the physical ceiling that flags multi-band / data issues.
        with pytest.raises(ValueError, match="ceiling"):
            spb.fit_quality_factor(28.0, 0.985)


class TestGuidance:
    def test_flags_under_doped_material(self):
        # Pick a point below the optimum Seebeck for its fitted beta.
        beta = 0.5
        s_opt = spb.optimal_seebeck(beta)
        eta = spb.eta_from_seebeck(s_opt + 60.0)  # lower |S| would be over-doped...
        # Construct a low-|S| (over-doped) point that is still SPB-describable.
        s = spb.seebeck_uv_k(eta)
        z = spb.zt(eta, beta)
        g = spb.guidance(s, z)
        assert g.applicable
        assert g.direction in {"increase_seebeck", "decrease_seebeck", "at_optimum"}
        assert g.optimal_seebeck_uv_k == pytest.approx(s_opt, abs=2.0)

    def test_at_optimum_when_measured_equals_optimal(self):
        beta = 0.5
        eta = spb.optimal_eta(beta)
        g = spb.guidance(spb.seebeck_uv_k(eta), spb.zt(eta, beta))
        assert g.applicable
        assert g.direction == "at_optimum"

    def test_ceiling_case_is_not_applicable_and_explains(self):
        g = spb.guidance(28.0, 0.985)
        assert not g.applicable
        assert g.beta is None
        assert g.zt_ceiling is not None and g.zt_ceiling < 0.985
        assert "multi-band" in g.note
