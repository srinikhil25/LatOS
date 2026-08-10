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


class TestSpbPrior:
    """`make_spb_prior` — the bridge from SPB physics to the optimizer's prior.

    The engine wants zT as a function of the *synthesis knob*; SPB gives it as a
    function of the reduced Fermi level. These pin the join, and in particular
    that the prior carries the interior optimum a zero-mean GP cannot know
    about.
    """

    @staticmethod
    def _series(beta: float, intercept: float, slope: float, knob):
        import numpy as np

        knob = np.asarray(knob, dtype=float)
        eta = intercept + slope * knob
        seebeck = np.array([abs(spb.seebeck_uv_k(float(e))) for e in eta])
        zt_vals = np.array([spb.zt(float(e), beta) for e in eta])
        return knob, seebeck, zt_vals

    def test_round_trips_the_parameters_that_generated_it(self):
        """A series generated from SPB must be recovered by the fit."""
        x, s, z = self._series(0.4, -2.0, 1.0, [0.0, 1.0, 2.0, 3.0, 4.0])
        p = spb.make_spb_prior(x, s, z)
        assert p.beta == pytest.approx(0.4, rel=1e-3)
        assert p.eta_intercept == pytest.approx(-2.0, abs=1e-3)
        assert p.eta_slope == pytest.approx(1.0, abs=1e-3)
        assert p.n_used == 5
        assert p.n_excluded == 0

    def test_carries_an_interior_optimum(self):
        """The whole point: the prior knows the far edge is bad beforehand.

        A zero-mean GP reverts to a flat trend outside the data, so a sparse
        campaign drifts toward whatever it has not sampled. SPB says zT peaks at
        an intermediate carrier concentration, and that peak has to survive the
        eta(x) bridge or the prior is not worth having.
        """
        import numpy as np

        x, s, z = self._series(0.4, -2.0, 1.0, [0.0, 1.0, 2.0, 3.0, 4.0])
        p = spb.make_spb_prior(x, s, z)
        grid = np.linspace(0.0, 6.0, 61)
        values = p(grid)
        peak = int(np.argmax(values))
        assert 0 < peak < len(grid) - 1, "prior must peak inside the range"
        expected = (spb.optimal_eta(0.4) - p.eta_intercept) / p.eta_slope
        assert grid[peak] == pytest.approx(expected, abs=0.2)

    def test_accepts_a_column_of_a_multi_axis_design(self):
        import numpy as np

        x, s, z = self._series(0.4, -2.0, 1.0, [0.0, 1.0, 2.0, 3.0, 4.0])
        design = np.column_stack([np.full_like(x, 400.0), x])
        p = spb.make_spb_prior(design, s, z, axis=1)
        assert p.eta_slope == pytest.approx(1.0, abs=1e-3)
        assert p(design) == pytest.approx(p(x), rel=1e-9)

    def test_excludes_samples_above_the_single_band_ceiling(self):
        """Points SPB cannot describe are dropped and counted, never absorbed.

        Silently fitting through an impossible point would bake a wrong shape
        into every subsequent recommendation. The count is the signal.
        """
        import numpy as np

        x, s, z = self._series(0.4, -2.0, 1.0, [0.0, 1.0, 2.0, 3.0, 4.0])
        z = np.asarray(z, dtype=float).copy()
        z[2] = 50.0  # far above the SPB ceiling at that Seebeck
        p = spb.make_spb_prior(x, s, z)
        assert p.n_excluded == 1
        assert p.n_used == 4
        assert "excluded" in p.note

    def test_refuses_when_fewer_than_two_samples_survive(self):
        """One point fixes a level but not a trend; an invented slope is worse
        than no prior at all."""
        import numpy as np

        x, s, z = self._series(0.4, -2.0, 1.0, [0.0, 1.0, 2.0])
        z = np.asarray(z, dtype=float).copy()
        z[1:] = 50.0
        with pytest.raises(ValueError, match="at least two"):
            spb.make_spb_prior(x, s, z)

    def test_mismatched_lengths_are_rejected(self):
        import numpy as np

        with pytest.raises(ValueError, match="align"):
            spb.make_spb_prior(
                np.array([0.0, 1.0, 2.0]), np.array([200.0, 150.0]), np.array([0.5, 0.6, 0.7])
            )

    def test_steers_the_optimizer_to_the_physical_optimum(self):
        """End to end, with an honest noise figure.

        `measured_noise` matters here and is not incidental. On the 8 % default
        the noise floor sits above the best expected improvement, the engine
        concludes no experiment can beat the measurement, and — below ten
        samples — falls back to exploring the widest gap, discarding the prior
        entirely. That is correct conservatism given a pessimistic noise input,
        but it means a prior only pays off once repeatability is measured.
        """
        import numpy as np

        from latos.optimization import optimize

        x, s, z = self._series(0.4, -2.0, 1.0, [0.0, 1.0, 2.0, 3.0, 4.0])
        rng = np.random.default_rng(0)
        observed = z + rng.normal(0.0, 0.02, z.size)
        p = spb.make_spb_prior(x, s, z)

        result = optimize(
            x,
            observed,
            bounds=(0.0, 6.0),
            input_name="doping_pct",
            target_name="zT",
            prior_mean=p,
            measured_noise=0.02,
        )
        expected = (spb.optimal_eta(p.beta) - p.eta_intercept) / p.eta_slope
        assert result.recommendation.x == pytest.approx(expected, abs=0.3)
