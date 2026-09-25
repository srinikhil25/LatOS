"""Tests for the `acquisition` argument on `optimize`.

Three things are guarded here, in order of how badly they would hurt.

The first is a *reporting* bug of the kind already found once: `optimize_nd`
hardcoded "RBF" into its config while the surrogate used whatever the helper
defaulted to, so every result claimed RBF regardless. A config that names the
wrong acquisition is worse than a wrong default, because it looks auditable.

The second is the separation of concerns that makes the shoot-out mean anything:
`acquisition` must change where the next point goes and must not touch the
convergence machinery, which is calibrated in EI's units against the
measurement noise.

The third is the interaction that makes the whole question narrower than it
looks. At these sample sizes the reliability-gated fallback can override the
acquisition entirely, so a test pins that the override is what happens rather
than leaving it to be rediscovered.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization import OptimizationError, optimize
from latos.optimization.acquisitions import ACQUISITIONS, DEFAULT_ACQUISITION

# Sparse, noisy and peaked: the regime the alternatives exist for.
X = np.array([0.0, 0.5, 1.0, 0.25])
Y = np.array([0.300, 0.378, 0.293, 0.247])
KW = {
    "bounds": (0.0, 1.0),
    "input_name": "x",
    "target_name": "S",
    "direction": "maximize",
    "measured_noise": 0.06,
    "seed": 3,
}


class TestTheConfigNamesWhatActuallyRan:
    def test_every_arm_is_reported_back(self):
        for name in ACQUISITIONS:
            result = optimize(X, Y, acquisition=name, **KW)
            assert result.config.acquisition_function == name

    def test_the_default_is_recorded_rather_than_left_blank(self):
        # A blank field means "written before the alternatives existed". A
        # default run is not that, and must not be indistinguishable from it.
        assert optimize(X, Y, **KW).config.acquisition_function == DEFAULT_ACQUISITION

    def test_an_unknown_arm_is_refused(self):
        with pytest.raises(OptimizationError, match="acquisition must be one of"):
            optimize(X, Y, acquisition="vibes", **KW)


class TestTheDefaultIsUnchangedBehaviour:
    def test_the_default_arm_is_ei(self):
        assert DEFAULT_ACQUISITION == "ei"

    def test_naming_ei_explicitly_changes_nothing(self):
        implicit = optimize(X, Y, **KW)
        explicit = optimize(X, Y, acquisition="ei", **KW)
        assert explicit.recommendation.x == implicit.recommendation.x
        assert explicit.max_ei == implicit.max_ei
        assert explicit.converged == implicit.converged


class TestTheAcquisitionDoesNotMoveTheStoppingRule:
    """`max_ei`, `converged` and the epsilon-delta verdict are EI's business.

    `kg` is in units of improvement in the reported optimum and `mes` is in
    nats, so if any of them reached the convergence test its threshold would
    mean something different in every arm and no comparison would be possible.
    """

    def test_max_ei_is_identical_across_arms(self):
        baseline = optimize(X, Y, acquisition="ei", **KW)
        for name in ACQUISITIONS:
            other = optimize(X, Y, acquisition=name, **KW)
            assert other.max_ei == pytest.approx(baseline.max_ei), name
            assert other.noise_threshold == pytest.approx(baseline.noise_threshold), name

    def test_convergence_is_identical_across_arms(self):
        baseline = optimize(X, Y, acquisition="ei", **KW)
        for name in ACQUISITIONS:
            other = optimize(X, Y, acquisition=name, **KW)
            assert other.converged == baseline.converged, name
            assert other.epsilon_delta_met == baseline.epsilon_delta_met, name

    def test_the_reliability_grade_is_identical_across_arms(self):
        baseline = optimize(X, Y, acquisition="ei", **KW)
        for name in ACQUISITIONS:
            other = optimize(X, Y, acquisition=name, **KW)
            assert other.reliability.level == baseline.reliability.level, name


class TestTheFallbackOverridesTheAcquisition:
    """Measured 2026-09-10, and the main content of open problem 8.

    On this four-point campaign the noisy incumbent (0.378) sits above the
    entire posterior mean (max 0.305), so EI collapses to 1.7e-59 — 57 orders
    of magnitude below the 0.06 noise floor it is tested against. Exhaustion
    therefore fires on the first round, the data is graded exploratory at n = 4,
    and the engine hands the pick to `argmax(sd)`. The acquisition function
    never decides anything in the shipped configuration.
    """

    def test_ei_collapses_far_below_the_noise_floor(self):
        result = optimize(X, Y, **KW)
        assert result.max_ei < 1e-12
        assert result.max_ei < result.noise_threshold

    def test_the_data_is_still_graded_exploratory(self):
        assert optimize(X, Y, **KW).reliability.level == "exploratory"

    def test_every_arm_lands_on_the_same_point_in_the_shipped_config(self):
        picks = {
            name: optimize(X, Y, acquisition=name, explore_policy="max_std", **KW).recommendation.x
            for name in ACQUISITIONS
        }
        assert len(set(picks.values())) == 1, picks

    def test_removing_the_fallback_lets_the_arms_disagree(self):
        # The same call with the fallback off must NOT tie, or the arms would
        # be indistinguishable for a reason unrelated to the fallback and this
        # whole comparison would be measuring nothing.
        picks = {
            name: optimize(X, Y, acquisition=name, explore_policy="ei", **KW).recommendation.x
            for name in ACQUISITIONS
        }
        assert len(set(picks.values())) > 1, picks


class TestAFlatAcquisitionSurfaceDoesNotPickTheLeftEndpoint:
    """Measured 2026-09-10. `argmax` on an all-tied array returns index 0.

    `kg` under a `prior_mean` is identically zero at every candidate: the prior
    pins the fitted length-scale at its ceiling, the rank-one update direction
    goes constant across the grid, and every candidate's expected gain is the
    same. Before the guard the tool synthesised `bounds[0]` and reported it as a
    knowledge-gradient recommendation. EI is computed on every path, so it is
    the available tie-break that is not an arbitrary corner of the box.
    """

    PRIOR = staticmethod(lambda x: 0.3 + 0.1 * np.asarray(x))

    def test_kg_under_a_prior_defers_to_ei_instead_of_the_low_bound(self):
        kw = {**KW, "explore_policy": "ei", "prior_mean": self.PRIOR}
        kg = optimize(X, Y, acquisition="kg", **kw).recommendation.x
        ei = optimize(X, Y, acquisition="ei", **kw).recommendation.x
        assert kg == ei
        assert kg != KW["bounds"][0]

    def test_kg_still_decides_for_itself_without_a_prior(self):
        # The guard must not have neutered the arm in the ordinary case.
        kw = {**KW, "explore_policy": "ei"}
        kg = optimize(X, Y, acquisition="kg", **kw).recommendation.x
        ei = optimize(X, Y, acquisition="ei", **kw).recommendation.x
        assert kg != ei
