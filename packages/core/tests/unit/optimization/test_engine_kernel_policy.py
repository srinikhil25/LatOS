"""Tests for the kernel choice and the exploration fallback policy.

Both options shipped without tests, which is why their defaults sat unexamined.
Two kinds of test live here.

The first guards a *bug*: `optimize_nd` used to hardcode "RBF" into the reported
config while its surrogate took whatever the helper default was. A result that
misnames its own model is worse than a wrong default, because it is invisible.

The second guards two *measured decisions*. Both defaults were challenged on
2026-09-02 and both survived on evidence, against the direction the literature
pointed. The numbers are in RESULTS_LOG.md; these tests exist so a future change
has to argue with the measurement rather than with a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization import OptimizationError, optimize, optimize_nd
from latos.optimization.engine import (
    _DEFAULT_EXPLORE_POLICY,
    _DEFAULT_KERNEL,
    _EXPLORE_POLICIES,
    _KERNELS,
    _kernel_label,
    _widest_gap_index,
)

X_1D = np.array([1.0, 2.0, 3.0, 4.0])
Y_1D = np.array([1.0, 3.0, 2.5, 1.2])
X_ND = np.array(
    [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.8, 0.2, 0.6], [0.3, 0.9, 0.1], [0.6, 0.4, 0.9]]
)
Y_ND = np.array([1.0, 2.0, 1.5, 0.5, 2.2])


def _nd(**kwargs):
    return optimize_nd(
        X_ND,
        Y_ND,
        bounds=((0.0, 1.0),) * 3,
        input_names=("a", "b", "c"),
        target_name="t",
        with_reliability=False,
        **kwargs,
    )


class TestMeasuredDefaults:
    """The shipped defaults, each pinned to the measurement that justifies it.

    Changing either is allowed — re-running the sweep and updating RESULTS_LOG.md
    is the price. Changing one silently is not.
    """

    def test_kernel_default_is_rbf(self):
        # Matern 5/2 is the field standard and lost here: on Branin it won only
        # 2 of 8 seeds and tripled the worst-case regret (0.351 -> 1.139).
        assert _DEFAULT_KERNEL == "rbf"

    def test_explore_policy_default_is_max_std(self):
        # "ei" and "ucb" both reached a worst-case regret of 6.02 on Forrester
        # from n = 4 — campaigns that never found the optimum at all — against
        # 0.17 for max_std.
        assert _DEFAULT_EXPLORE_POLICY == "max_std"

    def test_defaults_are_members_of_their_option_lists(self):
        assert _DEFAULT_KERNEL in _KERNELS
        assert _DEFAULT_EXPLORE_POLICY in _EXPLORE_POLICIES


class TestKernelLabel:
    """The reported name has to follow the kernel actually fitted."""

    @pytest.mark.parametrize(
        ("kernel", "ard", "expected"),
        [
            ("rbf", False, "ConstantKernel * RBF"),
            ("rbf", True, "ConstantKernel * RBF(ARD)"),
            ("matern52", False, "ConstantKernel * Matern(nu=5/2)"),
            ("matern52", True, "ConstantKernel * Matern(nu=5/2)(ARD)"),
        ],
    )
    def test_label(self, kernel, ard, expected):
        assert _kernel_label(kernel, ard=ard) == expected


class TestKernelIsThreadedNotHardcoded:
    """Regression: the N-D path once reported RBF whatever it fitted."""

    @pytest.mark.parametrize("kernel", ["rbf", "matern52"])
    def test_nd_reports_the_kernel_it_was_given(self, kernel):
        assert _kernel_label(kernel, ard=True) == _nd(kernel=kernel).config.kernel

    @pytest.mark.parametrize("kernel", ["rbf", "matern52"])
    def test_1d_reports_the_kernel_it_was_given(self, kernel):
        result = optimize(
            X_1D, Y_1D, bounds=(1.0, 4.0), input_name="a", target_name="t", kernel=kernel
        )
        assert result.config.kernel == _kernel_label(kernel)

    def test_nd_kernel_changes_the_surrogate(self):
        """A label that moves while the model does not would still be a lie."""
        rbf, matern = (_nd(kernel=k) for k in ("rbf", "matern52"))
        assert rbf.cand_mean != matern.cand_mean

    def test_nd_isotropic_label_drops_ard(self):
        assert _nd(kernel="rbf", isotropic=True).config.kernel == "ConstantKernel * RBF"


class TestRejectsUnknownChoices:
    def test_unknown_kernel_1d(self):
        with pytest.raises(OptimizationError, match="kernel must be one of"):
            optimize(X_1D, Y_1D, bounds=(1.0, 4.0), input_name="a", target_name="t", kernel="rbf52")

    def test_unknown_explore_policy(self):
        with pytest.raises(OptimizationError, match="explore_policy must be one of"):
            optimize(
                X_1D,
                Y_1D,
                bounds=(1.0, 4.0),
                input_name="a",
                target_name="t",
                explore_policy="wander",
            )


class TestExplorePolicyChangesTheFallback:
    """All three policies must remain reachable, whatever the default is."""

    @pytest.mark.parametrize("policy", ["max_std", "ei", "ucb"])
    def test_policy_is_accepted(self, policy):
        result = optimize(
            X_1D, Y_1D, bounds=(1.0, 4.0), input_name="a", target_name="t", explore_policy=policy
        )
        assert 1.0 <= float(result.recommendation.x) <= 4.0


class TestTheWidestGapFallback:
    """Item 8b, added 2026-09-10. NOT the default.

    `max_std` was described in `engine.py` as "the largest unmeasured gap" for
    months, and on an over-smoothed posterior it is not that: the sd surface
    goes flat to seven significant figures — measured range
    [0.00368, 0.00368] with eight grid points tied — so which point wins is
    decided by where a tie falls, and the design it produced spent 2 of 9 picks
    re-measuring endpoints already held. `widest_gap` is that description
    implemented literally.

    Added as a fourth policy rather than as an edit to `max_std`, because three
    defaults have been challenged in this project and three have been kept on
    evidence. A one-noise-level, one-shape-family measurement is not enough to
    move one.
    """

    def test_it_is_selectable_and_is_not_the_default(self):
        assert "widest_gap" in _EXPLORE_POLICIES
        assert _DEFAULT_EXPLORE_POLICY != "widest_gap"

    def test_it_bisects_the_only_gap(self):
        grid = np.linspace(0.0, 1.0, 1001)
        assert grid[_widest_gap_index(grid, np.array([0.5]))] == pytest.approx(0.25, abs=1e-3)

    def test_it_finds_an_interior_gap(self):
        grid = np.linspace(0.0, 1.0, 1001)
        got = grid[_widest_gap_index(grid, np.array([0.0, 0.1, 0.2]))]
        assert got == pytest.approx(0.6, abs=1e-3)

    def test_the_search_box_bounds_the_intervals(self):
        """An unsampled end of the range must compete with an interior gap.

        With every observation crowded at the top, the answer is the middle of
        the empty stretch below them — which is only reachable if the box edge
        is treated as an interval boundary.
        """
        grid = np.linspace(0.0, 1.0, 1001)
        got = grid[_widest_gap_index(grid, np.array([0.9, 0.95, 1.0]))]
        assert got == pytest.approx(0.45, abs=1e-3)

    def test_it_can_never_return_a_point_already_measured(self):
        """The property that distinguishes it from `argmax(std)`, and the one
        the measured duplicate-picking cost."""
        grid = np.linspace(0.0, 1.0, 1001)
        rng = np.random.default_rng(0)
        for _ in range(40):
            obs = np.sort(rng.uniform(0.0, 1.0, size=rng.integers(1, 9)))
            got = grid[_widest_gap_index(grid, obs)]
            assert np.min(np.abs(obs - got)) > 1e-3

    def test_it_spends_every_sample_on_a_new_composition(self):
        """End to end, on the shape where `max_std` wasted two of nine picks."""

        def peak(t):
            return 0.3 + 1.2 * np.exp(-((np.asarray(t, dtype=float) - 0.72) ** 2) / (2 * 0.09**2))

        rng = np.random.default_rng(0)
        xs = [0.0, 0.5, 1.0]
        ys = [float(peak(p)) + rng.normal(0, 0.15) for p in xs]
        while len(xs) < 12:
            result = optimize(
                np.array(xs),
                np.array(ys),
                bounds=(0.0, 1.0),
                input_name="x",
                target_name="S",
                measured_noise=0.15,
                seed=0,
                explore_policy="widest_gap",
                with_reliability=True,
            )
            xs.append(float(result.recommendation.x))
            ys.append(float(peak(xs[-1])) + rng.normal(0, 0.15))

        assert len(set(np.round(xs, 6))) == 12  # max_std averaged 10.0 of 12


class TestNonFiniteInputIsRejectedByName:
    """Re-found 2026-09-17; the July fix for this was never merged.

    `pytest.raises(ValueError)` would not have caught the regression:
    scikit-learn's own error is also a ValueError. The type that matters is
    `OptimizationError`, because that is the one the server turns into a 400.
    """

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            ([0.0, 0.5, 1.0], [1.0, np.nan, 2.0]),
            ([0.0, 0.5, 1.0], [1.0, np.inf, 2.0]),
            ([0.0, np.nan, 1.0], [1.0, 1.5, 2.0]),
            ([0.0, 0.5, -np.inf], [1.0, 1.5, 2.0]),
        ],
    )
    def test_optimize(self, x, y):
        with pytest.raises(OptimizationError, match="finite"):
            optimize(
                np.array(x),
                np.array(y),
                bounds=(0.0, 1.0),
                input_name="x",
                target_name="y",
                seed=0,
            )
