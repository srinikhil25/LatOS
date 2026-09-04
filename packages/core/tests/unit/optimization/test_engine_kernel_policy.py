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
