"""The categorical-axis guard on `optimize_nd`.

A synthesis parameter can only reach the engine as a float, so a knob with no
ordering — etching atmosphere Air / Ar / N2 — gets encoded as 0/1/2, and the GP
happily interpolates it. That is not hypothetical: a replay over the MXene TEM
data recommended a gas index of 0.55, a recipe with no meaning, reported with a
confidence interval like any other answer.

The guard is deliberately two-sided, and these tests pin both sides:

- an axis the caller DECLARES categorical raises, because intent is known;
- an axis that merely LOOKS encoded warns, because "anneal 1, 2, 3 h" has the
  same shape as "gas 0, 1, 2" and the engine sees only floats.

The second half is why the false-positive case below is asserted as a warning
rather than fixed: refusing there would block a legitimate campaign, and a
wrongly-blocked run costs more than a notice a researcher can read past.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization import OptimizationError, optimize_nd
from latos.optimization.engine import _encoded_axis_warning

# Three temperatures crossed with three encoded gases: the shape of the run
# that produced "gas 0.55". Temperature is a real quantity, gas is a name.
_TEMP = np.array([40, 50, 60, 40, 50, 60, 40, 50, 60], dtype=float)
_GAS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=float)
_Y = np.array([1.0, 1.4, 1.2, 1.9, 2.4, 2.0, 1.1, 1.3, 1.0])
_X = np.column_stack([_TEMP, _GAS])
_BOUNDS = [(40.0, 60.0), (0.0, 2.0)]
_NAMES = ["temp_C", "gas"]


def _run(**kwargs):
    return optimize_nd(_X, _Y, bounds=_BOUNDS, input_names=_NAMES, target_name="zT", **kwargs)


class TestDeclaredCategoricalIsRefused:
    def test_raises_and_names_the_axis(self):
        with pytest.raises(OptimizationError) as exc:
            _run(axis_kinds=["continuous", "categorical"])
        message = str(exc.value)
        assert "'gas'" in message
        # The error has to say what to do instead, or it just blocks the user.
        assert "one campaign per level" in message

    def test_every_categorical_axis_is_listed(self):
        with pytest.raises(OptimizationError) as exc:
            _run(axis_kinds=["categorical", "categorical"])
        assert "'temp_C'" in str(exc.value) and "'gas'" in str(exc.value)

    def test_all_continuous_is_allowed(self):
        result = _run(axis_kinds=["continuous", "continuous"])
        assert len(result.recommendation.x) == 2

    def test_wrong_length_is_rejected(self):
        with pytest.raises(OptimizationError, match="axis_kinds has 1 entries"):
            _run(axis_kinds=["continuous"])

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(OptimizationError, match="axis_kinds must each be one of"):
            _run(axis_kinds=["continuous", "ordinal"])

    def test_omitting_it_does_not_refuse(self):
        # The default must stay permissive: every existing caller passes nothing.
        assert len(_run().recommendation.x) == 2


class TestUndeclaredEncodedAxisWarns:
    def test_the_gas_axis_is_flagged(self):
        result = _run()
        assert len(result.axis_warnings) == 1
        warning = result.axis_warnings[0]
        assert "'gas'" in warning
        # The recommendation genuinely landed between levels — that is the whole
        # trigger, and the number the user would otherwise have gone and made.
        assert 0.05 < result.recommendation.x[1] % 1.0 < 0.95

    def test_the_continuous_axis_is_not_flagged(self):
        # 40/50/60 are whole numbers too. Spacing is what separates them from
        # an encoding, and temperature must never be accused of being a name.
        assert all("temp_C" not in w for w in _run().axis_warnings)

    def test_a_clean_two_axis_run_is_silent(self):
        dop = np.linspace(0.0, 5.0, 5)
        temp = np.linspace(300.0, 600.0, 5)
        dd, tt = np.meshgrid(dop, temp, indexing="ij")
        x = np.column_stack([dd.ravel(), tt.ravel()])
        y = np.exp(-((x[:, 0] - 3.0) ** 2) / 2.0) * (x[:, 1] / 600.0)
        result = optimize_nd(
            x,
            y,
            bounds=[(0.0, 5.0), (300.0, 600.0)],
            input_names=["doping", "temp_K"],
            target_name="zT",
        )
        assert result.axis_warnings == ()


class TestDetector:
    """`_encoded_axis_warning` directly — the trigger is two conditions, not one."""

    @staticmethod
    def _fire(values, rec) -> bool:
        return _encoded_axis_warning(np.array(values, dtype=float), rec, "ax") is not None

    def test_unit_spaced_levels_with_an_off_level_recommendation(self):
        assert self._fire([0, 0, 1, 1, 2, 2], 1.5)

    def test_silent_when_the_recommendation_lands_on_a_level(self):
        # Nothing to act on differently, so nothing to say.
        assert not self._fire([0, 0, 1, 1, 2, 2], 1.02)

    def test_silent_when_levels_are_not_one_apart(self):
        assert not self._fire([40, 50, 60], 52.4)

    def test_silent_on_a_genuinely_continuous_column(self):
        assert not self._fire([1.1, 2.7, 3.3, 4.9, 5.2], 3.9)

    def test_silent_above_the_level_cap(self):
        # Seven levels is a swept integer variable, not a handful of names.
        assert not self._fire(list(range(7)), 3.5)

    def test_silent_on_a_constant_column(self):
        assert not self._fire([2, 2, 2], 2.0)

    def test_a_binary_flag_is_flagged(self):
        # Two levels is the commonest encoding of all: with/without, on/off.
        assert self._fire([0, 0, 1, 1], 0.5)

    def test_a_real_quantity_at_1_2_3_is_a_known_false_positive(self):
        # Documented, not accidental. The engine cannot distinguish an anneal at
        # 1, 2 and 3 hours from three named gases, and the warning text says so.
        # This is exactly why the undeclared path warns instead of raising.
        assert self._fire([1, 1, 2, 2, 3, 3], 2.4)

    def test_the_warning_says_it_may_be_nothing(self):
        text = _encoded_axis_warning(np.array([0.0, 1.0, 2.0]), 1.5, "gas")
        assert text is not None
        assert "If they are real quantities, nothing is wrong here." in text
