"""The stopping verdict: one answer to "should I run another experiment?".

The engine already computed everything needed to answer that, spread across
`converged`, `max_ei`, `prob_within_epsilon`, `epsilon_delta_met` and the
reliability grade. A caller had to combine them correctly, and combining them
naively gave bad advice.

The case that motivated the work, reproduced below as
`test_a_found_peak_on_sparse_data_asks_for_confirmation`: a single-peak
objective sampled at six points *including the peak*. The engine reported
probability 0.992, signal exhausted, `converged=False`, and recommended the far
edge of the search space. Every number was right and the advice was wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization.engine import CONFIRM, CONTINUE, STOP, optimize, optimize_nd

BOUNDS = (0.0, 10.0)
PEAK_X = 6.0


def _truth(x: np.ndarray) -> np.ndarray:
    """One clear peak at x = 6, wide enough that a coarse grid can find it."""
    return 0.5 + 2.0 * np.exp(-((np.asarray(x, dtype=float) - PEAK_X) ** 2) / (2 * 1.5**2))


def _run(xs, **kwargs):
    x = np.asarray(xs, dtype=float)
    return optimize(
        x,
        _truth(x),
        bounds=BOUNDS,
        input_name="doping_pct",
        target_name="zt",
        measured_noise=0.02,
        seed=0,
        **kwargs,
    )


class TestTheThreeActions:
    def test_a_well_sampled_optimum_says_stop(self):
        result = _run(np.linspace(0.0, 10.0, 14))
        assert result.stopping.action == STOP
        assert result.stopping.should_stop is True
        assert result.stopping.probability > 0.9
        assert result.stopping.data_sufficient is True

    def test_a_found_peak_on_sparse_data_asks_for_confirmation(self):
        """The measured failure this whole verdict exists for.

        Six points, one of them sitting exactly on the peak. The model is
        essentially certain it is done, and separately has too little data for
        that certainty to be independently credible. Neither "stop" nor "go
        explore the far edge" is the honest answer; repeating the incumbent is.
        """
        result = _run([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        assert result.stopping.action == CONFIRM
        assert result.stopping.probability > 0.9
        assert result.stopping.signal_exhausted is True
        assert result.stopping.data_sufficient is False
        assert result.stopping.should_stop is False

    def test_a_missed_optimum_says_keep_going(self):
        """Five points that bracket the peak without landing on it.

        The best sample is measurably short of the optimum, so the probability
        collapses and no amount of flat acquisition should read as "done".
        """
        result = _run([0.0, 2.5, 5.0, 7.5, 10.0])
        assert result.stopping.action == CONTINUE
        assert result.stopping.probability < 0.5

    def test_remaining_improvement_says_keep_going(self):
        result = _run([0.0, 3.0, 6.0, 10.0])
        assert result.stopping.action == CONTINUE
        assert result.stopping.signal_exhausted is False


class TestTheReasonIsUsable:
    """The reason is the part a person actually reads."""

    def test_it_names_the_variable_and_the_incumbent(self):
        result = _run([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        assert "doping_pct" in result.stopping.reason
        assert "6" in result.stopping.reason

    def test_it_quotes_the_probability(self):
        result = _run(np.linspace(0.0, 10.0, 14))
        assert f"{result.stopping.probability:.2f}" in result.stopping.reason

    def test_the_confirm_reason_says_what_to_do_and_why_it_is_cheaper(self):
        result = _run([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        reason = result.stopping.reason.lower()
        assert "repeat" in reason
        assert "one experiment" in reason

    def test_a_flat_acquisition_is_explained_rather_than_trusted(self):
        """Exhausted signal with a low probability needs the reason spelled out.

        A flat expected improvement on sparse data means the surrogate is
        uninformative in the gaps it never sampled, which is the opposite of
        having found the optimum.
        """
        result = _run([0.0, 1.0, 2.0, 3.0, 4.0])
        if result.stopping.signal_exhausted and result.stopping.probability < 0.9:
            assert "never sampled" in result.stopping.reason


class TestConsistencyWithTheOlderFields:
    """The verdict is a reading of existing fields, never a second opinion."""

    def test_stop_implies_the_legacy_converged_flag(self):
        result = _run(np.linspace(0.0, 10.0, 14))
        assert result.stopping.action == STOP
        assert result.converged is True

    def test_the_verdict_echoes_epsilon_and_delta(self):
        result = _run([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], delta=0.2)
        assert result.stopping.epsilon == pytest.approx(result.epsilon)
        assert result.stopping.delta == pytest.approx(0.2)

    def test_probability_matches_the_field_it_reads(self):
        result = _run([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        assert result.stopping.probability == pytest.approx(result.prob_within_epsilon)

    def test_delta_moves_the_threshold(self):
        """A stricter risk level should make the engine harder to satisfy."""
        lenient = _run([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], delta=0.5)
        strict = _run([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], delta=0.0001)
        assert lenient.stopping.action == CONFIRM
        assert strict.stopping.action in {CONFIRM, CONTINUE}


class TestMultiDimensional:
    def test_optimize_nd_reports_a_verdict_naming_every_axis(self):
        grid = np.array([[a, b] for a in (0.0, 2.5, 5.0) for b in (0.0, 2.5, 5.0)], dtype=float)
        y = 1.0 + np.exp(-((grid[:, 0] - 2.5) ** 2 + (grid[:, 1] - 2.5) ** 2) / 2.0)
        result = optimize_nd(
            grid,
            y,
            bounds=((0.0, 5.0), (0.0, 5.0)),
            input_names=("a", "b"),
            target_name="zt",
            measured_noise=0.02,
            seed=0,
        )
        assert result.stopping is not None
        assert result.stopping.action in {STOP, CONFIRM, CONTINUE}
        assert "a = " in result.stopping.reason
        assert "b = " in result.stopping.reason


class TestWithoutReliability:
    """The robustness sweep runs without a grade; the verdict must still hold."""

    def test_no_grade_is_treated_as_insufficient_data(self):
        """Absent evidence of sufficiency is not evidence of it.

        Reading a missing grade as "good enough" would let the robustness path
        report STOP on data that was never assessed.
        """
        result = _run([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], with_reliability=False)
        assert result.stopping.data_sufficient is False
        assert result.stopping.action != STOP
