"""Tests for the multi-dimensional path, `latos.optimization.engine.optimize_nd`.

The most important test in this file is `TestOneDimensionalPathUnchanged`: the
1-D `optimize()` is what produced every frozen pre-registration on disk, and one
of those records is printed on a submitted midterm slide. Generalising the shared
helpers to accept (n, d) must not move those numbers by so much as a rounding
step, so they are pinned here explicitly rather than trusted to stay put.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.optimization import OptimizationError, optimize, optimize_nd

# The frozen drop-impact record: four loadings, the recommendation and interval
# that were committed to disk before the fifth sample was made.
FROZEN_X = np.array([40.169, 44.946, 50.025, 54.956])
FROZEN_Y = np.array([71.666667, 57.0, 36.666667, 57.0])


def _grid_2d(n_dop: int = 5, n_temp: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A separable 2-D surface with a known interior optimum.

    zT-like: a peak in doping at 3, rising monotonically with temperature.
    """
    dop = np.linspace(0.0, 5.0, n_dop)
    temp = np.linspace(300.0, 600.0, n_temp)
    dd, tt = np.meshgrid(dop, temp, indexing="ij")
    x = np.column_stack([dd.ravel(), tt.ravel()])
    y = np.exp(-((x[:, 0] - 3.0) ** 2) / 2.0) * (x[:, 1] / 600.0)
    return x, y


class TestOneDimensionalPathUnchanged:
    """The generalisation must be invisible to every existing caller."""

    def test_frozen_record_reproduces_exactly(self):
        r = optimize(
            FROZEN_X,
            FROZEN_Y,
            bounds=(float(FROZEN_X.min()), float(FROZEN_X.max())),
            input_name="particle_wt_pct",
            target_name="peak_force_n",
            direction="minimize",
        )
        lo, hi = r.recommendation.predictive_interval_95
        assert r.recommendation.x == pytest.approx(47.451, abs=0.01)
        assert lo == pytest.approx(31.215, abs=0.01)
        assert hi == pytest.approx(57.425, abs=0.01)
        assert r.max_ei == pytest.approx(0.68, abs=0.01)
        assert r.noise_threshold == pytest.approx(4.45, abs=0.01)
        assert r.reliability is not None
        assert (r.reliability.loo_inside, r.reliability.loo_total) == (3, 4)

    def test_reliability_reports_one_dimension(self):
        r = optimize(
            FROZEN_X,
            FROZEN_Y,
            bounds=(40.0, 55.0),
            input_name="wt",
            target_name="peak_force_n",
            direction="minimize",
        )
        assert r.reliability is not None
        assert r.reliability.n_dims == 1
        # The dimension caveat belongs only on multi-axis runs.
        assert "do not yet scale with dimension" not in r.reliability.note

    def test_scalar_length_scale_still_scalar(self):
        r = optimize(
            FROZEN_X,
            FROZEN_Y,
            bounds=(40.0, 55.0),
            input_name="wt",
            target_name="peak_force_n",
            direction="minimize",
        )
        assert isinstance(r.config.length_scale, float)
        assert np.isfinite(r.config.length_scale)


class TestGuards:
    def test_rejects_wrong_number_of_names(self):
        x, y = _grid_2d()
        with pytest.raises(OptimizationError, match="input_names"):
            optimize_nd(
                x, y, bounds=[(0, 5), (300, 600)], input_names=("doping",), target_name="zt"
            )

    def test_rejects_too_few_points_for_the_dimension(self):
        x = np.array([[0.0, 300.0], [1.0, 400.0], [2.0, 500.0]])
        with pytest.raises(OptimizationError, match="at least"):
            optimize_nd(
                x,
                np.array([0.1, 0.2, 0.3]),
                bounds=[(0, 5), (300, 600)],
                input_names=("doping", "temp"),
                target_name="zt",
            )

    def test_rejects_degenerate_bounds(self):
        x, y = _grid_2d()
        with pytest.raises(OptimizationError, match="high > low"):
            optimize_nd(
                x, y, bounds=[(0, 5), (600, 600)], input_names=("doping", "temp"), target_name="zt"
            )

    def test_rejects_non_finite(self):
        x, y = _grid_2d()
        y = y.copy()
        y[0] = np.nan
        with pytest.raises(OptimizationError, match="finite"):
            optimize_nd(
                x, y, bounds=[(0, 5), (300, 600)], input_names=("doping", "temp"), target_name="zt"
            )

    def test_rejects_row_count_mismatch(self):
        x, y = _grid_2d()
        with pytest.raises(OptimizationError, match="values"):
            optimize_nd(
                x,
                y[:-1],
                bounds=[(0, 5), (300, 600)],
                input_names=("doping", "temp"),
                target_name="zt",
            )


class TestTwoDimensional:
    def test_recommends_inside_the_box(self):
        x, y = _grid_2d()
        r = optimize_nd(
            x, y, bounds=[(0, 5), (300, 600)], input_names=("doping", "temp"), target_name="zt"
        )
        rec = r.recommendation.x
        assert len(rec) == 2
        assert 0.0 <= rec[0] <= 5.0
        assert 300.0 <= rec[1] <= 600.0

    def test_uses_an_ard_kernel_with_one_scale_per_axis(self):
        x, y = _grid_2d()
        r = optimize_nd(
            x, y, bounds=[(0, 5), (300, 600)], input_names=("doping", "temp"), target_name="zt"
        )
        assert r.config.n_dims == 2
        assert "ARD" in r.config.kernel
        assert len(r.config.length_scales) == 2
        assert all(np.isfinite(v) for v in r.config.length_scales)

    def test_separates_a_sharp_axis_from_a_flat_one(self):
        """On a surface that varies fast in x0 and slowly in x1, ARD should
        assign the shorter length-scale to x0. This is the whole reason for
        an anisotropic kernel: an isotropic one cannot say which axis matters."""
        dop = np.linspace(0.0, 5.0, 7)
        temp = np.linspace(300.0, 600.0, 7)
        dd, tt = np.meshgrid(dop, temp, indexing="ij")
        x = np.column_stack([dd.ravel(), tt.ravel()])
        # Three full cycles along axis 0, essentially flat along axis 1.
        y = np.sin(x[:, 0] * 3.0) + 0.01 * (x[:, 1] / 600.0)
        r = optimize_nd(
            x, y, bounds=[(0, 5), (300, 600)], input_names=("fast", "flat"), target_name="prop"
        )
        fast, flat = r.config.length_scales
        assert fast < flat

    def test_reliability_flags_the_dimension_and_its_caveat(self):
        x, y = _grid_2d()
        r = optimize_nd(
            x, y, bounds=[(0, 5), (300, 600)], input_names=("doping", "temp"), target_name="zt"
        )
        assert r.reliability is not None
        assert r.reliability.n_dims == 2
        assert r.reliability.loo_total == x.shape[0]
        assert "do not yet scale with dimension" in r.reliability.note

    def test_candidate_set_is_a_power_of_two_and_inside_bounds(self):
        x, y = _grid_2d()
        r = optimize_nd(
            x,
            y,
            bounds=[(0, 5), (300, 600)],
            input_names=("doping", "temp"),
            target_name="zt",
            n_candidates=512,
        )
        cand = np.asarray(r.candidates)
        assert cand.shape == (512, 2)
        assert cand[:, 0].min() >= 0.0 and cand[:, 0].max() <= 5.0
        assert cand[:, 1].min() >= 300.0 and cand[:, 1].max() <= 600.0
        assert len(r.cand_mean) == len(r.cand_ei) == 512

    def test_is_reproducible_for_a_fixed_seed(self):
        x, y = _grid_2d()
        kw = {
            "bounds": [(0, 5), (300, 600)],
            "input_names": ("doping", "temp"),
            "target_name": "zt",
            "seed": 3,
        }
        a = optimize_nd(x, y, **kw)
        b = optimize_nd(x, y, **kw)
        assert a.recommendation.x == b.recommendation.x
        assert a.max_ei == pytest.approx(b.max_ei)

    def test_physical_bounds_clamp_the_interval(self):
        x, y = _grid_2d()
        r = optimize_nd(
            x,
            y,
            bounds=[(0, 5), (300, 600)],
            input_names=("doping", "temp"),
            target_name="zt",
            y_min=0.0,
            y_max=4.0,
        )
        lo, hi = r.recommendation.predictive_interval_95
        assert lo >= 0.0
        assert hi <= 4.0
        assert min(r.cand_lower) >= 0.0

    def test_minimization_picks_a_low_region(self):
        x, y = _grid_2d()
        r = optimize_nd(
            x,
            y,
            bounds=[(0, 5), (300, 600)],
            input_names=("doping", "temp"),
            target_name="zt",
            direction="minimize",
        )
        assert r.best_y == pytest.approx(float(y.min()))


class TestOneAxisThroughTheNdPath:
    """d = 1 must still be a legal, sane call even though `optimize()` is the
    production route for it."""

    def test_single_axis_runs_and_recommends_in_range(self):
        x = FROZEN_X.reshape(-1, 1)
        r = optimize_nd(
            x,
            FROZEN_Y,
            bounds=[(40.0, 55.0)],
            input_names=("wt",),
            target_name="peak_force_n",
            direction="minimize",
        )
        assert r.config.n_dims == 1
        assert "ARD" not in r.config.kernel
        assert 40.0 <= r.recommendation.x[0] <= 55.0
        assert r.reliability is not None
        assert r.reliability.n_dims == 1
