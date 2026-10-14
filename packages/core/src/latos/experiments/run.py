"""Hand the prepared arrays to the optimizer, and nothing else.

One rule decides which optimizer function is called: one design variable goes
to `optimize()`, more than one goes to `optimize_nd()`. The two are not
interchangeable at d = 1. Their length-scale floors differ, only the first has
an acquisition choice and an exploration policy, and every one-variable record
already frozen was produced by the first. So the split is confined to this
module, and neither function is changed.

Every optimizer default is left alone: the acquisition, the exploration policy,
the kernel, the exploration sweetener, the risk level and the grid size are the
optimizer's own. All but the exploration policy are recorded in the frozen
configuration it returns; that one is not, so a record made through here was
made under the optimizer's default policy.
"""

from __future__ import annotations

from latos.experiments.prepare import PreparedOptimization
from latos.experiments.validation import InsufficientConditions
from latos.optimization import engine as _engine
from latos.optimization.engine import OptimizationResult, OptimizationResultND

__all__ = ["NOISE_LEVELS", "min_conditions", "run_optimization"]

# The optimizer's own minimums, repeated here so too little data can be refused
# before the call. A test holds these to the optimizer's constants.
_MIN_CONDITIONS = 3
_ND_EXTRA_CONDITIONS = 2

# "declared" passes the prepared prediction noise. "engine_median" passes none,
# so the optimizer takes the median of the per-point uncertainties itself; it
# exists only so an older pipeline can be reproduced exactly while it migrates.
NOISE_LEVELS = ("declared", "engine_median")


def min_conditions(n_dims: int) -> int:
    """The fewest distinct conditions a surrogate over `n_dims` axes needs."""
    if n_dims == 1:
        return _MIN_CONDITIONS
    return max(_MIN_CONDITIONS, n_dims + _ND_EXTRA_CONDITIONS)


def run_optimization(
    prepared: PreparedOptimization,
    *,
    seed: int = 0,
    noise_level: str = "declared",
    objective_aggregation: str | None = None,
) -> OptimizationResult | OptimizationResultND:
    """Run one optimization round on the prepared arrays.

    Args:
        prepared: The output of `prepare_optimization`.
        seed: The optimizer's random seed.
        noise_level: One of `NOISE_LEVELS`.
        objective_aggregation: Overrides the recorded aggregation label.
            Transitional, for reproducing an older pipeline's records.

    Returns:
        The optimizer's own result type: `OptimizationResult` for one design
        variable, `OptimizationResultND` for more.

    Raises:
        InsufficientConditions: If there are too few distinct conditions.
        ValueError: If `noise_level` is not recognised.
    """
    if noise_level not in NOISE_LEVELS:
        raise ValueError(f"noise_level must be one of {NOISE_LEVELS}; got {noise_level!r}")
    need = min_conditions(prepared.n_dims)
    have = int(prepared.y.size)
    if have < need:
        raise InsufficientConditions(have, need)

    measured_noise = prepared.prediction_noise if noise_level == "declared" else None
    aggregation = objective_aggregation or prepared.objective_aggregation

    if prepared.n_dims == 1:
        return _engine.optimize(
            prepared.x[:, 0],
            prepared.y,
            bounds=prepared.bounds[0],
            input_name=prepared.input_names[0],
            target_name=prepared.target_name,
            direction=prepared.direction,
            y_transform=prepared.fit_space,
            y_min=prepared.y_min,
            y_max=prepared.y_max,
            measured_noise=measured_noise,
            point_noise=prepared.sigma,
            seed=seed,
            objective_aggregation=aggregation,
        )
    return _engine.optimize_nd(
        prepared.x,
        prepared.y,
        bounds=prepared.bounds,
        input_names=prepared.input_names,
        axis_kinds=(_engine.AXIS_CONTINUOUS,) * prepared.n_dims,
        target_name=prepared.target_name,
        direction=prepared.direction,
        y_transform=prepared.fit_space,
        y_min=prepared.y_min,
        y_max=prepared.y_max,
        measured_noise=measured_noise,
        point_noise=prepared.sigma,
        seed=seed,
        objective_aggregation=aggregation,
    )
