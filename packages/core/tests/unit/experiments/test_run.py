"""Dispatch to the optimizer: which function, and with what."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from latos.experiments import (
    InsufficientConditions,
    build_observations,
    min_conditions,
    prepare_optimization,
    reduce_observations,
    run_optimization,
)
from latos.experiments.prepare import PreparedOptimization
from latos.optimization import engine
from latos.optimization.engine import OptimizationResult
from tests.unit.experiments._examples import (
    make_definition,
    make_nd_definition,
    make_nd_table,
    make_row,
    reduced_set,
)


def _prepared_1d() -> PreparedOptimization:
    definition = make_definition()
    rows = (
        make_row("a", 0.0, -2.6, 0.40),
        make_row("b", 0.25, 2.0, 0.02),
        make_row("c", 0.5, 2.1, 0.08),
        make_row("d", 0.75, 2.2, 0.04),
        make_row("e", 1.0, 1.6, 0.03),
    )
    return prepare_optimization(definition, reduced_set(definition, *rows))


def _prepared_nd(n_dims: int) -> PreparedOptimization:
    definition = make_nd_definition(n_dims)
    rng = np.random.default_rng(n_dims)
    points = [
        (tuple(float(v) for v in rng.uniform(0.0, 10.0, n_dims)), float(i))
        for i in range(n_dims + 3)
    ]
    observations, excluded = build_observations(definition, make_nd_table(n_dims, points))
    return prepare_optimization(definition, reduce_observations(definition, observations, excluded))


@pytest.fixture
def spies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {
        "optimize": [],
        "optimize_nd": [],
    }

    def record(name: str) -> Any:
        def spy(*args: Any, **kwargs: Any) -> str:
            calls[name].append((args, kwargs))
            return name

        return spy

    monkeypatch.setattr(engine, "optimize", record("optimize"))
    monkeypatch.setattr(engine, "optimize_nd", record("optimize_nd"))
    return calls


def test_one_design_variable_goes_to_optimize(spies: dict[str, list[Any]]) -> None:
    prepared = _prepared_1d()
    assert run_optimization(prepared, seed=7) == "optimize"
    assert not spies["optimize_nd"]
    ((args, kwargs),) = spies["optimize"]

    x, y = args
    assert x.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]  # one axis, flat
    assert y.tolist() == [2.6, 2.0, 2.1, 2.2, 1.6]  # magnitudes
    # The arrays are handed over as they are, not copied.
    assert kwargs.pop("point_noise") is prepared.sigma
    assert kwargs == {
        "bounds": (0.0, 1.0),
        "input_name": "additive_fraction",
        "target_name": "abs_signal",
        "direction": "maximize",
        "y_transform": "identity",
        "y_min": None,
        "y_max": None,
        "measured_noise": prepared.prediction_noise,
        "seed": 7,
        "objective_aggregation": "magnitude",
    }
    assert kwargs["measured_noise"] == pytest.approx(0.04)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_more_than_one_design_variable_goes_to_optimize_nd(
    spies: dict[str, list[Any]], n_dims: int
) -> None:
    prepared = _prepared_nd(n_dims)
    assert run_optimization(prepared) == "optimize_nd"
    assert not spies["optimize"]
    ((args, kwargs),) = spies["optimize_nd"]

    assert args[0].shape == (n_dims + 3, n_dims)
    assert kwargs["axis_kinds"] == ("continuous",) * n_dims
    assert kwargs["input_names"] == tuple(f"x{i}" for i in range(n_dims))
    assert kwargs["bounds"] == ((0.0, 10.0),) * n_dims
    assert kwargs["target_name"] == "yield_pct"
    assert kwargs["objective_aggregation"] == "identity"
    assert kwargs["point_noise"] is prepared.sigma
    assert kwargs["measured_noise"] == prepared.prediction_noise
    assert kwargs["seed"] == 0


def test_the_transitional_options_change_only_what_they_name(spies: dict[str, list[Any]]) -> None:
    prepared = _prepared_1d()
    run_optimization(prepared, noise_level="engine_median", objective_aggregation="abs_slope")
    ((_, kwargs),) = spies["optimize"]
    assert kwargs["measured_noise"] is None
    assert kwargs["point_noise"] is prepared.sigma
    assert kwargs["objective_aggregation"] == "abs_slope"

    with pytest.raises(ValueError, match="noise_level"):
        run_optimization(prepared, noise_level="guess")


def test_the_minimums_match_the_optimizers_own() -> None:
    assert min_conditions(1) == engine._MIN_POINTS
    for n_dims in (2, 3, 6):
        assert min_conditions(n_dims) == max(engine._MIN_POINTS, n_dims + engine._ND_MIN_EXTRA)


def test_too_few_conditions_are_refused_before_the_optimizer_is_called(
    spies: dict[str, list[Any]],
) -> None:
    definition = make_definition()
    rows = (make_row("a", 0.1, 1.0, 0.1), make_row("b", 0.4, 2.0, 0.1))
    prepared = prepare_optimization(definition, reduced_set(definition, *rows))
    with pytest.raises(InsufficientConditions) as caught:
        run_optimization(prepared)
    assert (caught.value.have, caught.value.need) == (2, 3)
    assert not spies["optimize"]


def test_the_uncertainties_reach_the_surrogate() -> None:
    """The real optimizer, not a spy: sigma and the noise level arrive as declared."""
    prepared = _prepared_1d()
    result = run_optimization(prepared)

    assert isinstance(result, OptimizationResult)
    config = result.config
    assert config.objective == "abs_signal"
    assert config.input_name == "additive_fraction"
    assert config.bounds == (0.0, 1.0)
    assert config.objective_aggregation == "magnitude"
    assert config.n_observations == 5
    # The prediction noise sets the shared level ...
    assert config.noise_std == pytest.approx(0.04)
    assert result.noise_measured is True
    # ... and each condition's own sigma is its weight, as sigma / noise_std.
    assert config.point_noise_used is True
    assert config.point_noise_scale == pytest.approx([10.0, 0.5, 2.0, 1.0, 0.75])

    recommendation = result.recommendation
    low, high = recommendation.predictive_interval_95
    assert 0.0 <= recommendation.x <= 1.0
    assert low < recommendation.predicted_mean < high
    assert result.stopping is not None
    assert result.reliability is not None
    assert result.reliability.level == "exploratory"  # five conditions
    assert result.stopping.should_stop is False


def test_the_optimizers_own_functions_are_what_runs() -> None:
    """Nothing was wrapped or replaced: the dispatch targets are the originals."""
    from latos.experiments import run as run_module

    assert run_module._engine is engine
    assert run_module._engine.optimize is engine.optimize
    assert run_module._engine.optimize_nd is engine.optimize_nd
