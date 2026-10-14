"""Turn reduced conditions into exactly what the optimizer is handed.

Pure, and it does no I/O. This is the one place the objective is projected:

==========================  ==============  =========  =====================
objective                   y               direction  label
==========================  ==============  =========  =====================
maximize / minimize         y_raw           the goal   ``<name>``
the same, with magnitude    |y_raw|         the goal   ``abs_<name>``
target t                    |y_raw - t|     minimize   ``|<name> - <t>|``
==========================  ==============  =========  =====================

`y_raw` is copied through untouched, so a frozen record always holds both the
signed value that was measured and the value that was optimized, and `y` can be
recomputed from `y_raw` and the declared objective alone.

The uncertainty of a projected value is taken to equal that of `y_raw`. That is
exact for the identity and first-order for the two folds away from the fold
itself. Near zero the fold changes the shape of the distribution, which no
single sigma can represent.

When a magnitude objective is given signed values of both signs, a message says
so. The magnitude then falls to zero somewhere between two conditions, and a
surrogate fitted to magnitudes has no way to know it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from latos.experiments.definition import (
    ExperimentDefinition,
    Goal,
    VariableDefinition,
    VariableKind,
)
from latos.experiments.identity import definition_sha256
from latos.experiments.observations import ObservationSet, ReducedObservation
from latos.experiments.validation import UnsupportedDesignSpace

__all__ = [
    "AGGREGATION_IDENTITY",
    "AGGREGATION_MAGNITUDE",
    "AGGREGATION_TARGET",
    "PREDICTION_NOISE_ENGINE_MEDIAN",
    "PREDICTION_NOISE_GLOSSARY",
    "PREDICTION_NOISE_LOWER_BOUND",
    "PREDICTION_NOISE_NOT_PASSED",
    "PREDICTION_NOISE_POOLED",
    "PREDICTION_NOISE_UNDETERMINED",
    "PreparedOptimization",
    "prepare_optimization",
    "project",
    "require_continuous_design",
]

AGGREGATION_IDENTITY = "identity"
AGGREGATION_MAGNITUDE = "magnitude"
AGGREGATION_TARGET = "target_distance"

PREDICTION_NOISE_POOLED = "pooled_condition"
PREDICTION_NOISE_LOWER_BOUND = "lower_bound"
PREDICTION_NOISE_UNDETERMINED = "undetermined"
PREDICTION_NOISE_NOT_PASSED = "not_passed_degenerate"
PREDICTION_NOISE_ENGINE_MEDIAN = "engine_median"

PREDICTION_NOISE_GLOSSARY: Mapping[str, str] = {
    PREDICTION_NOISE_POOLED: (
        "The pooled specimen-to-specimen standard deviation, divided by the square root of "
        "the declared number of specimens per condition."
    ),
    PREDICTION_NOISE_LOWER_BOUND: (
        "The median of the per-condition uncertainties handed to the optimizer (a condition "
        "without one is given the largest known), divided by the square root of the declared "
        "number of specimens per condition. No pooled specimen-to-specimen spread is "
        "available, and a within-specimen uncertainty leaves that variation out, so the "
        "predictive interval is too narrow by an amount that cannot be known until a "
        "condition is replicated."
    ),
    PREDICTION_NOISE_UNDETERMINED: (
        "No uncertainty was known for any condition. Each was given the spread of the "
        "conditions' own values (or 1 response unit if they are all equal), and the median "
        "of those, divided by the square root of the declared number of specimens per "
        "condition, was used. That spread includes real differences between conditions, so "
        "this is an assumption about the noise, not a measurement of it."
    ),
    PREDICTION_NOISE_NOT_PASSED: (
        "The median uncertainty was vanishingly small (below 1e-6 of the mean magnitude of "
        "the values), or none was positive, so none was passed and the optimizer's own "
        "relative assumption applies."
    ),
    PREDICTION_NOISE_ENGINE_MEDIAN: (
        "Not passed; the optimizer took the median of the per-point uncertainties itself."
    ),
}

# Below this share of the signal a claimed uncertainty is not precision, it is
# the absence of an estimate.
_DEGENERATE_SIGMA_FRACTION = 1e-6

_REFUSALS: Mapping[VariableKind, str] = {
    VariableKind.INTEGER: (
        "the optimizer returns a real number and nothing rounds it. Declare the axis "
        "continuous if a fractional value can be made, or run one campaign per level"
    ),
    VariableKind.CATEGORICAL: (
        "a Gaussian process interpolates between numbers, so it would recommend a value "
        "between two levels, which is not something that can be made. Run one campaign per "
        "level and compare them"
    ),
    VariableKind.ORDINAL: (
        "an ordinal axis has order but no metric, and a Gaussian process needs a distance"
    ),
}


@dataclass(frozen=True, slots=True)
class PreparedOptimization:
    """Exactly the arrays and settings one optimization round uses.

    Attributes:
        x: The design matrix, shape (n, d), one row per condition.
        y: The projected objective values the optimizer sees.
        y_raw: The signed values before projection, element-identical to the
            reducer's output.
        sigma: The per-condition 1-sigma handed to the optimizer, substitutes
            included; None when no uncertainty is passed.
        sigma_sources: The reducer's label for each condition's sigma.
        prediction_noise: The 1-sigma of one future observation made per the
            declared protocol; None when not passed.
        prediction_noise_source: A label from `PREDICTION_NOISE_GLOSSARY`.
        bounds: The search range of each design variable.
        input_names: The design variable names, in axis order.
        target_name: The label the optimizer reports under.
        direction: ``maximize`` or ``minimize``.
        fit_space: ``identity`` or ``log``, as declared.
        y_min: Lower physical bound of the optimized quantity, or None.
        y_max: Upper physical bound, or None.
        objective_aggregation: How `y` was obtained from `y_raw`.
        n_dims: The number of design variables.
        messages: Notes for the report and the record.
    """

    x: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    sigma: np.ndarray | None
    sigma_sources: tuple[str, ...]
    prediction_noise: float | None
    prediction_noise_source: str
    bounds: tuple[tuple[float, float], ...]
    input_names: tuple[str, ...]
    target_name: str
    direction: str
    fit_space: str
    y_min: float | None
    y_max: float | None
    objective_aggregation: str
    n_dims: int
    messages: tuple[str, ...]


def project(y_raw: np.ndarray, *, goal: str, magnitude: bool, target: float | None) -> np.ndarray:
    """Project signed values onto the quantity the objective optimizes.

    Returns a new array; `y_raw` is never modified.

    Raises:
        ValueError: If the goal is ``target`` and no target is given.
    """
    values = np.asarray(y_raw, dtype=float)
    if goal == Goal.TARGET.value:
        if target is None:
            raise ValueError("a target objective needs a target value")
        return np.asarray(np.abs(values - float(target)), dtype=float)
    if magnitude:
        return np.asarray(np.abs(values), dtype=float)
    return values.copy()


def require_continuous_design(definition: ExperimentDefinition) -> tuple[VariableDefinition, ...]:
    """The design variables, once every one of them is known to be continuous.

    Raises:
        UnsupportedDesignSpace: Naming each design variable that is not.
    """
    design = definition.design_variables()
    refused = [
        f"design variable {v.name!r} is {v.kind.value}: {_REFUSALS[v.kind]}"
        for v in design
        if v.kind is not VariableKind.CONTINUOUS
    ]
    if refused:
        raise UnsupportedDesignSpace(
            "This design space can be described and recorded, but not optimized. "
            + "; ".join(refused)
            + "."
        )
    return design


def prepare_optimization(
    definition: ExperimentDefinition, observation_set: ObservationSet
) -> PreparedOptimization:
    """Build the optimizer's inputs from the reduced conditions.

    Raises:
        UnsupportedDesignSpace: If any design variable is not continuous.
        ValueError: If `observation_set` was built under another definition.
    """
    if observation_set.definition_hash != definition_sha256(definition):
        raise ValueError(
            "The observation set was built under a different definition than the one given."
        )
    design = require_continuous_design(definition)
    reduced = observation_set.reduced
    objective = definition.objective
    response = definition.objective_response()

    x = np.asarray([[float(v) for v in condition.x] for condition in reduced], dtype=float).reshape(
        len(reduced), len(design)
    )
    y_raw = np.asarray([condition.y_raw for condition in reduced], dtype=float)
    y = project(
        y_raw, goal=objective.goal.value, magnitude=objective.magnitude, target=objective.target
    )

    if objective.goal is Goal.TARGET:
        label = f"|{response.name} - {objective.target:g}|"
        direction, aggregation = Goal.MINIMIZE.value, AGGREGATION_TARGET
    elif objective.magnitude:
        label = f"abs_{response.name}"
        direction, aggregation = objective.goal.value, AGGREGATION_MAGNITUDE
    else:
        label = response.name
        direction, aggregation = objective.goal.value, AGGREGATION_IDENTITY

    messages: list[str] = []
    sigma, degenerate_note = _fill_sigma(reduced, y)
    if degenerate_note is not None:
        messages.append(degenerate_note)
    noise, noise_source = _prediction_noise(definition, observation_set, sigma)
    sign_note = _sign_change_note(definition, reduced)
    if sign_note is not None:
        messages.append(sign_note)

    # A projected quantity keeps no physical bound in this version; only an
    # unprojected objective inherits the response's valid range.
    unprojected = aggregation == AGGREGATION_IDENTITY
    y_min, y_max = response.valid_range if unprojected else (None, None)

    for array in (x, y, y_raw, sigma):
        if array is not None:
            array.setflags(write=False)
    return PreparedOptimization(
        x=x,
        y=y,
        y_raw=y_raw,
        sigma=sigma,
        sigma_sources=tuple(condition.sigma_source for condition in reduced),
        prediction_noise=noise,
        prediction_noise_source=noise_source,
        bounds=tuple((float(v.bounds[0]), float(v.bounds[1])) for v in design if v.bounds),
        input_names=tuple(v.name for v in design),
        target_name=label,
        direction=direction,
        fit_space=objective.fit_space.value,
        y_min=y_min,
        y_max=y_max,
        objective_aggregation=aggregation,
        n_dims=len(design),
        messages=tuple(messages),
    )


def _fill_sigma(
    reduced: Sequence[ReducedObservation], y: np.ndarray
) -> tuple[np.ndarray | None, str | None]:
    """The per-condition sigma handed to the optimizer, or None, with a note.

    A condition with no estimate is given the largest known sigma in the
    campaign rather than the smallest: treating "unknown" as "excellent" is how
    one under-measured point comes to dominate a surface. When nothing is known
    anywhere, every condition gets the spread of `y`, which is an assumption and
    is labelled as one.
    """
    if not reduced:
        return None, None
    known = [c.sigma for c in reduced if c.sigma is not None]
    fallback = max(known) if known else float(np.std(y)) or 1.0
    sigma = np.asarray([c.sigma if c.sigma is not None else fallback for c in reduced], dtype=float)

    scale = float(np.mean(np.abs(y)))
    degenerate = scale > 0 and float(np.median(sigma)) < _DEGENERATE_SIGMA_FRACTION * scale
    if degenerate or not bool(np.any(sigma > 0)):
        return None, (
            "The median uncertainty was vanishingly small, or none was positive, so this "
            "campaign carries no "
            "measurement-noise estimate and the optimizer's default assumption applies. "
            "Real replicate scatter would not look like this; check that the recorded "
            "values are measurements rather than computed values."
        )
    return sigma, None


def _prediction_noise(
    definition: ExperimentDefinition, observation_set: ObservationSet, sigma: np.ndarray | None
) -> tuple[float | None, str]:
    """The 1-sigma of one future observation, and what that number rests on."""
    if sigma is None:
        return None, PREDICTION_NOISE_NOT_PASSED
    root_m = math.sqrt(definition.specimens_per_condition)
    if observation_set.pooled_sd is not None:
        return observation_set.pooled_sd / root_m, PREDICTION_NOISE_POOLED
    median = float(np.median(sigma)) / root_m
    if any(c.sigma is not None for c in observation_set.reduced):
        return median, PREDICTION_NOISE_LOWER_BOUND
    return median, PREDICTION_NOISE_UNDETERMINED


def _where(conditions: Sequence[ReducedObservation], names: Sequence[str]) -> str:
    return "; ".join(
        ", ".join(f"{name} = {value:g}" for name, value in zip(names, c.x, strict=True))
        for c in conditions
    )


def _sign_change_note(
    definition: ExperimentDefinition, reduced: Sequence[ReducedObservation]
) -> str | None:
    """Say so when a magnitude objective is given values of both signs."""
    if not definition.objective.magnitude:
        return None
    negative = [c for c in reduced if c.y_raw < 0]
    positive = [c for c in reduced if c.y_raw > 0]
    if not negative or not positive:
        return None
    names = [v.name for v in definition.design_variables()]
    response = definition.objective.response
    return (
        f"The signed {response} changes sign across the measured conditions: negative at "
        f"{_where(negative, names)}; positive at {_where(positive, names)}. The objective "
        "optimizes its magnitude, which must fall to zero somewhere between conditions of "
        "opposite sign. The surrogate is fitted to the magnitudes and cannot see that "
        "minimum, so its predictions between those conditions are not reliable. Nothing was "
        "discarded or altered: the signed values are kept in the record."
    )
