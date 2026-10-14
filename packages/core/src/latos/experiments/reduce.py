"""Reduce specimens to conditions: one signed value and one labelled uncertainty.

Pure, and it does no I/O.

**Reduce on raw values, project afterwards.** Specimens made at one condition
are averaged on their *signed* values. Any projection the objective asks for (a
magnitude, a distance to a target) is applied later, once, to the reduced value.
The two orders differ exactly where it matters: two specimens at +a and -b have
a mean magnitude of (a + b)/2 and a magnitude of the mean of |a - b|/2, and only
the second is an estimate of the response at that condition.

**Every uncertainty says what it is.** Each condition's sigma comes from the
first row below that applies:

1. the condition was replicated, and a pooled specimen-to-specimen spread
   exists: pooled / sqrt(n);
2. the condition was made once, and a pooled spread exists: the pooled spread;
3. no pooled spread; made once; the producer declared a CONDITION-scope
   uncertainty: that value;
4. no pooled spread; made once; the producer declared a SPECIMEN-scope
   uncertainty: that value, labelled a lower bound;
5. anything else: no estimate.

A within-specimen uncertainty is never added to the between-specimen spread.
The spread computed from specimen values already contains it on average.

Nothing is invented: there is no prior from literature or from another
campaign. When replicates appear anywhere, the pooled spread takes over every
condition at the next reduction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from scipy.stats import chi2

from latos.core.enums import Severity
from latos.experiments.definition import (
    ExperimentDefinition,
    ResponseDefinition,
    VariableDefinition,
    VariableKind,
)
from latos.experiments.identity import definition_sha256
from latos.experiments.observations import (
    EXCLUDED_OUTSIDE_BOUNDS,
    ConditionIssue,
    ExcludedObservation,
    Observation,
    ObservationSet,
    ReducedObservation,
)
from latos.experiments.values import UncertaintyScope

__all__ = [
    "MIN_POOLED_DF",
    "SIGMA_DECLARED",
    "SIGMA_EMPIRICAL",
    "SIGMA_GLOSSARY",
    "SIGMA_LOWER_BOUND",
    "SIGMA_POOLED",
    "SIGMA_UNDETERMINED",
    "reduce_observations",
]

SIGMA_EMPIRICAL = "empirical_condition"
SIGMA_POOLED = "pooled_condition"
SIGMA_DECLARED = "declared_condition"
SIGMA_LOWER_BOUND = "lower_bound_specimen"
SIGMA_UNDETERMINED = "undetermined"

SIGMA_GLOSSARY: Mapping[str, str] = {
    SIGMA_EMPIRICAL: (
        "The pooled specimen-to-specimen standard deviation divided by the square root of "
        "the number of specimens at this condition. A condition-scope uncertainty."
    ),
    SIGMA_POOLED: (
        "The pooled specimen-to-specimen standard deviation, measured at other conditions "
        "and applied to this one, which was made once. A condition-scope uncertainty."
    ),
    SIGMA_DECLARED: (
        "A condition-scope uncertainty declared by the producer for a condition made once."
    ),
    SIGMA_LOWER_BOUND: (
        "A within-specimen uncertainty of the one specimen at this condition. It leaves out "
        "specimen-to-specimen variation, so it is a lower bound on the condition's uncertainty."
    ),
    SIGMA_UNDETERMINED: (
        "No estimate exists for this condition. A substitute was used so the surrogate could "
        "be fitted; it is an assumption, not a measurement."
    ),
}

# Below this many degrees of freedom the pooled spread is still used, with a
# caveat: a wide honest estimate beats an assumed one.
MIN_POOLED_DF = 3

_NUMERIC_KINDS = (VariableKind.CONTINUOUS, VariableKind.INTEGER)
_MIN_REPLICATES = 2


def reduce_observations(
    definition: ExperimentDefinition,
    observations: Sequence[Observation],
    excluded: Sequence[ExcludedObservation],
) -> ObservationSet:
    """Group specimens into conditions and reduce each one.

    Args:
        definition: The experiment the observations were built under.
        observations: The specimens that entered.
        excluded: The rows that did not; carried through unchanged.

    Returns:
        The `ObservationSet`, with observations ordered by their design values.
    """
    design = definition.design_variables()
    response = definition.objective_response()
    ordered = sorted(observations, key=lambda obs: _design_key(obs, design))
    groups = _group(ordered, design)
    values = [[obs.responses[response.name].value for obs in group] for group in groups]

    pooled_sd, pooled_df, messages = _pool(values, response.unit)
    reduced = tuple(
        _reduce_group(group, vals, design, response.name, pooled_sd)
        for group, vals in zip(groups, values, strict=True)
    )

    collapsed = len(ordered) - len(groups)
    if collapsed > 0:
        messages.append(
            f"{len(ordered)} sample(s) collapsed onto {len(groups)} design point(s); "
            f"{collapsed} were replicates of a condition already present."
        )
    outside = [row.observation for row in excluded if row.category == EXCLUDED_OUTSIDE_BOUNDS]
    kept = [obs for obs in outside if obs is not None]
    if kept:
        messages.append(_outside_note(kept, design, response))
    return ObservationSet(
        definition_hash=definition_sha256(definition),
        observations=tuple(ordered),
        excluded=tuple(excluded),
        reduced=reduced,
        messages=tuple(messages),
        pooled_sd=pooled_sd,
        pooled_df=pooled_df,
    )


def _outside_note(
    kept: Sequence[Observation],
    design: Sequence[VariableDefinition],
    response: ResponseDefinition,
) -> str:
    """Say which complete specimens the design bounds kept out of the fit."""
    listed = []
    for obs in kept:
        axes = [(v.name, _axis_value(obs, v)) for v in design]
        where = ", ".join(
            f"{name} = {at if isinstance(at, str) else format(at, 'g')}" for name, at in axes
        )
        value = obs.responses[response.name].value
        listed.append(f"{obs.sample_id} ({where}; {response.name} = {value:+g} {response.unit})")
    return (
        f"{len(kept)} specimen(s) passed the same checks as every other accepted specimen "
        f"and lie outside the design bounds: {'; '.join(listed)}. They are kept in the record "
        "and are not in the fit; the bounds alone keep them out."
    )


def _axis_value(obs: Observation, variable: VariableDefinition) -> float | str:
    raw = obs.inputs[variable.name]
    if variable.kind in _NUMERIC_KINDS:
        return float(raw)  # type: ignore[arg-type]  # build admitted only finite numbers
    return str(raw)


def _design_key(obs: Observation, design: Sequence[VariableDefinition]) -> tuple[float | str, ...]:
    return tuple(_axis_value(obs, variable) for variable in design)


def _same_condition(a: Observation, b: Observation, design: Sequence[VariableDefinition]) -> bool:
    """Whether two specimens differ by no more than each axis's precision."""
    for variable in design:
        left, right = _axis_value(a, variable), _axis_value(b, variable)
        if isinstance(left, str) or isinstance(right, str) or variable.precision is None:
            if left != right:
                return False
        elif abs(left - right) > variable.precision:
            return False
    return True


def _group(
    ordered: Sequence[Observation], design: Sequence[VariableDefinition]
) -> list[list[Observation]]:
    """Connected components of "same condition" (single linkage).

    On one axis this is the chain along the sorted list. A run of specimens each
    just inside the precision of its neighbour can therefore merge into a group
    wider than the precision. That is accepted: fixed bins would split a genuine
    pair that straddles a bin edge, which is the worse failure.
    """
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if _same_condition(ordered[i], ordered[j], design):
                parent[find(j)] = find(i)

    groups: dict[int, list[Observation]] = {}
    for i, obs in enumerate(ordered):
        groups.setdefault(find(i), []).append(obs)
    return list(groups.values())


def _pool(values: Sequence[Sequence[float]], unit: str) -> tuple[float | None, int, list[str]]:
    """The pooled specimen-to-specimen spread over every replicated condition.

    Two replicates give one degree of freedom, and a variance on one degree of
    freedom is close to worthless. Pooling every replicated condition gives the
    sum of their degrees of freedom for no extra experiments.
    """
    ssq = 0.0
    dof = 0
    replicated = 0
    for group in values:
        if len(group) < _MIN_REPLICATES:
            continue
        mean = math.fsum(group) / len(group)
        ssq += math.fsum((v - mean) ** 2 for v in group)
        dof += len(group) - 1
        replicated += 1
    if dof < 1:
        return None, 0, []

    pooled_sd = math.sqrt(ssq / dof)
    if pooled_sd <= 0.0:
        return (
            None,
            0,
            [
                "Replicates at the same condition gave identical values to full precision. "
                "That is not agreement, it is a sign the recorded numbers are computed "
                "rather than measured. The replicate estimate was discarded."
            ],
        )
    detail = (
        f"Replicate scatter measured at {replicated} condition(s): pooled standard "
        f"deviation {pooled_sd:.4f} {unit} on {dof} degree(s) of freedom. It is a "
        f"specimen-to-specimen spread. Every condition's uncertainty handed to the optimizer "
        f"is taken from it, in place of any declared value, and divided by the square root "
        f"of the declared number of specimens per condition it is the prediction noise, in "
        f"place of the engine's assumed default."
    )
    if dof < MIN_POOLED_DF:
        # A chi-square interval for a standard deviation, assuming normal scatter.
        low = math.sqrt(dof / float(chi2.ppf(0.975, dof)))
        high = math.sqrt(dof / float(chi2.ppf(0.025, dof)))
        detail += (
            f" With only {dof} degree(s) of freedom this estimate is itself very uncertain: "
            f"a 95% confidence interval for the true spread runs from about {low:.2g} to "
            f"{high:.2g} times it (chi-square, assuming normal scatter). Treat it as an order "
            f"of magnitude, and replicate a second condition to sharpen it."
        )
    return pooled_sd, dof, [detail]


def _sigma(
    group: Sequence[Observation], response_name: str, pooled_sd: float | None
) -> tuple[float | None, str]:
    n = len(group)
    if pooled_sd is not None:
        if n >= _MIN_REPLICATES:
            return pooled_sd / math.sqrt(n), SIGMA_EMPIRICAL
        return pooled_sd, SIGMA_POOLED
    if n == 1:
        declared = group[0].responses[response_name]
        if declared.uncertainty is not None:
            if declared.uncertainty_scope is UncertaintyScope.CONDITION:
                return declared.uncertainty, SIGMA_DECLARED
            return declared.uncertainty, SIGMA_LOWER_BOUND
    return None, SIGMA_UNDETERMINED


def _reduce_group(
    group: Sequence[Observation],
    values: Sequence[float],
    design: Sequence[VariableDefinition],
    response_name: str,
    pooled_sd: float | None,
) -> ReducedObservation:
    x: list[float | str] = []
    for variable in design:
        axis = [_axis_value(obs, variable) for obs in group]
        numbers = [v for v in axis if not isinstance(v, str)]
        x.append(math.fsum(numbers) / len(numbers) if len(numbers) == len(axis) else axis[0])

    sigma, source = _sigma(group, response_name, pooled_sd)
    issues: list[ConditionIssue] = []
    # Zeros carry no sign. No threshold is applied: any strict disagreement is
    # reported, and the condition is still reduced by its signed mean.
    if len(group) >= _MIN_REPLICATES and any(v > 0 for v in values) and any(v < 0 for v in values):
        listed = ", ".join(
            f"{obs.sample_id} = {value:+g}" for obs, value in zip(group, values, strict=True)
        )
        issues.append(
            ConditionIssue(
                code="sign_instability",
                severity=Severity.WARNING,
                message=(
                    f"Specimens at this condition disagree in the sign of {response_name}: "
                    f"{listed}. They were reduced by their signed mean and nothing was "
                    "discarded, but the condition is not settled."
                ),
            )
        )
    return ReducedObservation(
        x=tuple(x),
        y_raw=math.fsum(values) / len(values),
        sigma=sigma,
        sigma_source=source,
        n_specimens=len(group),
        sample_ids=tuple(obs.sample_id for obs in group),
        issues=tuple(issues),
    )
