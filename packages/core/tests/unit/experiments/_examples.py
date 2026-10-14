"""Neutral example definitions and tables for the generic-layer tests.

Nothing here belongs to any real technique: the generic layer must work for an
arbitrary experiment, so its tests are written against an invented one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from latos.core.models import ValidationIssue
from latos.experiments import (
    ExperimentDefinition,
    Goal,
    ObjectiveDefinition,
    ObservationSet,
    Record,
    RecordTable,
    ResponseDefinition,
    UncertaintyScope,
    VariableDefinition,
    VariableKind,
    VariableRole,
    build_observations,
    reduce_observations,
)

PRODUCER = "bench-table/1"

X = VariableDefinition(
    name="additive_fraction",
    role=VariableRole.DESIGN,
    kind=VariableKind.CONTINUOUS,
    unit="1",
    column="additive_fraction",
    bounds=(0.0, 1.0),
    precision=0.005,
)
BAKE = VariableDefinition(
    name="bake_time_min",
    role=VariableRole.CONTROL,
    kind=VariableKind.CONTINUOUS,
    unit="min",
    column="bake_time_min",
    expected_value=60.0,
    tolerance=2.0,
)
ATMOSPHERE = VariableDefinition(
    name="atmosphere",
    role=VariableRole.CONTROL,
    kind=VariableKind.CATEGORICAL,
    unit="1",
    column="atmosphere",
    expected_value="Ar",
)
PASSES = VariableDefinition(
    name="passes",
    role=VariableRole.CONTROL,
    kind=VariableKind.INTEGER,
    unit="1",
    column="passes",
    expected_value=3,
)
GLOVEBOX = VariableDefinition(
    name="glovebox",
    role=VariableRole.CONTROL,
    kind=VariableKind.CATEGORICAL,
    unit="1",
    column=None,
    expected_value="yes",
)
HUMIDITY = VariableDefinition(
    name="humidity_pct",
    role=VariableRole.ENVIRONMENT,
    kind=VariableKind.CONTINUOUS,
    unit="%",
    column="humidity_pct",
)
BATCH = VariableDefinition(
    name="batch",
    role=VariableRole.METADATA,
    kind=VariableKind.CATEGORICAL,
    unit="1",
    column="batch",
)
SIGNAL = ResponseDefinition(
    name="signal",
    unit="a.u.",
    column="signal",
    uncertainty_column="signal_u",
    uncertainty_scope=UncertaintyScope.SPECIMEN,
)
SIDE = ResponseDefinition(
    name="side_signal", unit="a.u.", column="side_signal", valid_range=(0.0, None)
)

VARIABLES = (X, BAKE, ATMOSPHERE, PASSES, GLOVEBOX, HUMIDITY, BATCH)
RESPONSES = (SIGNAL, SIDE)
COLUMNS = (
    "sample_id",
    "additive_fraction",
    "bake_time_min",
    "atmosphere",
    "passes",
    "humidity_pct",
    "batch",
    "signal",
    "signal_u",
    "side_signal",
)


def make_definition(
    *,
    variables: tuple[VariableDefinition, ...] = VARIABLES,
    responses: tuple[ResponseDefinition, ...] = RESPONSES,
    objective: ObjectiveDefinition | None = None,
    **fields: Any,
) -> ExperimentDefinition:
    """The 1-D example: maximize the magnitude of a signed signal."""
    settings: dict[str, Any] = {
        "name": "example-campaign",
        "version": "1.0.0",
        "title": "An invented campaign",
        "producer": PRODUCER,
        "sample_key": "sample_id",
    }
    settings.update(fields)
    return ExperimentDefinition(
        variables=variables,
        responses=responses,
        objective=objective
        or ObjectiveDefinition(response="signal", goal=Goal.MAXIMIZE, magnitude=True),
        **settings,
    )


def make_row(
    sample_id: str | None,
    x: Any,
    y: Any,
    u: Any = None,
    *,
    issues: tuple[ValidationIssue, ...] = (),
    detail: Mapping[str, Any] | None = None,
    **extra: Any,
) -> Record:
    """One specimen of the 1-D example, with every control at its expected value."""
    values: dict[str, Any] = {
        "sample_id": sample_id,
        "additive_fraction": x,
        "bake_time_min": 60.0,
        "atmosphere": "Ar",
        "passes": 3,
        "humidity_pct": 40.0,
        "batch": "b1",
        "signal": y,
        "signal_u": u,
        "side_signal": 1.0,
    }
    values.update(extra)
    return Record(
        record_id=str(sample_id or "unnamed-row"),
        values=values,
        provenance={"signal": f"{PRODUCER}: invented value"},
        issues=issues,
        producer_detail=detail,
    )


def make_table(
    *rows: Record, producer: str = PRODUCER, columns: tuple[str, ...] = COLUMNS
) -> RecordTable:
    return RecordTable(producer=producer, columns=columns, rows=tuple(rows))


def reduced_set(definition: ExperimentDefinition, *rows: Record) -> ObservationSet:
    """Build and reduce `rows` under `definition`."""
    observations, excluded = build_observations(definition, make_table(*rows))
    return reduce_observations(definition, observations, excluded)


def make_nd_definition(n_dims: int, **fields: Any) -> ExperimentDefinition:
    """A d-variable example: maximize an unsigned property."""
    variables = tuple(
        VariableDefinition(
            name=f"x{i}",
            role=VariableRole.DESIGN,
            kind=VariableKind.CONTINUOUS,
            unit="1",
            column=f"x{i}",
            bounds=(0.0, 10.0),
            precision=0.1,
        )
        for i in range(n_dims)
    )
    settings: dict[str, Any] = {
        "name": f"example-{n_dims}d",
        "version": "1.0.0",
        "title": f"An invented {n_dims}-variable campaign",
        "producer": PRODUCER,
        "sample_key": "sample_id",
    }
    settings.update(fields)
    return ExperimentDefinition(
        variables=variables,
        responses=(
            ResponseDefinition(
                name="yield_pct",
                unit="%",
                column="yield_pct",
                uncertainty_column="yield_u",
                uncertainty_scope=UncertaintyScope.SPECIMEN,
            ),
        ),
        objective=ObjectiveDefinition(response="yield_pct", goal=Goal.MAXIMIZE),
        **settings,
    )


def make_nd_table(n_dims: int, points: list[tuple[tuple[float, ...], float]]) -> RecordTable:
    """A table for `make_nd_definition`, one specimen per (x, y) point."""
    columns = ("sample_id", *(f"x{i}" for i in range(n_dims)), "yield_pct", "yield_u")
    rows = []
    for index, (x, y) in enumerate(points):
        values: dict[str, Any] = {"sample_id": f"s{index}", "yield_pct": y, "yield_u": 0.5}
        values.update({f"x{i}": x[i] for i in range(n_dims)})
        rows.append(Record(record_id=f"s{index}", values=values))
    return RecordTable(producer=PRODUCER, columns=columns, rows=tuple(rows))
