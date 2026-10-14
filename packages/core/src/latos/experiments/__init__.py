"""Experiment definitions and the path from a record table to the optimizer.

One definition system and one pipeline for every experiment::

    record table (one row per specimen)  +  ExperimentDefinition
        -> build_observations     each row checked; exclusions kept, labelled, with reasons
        -> reduce_observations    specimens reduced to conditions, on signed values
        -> prepare_optimization   the objective projected; uncertainties labelled
        -> run_optimization       one design variable -> optimize(); more -> optimize_nd()
        -> freeze_v2              the recommendation committed before the specimen exists

The modules here are experiment-agnostic. They know nothing of any technique,
instrument or material: a producer outside this layer turns one pipeline's data
into a record table, and the definition names the columns to read. Nothing here
imports `latos.ingestion` or `latos.analysis`; a test enforces that.
"""

from __future__ import annotations

from latos.experiments.build import build_observations
from latos.experiments.definition import (
    ExperimentDefinition,
    FitSpace,
    Goal,
    ObjectiveDefinition,
    ResponseDefinition,
    VariableDefinition,
    VariableKind,
    VariableRole,
)
from latos.experiments.identity import (
    canonical_definition,
    definition_from_dict,
    definition_sha256,
    definition_to_dict,
)
from latos.experiments.observations import (
    EXCLUDED_OUTSIDE_BOUNDS,
    EXCLUDED_REJECTED,
    EXCLUSION_GLOSSARY,
    ConditionIssue,
    ExcludedObservation,
    Observation,
    ObservationSet,
    ReducedObservation,
)
from latos.experiments.prepare import PreparedOptimization, prepare_optimization
from latos.experiments.records import Record, RecordTable, producer_detail_sha256
from latos.experiments.reduce import reduce_observations
from latos.experiments.revision import SourceRevision, runtime_environment, source_revision
from latos.experiments.run import min_conditions, run_optimization
from latos.experiments.validation import (
    DefinitionError,
    InsufficientConditions,
    TableMismatch,
    UnsupportedDesignSpace,
    definition_warnings,
    validate_definition,
)
from latos.experiments.values import MeasurementValue, UncertaintyScope

__all__ = [
    # Definition
    "ExperimentDefinition",
    "VariableDefinition",
    "ResponseDefinition",
    "ObjectiveDefinition",
    "VariableRole",
    "VariableKind",
    "Goal",
    "FitSpace",
    "validate_definition",
    "definition_warnings",
    # Identity
    "canonical_definition",
    "definition_sha256",
    "definition_to_dict",
    "definition_from_dict",
    # Values and records
    "MeasurementValue",
    "UncertaintyScope",
    "Record",
    "RecordTable",
    "producer_detail_sha256",
    # Observations
    "Observation",
    "ExcludedObservation",
    "EXCLUDED_REJECTED",
    "EXCLUDED_OUTSIDE_BOUNDS",
    "EXCLUSION_GLOSSARY",
    "ConditionIssue",
    "ReducedObservation",
    "ObservationSet",
    # Source revision
    "SourceRevision",
    "runtime_environment",
    "source_revision",
    # Pipeline
    "build_observations",
    "reduce_observations",
    "PreparedOptimization",
    "prepare_optimization",
    "run_optimization",
    "min_conditions",
    # Errors
    "DefinitionError",
    "TableMismatch",
    "UnsupportedDesignSpace",
    "InsufficientConditions",
]
