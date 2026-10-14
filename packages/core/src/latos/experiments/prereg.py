"""Freeze a recommendation, with everything needed to check it later.

A recommendation is prospective only if the data, the declared experiment, the
model and the prediction are committed **before** the recommended specimen
exists. `freeze_v2` writes that commitment as one JSON record and a readable
note beside it. `verify_record` recomputes what the record says about itself.

What a version 2 record holds
-----------------------------
* **The experiment.** The whole definition, inline, with its identity hash and
  the producer contract it reads. The record can be understood, and the hash
  recomputed, without the code that declared the experiment.
* **The observations the model saw.** One row per condition: the design values,
  the signed value `y_raw`, the projected value `y` the optimizer was handed,
  the uncertainty and a label saying what that uncertainty is. A digest covers
  all of them. `y_raw` is in the digest because under a magnitude objective
  flipping its sign leaves `y` unchanged, and nothing else would notice.
* **Every specimen behind those rows**, and every row that was left out, with
  the reasons. Each carries its producer's own detail and a digest of it. This
  module stores that detail and never reads inside it. A row left out is
  labelled: either it was rejected, or it is a complete specimen that the
  design bounds do not cover, which is kept with its values and labelled
  uncertainties like any specimen in the fit.
* **The model and the prediction**: the optimizer's frozen configuration, the
  recommended condition, the predicted value and both intervals, what the
  acquisition saw, the stopping verdict and the reliability grade.
* **The source revision**: the Git commit of the code that computed all of
  this, and whether the working tree was exactly that commit. With the software
  version, the definition hash and the producer contract, it says what would
  have to be checked out to compute the record again. A record frozen from a
  tree that was not clean says so; it is not refused here, because whether that
  is acceptable is the caller's decision.
* **A seal** over the whole record.

One-variable records are built by calling the version 1 writer and adding to
its output, so every key the existing screens read is written by the one writer
that has always written it. Records over several variables build the same keys
from the multi-variable result.

What verification can and cannot show
-------------------------------------
`verify_record` recomputes the observation digest, the definition hash, `y`
from `y_raw` and the objective the inline definition declares, each
condition's value and design point from its specimens, each producer-detail
digest, and the seal. For a one-variable record it also takes the prediction,
both intervals, the expected improvement at every grid point, the choice of
the recommended point and the stopping verdict back from the frozen model, and
it checks that the source revision is stated and coherent. An edit to any
part of the file changes at least one of them.

It cannot, by itself, show that a file was not rewritten wholesale: whoever
rewrites a record can reseal it. That needs the seal to exist somewhere the
file's editor cannot reach. `freeze_v2` returns the path of a record whose seal
is short enough to write in a lab notebook or send to a colleague at the moment
of freezing, and `verify_record` accepts that noted seal to compare against.

Write-once, and all or nothing
------------------------------
A record is never replaced. Freezing the same experiment name and version with
a different definition is refused (`VersionCollision`): a version states what
was declared, and two declarations cannot share one.

A freeze either leaves a complete record under its final name or leaves no
file that can be taken for one. Both files are first written in full under
names that start with a dot and end in ``.partial``, which no reader takes for
a record, and the JSON is read back and compared with what was meant before
anything is published. Then the note is given its final name, and last the
record itself: that one step is the freeze. Neither step can replace a file
that is already there. When anything fails before the record is published,
whatever this attempt published or half-wrote is removed again. A process that
is killed between the steps can still leave such files; `interrupted_freeze_files`
finds them, so the campaign command can stop and say what they are.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from latos import __version__
from latos.experiments._canonical import canonical_json, sha256_hex
from latos.experiments.definition import ExperimentDefinition, Goal, VariableRole
from latos.experiments.identity import (
    CANONICAL_FORM,
    definition_from_dict,
    definition_sha256,
    definition_to_dict,
)
from latos.experiments.observations import (
    EXCLUDED_OUTSIDE_BOUNDS,
    EXCLUSION_GLOSSARY,
    ExcludedObservation,
    Observation,
    ObservationSet,
)
from latos.experiments.prepare import (
    PREDICTION_NOISE_GLOSSARY,
    PREDICTION_NOISE_LOWER_BOUND,
    PREDICTION_NOISE_NOT_PASSED,
    PREDICTION_NOISE_POOLED,
    PREDICTION_NOISE_UNDETERMINED,
    PreparedOptimization,
    project,
)
from latos.experiments.records import producer_detail_sha256
from latos.experiments.reduce import MIN_POOLED_DF, SIGMA_GLOSSARY, SIGMA_LOWER_BOUND
from latos.experiments.revision import (
    _MAX_LISTED_PATHS as _LISTED,
)
from latos.experiments.revision import (
    WORKING_TREE_CLEAN,
    WORKING_TREE_DIRTY,
    WORKING_TREE_STATES,
    WORKING_TREE_UNKNOWN,
    SourceRevision,
    runtime_environment,
    source_revision,
)
from latos.experiments.validation import definition_warnings
from latos.experiments.values import UncertaintyScope
from latos.optimization import prereg as _v1
from latos.optimization.acquisitions import expected_improvement
from latos.optimization.engine import (
    _CI95,
    _DEFAULT_EXPLORE_POLICY,
    _RELIABILITY_CALIBRATED_N,
    _RELIABILITY_INDICATIVE_N,
    OptimizationResult,
    OptimizationResultND,
)

__all__ = [
    "OBSERVATION_FORM",
    "PARTIAL_SUFFIX",
    "SCHEMA_VERSION",
    "SEAL_FORM",
    "RecordError",
    "VersionCollision",
    "build_record_v2",
    "freeze_v2",
    "interrupted_freeze_files",
    "observations_digest_nd",
    "read_record",
    "record_seal",
    "render_note",
    "verify_record",
]

SCHEMA_VERSION = 2
RECORD_KIND = "latos.bo.prereg"

OBSERVATION_FORM = (
    "latos.obs.nd.v1: rows of x0..x(d-1), y, y_raw[, sigma] as Python float repr, "
    "tab-separated, sorted ascending, newline-joined, UTF-8, SHA-256"
)
SEAL_FORM = (
    "latos.record.seal.v1: SHA-256 of the record without its seal, as JSON with sorted "
    "keys, separators (',', ':'), no ASCII escaping, UTF-8"
)

BACKEND_1D = "optimize"
BACKEND_ND = "optimize_nd"

# What `read_record` insists on. A record without these cannot be scored
# against a later measurement.
_REQUIRED = (
    ("observations", "y_raw"),
    ("projection",),
    ("experiment", "definition"),
    ("prediction_at_recommendation", "condition"),
)

# As in the version 1 writer: bounds the search for an unused file name.
_MAX_NAME_ATTEMPTS = 1000

# A condition's value is the mean of its specimens. Recomputed from the record
# it may differ in the last bit from the reducer's own arithmetic.
_MEAN_REL = 1e-12
_MEAN_ABS = 1e-12

# Numbers a record states that follow from other numbers it states, compared
# after the optimizer's own arithmetic and a round trip through JSON. Far
# tighter than anything a reader of the record could act on.
_DERIVED_REL = 1e-9
_DERIVED_ABS = 1e-12

# The acquisition and the reliability grade the selection rule turns on.
_ACQUISITION_EI = "ei"
_EXPLORATORY = "exploratory"
_FALLBACK_LARGEST_SD = "max_std"

# How the optimizer's verdict sentence opens for each action.
_VERDICT_OPENINGS = {
    "stop": "Stop.",
    "confirm": "Confirm before stopping.",
    "continue": "Keep going.",
}

# Expected improvement recomputed from a band taken back from JSON: the
# standard deviation is recovered to about 1e-15 of the mean, which EI then
# carries through a normal CDF. Far tighter than any doubling or rescaling.
_EI_REL = 1e-6
_EI_ABS = 1e-9

# Prediction noise taken from measurements: a pooled spread, or the median of
# within-specimen uncertainties (a lower bound, but measured).
_MEASURED_NOISE_SOURCES = frozenset({PREDICTION_NOISE_POOLED, PREDICTION_NOISE_LOWER_BOUND})


class RecordError(ValueError):
    """A file is not a version 2 record, or lacks something one must have."""


class VersionCollision(ValueError):  # noqa: N818 — the agreed public name
    """An experiment name and version were frozen before under another definition."""


# ─── Digests ────────────────────────────────────────────────────────
def observations_digest_nd(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    y_raw: Sequence[float],
    *,
    sigma: Sequence[float] | None = None,
) -> str:
    """SHA-256 of the observations a model was fitted to, in form `OBSERVATION_FORM`.

    Rows are sorted, so the same conditions listed in another order give the
    same digest. `sigma` is included exactly when per-condition uncertainties
    were handed to the optimizer.

    Raises:
        ValueError: If the columns are not all the same length.
    """
    n = len(y)
    if len(x) != n or len(y_raw) != n or (sigma is not None and len(sigma) != n):
        raise ValueError("x, y, y_raw and sigma must all have one entry per condition")
    rows: list[tuple[float, ...]] = []
    for index in range(n):
        row = [float(v) for v in x[index]] + [float(y[index]), float(y_raw[index])]
        if sigma is not None:
            row.append(float(sigma[index]))
        rows.append(tuple(row))
    text = "\n".join("\t".join(repr(v) for v in row) for row in sorted(rows))
    return sha256_hex(text)


def record_seal(record: Mapping[str, Any]) -> str:
    """SHA-256 over the whole record apart from its own seal, in form `SEAL_FORM`."""
    body = {key: value for key, value in record.items() if key != "seal"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_hex(text)


# ─── Building ───────────────────────────────────────────────────────
def _issues(observation: Observation | ExcludedObservation) -> list[dict[str, str]]:
    source = (
        observation.issues if isinstance(observation, Observation) else observation.record.issues
    )
    return [
        {"field": issue.field, "severity": issue.severity.value, "message": issue.message}
        for issue in source
    ]


def _plain(detail: Mapping[str, Any] | None) -> Any:
    """Producer detail as plain JSON data, exactly as its digest sees it."""
    return None if detail is None else json.loads(canonical_json(detail))


def _specimen(observation: Observation) -> dict[str, Any]:
    return {
        "sample_id": observation.sample_id,
        "inputs": dict(observation.inputs),
        "metadata": dict(observation.metadata),
        "responses": {
            name: {
                "value": value.value,
                "unit": value.unit,
                "uncertainty": value.uncertainty,
                "uncertainty_scope": (
                    None if value.uncertainty_scope is None else value.uncertainty_scope.value
                ),
                "provenance": value.provenance,
            }
            for name, value in observation.responses.items()
        },
        "issues": _issues(observation),
        "producer_detail": _plain(observation.producer_detail),
        "producer_detail_sha256": producer_detail_sha256(observation.producer_detail),
    }


def _excluded(row: ExcludedObservation) -> dict[str, Any]:
    out: dict[str, Any] = {
        "record_id": row.record.record_id,
        "sample_id": row.sample_id,
        "category": row.category,
        "reasons": list(row.reasons),
        "values": dict(row.record.values),
        "issues": _issues(row),
        "producer_detail": _plain(row.record.producer_detail),
        "producer_detail_sha256": producer_detail_sha256(row.record.producer_detail),
    }
    if row.observation is not None:
        # A complete specimen the design bounds do not cover: kept as a
        # specimen is kept, with its labelled uncertainty.
        whole = _specimen(row.observation)
        out.update({key: whole[key] for key in ("inputs", "metadata", "responses", "issues")})
    return out


def _experiment_block(definition: ExperimentDefinition) -> dict[str, Any]:
    return {
        "name": definition.name,
        "version": definition.version,
        "title": definition.title,
        "producer": definition.producer,
        "definition_sha256": definition_sha256(definition),
        "definition_canonical_form": CANONICAL_FORM,
        "definition": definition_to_dict(definition),
        "warnings": list(definition_warnings(definition)),
        "controls": [
            {
                "name": variable.name,
                "expected_value": variable.expected_value,
                "tolerance": variable.tolerance,
                "unit": variable.unit,
                # A control with no column is declared, not checked.
                "verified": variable.column is not None,
            }
            for variable in definition.variables
            if variable.role is VariableRole.CONTROL
        ],
    }


def _observation_block(
    observation_set: ObservationSet, prepared: PreparedOptimization
) -> dict[str, Any]:
    x = prepared.x.tolist()
    y = prepared.y.tolist()
    y_raw = prepared.y_raw.tolist()
    sigma = None if prepared.sigma is None else prepared.sigma.tolist()
    reduced = observation_set.reduced
    return {
        "canonical_form": OBSERVATION_FORM,
        "sha256": observations_digest_nd(x, y, y_raw, sigma=sigma),
        "input_names": list(prepared.input_names),
        "x": x,
        "y": y,
        "y_raw": y_raw,
        "sigma": sigma,
        "sigma_sources": list(prepared.sigma_sources),
        "sigma_glossary": {
            label: SIGMA_GLOSSARY[label] for label in sorted(set(prepared.sigma_sources))
        },
        "n_specimens": [condition.n_specimens for condition in reduced],
        "sample_ids": [list(condition.sample_ids) for condition in reduced],
        "condition_issues": [
            [
                {"code": issue.code, "severity": issue.severity.value, "message": issue.message}
                for issue in condition.issues
            ]
            for condition in reduced
        ],
        "pooled_sd": observation_set.pooled_sd,
        "pooled_df": observation_set.pooled_df,
        "min_pooled_df": MIN_POOLED_DF,
    }


def _nd_base_record(result: OptimizationResultND, prior_best: float) -> dict[str, Any]:
    """The version 1 keys, built from a multi-variable result.

    Three one-variable keys have no meaning here and are left out: the scalar
    recommended x, the single input variable, and the version 1 training block.
    """
    config = result.config
    recommendation = result.recommendation
    frozen = asdict(config)
    del frozen["created_at"]
    low, high = recommendation.predictive_interval_95
    record: dict[str, Any] = {
        "kind": RECORD_KIND,
        "created_at": config.created_at.isoformat(),
        "latos_version": __version__,
        "objective": {
            "property": config.objective,
            "direction": config.direction,
            "y_transform": config.y_transform,
            "aggregation": config.objective_aggregation,
            "search_bounds": [list(pair) for pair in config.bounds],
        },
        "frozen_config": frozen,
        "prediction_at_recommendation": {
            "predicted_mean": recommendation.predicted_mean,
            "ci95_model": recommendation.ci95,
            "predictive_sd": recommendation.predictive_sd,
            "ci95_predictive": recommendation.ci95_predictive,
            "predictive_interval_95": [low, high],
        },
        "prior_best": prior_best,
        "converged": result.converged,
        "stopping_claim": {
            "epsilon": result.epsilon,
            "delta": result.delta,
            "probability_within_epsilon": result.prob_within_epsilon,
            "met": result.epsilon_delta_met,
            "best_measured": result.best_y,
            "n_unreliable_observations": result.n_unreliable,
        },
        "validation_criteria": {
            "calibration": "measured value falls within predictive_interval_95",
            "improvement": "measured value exceeds prior_best",
            "stopping_claim": (
                "the best value found over the campaign improves on the frozen "
                "best_measured by no more than epsilon"
            ),
        },
    }
    if result.reliability is not None:
        record["reliability"] = asdict(result.reliability)
    return record


def _check_same_data(
    prepared: PreparedOptimization, result: OptimizationResult | OptimizationResultND
) -> None:
    """Refuse a result that was not computed from the prepared observations."""

    def rows(points: Any, values: Any) -> list[tuple[float, ...]]:
        return sorted(
            (*(float(v) for v in np.ravel(point)), float(value))
            for point, value in zip(points, values, strict=True)
        )

    seen = rows(result.observed_x, result.observed_y)
    if seen != rows(prepared.x.tolist(), prepared.y.tolist()):
        raise ValueError(
            "The optimization result was not computed from these prepared observations, "
            "so freezing them together would record data the model never saw."
        )


def build_record_v2(
    definition: ExperimentDefinition,
    observation_set: ObservationSet,
    prepared: PreparedOptimization,
    result: OptimizationResult | OptimizationResultND,
    *,
    prior_best: float | None = None,
    source: SourceRevision | None = None,
) -> dict[str, Any]:
    """Assemble a version 2 record as plain JSON data, sealed.

    Args:
        definition: The experiment the observations were built under.
        observation_set: The output of `reduce_observations`.
        prepared: The output of `prepare_optimization` for that set.
        result: What `run_optimization` returned for `prepared`.
        prior_best: The baseline an outcome must beat, in the optimized
            quantity. Defaults to the optimizer's own `best_y`: the best value
            among the fitted conditions, which leaves out specimens kept
            outside the search bounds.
        source: The revision of the source that computed `result`. Asked of
            Git now when not given; give it when it was already read, so the
            record states the revision that was checked and not a later one.

    Raises:
        ValueError: If the four arguments do not belong together.
    """
    if observation_set.definition_hash != definition_sha256(definition):
        raise ValueError("The observation set was built under a different definition.")
    reduced_raw = [condition.y_raw for condition in observation_set.reduced]
    if prepared.y_raw.tolist() != reduced_raw:
        raise ValueError("The prepared arrays were not made from this observation set.")
    _check_same_data(prepared, result)

    best = result.best_y if prior_best is None else prior_best
    one_variable = isinstance(result, OptimizationResult)
    if isinstance(result, OptimizationResult):
        record = _v1.build_record(result, prior_best=best)
    else:
        record = _nd_base_record(result, best)

    objective = definition.objective
    response = definition.objective_response()
    config = result.config
    recommended = result.recommendation.x
    point = [recommended] if isinstance(recommended, float) else list(recommended)

    record["schema_version"] = SCHEMA_VERSION
    record["backend"] = BACKEND_1D if one_variable else BACKEND_ND
    record["experiment"] = _experiment_block(definition)
    record["objective"]["input_variables"] = list(prepared.input_names)
    record["projection"] = {
        "response": objective.response,
        "goal": objective.goal.value,
        "magnitude": objective.magnitude,
        "target": objective.target,
        "fit_space_declared": objective.fit_space.value,
        "is_transformed": config.y_transform != "identity",
        "y_min": prepared.y_min,
        "y_max": prepared.y_max,
        "response_unit": response.unit,
    }
    record["observations"] = _observation_block(observation_set, prepared)
    record["specimens"] = [_specimen(o) for o in observation_set.observations]
    record["excluded"] = [_excluded(row) for row in observation_set.excluded]
    record["exclusion_glossary"] = {
        label: EXCLUSION_GLOSSARY[label]
        for label in sorted({row.category for row in observation_set.excluded})
    }
    record["source_revision"] = (source_revision() if source is None else source).to_dict()
    record["environment"] = runtime_environment()
    record["prediction_noise"] = {
        "prepared": prepared.prediction_noise,
        "applied": config.noise_std,
        "source": prepared.prediction_noise_source,
        "meaning": PREDICTION_NOISE_GLOSSARY[prepared.prediction_noise_source],
        "specimens_per_condition": definition.specimens_per_condition,
    }
    record["prediction_at_recommendation"]["condition"] = dict(
        zip(prepared.input_names, point, strict=True)
    )
    record["acquisition"] = _acquisition_block(result, prepared)
    if result.stopping is not None:
        record["stopping_verdict"] = asdict(result.stopping)
    messages = [*observation_set.messages, *prepared.messages]
    if isinstance(result, OptimizationResultND):
        messages += result.axis_warnings
    record["messages"] = messages
    # Rules for scoring the outcome later. Nothing scores a version 2 record
    # yet, and the criteria say so rather than describe a check as made.
    record["validation_criteria"]["unit_check"] = (
        f"when an outcome is scored, the signed {response.name} in {response.unit} is "
        "projected as the frozen objective declares before it is compared with the "
        "prediction; no code applies this rule yet"
    )
    record["validation_criteria"]["condition_check"] = (
        "when an outcome is scored, the specimen made counts as the recommended condition "
        "only when every design variable is within its declared precision of the recommended "
        "value, or equal to it where no precision is declared; no code applies this rule yet"
    )

    # Through JSON once, so what is sealed is exactly what will be read back.
    plain: dict[str, Any] = json.loads(json.dumps(record, ensure_ascii=False))
    plain["seal"] = {"canonical_form": SEAL_FORM, "sha256": record_seal(plain)}
    return plain


def _acquisition_block(
    result: OptimizationResult | OptimizationResultND, prepared: PreparedOptimization
) -> dict[str, Any]:
    """What the acquisition saw, as far as the optimizer reports it."""
    block: dict[str, Any] = {
        "max_expected_improvement": result.max_ei,
        "noise_threshold": result.noise_threshold,
        # The optimizer counts any noise it is handed as measured. A noise made
        # up where no uncertainty was known is an assumption, and is not.
        "noise_threshold_from_measured_noise": (
            result.noise_measured and prepared.prediction_noise_source in _MEASURED_NOISE_SOURCES
        ),
        "xi": result.config.xi,
        "xi_absolute": result.config.xi_absolute,
    }
    if isinstance(result, OptimizationResult):
        block["function"] = result.config.acquisition_function
        # The posterior and the acquisition over the search range, as the
        # optimizer evaluated them: what the model believed when it chose.
        block["grid"] = {
            "x": list(result.grid_x),
            "mean": list(result.grid_mean),
            "lower_95": list(result.grid_lower),
            "upper_95": list(result.grid_upper),
            "expected_improvement": list(result.grid_ei),
        }
    else:
        block["maximised_by"] = result.config.acquisition
        block["n_candidates"] = result.config.n_candidates
    return block


# ─── Writing ────────────────────────────────────────────────────────
def _check_collision(directory: Path, record: Mapping[str, Any]) -> None:
    experiment = record["experiment"]
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("prereg_*.json")):
        if ".outcome." in path.name:
            continue
        try:
            other = json.loads(path.read_text(encoding="utf-8"))
            earlier = other["experiment"]
            same = (
                other.get("schema_version") == SCHEMA_VERSION
                and earlier["name"] == experiment["name"]
                and earlier["version"] == experiment["version"]
            )
            earlier_hash = earlier["definition_sha256"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            # Unreadable files and version 1 records carry no definition to
            # collide with, and one damaged file must not block a freeze.
            continue
        if same and earlier_hash != experiment["definition_sha256"]:
            raise VersionCollision(
                f"{experiment['name']} {experiment['version']} was frozen earlier with a "
                f"different definition ({path.name}: {earlier_hash}; now: "
                f"{experiment['definition_sha256']}). Bump the version or revert the definition."
            )


PARTIAL_SUFFIX = ".partial"


def _partial(final: Path) -> Path:
    """Where `final` is written before it is published. No reader takes it for a record."""
    return final.with_name(f".{final.name}{PARTIAL_SUFFIX}")


def interrupted_freeze_files(directory: Path) -> tuple[Path, ...]:
    """Files in `directory` that a freeze which did not complete can leave behind.

    Each is either a file still under its ``.partial`` name, or a note whose
    record was never published beside it. None of them is a record, and what
    they state was never confirmed as frozen.

    Raises:
        OSError: If `directory` exists and cannot be listed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return ()
    found = []
    for path in sorted(directory.iterdir()):
        unfinished = (
            path.name.startswith(".prereg_")
            and path.name.endswith(PARTIAL_SUFFIX)
            # A partial that could not be removed after its file was published
            # is that published file under a second name, and states nothing new.
            and not _same_file(path, path.with_name(path.name[1 : -len(PARTIAL_SUFFIX)]))
        )
        orphan_note = (
            path.name.startswith("prereg_")
            and path.suffix == ".md"
            and ".outcome." not in path.name
            and not path.with_suffix(".json").exists()
        )
        if unfinished or orphan_note:
            found.append(path)
    return tuple(found)


def _unused_path(directory: Path, stamp: str) -> Path:
    """`prereg_<stamp>.json`, suffixed until neither it, its note nor their partials exist."""

    def taken(candidate: Path) -> bool:
        names = (candidate, candidate.with_suffix(".md"))
        return any(path.exists() or _partial(path).exists() for path in names)

    candidate = _v1.unused_record_path(directory, stamp)
    suffix = 2
    while taken(candidate):
        candidate = directory / f"prereg_{stamp}_{suffix}.json"
        suffix += 1
    return candidate


def _same_file(a: Path, b: Path) -> bool:
    """Whether `a` and `b` are one file under two names. False when either is missing."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _discard(paths: Sequence[Path]) -> None:
    """Remove files this attempt made, as far as that is possible.

    Whatever cannot be removed is found later by `interrupted_freeze_files`.
    """
    for path in paths:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def _write_once(record: Mapping[str, Any], path: Path) -> Path:
    """Publish the JSON and its note, all or nothing, refusing to replace either.

    Raises:
        FileExistsError: If the JSON or the note is already there.
        RecordError: If what was written does not read back as what was meant.
        OSError: If writing or publishing failed. Nothing is left published.
    """
    note = path.with_suffix(".md")
    text = json.dumps(record, indent=2, ensure_ascii=False)
    note_text = render_note(record)  # made before anything is written
    path.parent.mkdir(parents=True, exist_ok=True)
    for target in (path, note):
        if target.exists():
            raise FileExistsError(
                f"{target} already exists, and a frozen pre-registration is never overwritten."
            )
    partials = (_partial(path), _partial(note))
    for target in partials:
        if target.exists():
            raise FileExistsError(
                f"{target} already exists: a freeze under this name did not complete. It is "
                "left as it is."
            )
    # Only what this attempt itself created is ever removed again.
    made: list[Path] = []
    try:
        for target, content in zip(partials, (text, note_text), strict=True):
            handle = target.open("x", encoding="utf-8")
            made.append(target)
            with handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        if json.loads(partials[0].read_text(encoding="utf-8")) != json.loads(text):
            raise RecordError(f"{partials[0].name} does not read back as the record that was meant")
        # A hard link takes the final name and fails if it is taken, on every
        # platform. A rename would replace an existing file on some.
        os.link(partials[1], note)
        made.append(note)
    except BaseException:
        _discard(made)
        raise
    try:
        os.link(partials[0], path)  # the freeze: from here the record exists
    except BaseException:
        # Interrupted at the very moment of the freeze: if the record was
        # published, it keeps its note; if not, nothing this attempt made stays.
        if not _same_file(partials[0], path):
            _discard(made)
        raise
    _discard(partials)  # the published names stay
    return path


def freeze_v2(
    definition: ExperimentDefinition,
    observation_set: ObservationSet,
    prepared: PreparedOptimization,
    result: OptimizationResult | OptimizationResultND,
    directory: Path,
    *,
    prior_best: float | None = None,
    source: SourceRevision | None = None,
) -> Path:
    """Freeze a recommendation into `directory` and return the JSON path.

    The file is named for the moment the recommendation was produced. It is
    published all or nothing, beside its note, and never in place of a file
    that is already there (see "Write-once, and all or nothing" above). Read it
    back with `read_record` to get the seal.

    The record states the source revision that computed it, clean or not. This
    function does not refuse a tree that is not clean: a caller for whom that
    matters reads the revision first, decides, and passes it as `source`.

    Raises:
        ValueError: If the arguments do not belong together.
        VersionCollision: If this experiment name and version were frozen in
            `directory` before under a different definition.
        FileExistsError: Only if every candidate file name was taken.
        RecordError: If what was written did not read back as the record.
        OSError: If the record could not be written or published. No record
            is left under its final name.
    """
    directory = Path(directory)
    record = build_record_v2(
        definition, observation_set, prepared, result, prior_best=prior_best, source=source
    )
    _check_collision(directory, record)
    stamp = result.config.created_at.strftime("%Y%m%dT%H%M%SZ")
    for _ in range(_MAX_NAME_ATTEMPTS):
        try:
            return _write_once(record, _unused_path(directory, stamp))
        except FileExistsError:
            continue  # another writer took the name between the check and the write
    raise FileExistsError(f"No free pre-registration name for {stamp} in {directory}")


# ─── Reading and verifying ──────────────────────────────────────────
def read_record(path: Path) -> dict[str, Any]:
    """Read a version 2 record.

    Raises:
        RecordError: If the file is not JSON, is not a version 2 record, or
            lacks a block every version 2 record has.
    """
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RecordError(f"{path.name} is not JSON: {exc}") from exc
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        found = record.get("schema_version") if isinstance(record, dict) else None
        raise RecordError(
            f"{path.name} is not a version {SCHEMA_VERSION} record (schema_version: {found!r})"
        )
    for keys in _REQUIRED:
        block: Any = record
        for key in keys:
            if not isinstance(block, dict) or key not in block:
                raise RecordError(f"{path.name} lacks {'.'.join(keys)}")
            block = block[key]
    return record


def _verify_seal(record: Mapping[str, Any], expected_seal: str | None) -> list[str]:
    seal = record.get("seal")
    if not isinstance(seal, dict) or "sha256" not in seal:
        return ["the record carries no seal"]
    problems = []
    # The seal block lies outside what the seal covers, so what it may hold
    # is checked here: its digest and the one label saying how it was made.
    if set(seal) != {"canonical_form", "sha256"} or seal["canonical_form"] != SEAL_FORM:
        problems.append("the seal block is not one a freeze writes: it was changed or added to")
    if record_seal(record) != seal["sha256"]:
        problems.append("the record does not match its seal: it was changed after it was frozen")
    if expected_seal is not None and seal["sha256"] != expected_seal.strip().lower():
        problems.append(
            "the record's seal is not the one noted when it was frozen: this is not that record"
        )
    return problems


def _verify_definition(record: Mapping[str, Any]) -> list[str]:
    experiment = record["experiment"]
    definition = definition_from_dict(experiment["definition"])
    problems = []
    if definition_sha256(definition) != experiment["definition_sha256"]:
        problems.append("the inline definition does not hash to the recorded definition_sha256")
    for key in ("name", "version", "producer"):
        if experiment.get(key) != experiment["definition"].get(key):
            problems.append(f"experiment.{key} differs from the inline definition")
    if experiment.get("definition_canonical_form") != CANONICAL_FORM:
        problems.append("experiment.definition_canonical_form is not the form the hash is taken in")
    if experiment.get("warnings") != list(definition_warnings(definition)):
        problems.append("experiment.warnings are not the warnings the inline definition raises")
    return problems


def _verify_observations(record: Mapping[str, Any]) -> list[str]:
    block = record["observations"]
    problems = []
    digest = observations_digest_nd(block["x"], block["y"], block["y_raw"], sigma=block["sigma"])
    if digest != block["sha256"]:
        problems.append("the observations do not hash to the recorded digest")
    if block.get("canonical_form") != OBSERVATION_FORM:
        problems.append("observations.canonical_form is not the form the digest is taken in")
    if block.get("min_pooled_df") != MIN_POOLED_DF:
        problems.append("observations.min_pooled_df is not the threshold this code applies")

    projection = record["projection"]
    recomputed = project(
        np.asarray(block["y_raw"], dtype=float),
        goal=projection["goal"],
        magnitude=projection["magnitude"],
        target=projection["target"],
    ).tolist()
    if recomputed != block["y"]:
        problems.append("y is not what the declared objective gives from y_raw")

    training = record.get("training_data")
    if training is not None:
        # The version 1 mirror of the same data, which older screens read.
        scale = record["frozen_config"].get("point_noise_scale")
        if _v1.observations_digest(training["x"], training["y"], sigma=scale) != training["sha256"]:
            problems.append("the version 1 training block does not hash to its digest")
        if sorted(zip(training["x"], training["y"], strict=True)) != sorted(
            zip((row[0] for row in block["x"]), block["y"], strict=True)
        ):
            problems.append("the version 1 training block differs from the observations")
    return problems


def _verify_specimens(record: Mapping[str, Any]) -> list[str]:
    problems = []
    for kind in ("specimens", "excluded"):
        for row in record[kind]:
            label = row.get("sample_id") or row.get("record_id")
            if producer_detail_sha256(row["producer_detail"]) != row["producer_detail_sha256"]:
                problems.append(f"the producer detail of {label!r} does not hash to its digest")
    # A sample id on more than one row rejects every such row, so rejected rows
    # may share an id. An accepted specimen's id is on no other row.
    accepted = [row["sample_id"] for row in record["specimens"]] + [
        row["sample_id"] for row in record["excluded"] if row["category"] == EXCLUDED_OUTSIDE_BOUNDS
    ]
    named = [row["sample_id"] for row in [*record["specimens"], *record["excluded"]]]
    repeated = sorted({i for i in accepted if i is not None and named.count(i) > 1})
    if repeated:
        problems.append(f"the record holds these sample ids more than once: {', '.join(repeated)}")
    for row in record["excluded"]:
        if row["category"] != EXCLUDED_OUTSIDE_BOUNDS and not row["reasons"]:
            label = row.get("sample_id") or row.get("record_id")
            problems.append(f"the rejected row {label!r} gives no reason")
    problems += _verify_condition_inputs(record)

    response = record["projection"]["response"]
    values = {row["sample_id"]: row["responses"][response]["value"] for row in record["specimens"]}
    block = record["observations"]
    used: list[str] = []
    for index, ids in enumerate(block["sample_ids"]):
        used += ids
        if block["n_specimens"][index] != len(ids) or any(i not in values for i in ids):
            problems.append(f"condition {index + 1} names specimens the record does not hold")
            continue
        mean = sum(values[i] for i in ids) / len(ids)
        if not math.isclose(mean, block["y_raw"][index], rel_tol=_MEAN_REL, abs_tol=_MEAN_ABS):
            problems.append(f"y_raw of condition {index + 1} is not the mean of its specimens")
    if sorted(used) != sorted(values):
        problems.append("the specimens listed are not exactly the specimens the conditions use")

    prediction = record["prediction_at_recommendation"]
    if "x" in prediction and list(prediction["condition"].values()) != [prediction["x"]]:
        problems.append("the recommended condition differs from the recommended x")
    return problems


def _slack(limit: float) -> float:
    """How far a mean of values at `limit` may round past it."""
    return max(_MEAN_ABS, _MEAN_REL * abs(limit))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _verify_condition_inputs(record: Mapping[str, Any]) -> list[str]:
    """Whether each condition sits where its specimens were made, inside the search bounds.

    A numeric design value of a condition is the mean of its specimens' own
    values; a value that is not a number is not compared here.
    """
    block = record["observations"]
    names = block["input_names"]
    inputs = {row["sample_id"]: row["inputs"] for row in record["specimens"]}
    stated = record["objective"]["search_bounds"]
    bounds = [stated] if len(names) == 1 else stated
    problems = []
    for index, (point, ids) in enumerate(zip(block["x"], block["sample_ids"], strict=True)):
        if not all(i in inputs for i in ids):
            continue  # said by the caller
        for axis, name in enumerate(names):
            made = [inputs[i].get(name) for i in ids]
            if not (made and all(_is_number(v) for v in made) and _is_number(point[axis])):
                continue
            mean = sum(made) / len(made)
            if not math.isclose(mean, point[axis], rel_tol=_MEAN_REL, abs_tol=_MEAN_ABS):
                problems.append(f"condition {index + 1} is not at the mean {name} of its specimens")
            limits = bounds[axis] if axis < len(bounds) else None
            if limits is not None and not (
                limits[0] - _slack(limits[0]) <= point[axis] <= limits[1] + _slack(limits[1])
            ):
                problems.append(f"condition {index + 1} lies outside the search bounds of {name}")
    return problems


def _verify_identity(record: Mapping[str, Any]) -> list[str]:
    """Whether the record is what it says it is, under the definition it carries.

    Everything here is derived again from the inline definition: the title,
    the declared controls, the design variables the observations are named
    for, the backend those call for, the search bounds and the objective.
    """
    problems = []
    if record.get("kind") != RECORD_KIND:
        problems.append(f"the record's kind is {record.get('kind')!r}, not {RECORD_KIND!r}")
    experiment = record["experiment"]
    definition = definition_from_dict(experiment["definition"])
    built = _experiment_block(definition)
    for key in ("title", "controls"):
        if experiment.get(key) != built[key]:
            problems.append(f"experiment.{key} is not what the inline definition declares")

    names = [variable.name for variable in definition.design_variables()]
    if record["observations"]["input_names"] != names:
        problems.append("the observations are not named for the design variables declared")
    if record["objective"].get("input_variables") != names:
        problems.append("objective.input_variables are not the design variables declared")
    if list(record["prediction_at_recommendation"]["condition"]) != names:
        problems.append("the recommended condition is not stated for the design variables declared")
    if record.get("backend") != (BACKEND_1D if len(names) == 1 else BACKEND_ND):
        problems.append(
            f"backend {record.get('backend')!r} is not the one {len(names)} design "
            "variable(s) call for"
        )
    bounds = [list(v.bounds) if v.bounds else None for v in definition.design_variables()]
    stated = record["objective"]["search_bounds"]
    if (stated if len(names) > 1 else [stated]) != bounds:
        problems.append("the search bounds are not the bounds the inline definition declares")

    objective = definition.objective
    response = definition.objective_response()
    projection = record["projection"]
    # As `prepare_optimization` derives them: a target is a distance to
    # minimize, and only an unprojected objective keeps the response's range.
    unprojected = objective.goal is not Goal.TARGET and not objective.magnitude
    low, high = response.valid_range if unprojected else (None, None)
    declared = {
        "response": objective.response,
        "goal": objective.goal.value,
        "magnitude": objective.magnitude,
        "target": objective.target,
        "fit_space_declared": objective.fit_space.value,
        "response_unit": response.unit,
        "is_transformed": objective.fit_space.value != "identity",
        "y_min": low,
        "y_max": high,
    }
    problems += [
        f"projection.{key} is not what the inline definition declares"
        for key, value in declared.items()
        if projection.get(key) != value
    ]
    direction = Goal.MINIMIZE.value if objective.goal is Goal.TARGET else objective.goal.value
    if record["objective"].get("direction") != direction:
        problems.append("objective.direction is not what the inline definition's goal gives")
    if record["objective"].get("y_transform") != objective.fit_space.value:
        problems.append("objective.y_transform is not the fit space the definition declares")
    noise = record["prediction_noise"]
    if noise.get("specimens_per_condition") != definition.specimens_per_condition:
        problems.append(
            "prediction_noise.specimens_per_condition is not what the definition declares"
        )
    count = len(record["observations"]["y"])
    if record["frozen_config"].get("n_observations") != count:
        problems.append("frozen_config.n_observations is not the number of conditions fitted")
    return problems


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_DERIVED_REL, abs_tol=_DERIVED_ABS)


def _verify_prediction(record: Mapping[str, Any]) -> list[str]:
    """Whether the prediction, its intervals and the choice of x agree with the frozen grid.

    For a one-variable record the optimizer's whole grid is frozen: the model
    mean, the 95% model band and the expected improvement at every point. The
    recommended point must be one of those points, its prediction must be the
    model at that point, the predictive interval must be the model band
    widened by the frozen prediction noise, and the point must be the one the
    selection rule picks. Records over several variables freeze no grid, and
    only what they state about themselves is compared.
    """
    if record.get("backend") != BACKEND_1D:
        return []
    problems: list[str] = []
    prediction = record["prediction_at_recommendation"]
    config = record["frozen_config"]
    acquisition = record["acquisition"]
    grid = acquisition["grid"]
    xs = grid["x"]
    columns = ("mean", "lower_95", "upper_95", "expected_improvement")
    if any(len(grid[column]) != len(xs) for column in columns) or len(xs) != config["grid_size"]:
        return ["the frozen grid's columns are not all of the frozen grid size"]
    low, high = record["objective"]["search_bounds"]
    expected = np.linspace(low, high, len(xs)).tolist()
    if not all(_close(a, b) for a, b in zip(xs, expected, strict=True)):
        problems.append("the frozen grid is not the evenly spaced grid over the search bounds")
    x = prediction["x"]
    if x not in xs:
        return [*problems, "the recommended x is not a point of the frozen grid"]
    index = xs.index(x)

    mean, lower, upper = (grid[c][index] for c in ("mean", "lower_95", "upper_95"))
    if not _close(prediction["predicted_mean"], mean):
        problems.append("the predicted mean is not the model's mean at the recommended x")
    if not _close(prediction["ci95_model"], (upper - lower) / 2.0):
        problems.append("the model interval is not the frozen model band at the recommended x")
    noise = config["noise_std"]
    if record["prediction_noise"]["applied"] != noise:
        problems.append("the prediction noise applied is not the noise the model was given")
    problems += _verify_intervals(record, mean, lower, upper, noise)
    problems += _verify_selection(record, index)
    return problems


def _verify_intervals(
    record: Mapping[str, Any], mean: float, lower: float, upper: float, noise: float
) -> list[str]:
    """The predictive interval: the model band at x, widened by the prediction noise."""
    prediction = record["prediction_at_recommendation"]
    low, high = prediction["predictive_interval_95"]
    problems = []
    if not _close(prediction["ci95_predictive"], (high - low) / 2.0):
        problems.append("ci95_predictive is not the half-width of the predictive interval")
    if not (low <= lower <= mean <= upper <= high):
        problems.append("the predictive interval does not contain the model band at x")
    projection = record["projection"]
    clamped = projection["y_min"] is not None or projection["y_max"] is not None
    if projection["is_transformed"] or clamped:
        return problems  # the band is not symmetric, so only its order is checked
    model_sd = (upper - lower) / (2.0 * _CI95)
    predictive_sd = prediction["predictive_sd"]
    if not _close(predictive_sd, math.sqrt(model_sd**2 + noise**2)):
        problems.append(
            "predictive_sd is not the model's standard deviation at x combined with the "
            "prediction noise"
        )
    if not (
        _close(low, mean - _CI95 * predictive_sd) and _close(high, mean + _CI95 * predictive_sd)
    ):
        problems.append("the predictive interval is not the prediction +/- 1.96 predictive_sd")
    return problems


def _verify_selection(record: Mapping[str, Any], index: int) -> list[str]:
    """Whether the recommended point is the one the frozen selection rule picks.

    The campaign path asks for expected improvement (EI) and leaves the
    optimizer's fallback as it is shipped: when the largest EI is below the
    noise threshold and the reliability grade is exploratory, the point of
    largest model uncertainty is taken instead. Which of the two applied
    follows from the frozen numbers. A record made with another acquisition
    is not checked here, because its acquisition values are not frozen.
    """
    acquisition = record["acquisition"]
    grid = acquisition["grid"]
    ei = grid["expected_improvement"]
    problems = []
    if acquisition["max_expected_improvement"] != max(ei):
        problems.append("max_expected_improvement is not the largest expected improvement frozen")
    if acquisition["noise_threshold"] != record["frozen_config"]["noise_std"]:
        problems.append("the noise threshold is not the noise the model was given")
    if acquisition.get("function") != _ACQUISITION_EI:
        return problems
    problems += _verify_expected_improvement(record)
    reliability = record.get("reliability") or {}
    exhausted = acquisition["max_expected_improvement"] < acquisition["noise_threshold"]
    exploring = exhausted and reliability.get("level") == _EXPLORATORY
    if not exploring:
        if index != ei.index(max(ei)):
            problems.append(
                "the recommended x is not the point of largest expected improvement, which "
                "the frozen numbers call for"
            )
        return problems
    if _DEFAULT_EXPLORE_POLICY != _FALLBACK_LARGEST_SD or not _band_is_plain(record):
        return problems  # a fallback, or a band, this check cannot recompute
    # The fallback takes the largest model standard deviation, which on an
    # untransformed and unclamped band is the widest band. Widths taken back
    # from the frozen band differ from the model's own in the last bits.
    widths = [hi - lo for lo, hi in zip(grid["lower_95"], grid["upper_95"], strict=True)]
    widest = max(widths)
    if widths[index] < widest - (_DERIVED_REL * widest + _DERIVED_ABS):
        problems.append(
            "the recommended x is not the point of largest model uncertainty, which the "
            "frozen numbers call for"
        )
    return problems


def _band_is_plain(record: Mapping[str, Any]) -> bool:
    """Whether the frozen band is the model's mean +/- 1.96 sd, unbent and unclipped."""
    projection = record["projection"]
    return (
        not projection["is_transformed"]
        and projection["y_min"] is None
        and projection["y_max"] is None
    )


def _verify_expected_improvement(record: Mapping[str, Any]) -> list[str]:
    """Whether the frozen expected improvement follows from the frozen model.

    Recomputed at every grid point from the model mean and standard deviation
    taken back from the frozen band, against the best fitted value and the
    frozen `xi`, by the optimizer's own formula, with the sign flipped for a
    minimized objective as the optimizer flips it. Only where the band is
    plain, as on the campaign path; a transformed or clamped band does not
    give the model's own numbers back.
    """
    acquisition = record["acquisition"]
    grid = acquisition["grid"]
    problems: list[str] = []
    direction = record["objective"]["direction"]
    if not _band_is_plain(record) or direction not in ("maximize", "minimize"):
        return problems
    # `xi` is a fraction of the observed spread, floored at the noise; in an
    # untransformed fit space the spread is that of the fitted values.
    fitted = np.asarray(record["observations"]["y"], dtype=float)
    noise = record["frozen_config"]["noise_std"]
    expected_xi = float(acquisition["xi"]) * max(float(np.std(fitted)), float(noise))
    xi = acquisition.get("xi_absolute")
    if not (_is_number(xi) and _close(float(xi), expected_xi)):
        problems.append("acquisition.xi_absolute is not xi applied to the observed spread")
        xi = expected_xi
    # The optimizer maximizes; a minimized objective enters with its sign flipped.
    sign = 1.0 if direction == "maximize" else -1.0
    best = record["stopping_claim"]["best_measured"]
    lower = np.asarray(grid["lower_95"], dtype=float)
    upper = np.asarray(grid["upper_95"], dtype=float)
    mean = np.asarray(grid["mean"], dtype=float)
    sd = (upper - lower) / (2.0 * _CI95)
    again = expected_improvement(sign * mean, sd, sign * float(best), float(xi))
    frozen = np.asarray(grid["expected_improvement"], dtype=float)
    scale = max(float(np.max(np.abs(frozen))), float(np.max(sd)))
    if not np.allclose(frozen, again, rtol=_EI_REL, atol=_EI_ABS * scale):
        problems.append("the frozen expected improvement is not what the frozen model gives")
    return problems


def _verify_stopping(record: Mapping[str, Any]) -> list[str]:
    """Whether the stopping verdict follows from the numbers frozen beside it."""
    verdict = record.get("stopping_verdict")
    if verdict is None:
        return []
    claim = record["stopping_claim"]
    acquisition = record["acquisition"]
    reliability = record.get("reliability")
    exhausted = acquisition["max_expected_improvement"] < acquisition["noise_threshold"]
    sufficient = reliability is not None and reliability["level"] != _EXPLORATORY
    met = claim["probability_within_epsilon"] >= 1.0 - claim["delta"]
    problems = []
    stated = {
        "probability": claim["probability_within_epsilon"],
        "epsilon": claim["epsilon"],
        "delta": claim["delta"],
        "signal_exhausted": exhausted,
        "data_sufficient": sufficient,
    }
    problems += [
        f"stopping_verdict.{key} does not agree with the numbers frozen beside it"
        for key, value in stated.items()
        if verdict[key] != value
    ]
    if claim["met"] != met:
        problems.append("stopping_claim.met does not follow from its probability and delta")
    if exhausted and met and sufficient:
        action = "stop"
    elif met and not sufficient:
        action = "confirm"
    else:
        action = "continue"
    if verdict["action"] != action:
        problems.append(
            f"the stopping action {verdict['action']!r} is not the one the frozen numbers give "
            f"({action!r})"
        )
    opening = _VERDICT_OPENINGS.get(verdict["action"])
    if opening is None or not str(verdict.get("reason", "")).startswith(opening):
        problems.append("the stopping verdict's sentence is not the one its action is given")
    problems += _verify_grade(record)
    exploratory = reliability is not None and reliability["level"] == _EXPLORATORY
    if record["converged"] != (exhausted and not exploratory):
        problems.append("converged does not follow from the frozen numbers")
    fitted = record["observations"]["y"]
    direction = record["objective"].get("direction")
    if fitted and direction in ("maximize", "minimize"):
        best = max(fitted) if direction == "maximize" else min(fitted)
        if claim["best_measured"] != best:
            problems.append(
                "stopping_claim.best_measured is not the best value among the fitted conditions"
            )
    return problems


def _verify_specimen_sigma(record: Mapping[str, Any]) -> list[str]:
    """Whether a condition's within-specimen sigma is its one specimen's own uncertainty."""
    block = record["observations"]
    if block["sigma"] is None:
        return []
    response = record["projection"]["response"]
    stated = {
        row["sample_id"]: row["responses"][response]["uncertainty"] for row in record["specimens"]
    }
    problems = []
    for index, (source, ids) in enumerate(
        zip(block["sigma_sources"], block["sample_ids"], strict=True)
    ):
        if source != SIGMA_LOWER_BOUND or len(ids) != 1:
            continue
        own: Any = stated.get(ids[0])
        if not (_is_number(own) and _close(float(own), float(block["sigma"][index]))):
            problems.append(
                f"the sigma of condition {index + 1} is not the uncertainty its specimen states"
            )
    return problems


def _verify_grade(record: Mapping[str, Any]) -> list[str]:
    """Whether the reliability grade is one its own count of conditions allows.

    The grade starts from the count of conditions and can only be lowered by
    the later tests, so no grade above the count's own can be genuine. The
    leave-one-out numbers need a refit and are the seal's to hold.
    """
    reliability = record.get("reliability")
    if reliability is None:
        return []
    count = len(record["observations"]["y"])
    problems = []
    if reliability["n_observations"] != count:
        problems.append("reliability.n_observations is not the number of conditions fitted")
    allowed = (
        {_EXPLORATORY}
        if count < _RELIABILITY_INDICATIVE_N
        else {_EXPLORATORY, "indicative"}
        if count < _RELIABILITY_CALIBRATED_N
        else {_EXPLORATORY, "indicative", "calibrated"}
    )
    if reliability["level"] not in allowed:
        problems.append(
            f"the reliability grade {reliability['level']!r} is above what {count} "
            "condition(s) allow"
        )
    return problems


def _explained(glossary: Any, labels: set[str]) -> bool:
    """Whether `glossary` explains exactly `labels`, each in words."""
    return (
        isinstance(glossary, dict)
        and set(glossary) == labels
        and all(isinstance(text, str) and text.strip() for text in glossary.values())
    )


def _verify_notes(record: Mapping[str, Any]) -> list[str]:
    """Whether the record explains every label it uses, and its noise follows from its data.

    The wording of each explanation is the seal's to hold; what is checked is
    that none is missing or left over, and that the prediction noise is what
    its stated source gives from the observations.
    """
    problems = []
    block = record["observations"]
    count = len(block["y"])
    columns = ["x", "y_raw", "sigma_sources", "n_specimens", "sample_ids", "condition_issues"]
    if block["sigma"] is not None:
        columns.append("sigma")
    problems += [
        f"observations.{column} does not have one entry per condition"
        for column in columns
        if len(block[column]) != count
    ]
    if not _explained(block["sigma_glossary"], set(block["sigma_sources"])):
        problems.append("the sigma glossary does not explain exactly the uncertainty labels used")
    categories = {row["category"] for row in record["excluded"]}
    if not _explained(record["exclusion_glossary"], categories):
        problems.append("the exclusion glossary does not explain exactly the categories used")
    for key in ("calibration", "unit_check", "condition_check"):
        text = record["validation_criteria"].get(key)
        if not (isinstance(text, str) and text.strip()):
            problems.append(f"validation_criteria.{key} is missing")

    noise = record["prediction_noise"]
    meaning = noise.get("meaning")
    if not (isinstance(meaning, str) and meaning.strip()):
        problems.append("the prediction noise does not say what it rests on")
    sigma = block["sigma"]
    root_m = math.sqrt(noise["specimens_per_condition"])
    if sigma is None:
        expected, sources = None, {PREDICTION_NOISE_NOT_PASSED}
    elif block["pooled_sd"] is not None:
        expected, sources = block["pooled_sd"] / root_m, {PREDICTION_NOISE_POOLED}
    else:
        expected = float(np.median(sigma)) / root_m
        sources = {PREDICTION_NOISE_LOWER_BOUND, PREDICTION_NOISE_UNDETERMINED}
    if noise["source"] not in sources:
        problems.append(
            f"the prediction noise source {noise['source']!r} is not what the observations give"
        )
    problems += _verify_specimen_sigma(record)
    measured = record["acquisition"].get("noise_threshold_from_measured_noise")
    if measured and noise["source"] not in _MEASURED_NOISE_SOURCES:
        problems.append(
            "the noise threshold is said to come from measured noise, and its source "
            f"{noise['source']!r} is not a measurement"
        )
    stated = noise["prepared"]
    if (stated is None) != (expected is None) or (
        stated is not None and expected is not None and not _close(stated, expected)
    ):
        problems.append("the prediction noise is not what its stated source gives")
    return problems


_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# The keys of the environment block that describe the interpreter, not a library.
_INTERPRETER_KEYS = ("python", "implementation", "platform", "machine")


def _verify_source(record: Mapping[str, Any]) -> list[str]:
    """Whether the record states a coherent source revision.

    A tree that was not clean is not a discrepancy: the record then says that
    its commit cannot be taken as the code that ran. What is checked is that
    the statement is there and does not contradict itself. Whether the commit
    exists is for a caller that can read the repository.
    """
    block = record.get("source_revision")
    if not isinstance(block, dict):
        return ["the record does not say which source revision produced it"]
    problems = []
    if not isinstance(record.get("environment"), dict):
        problems.append("the record does not say what environment it was computed in")
    commit = block.get("git_commit")
    state = block.get("working_tree")
    if state not in WORKING_TREE_STATES:
        problems.append(f"the recorded working-tree state {state!r} is not one a record can hold")
    if commit is not None and not (isinstance(commit, str) and _COMMIT_RE.fullmatch(commit)):
        problems.append("the recorded git commit is not a commit hash")
    if state == WORKING_TREE_CLEAN and (commit is None or block.get("changed_path_count")):
        problems.append("the record claims a clean working tree without a commit to match")
    count, paths, note = (
        block.get("changed_path_count"),
        block.get("changed_paths"),
        block.get("note"),
    )
    if not (
        isinstance(count, int) and isinstance(paths, list) and len(paths) == min(count, _LISTED)
    ):
        problems.append("the changed paths listed do not agree with their count")
    elif state == WORKING_TREE_DIRTY and count == 0:
        problems.append("the record claims a dirty working tree with no path that differs")
    elif state == WORKING_TREE_CLEAN and (paths or note):
        problems.append("the record claims a clean working tree and lists a difference")
    if state == WORKING_TREE_UNKNOWN and not (isinstance(note, str) and note.strip()):
        problems.append("the record states an unknown working tree without saying why")
    return problems


def verify_record(
    record: Mapping[str, Any], *, expected_seal: str | None = None
) -> tuple[str, ...]:
    """Recompute what a record claims about itself and return every discrepancy.

    Args:
        record: A record as `read_record` returns it.
        expected_seal: The seal noted elsewhere when the record was frozen. If
            given, the record's seal must be that one.

    Nothing is written, and nothing outside `record` is read.

    Returns:
        One sentence per discrepancy. Empty means the record is internally
        consistent and, if `expected_seal` was given, its body is the body
        that was frozen. The seal covers everything but the seal block, whose
        two entries are checked against what a freeze writes.
    """
    problems: list[str] = []
    checks = (
        ("seal", lambda r: _verify_seal(r, expected_seal)),
        ("definition", _verify_definition),
        ("identity", _verify_identity),
        ("observations", _verify_observations),
        ("specimens", _verify_specimens),
        ("prediction", _verify_prediction),
        ("stopping verdict", _verify_stopping),
        ("notes and limitations", _verify_notes),
        ("source revision", _verify_source),
    )
    for name, check in checks:
        try:
            problems += check(record)
        except Exception as exc:
            # A block damaged past recomputing is itself the finding.
            problems.append(f"the record's {name} cannot be checked: {type(exc).__name__}: {exc}")
    return tuple(problems)


# ─── The note ───────────────────────────────────────────────────────
def _number(value: Any) -> str:
    return "not available" if value is None else f"{value:.4g}"


def _note_observations(record: Mapping[str, Any]) -> list[str]:
    block = record["observations"]
    projection = record["projection"]
    unit = projection["response_unit"]
    names = block["input_names"]
    lines = [
        "## Observations the model saw (frozen)",
        f"- SHA-256: `{block['sha256']}`",
        "",
        f"| {' | '.join(names)} | {projection['response']} ({unit}), signed | optimized value "
        f"({unit}) | sigma ({unit}) | what sigma is | specimens |",
        "|" + "---|" * (len(names) + 5),
    ]
    for index, point in enumerate(block["x"]):
        sigma = None if block["sigma"] is None else block["sigma"][index]
        lines.append(
            "| "
            + " | ".join(f"{v:.6g}" for v in point)
            + f" | {block['y_raw'][index]:+.4g} | {block['y'][index]:.4g} | {_number(sigma)} "
            f"| {block['sigma_sources'][index]} | {', '.join(block['sample_ids'][index])} |"
        )
    lines.append("")
    lines += [f"- `{label}`: {meaning}" for label, meaning in block["sigma_glossary"].items()]
    for issues in block["condition_issues"]:
        lines += [f"- **{issue['severity'].upper()}** {issue['message']}" for issue in issues]
    return lines


def _note_exclusions(record: Mapping[str, Any]) -> list[str]:
    outside = [row for row in record["excluded"] if row["category"] == EXCLUDED_OUTSIDE_BOUNDS]
    rejected = [row for row in record["excluded"] if row["category"] != EXCLUDED_OUTSIDE_BOUNDS]
    lines: list[str] = []
    if outside:
        lines += [
            "",
            "## Specimens kept outside the design bounds (not in the model)",
            f"- {EXCLUSION_GLOSSARY[EXCLUDED_OUTSIDE_BOUNDS]}",
        ]
        response = record["projection"]["response"]
        unit = record["projection"]["response_unit"]
        for row in outside:
            value = row["responses"][response]
            scope = value["uncertainty_scope"]
            said = (
                "no scope stated"
                if scope is None
                else f"{scope} scope; a lower bound"
                if scope == UncertaintyScope.SPECIMEN
                else f"{scope} scope"
            )
            lines.append(
                f"- **{row['sample_id']}**: {'; '.join(row['reasons'])}. "
                f"{response} = {value['value']:+.4g} {unit}, uncertainty "
                f"{_number(value['uncertainty'])} {unit} ({said})"
            )
            lines += [
                f"  - **{issue['severity'].upper()}** {issue['message']}" for issue in row["issues"]
            ]
    lines += ["", "## Left out of the model"]
    if not rejected:
        return [*lines, "- Nothing was rejected."]
    for row in rejected:
        label = row["sample_id"] or row["record_id"]
        lines.append(f"- **{label}**: " + "; ".join(row["reasons"]))
    return lines


def _note_source(record: Mapping[str, Any]) -> list[str]:
    experiment = record["experiment"]
    block = record["source_revision"]
    state = block["working_tree"]
    lines = [
        "",
        "## Source revision (frozen)",
        f"- Git commit: `{block['git_commit'] or 'not available'}`",
    ]
    if state == WORKING_TREE_CLEAN:
        lines.append(
            "- Working tree: clean. No tracked or untracked file differed from this commit; "
            "files Git ignores are not compared."
        )
    else:
        why = block["note"] or f"{block['changed_path_count']} path(s) differ from the commit"
        lines.append(
            f"- Working tree: **{state.upper()}** ({why}). The commit alone cannot be taken "
            "as the code that computed this record."
        )
    environment = record["environment"]
    libraries = ", ".join(
        f"{name} {version or 'not installed'}"
        for name, version in environment.items()
        if name not in _INTERPRETER_KEYS
    )
    lines += [
        f"- latos {record['latos_version']}",
        f"- Definition SHA-256: `{experiment['definition_sha256']}`",
        f"- Producer contract: `{experiment['producer']}`",
        f"- Computed with {environment['implementation']} {environment['python']} on "
        f"{environment['platform']} ({environment['machine']}); {libraries}",
    ]
    return lines


def _control_line(control: Mapping[str, Any]) -> str:
    tolerance = control["tolerance"]
    unit = "" if control["unit"] in (None, "", "1") else control["unit"]  # "1": a count
    if tolerance is None:
        wanted = f"expected {control['expected_value']!r} {unit}"
    elif tolerance == 0:
        # A required value, not a quantity measured with no uncertainty.
        wanted = f"required to equal {control['expected_value']!r} {unit}, with no tolerance"
    else:
        wanted = f"expected {control['expected_value']!r} ± {tolerance:g} {unit}"
    checked = (
        "checked against the value recorded for each specimen; a specimen with none recorded "
        "is kept, and the JSON record carries a warning on it"
        if control["verified"]
        else "declared, not checked"
    )
    return f"- `{control['name']}`: {' '.join(wanted.split())}; {checked}"


def render_note(record: Mapping[str, Any]) -> str:
    """The record as a note a person can read. The JSON remains the record."""
    experiment = record["experiment"]
    objective = record["objective"]
    projection = record["projection"]
    prediction = record["prediction_at_recommendation"]
    noise = record["prediction_noise"]
    low, high = prediction["predictive_interval_95"]
    condition = ", ".join(
        f"`{name}` = **{value:.4g}**" for name, value in prediction["condition"].items()
    )
    folded = ", as a magnitude" if projection["magnitude"] else ""
    target = "" if projection["target"] is None else f" (target {projection['target']:g})"
    unit = projection["response_unit"]
    # The noise and the stopping tolerance are in the fit space's units, which
    # are the response's own only when the fit space is not transformed.
    noise_unit = "(in fit-space units)" if projection["is_transformed"] else unit

    lines = [
        "# Pre-registered recommendation",
        "",
        f"_Recommendation computed {record['created_at']}. It is committed when its seal "
        "(below) is written down where this file's editor cannot reach, before the "
        "recommended specimen is made. Nothing in this file shows when that was done._",
        "",
        "## Experiment (frozen)",
        f"- **{experiment['title']}** ({experiment['name']} {experiment['version']})",
        f"- Definition SHA-256: `{experiment['definition_sha256']}`",
        f"- Producer contract: `{experiment['producer']}`",
        f"- Backend: `{record['backend']}`, latos {record['latos_version']}",
        "",
        "## Objective",
        f"- {projection['goal'].capitalize()} **{projection['response']}** "
        f"({projection['response_unit']}){folded}{target}",
        f"- Optimized as `{objective['property']}`; fit space: {objective['y_transform']}",
        f"- Search bounds: {objective['search_bounds']}",
        "",
        "## Prediction at the recommended condition (as frozen)",
        f"- Recommended: {condition}",
        f"- Predicted {objective['property']} = **{prediction['predicted_mean']:.4g} {unit}**",
        f"- 95% predictive interval: **[{low:.4g}, {high:.4g}] {unit}**: the model band below "
        "widened by the prediction noise, and the band a new measurement is to be compared "
        f"with. {_PREDICTIVE_CAVEAT.get(noise['source'], '')}".rstrip(),
        f"- 95% model interval: ± {prediction['ci95_model']:.4g} {unit}: the model's "
        "uncertainty about the mean response here, without measurement noise. The grid's "
        "lower_95 and upper_95 in the acquisition block are this band at every grid point, "
        "not predictive intervals.",
        f"- Prior best: {record['prior_best']:.4g} {unit}{_prior_best_basis(record)}",
        "",
        "## Prediction noise",
        f"- {_number(noise['applied'])} {noise_unit} applied by the optimizer "
        f"(prepared: {_number(noise['prepared'])}); source: `{noise['source']}`",
        f"- {noise['meaning']}",
        "",
        *_note_observations(record),
        *_note_exclusions(record),
        "",
        "## Controls",
    ]
    lines += [_control_line(control) for control in experiment["controls"]] or ["- None declared."]

    lines += ["", "## Stopping and reliability (self-assessed at freeze time)"]
    lines += _note_stopping(record, unit, noise_unit)
    if record["messages"] or experiment["warnings"]:
        lines += ["", "## Notes and warnings"]
        lines += [f"- {message}" for message in [*experiment["warnings"], *record["messages"]]]
    lines += _note_source(record)
    lines += [
        "",
        "## Seal",
        f"- `{record['seal']['sha256']}`",
        "- Write this down somewhere this file's editor cannot reach, at the time of the "
        "freeze. It covers the JSON record except its own seal block. This note is not "
        "sealed, and the seal printed here can be changed together with it. The record is "
        "the one frozen only if it verifies against the seal written down at the freeze "
        "(`verify --seal`).",
    ]
    return "\n".join(lines) + "\n"


# What the predictive interval's width rests on, by the source of its noise.
_PREDICTIVE_CAVEAT = {
    PREDICTION_NOISE_LOWER_BOUND: (
        "Its noise is a lower bound here, so the band is too narrow (see Prediction noise)."
    ),
    PREDICTION_NOISE_UNDETERMINED: (
        "Its noise is an assumption here, not a measurement (see Prediction noise)."
    ),
    PREDICTION_NOISE_NOT_PASSED: (
        "No measured noise was passed; the optimizer's own assumption widened it."
    ),
}


def _prior_best_basis(record: Mapping[str, Any]) -> str:
    """What the prior best is, said only when the record shows it."""
    claim = record.get("stopping_claim") or {}
    if claim.get("best_measured") == record["prior_best"]:
        return (
            ": the best value among the fitted conditions. Specimens kept outside the search "
            "bounds are not among them."
        )
    return ": the baseline given when the record was frozen."


def _note_stopping(record: Mapping[str, Any], unit: str, noise_unit: str) -> list[str]:
    """The stopping verdict and reliability grade, with what their words mean here.

    The verdict's own sentence is the optimizer's and is frozen as written;
    what it leaves unsaid about this record is said beside it.
    """
    verdict = record.get("stopping_verdict")
    reliability = record.get("reliability")
    claim = record.get("stopping_claim")
    lines: list[str] = []
    if verdict is not None:
        lines.append(f"- **{verdict['action']}**: {verdict['reason']}")
    if claim is not None and claim.get("probability_within_epsilon") is not None:
        bar = 1.0 - claim["delta"]
        grid = record["frozen_config"].get("grid_size")
        over = f"a {grid}-point grid across the search range" if grid else "the search range"
        lines.append(
            "- Stopping claim (frozen): at the best measured condition (best value "
            f"{claim['best_measured']:.4g} {unit}, among the conditions the model was fitted "
            f"to), the model's value is within epsilon = {claim['epsilon']:.4g} {noise_unit} "
            f"of the best value the model draws over {over}, with probability "
            f"{claim['probability_within_epsilon']:.2f} under the frozen model "
            f"({'meets' if claim['met'] else 'below'} the {bar:.0%} bar). "
            "In the verdict, the measurement noise and the noise threshold are the applied "
            "prediction noise above, and improvement is the model's expected improvement "
            "over that grid."
        )
    if reliability is None:
        return lines
    few = reliability["n_observations"] < _RELIABILITY_INDICATIVE_N
    if verdict is not None and verdict["action"] == "confirm" and few:
        lines.append(
            f"- With fewer than {_RELIABILITY_INDICATIVE_N} distinct conditions the grade is "
            "exploratory whatever is measured, and a repeat at a composition already made "
            "adds no condition. A repeat of the incumbent therefore cannot by itself make "
            "the data sufficient for this claim."
        )
    lines.append(
        f"- Reliability: **{reliability['level'].upper()}** "
        f"({reliability['n_observations']} conditions; leave-one-out "
        f"{reliability['loo_inside']}/{reliability['loo_total']} inside the 95% band). "
        f"{reliability['note']}"
    )
    lines.append(
        f"- The condition counts behind the grade ({_RELIABILITY_INDICATIVE_N} and "
        f"{_RELIABILITY_CALIBRATED_N}) were set with reference to a calibration study on "
        "another material system (P3HT/CNT). As the optimizer records that study, a nominal "
        "95% interval covered about 50% of held-out points at 6 conditions, about 85% at 25 "
        "and about 92% at 70 or more: no tier was shown to reach the nominal coverage. None "
        "of this has been validated for this material system."
    )
    if record["frozen_config"].get("point_noise_used"):
        lines.append(
            "- The leave-one-out refits used one shared noise level, not the per-condition "
            "uncertainties the model itself was given."
        )
    return lines
