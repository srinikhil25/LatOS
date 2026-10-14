"""Freezing a recommendation, reading it back, and checking it against itself."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy
import sklearn

from latos import __version__
from latos.core.enums import Severity
from latos.core.models import ValidationIssue, utc_now
from latos.experiments import (
    EXCLUSION_GLOSSARY,
    Goal,
    ObjectiveDefinition,
    SourceRevision,
    build_observations,
    definition_from_dict,
    definition_sha256,
    prepare_optimization,
    producer_detail_sha256,
    reduce_observations,
    run_optimization,
    runtime_environment,
)
from latos.experiments import prereg as experiments_prereg
from latos.experiments.prereg import (
    OBSERVATION_FORM,
    SCHEMA_VERSION,
    RecordError,
    VersionCollision,
    _write_once,
    build_record_v2,
    freeze_v2,
    interrupted_freeze_files,
    observations_digest_nd,
    read_record,
    record_seal,
    render_note,
    verify_record,
)
from latos.optimization import prereg as v1
from latos.optimization.engine import OptimizationResult
from latos.optimization.validate import list_preregistrations
from tests.unit.experiments._examples import (
    BAKE,
    VARIABLES,
    X,
    make_definition,
    make_nd_definition,
    make_nd_table,
    make_row,
    reduced_set,
)

Campaign = tuple[Any, Any, Any, Any]

SOURCE = SourceRevision(git_commit="ab" * 20, working_tree="clean")


@pytest.fixture(autouse=True)
def committed_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A committed tree, so no test depends on the state of this repository."""
    monkeypatch.setattr("latos.experiments.prereg.source_revision", lambda: SOURCE)


ROWS = (
    make_row("a", 0.0, -2.6, 0.40, detail={"repeats": [-3.0, -2.6, -2.2], "ünïcode": "é"}),
    make_row("b", 0.25, 2.0, 0.02, detail={"repeats": [2.0, 2.0, 2.0], "nested": {"k": [1, None]}}),
    make_row("c", 0.5, 2.1, 0.08),
    make_row("d", 0.75, 2.2, 0.04, detail={"repeats": [2.1, 2.2, 2.3]}),
    make_row("e", 1.0, 1.6, 0.03, detail={"repeats": [1.5, 1.6, 1.7]}),
    make_row("late", 0.6, 9.9, 0.01, bake_time_min=70.0, detail={"repeats": [9.9]}),
    make_row(
        "unread",
        0.9,
        None,
        issues=(
            ValidationIssue(
                field="signal",
                severity=Severity.ERROR,
                message="the reading could not be taken",
                detected_at=utc_now(),
            ),
        ),
    ),
)


def _campaign(definition: Any = None, rows: tuple[Any, ...] = ROWS) -> Campaign:
    definition = definition or make_definition()
    observation_set = reduced_set(definition, *rows)
    prepared = prepare_optimization(definition, observation_set)
    return definition, observation_set, prepared, run_optimization(prepared)


@pytest.fixture(scope="module")
def campaign() -> Campaign:
    return _campaign()


@pytest.fixture
def record(campaign: Campaign) -> dict[str, Any]:
    return build_record_v2(*campaign)


# ─── Round trip ─────────────────────────────────────────────────────
def test_a_frozen_record_reads_back_and_verifies(tmp_path: Path, campaign: Campaign) -> None:
    path = freeze_v2(*campaign, tmp_path)

    stamp = campaign[3].config.created_at.strftime("%Y%m%dT%H%M%SZ")
    assert path == tmp_path / f"prereg_{stamp}.json"
    assert path.with_suffix(".md").is_file()
    record = read_record(path)
    assert verify_record(record) == ()
    assert verify_record(record, expected_seal=record["seal"]["sha256"]) == ()
    # What was written is what was built: nothing is lost on the way to disk.
    assert record == build_record_v2(*campaign)


def test_the_record_says_which_experiment_and_which_contract(
    record: dict[str, Any], campaign: Campaign
) -> None:
    definition = campaign[0]
    experiment = record["experiment"]
    assert record["kind"] == "latos.bo.prereg"
    assert record["schema_version"] == SCHEMA_VERSION == 2
    assert record["backend"] == "optimize"
    assert record["latos_version"] == __version__
    assert datetime.fromisoformat(record["created_at"]).tzinfo is not None

    assert (experiment["name"], experiment["version"]) == ("example-campaign", "1.0.0")
    assert experiment["producer"] == definition.producer == "bench-table/1"
    assert experiment["definition_sha256"] == definition_sha256(definition)
    assert experiment["definition_canonical_form"] == "latos.experiment.v1"
    # The whole definition is inline: the record needs no code to be understood.
    assert definition_from_dict(experiment["definition"]) == definition
    controls = {control["name"]: control for control in experiment["controls"]}
    assert controls["bake_time_min"] == {
        "name": "bake_time_min",
        "expected_value": 60.0,
        "tolerance": 2.0,
        "unit": "min",
        "verified": True,
    }
    assert controls["glovebox"]["verified"] is False  # declared, never recorded


def test_the_record_holds_exactly_the_observations_the_model_saw(
    record: dict[str, Any], campaign: Campaign
) -> None:
    _, _, prepared, result = campaign
    block = record["observations"]
    assert block["canonical_form"] == OBSERVATION_FORM
    assert block["input_names"] == ["additive_fraction"]
    assert block["x"] == [[0.0], [0.25], [0.5], [0.75], [1.0]]
    assert block["y_raw"] == [-2.6, 2.0, 2.1, 2.2, 1.6]  # signed, as measured
    assert block["y"] == [2.6, 2.0, 2.1, 2.2, 1.6]  # as optimized
    assert block["sigma"] == [0.40, 0.02, 0.08, 0.04, 0.03]
    assert block["sigma_sources"] == ["lower_bound_specimen"] * 5
    assert "lower bound" in block["sigma_glossary"]["lower_bound_specimen"]
    assert block["n_specimens"] == [1] * 5
    assert block["sample_ids"] == [["a"], ["b"], ["c"], ["d"], ["e"]]
    assert block["sha256"] == observations_digest_nd(
        prepared.x.tolist(),
        prepared.y.tolist(),
        prepared.y_raw.tolist(),
        sigma=prepared.sigma.tolist(),
    )
    # And they are what the optimizer reports having been given.
    assert [row[0] for row in block["x"]] == list(result.observed_x)
    assert block["y"] == list(result.observed_y)

    assert record["projection"] == {
        "response": "signal",
        "goal": "maximize",
        "magnitude": True,
        "target": None,
        "fit_space_declared": "identity",
        "is_transformed": False,
        "y_min": None,
        "y_max": None,
        "response_unit": "a.u.",
    }
    assert record["prediction_noise"]["source"] == "lower_bound"
    assert record["prediction_noise"]["prepared"] == pytest.approx(0.04)
    assert record["prediction_noise"]["applied"] == pytest.approx(0.04)
    assert "too narrow" in record["prediction_noise"]["meaning"]


def test_every_specimen_and_every_exclusion_is_in_the_record(record: dict[str, Any]) -> None:
    specimens = {row["sample_id"]: row for row in record["specimens"]}
    assert list(specimens) == ["a", "b", "c", "d", "e"]
    first = specimens["a"]
    assert first["inputs"]["additive_fraction"] == 0.0
    assert first["inputs"]["bake_time_min"] == 60.0
    assert first["metadata"] == {"batch": "b1"}
    assert first["responses"]["signal"] == {
        "value": -2.6,
        "unit": "a.u.",
        "uncertainty": 0.40,
        "uncertainty_scope": "specimen",
        "provenance": "bench-table/1: invented value",
    }

    excluded = {row["sample_id"]: row for row in record["excluded"]}
    assert list(excluded) == ["late", "unread"]
    assert excluded["late"]["reasons"] == [
        "control bake_time_min: recorded 70 min, expected 60 ± 2"
    ]
    assert excluded["late"]["values"]["signal"] == 9.9  # kept, with the value it had
    assert {row["category"] for row in record["excluded"]} == {"rejected"}
    assert "responses" not in excluded["late"]  # a rejected row is not offered as a specimen
    assert record["exclusion_glossary"] == {"rejected": EXCLUSION_GLOSSARY["rejected"]}
    assert excluded["unread"]["reasons"] == ["the reading could not be taken"]
    assert excluded["unread"]["issues"] == [
        {"field": "signal", "severity": "error", "message": "the reading could not be taken"}
    ]


def test_producer_detail_is_carried_whole_digested_and_never_interpreted(
    record: dict[str, Any],
) -> None:
    specimens = {row["sample_id"]: row for row in record["specimens"]}
    detail = {"repeats": [-3.0, -2.6, -2.2], "ünïcode": "é"}
    assert specimens["a"]["producer_detail"] == detail
    assert specimens["a"]["producer_detail_sha256"] == producer_detail_sha256(detail)
    # Any shape at all: nothing here looked for a key.
    assert specimens["b"]["producer_detail"] == {
        "repeats": [2.0, 2.0, 2.0],
        "nested": {"k": [1, None]},
    }
    assert (specimens["c"]["producer_detail"], specimens["c"]["producer_detail_sha256"]) == (
        None,
        None,
    )
    late = next(row for row in record["excluded"] if row["sample_id"] == "late")
    assert late["producer_detail"] == {"repeats": [9.9]}
    assert late["producer_detail_sha256"] == producer_detail_sha256({"repeats": [9.9]})


def test_the_record_holds_the_prediction_and_what_the_optimizer_said(
    record: dict[str, Any], campaign: Campaign
) -> None:
    result: OptimizationResult = campaign[3]
    recommendation = result.recommendation
    prediction = record["prediction_at_recommendation"]
    assert prediction["condition"] == {"additive_fraction": recommendation.x}
    assert prediction["x"] == recommendation.x
    assert prediction["predicted_mean"] == recommendation.predicted_mean
    assert prediction["predictive_interval_95"] == list(recommendation.predictive_interval_95)
    assert prediction["ci95_model"] == recommendation.ci95
    assert record["prior_best"] == result.best_y  # the default

    acquisition = record["acquisition"]
    assert acquisition["function"] == result.config.acquisition_function
    assert acquisition["max_expected_improvement"] == result.max_ei
    assert acquisition["noise_threshold"] == result.noise_threshold
    assert acquisition["grid"]["x"] == list(result.grid_x)
    assert acquisition["grid"]["expected_improvement"] == list(result.grid_ei)
    assert len(acquisition["grid"]["mean"]) == result.config.grid_size

    assert result.stopping is not None
    assert record["stopping_verdict"]["action"] == result.stopping.action
    assert record["stopping_verdict"]["reason"] == result.stopping.reason
    assert record["reliability"]["level"] == "exploratory"
    assert record["frozen_config"]["seed"] == 0
    assert record["frozen_config"]["point_noise_used"] is True
    # The sign change across conditions is in the record, as a note.
    assert any("changes sign across the measured conditions" in m for m in record["messages"])
    assert set(record["validation_criteria"]) >= {"calibration", "unit_check", "condition_check"}


def test_every_version_one_key_is_the_version_one_writers(
    record: dict[str, Any], campaign: Campaign
) -> None:
    result = campaign[3]
    old = json.loads(json.dumps(v1.build_record(result, prior_best=result.best_y)))
    for key, value in old.items():
        if isinstance(value, dict):
            # Version 2 may add keys to a block; it never changes one.
            assert {k: record[key][k] for k in value} == value, key
        else:
            assert record[key] == value, key


def test_the_existing_listing_reads_a_one_variable_record(
    tmp_path: Path, campaign: Campaign
) -> None:
    freeze_v2(*campaign, v1.prereg_dir(tmp_path))
    (entry,) = list_preregistrations(tmp_path)
    assert entry.recommended_x == campaign[3].recommendation.x
    assert entry.property_name == "abs_signal"
    assert entry.input_variable == "additive_fraction"
    assert entry.search_bounds == (0.0, 1.0)
    assert entry.reliability_level == "exploratory"


# ─── Tampering ──────────────────────────────────────────────────────
def _set(*path: Any, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(record: dict[str, Any]) -> None:
        block: Any = record
        for key in path[:-1]:
            block = block[key]
        block[path[-1]] = value

    return change


SEAL = "does not match its seal"


@pytest.mark.parametrize(
    ("change", "finding"),
    [
        # Under a magnitude objective this leaves y untouched. The digest sees it.
        (_set("observations", "y_raw", 0, value=2.6), "observations do not hash"),
        (_set("observations", "y_raw", 0, value=2.6), "y_raw of condition 1 is not the mean"),
        (_set("observations", "y", 1, value=2.5), "y is not what the declared objective gives"),
        (_set("observations", "sigma", 2, value=0.001), "observations do not hash"),
        (_set("observations", "x", 3, value=[0.7]), "observations do not hash"),
        (_set("observations", "sample_ids", 0, value=["z"]), "names specimens the record does not"),
        (
            _set("experiment", "definition", "variables", 0, "bounds", value=[0.2, 1.0]),
            "inline definition does not hash",
        ),
        (
            _set("experiment", "definition", "objective", "magnitude", value=False),
            "inline definition does not hash",
        ),
        (_set("experiment", "version", value="1.0.1"), "experiment.version differs"),
        (_set("experiment", "producer", value="bench-table/2"), "experiment.producer differs"),
        (
            _set("specimens", 0, "producer_detail", "repeats", value=[-2.6]),
            "producer detail of 'a' does not hash",
        ),
        (
            _set("excluded", 0, "producer_detail", "repeats", value=[1.0]),
            "producer detail of 'late' does not hash",
        ),
        (
            _set("specimens", 1, "responses", "signal", "value", value=2.4),
            "y_raw of condition 2 is not the mean",
        ),
        (_set("training_data", "y", 0, value=3.0), "version 1 training block"),
        (_set("prediction_at_recommendation", "x", value=0.123), "differs from the recommended x"),
        # Nothing but the seal covers the prediction itself.
        (_set("prediction_at_recommendation", "predicted_mean", value=9.0), SEAL),
        (_set("prediction_at_recommendation", "predictive_interval_95", value=[0.0, 99.0]), SEAL),
        (_set("stopping_verdict", "action", value="STOP"), SEAL),
        (_set("created_at", value="2020-01-01T00:00:00+00:00"), SEAL),
        (_set("excluded", 0, "reasons", value=[]), SEAL),
        (_set("specimens", 0, "responses", "signal", "uncertainty", value=0.001), SEAL),
        # The revision is a statement like any other: only the seal holds it.
        (_set("source_revision", "git_commit", value="cd" * 20), SEAL),
        (_set("source_revision", "working_tree", value="dirty"), SEAL),
        (_set("environment", "numpy", value="0.0.1"), SEAL),
        (_set("excluded", 0, "category", value="outside_design_bounds"), SEAL),
        (_set("seal", value=None), "carries no seal"),
    ],
)
def test_a_changed_record_is_caught(
    record: dict[str, Any], change: Callable[[dict[str, Any]], None], finding: str
) -> None:
    assert verify_record(record) == ()
    changed = copy.deepcopy(record)
    change(changed)
    problems = verify_record(changed)
    assert any(finding in problem for problem in problems), problems
    if finding != "carries no seal":
        assert any(SEAL in problem for problem in problems)  # the seal catches every edit


def test_a_resealed_rewrite_is_caught_only_by_the_seal_noted_at_freeze_time(
    record: dict[str, Any],
) -> None:
    noted = record["seal"]["sha256"]
    rewritten = copy.deepcopy(record)
    # A statement nothing else in the record repeats: only the seal holds it.
    rewritten["created_at"] = "2020-01-01T00:00:00+00:00"
    rewritten["seal"]["sha256"] = record_seal(rewritten)

    # The file agrees with itself again, which is all a file can show.
    assert verify_record(rewritten) == ()
    # Against the seal written down when it was frozen, it does not.
    (problem,) = verify_record(rewritten, expected_seal=noted)
    assert "not the one noted when it was frozen" in problem
    assert verify_record(record, expected_seal=noted.upper() + " ") == ()


def _grid_point(record: dict[str, Any], index: int) -> None:
    """Move the recommendation to another grid point, as a careless rewrite would."""
    x = record["acquisition"]["grid"]["x"][index]
    record["prediction_at_recommendation"]["x"] = x
    record["prediction_at_recommendation"]["condition"]["additive_fraction"] = x


def _shift_interval(record: dict[str, Any]) -> None:
    low, high = record["prediction_at_recommendation"]["predictive_interval_95"]
    record["prediction_at_recommendation"]["predictive_interval_95"] = [low + 0.1, high + 0.1]


def _move_condition(record: dict[str, Any], *, to: float, specimens: bool = False) -> None:
    """Move condition 4 to `to`, its digest redone, and its specimens too if asked."""
    block = record["observations"]
    block["x"][3] = [to]
    block["sha256"] = observations_digest_nd(
        block["x"], block["y"], block["y_raw"], sigma=block["sigma"]
    )
    if specimens:
        for row in record["specimens"]:
            if row["sample_id"] in block["sample_ids"][3]:
                row["inputs"]["additive_fraction"] = to


def _double_expected_improvement(record: dict[str, Any]) -> None:
    acquisition = record["acquisition"]
    acquisition["grid"]["expected_improvement"] = [
        2.0 * v for v in acquisition["grid"]["expected_improvement"]
    ]
    acquisition["max_expected_improvement"] *= 2.0


def _repeat_a_specimen(record: dict[str, Any]) -> None:
    # A fitted specimen's id twice. Rejected rows may share an id: a shared id
    # is what rejects them.
    record["specimens"].append(copy.deepcopy(record["specimens"][0]))


@pytest.mark.parametrize(
    ("change", "finding"),
    [
        (_set("prediction_at_recommendation", "predicted_mean", value=9.0), "predicted mean"),
        (_set("prediction_at_recommendation", "ci95_model", value=0.01), "model interval"),
        (_set("prediction_at_recommendation", "predictive_sd", value=0.01), "predictive_sd"),
        (_set("prediction_at_recommendation", "ci95_predictive", value=0.01), "ci95_predictive"),
        (_shift_interval, "predictive interval"),
        (lambda r: _grid_point(r, 57), "the recommended x is not the point of largest"),
        (_set("prediction_at_recommendation", "x", value=0.123), "not a point of the frozen grid"),
        (_set("acquisition", "max_expected_improvement", value=1e-9), "max_expected_improvement"),
        (_set("acquisition", "noise_threshold", value=5.0), "noise threshold"),
        (_set("acquisition", "grid", "x", 3, value=0.5), "evenly spaced grid"),
        (_set("frozen_config", "noise_std", value=0.5), "noise"),
        (_set("prediction_noise", "prepared", value=0.5), "prediction noise is not what"),
        (_set("prediction_noise", "source", value="pooled"), "prediction noise source"),
        (_set("prediction_noise", "meaning", value=""), "does not say what it rests on"),
        (_set("stopping_verdict", "action", value="stop"), "stopping action"),
        (_set("stopping_verdict", "signal_exhausted", value=False), "signal_exhausted"),
        (_set("stopping_claim", "met", value=True), "stopping_claim.met"),
        (_set("converged", value=True), "converged"),
        (_set("observations", "sigma_glossary", value={}), "sigma glossary"),
        (_set("exclusion_glossary", value={}), "exclusion glossary"),
        (_set("validation_criteria", "condition_check", value=""), "condition_check"),
        (_set("kind", value="latos.bo.other"), "kind"),
        (_set("backend", value="optimize_nd"), "backend"),
        (_set("experiment", "title", value="Another title"), "experiment.title"),
        (_set("experiment", "controls", value=[]), "experiment.controls"),
        (_set("objective", "search_bounds", value=[0.0, 2.0]), "search bounds"),
        (_set("projection", "magnitude", value=False), "projection.magnitude"),
        (_set("projection", "response_unit", value="V"), "projection.response_unit"),
        # The seal block is outside the seal; what it may hold is fixed.
        (_set("seal", "canonical_form", value="another form"), "seal block"),
        (lambda r: r["seal"].update(noted_by="someone"), "seal block"),
        (_set("experiment", "definition_canonical_form", value="x"), "definition_canonical_form"),
        (_set("experiment", "warnings", value=["an invented warning"]), "experiment.warnings"),
        (_set("observations", "canonical_form", value="x"), "observations.canonical_form"),
        (_set("observations", "min_pooled_df", value=1), "min_pooled_df"),
        (_repeat_a_specimen, "more than once: a"),
        (_set("excluded", 0, "reasons", value=[]), "the rejected row 'late' gives no reason"),
        (lambda r: _move_condition(r, to=0.7), "condition 4 is not at the mean additive_fraction"),
        (
            lambda r: _move_condition(r, to=2.0, specimens=True),
            "condition 4 lies outside the search bounds",
        ),
        (_double_expected_improvement, "expected improvement is not what the frozen model"),
        (_set("stopping_claim", "best_measured", value=9.0), "best_measured is not the best"),
        (
            _set("prediction_noise", "source", value="undetermined"),
            "said to come from measured noise",
        ),
        # Labels that would switch a recomputation off are tied to the definition.
        (_set("objective", "direction", value="minimize"), "objective.direction"),
        (_set("objective", "y_transform", value="log"), "objective.y_transform"),
        (_set("projection", "is_transformed", value=True), "projection.is_transformed"),
        (_set("projection", "y_min", value=-1000.0), "projection.y_min"),
        (_set("acquisition", "xi_absolute", value=None), "xi_absolute"),
        (_set("prediction_noise", "specimens_per_condition", value=3), "specimens_per_condition"),
        (_set("frozen_config", "n_observations", value=99), "frozen_config.n_observations"),
        (_set("reliability", "n_observations", value=99), "reliability.n_observations"),
        (_set("reliability", "level", value="calibrated"), "above what"),
        (_set("stopping_verdict", "reason", value="Stop. The optimum is found."), "sentence"),
        (_set("observations", "sigma", 1, value=0.5), "is not the uncertainty its specimen"),
    ],
)
def test_a_resealed_statement_that_no_longer_follows_from_the_record_is_caught(
    record: dict[str, Any], change: Callable[[dict[str, Any]], None], finding: str
) -> None:
    # Whoever rewrites a record can reseal it. What the record states in more
    # than one place, or derives from what it states, must still agree.
    changed = copy.deepcopy(record)
    change(changed)
    changed["seal"]["sha256"] = record_seal(changed)
    problems = verify_record(changed)
    assert any(finding in problem for problem in problems), problems


# Invented rows that put the optimizer on each branch of its selection rule,
# with the two rules pointing at different grid points.
_EI_ROWS = (
    make_row("a", 0.0, 1.0, 0.002),
    make_row("b", 0.45, 2.0, 0.002),
    make_row("c", 0.55, 2.0, 0.002),
    make_row("d", 1.0, 1.0, 0.002),
)
_FALLBACK_ROWS = (
    make_row("a", 0.0, 1.0, 0.3),
    make_row("b", 0.1, 1.1, 0.3),
    make_row("c", 0.2, 1.2, 0.3),
    make_row("d", 0.5, 2.0, 0.3),
)


@pytest.mark.parametrize(
    ("rows", "exploring"),
    [(_EI_ROWS, False), (_FALLBACK_ROWS, True)],
    ids=["expected-improvement", "largest-uncertainty-fallback"],
)
def test_the_rule_that_chose_x_is_the_one_the_frozen_numbers_call_for(
    rows: tuple[Any, ...], exploring: bool
) -> None:
    record = build_record_v2(*_campaign(rows=rows))
    acquisition = record["acquisition"]
    grid = acquisition["grid"]
    ei = grid["expected_improvement"]
    widths = [hi - lo for lo, hi in zip(grid["lower_95"], grid["upper_95"], strict=True)]
    index = grid["x"].index(record["prediction_at_recommendation"]["x"])

    # Which branch applied follows from numbers the record freezes.
    assert record["reliability"]["level"] == "exploratory"
    exhausted = acquisition["max_expected_improvement"] < acquisition["noise_threshold"]
    assert exhausted is exploring
    largest_ei, widest = ei.index(max(ei)), widths.index(max(widths))
    chosen, other = (widest, largest_ei) if exploring else (largest_ei, widest)
    assert index == chosen != other
    assert verify_record(record) == ()

    # The point the other rule would have picked, sealed again, is caught.
    moved = copy.deepcopy(record)
    _grid_point(moved, other)
    moved["seal"]["sha256"] = record_seal(moved)
    rule = "largest model uncertainty" if exploring else "largest expected improvement"
    assert any(f"is not the point of {rule}" in p for p in verify_record(moved))


def test_a_minimized_objective_has_its_expected_improvement_recomputed_too() -> None:
    minimized = make_definition(
        objective=ObjectiveDefinition(response="signal", goal=Goal.MINIMIZE)
    )
    record = build_record_v2(*_campaign(minimized, rows=_EI_ROWS))
    assert record["objective"]["direction"] == "minimize"
    assert verify_record(record) == ()
    _double_expected_improvement(record)
    record["seal"]["sha256"] = record_seal(record)
    assert any("expected improvement is not what" in p for p in verify_record(record))


def test_rejected_rows_may_share_the_id_that_rejected_them() -> None:
    rows = (*ROWS, make_row("twin", 0.3, 2.0, 0.05), make_row("twin", 0.35, 2.05, 0.05))
    record = build_record_v2(*_campaign(rows=rows))
    twins = [row for row in record["excluded"] if row["sample_id"] == "twin"]
    assert len(twins) == 2 and all(row["category"] == "rejected" for row in twins)
    assert verify_record(record) == ()


@pytest.mark.parametrize(
    "change",
    [
        _set("prediction_noise", "specimens_per_condition", value=0),
        _set("observations", "y_raw", 0, value=10**400),
        _set("observations", "sample_ids", 0, value=[]),
        _set("observations", "sample_ids", value=None),
        _set("created_at", value="\ud800"),  # cannot be encoded, so cannot be sealed
    ],
)
def test_a_record_damaged_in_any_way_gives_findings_not_a_crash(
    record: dict[str, Any], change: Callable[[dict[str, Any]], None]
) -> None:
    change(record)
    problems = verify_record(record)
    assert problems
    assert all(isinstance(problem, str) for problem in problems)


def test_a_block_damaged_past_recomputing_is_reported_not_raised(record: dict[str, Any]) -> None:
    del record["observations"]["y_raw"]
    problems = verify_record(record)
    assert any("the record's observations cannot be checked: KeyError" in p for p in problems)


# ─── The observation digest ─────────────────────────────────────────
def test_the_digest_is_the_documented_form() -> None:
    text = "0.0\t2.6\t-2.6\t0.4\n0.5\t2.0\t2.0\t0.02"
    digest = observations_digest_nd([[0.5], [0.0]], [2.0, 2.6], [2.0, -2.6], sigma=[0.02, 0.4])
    assert digest == hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Row order is not identity; every value is.
    assert digest == observations_digest_nd(
        [[0.0], [0.5]], [2.6, 2.0], [-2.6, 2.0], sigma=[0.4, 0.02]
    )
    assert digest != observations_digest_nd(
        [[0.0], [0.5]], [2.6, 2.0], [2.6, 2.0], sigma=[0.4, 0.02]
    )
    assert digest != observations_digest_nd([[0.0], [0.5]], [2.6, 2.0], [-2.6, 2.0])
    two_axes = observations_digest_nd([[1.0, 2.0]], [3.0], [3.0])
    assert two_axes == hashlib.sha256(b"1.0\t2.0\t3.0\t3.0").hexdigest()
    with pytest.raises(ValueError, match="one entry per condition"):
        observations_digest_nd([[0.0]], [1.0, 2.0], [1.0, 2.0])


# ─── Writing ────────────────────────────────────────────────────────
def test_a_second_freeze_never_replaces_the_first(tmp_path: Path, campaign: Campaign) -> None:
    first = freeze_v2(*campaign, tmp_path)
    before = first.read_bytes()
    second = freeze_v2(*campaign, tmp_path)
    assert second != first
    assert second.name == first.name.replace(".json", "_2.json")
    assert first.read_bytes() == before
    assert len(list(tmp_path.glob("prereg_*.json"))) == 2


def _fail_on_call(monkeypatch: pytest.MonkeyPatch, name: str, call: int) -> list[tuple[Any, ...]]:
    """Make `os.<name>` raise on its `call`-th use, as a full disk would."""
    real = getattr(os, name)
    calls: list[tuple[Any, ...]] = []

    def failing(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        if len(calls) == call:
            raise OSError(28, "No space left on device")
        return real(*args, **kwargs)

    monkeypatch.setattr(os, name, failing)
    return calls


@pytest.mark.parametrize(
    ("step", "call"),
    [("fsync", 1), ("fsync", 2), ("link", 1), ("link", 2)],
    ids=["record-written", "note-written", "note-published", "record-published"],
)
def test_a_freeze_that_fails_part_way_leaves_nothing_behind(
    tmp_path: Path, campaign: Campaign, monkeypatch: pytest.MonkeyPatch, step: str, call: int
) -> None:
    _fail_on_call(monkeypatch, step, call)
    with pytest.raises(OSError, match="No space left"):
        freeze_v2(*campaign, tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert interrupted_freeze_files(tmp_path) == ()


def test_a_freeze_killed_between_its_steps_leaves_files_that_are_found(
    tmp_path: Path, campaign: Campaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A killed process cleans nothing up: the partial files and the published
    # note stay, and the record itself was never given its name.
    monkeypatch.setattr(experiments_prereg, "_discard", lambda paths: None)
    _fail_on_call(monkeypatch, "link", 2)
    with pytest.raises(OSError):
        freeze_v2(*campaign, tmp_path)
    left = sorted(path.name for path in interrupted_freeze_files(tmp_path))
    # The record's partial and the published note. The note's partial is that
    # note under a second name, and says nothing more.
    assert len(left) == 2
    assert sum(name.endswith(".partial") for name in left) == 1
    (note,) = (name for name in left if name.endswith(".md"))
    assert not (tmp_path / note).with_suffix(".json").exists()
    assert list(tmp_path.glob("prereg_*.json")) == []  # nothing a reader takes for a record

    # A second attempt under the same name refuses to publish past them.
    monkeypatch.undo()
    monkeypatch.setattr("latos.experiments.prereg.source_revision", lambda: SOURCE)
    record = read_record(freeze_v2(*campaign, tmp_path))  # under another, unused name
    assert verify_record(record) == ()
    with pytest.raises(FileExistsError):
        _write_once(record, tmp_path / note.replace(".md", ".json"))


def test_verification_reads_the_record_and_changes_nothing(
    tmp_path: Path, campaign: Campaign
) -> None:
    path = freeze_v2(*campaign, tmp_path)
    record = read_record(path)
    kept = copy.deepcopy(record)
    files = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.iterdir()}
    assert verify_record(record, expected_seal=record["seal"]["sha256"]) == ()
    assert record == kept
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.iterdir()} == files


@pytest.mark.parametrize("taken", ["json", "md"])
@pytest.mark.parametrize("writer", [_write_once, v1.write_record], ids=["v2", "v1"])
def test_both_writers_refuse_to_overwrite(
    tmp_path: Path, record: dict[str, Any], writer: Callable[..., Path], taken: str
) -> None:
    path = tmp_path / "prereg_x.json"
    existing = path if taken == "json" else path.with_suffix(".md")
    existing.write_text("already here", encoding="utf-8")
    with pytest.raises(FileExistsError, match="never overwritten"):
        writer(record, path)
    assert existing.read_text(encoding="utf-8") == "already here"


def test_one_name_and_version_cannot_carry_two_definitions(
    tmp_path: Path, campaign: Campaign
) -> None:
    first = freeze_v2(*campaign, tmp_path)
    definition = campaign[0]

    changed = replace(definition, variables=(replace(X, precision=0.02), *definition.variables[1:]))
    with pytest.raises(VersionCollision) as caught:
        freeze_v2(*_campaign(changed), tmp_path)
    message = str(caught.value)
    assert "example-campaign 1.0.0 was frozen earlier with a different definition" in message
    assert first.name in message
    assert definition_sha256(definition) in message
    assert definition_sha256(changed) in message
    assert len(list(tmp_path.glob("prereg_*.json"))) == 1  # nothing was written

    # A new version may carry the new definition; a display change needs no bump.
    freeze_v2(*_campaign(replace(changed, version="1.1.0")), tmp_path)
    freeze_v2(*_campaign(replace(definition, title="A better title")), tmp_path)
    assert len(list(tmp_path.glob("prereg_*.json"))) == 3


def test_other_files_in_the_directory_do_not_block_a_freeze(
    tmp_path: Path, campaign: Campaign
) -> None:
    result = campaign[3]
    (tmp_path / "prereg_old.json").write_text(
        json.dumps(v1.build_record(result, prior_best=result.best_y)), encoding="utf-8"
    )
    (tmp_path / "prereg_broken.json").write_text("{ not json", encoding="utf-8")
    (tmp_path / "prereg_list.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "prereg_old.outcome.json").write_text("{}", encoding="utf-8")
    assert verify_record(read_record(freeze_v2(*campaign, tmp_path))) == ()


def test_arguments_that_do_not_belong_together_are_refused(campaign: Campaign) -> None:
    definition, observation_set, prepared, result = campaign
    other = replace(definition, specimens_per_condition=2)
    with pytest.raises(ValueError, match="built under a different definition"):
        build_record_v2(other, observation_set, prepared, result)

    fewer = _campaign(rows=ROWS[:4])
    with pytest.raises(ValueError, match="not made from this observation set"):
        build_record_v2(definition, observation_set, fewer[2], result)
    with pytest.raises(ValueError, match="not computed from these prepared observations"):
        build_record_v2(definition, observation_set, prepared, fewer[3])


# ─── Reading ────────────────────────────────────────────────────────
def test_only_a_complete_version_two_record_is_read(
    tmp_path: Path, record: dict[str, Any], campaign: Campaign
) -> None:
    path = tmp_path / "record.json"
    result = campaign[3]

    path.write_text(json.dumps(v1.build_record(result, prior_best=1.0)), encoding="utf-8")
    with pytest.raises(RecordError, match=r"not a version 2 record \(schema_version: None\)"):
        read_record(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(RecordError, match="not a version 2 record"):
        read_record(path)
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(RecordError, match="is not JSON"):
        read_record(path)

    for keys in (
        ("observations", "y_raw"),
        ("projection",),
        ("experiment", "definition"),
        ("prediction_at_recommendation", "condition"),
    ):
        damaged = copy.deepcopy(record)
        block = damaged
        for key in keys[:-1]:
            block = block[key]
        del block[keys[-1]]
        path.write_text(json.dumps(damaged), encoding="utf-8")
        with pytest.raises(RecordError, match="lacks " + ".".join(keys)):
            read_record(path)


# ─── The note ───────────────────────────────────────────────────────
def test_the_note_says_what_a_reader_needs(tmp_path: Path, campaign: Campaign) -> None:
    path = freeze_v2(*campaign, tmp_path)
    record = read_record(path)
    note = path.with_suffix(".md").read_text(encoding="utf-8")

    # When the commitment is made is the seal's to show, not the note's to assert.
    assert f"_Recommendation computed {record['created_at']}. It is committed when its seal" in note
    assert "before the recommended specimen is made. Nothing in this file shows when" in note
    assert "committed in advance" not in note
    assert f"Definition SHA-256: `{record['experiment']['definition_sha256']}`" in note
    assert "Producer contract: `bench-table/1`" in note
    assert "Maximize **signal** (a.u.), as a magnitude" in note
    assert "Optimized as `abs_signal`" in note
    assert "| signal (a.u.), signed | optimized value (a.u.) | sigma (a.u.) |" in note
    assert "| 0 | -2.6 | 2.6 | 0.4 | lower_bound_specimen | a |" in note  # signed and optimized
    assert "`lower_bound_specimen`:" in note
    assert "source: `lower_bound`" in note
    # Every number carries its unit, and a lower-bound noise says the band is too narrow.
    prediction = record["prediction_at_recommendation"]
    assert f"= **{prediction['predicted_mean']:.4g} a.u.**" in note
    assert "Its noise is a lower bound here, so the band is too narrow" in note
    assert f"95% model interval: ± {prediction['ci95_model']:.4g} a.u.: the model's" in note
    assert "not predictive intervals" in note
    assert f"Prior best: {record['prior_best']:.4g} a.u.: the best value among the fitted" in note
    assert "**late**: control bake_time_min: recorded 70 min, expected 60 ± 2" in note
    assert "`bake_time_min`: expected 60.0 ± 2 min; checked against the value recorded" in note
    assert "`glovebox`: expected 'yes'; declared, not checked" in note  # no unit of "1"
    assert "changes sign across the measured conditions" in note
    # The stopping claim, and what the grade rests on, are stated beside the verdict.
    claim = record["stopping_claim"]
    assert f"with probability {claim['probability_within_epsilon']:.2f} under the frozen" in note
    assert "Reliability: **EXPLORATORY**" in note
    assert "calibration study on another material system (P3HT/CNT)" in note
    assert f"`{record['seal']['sha256']}`" in note
    assert "It covers the JSON record except its own seal block. This note is not sealed" in note
    assert "only if it verifies against the seal written down at the freeze" in note
    assert "## Source revision (frozen)" in note
    assert f"- Git commit: `{SOURCE.git_commit}`" in note
    assert "- Working tree: clean. No tracked or untracked file differed from this commit" in note
    assert "reproduces the code" not in note
    assert f"- latos {__version__}" in note
    assert "## Specimens kept outside the design bounds" not in note  # there are none here


# ─── Several design variables ───────────────────────────────────────
def test_a_two_variable_record_round_trips(tmp_path: Path) -> None:
    definition = make_nd_definition(2)
    points = [
        ((1.0, 1.0), 10.0),
        ((9.0, 1.0), 14.0),
        ((1.0, 9.0), 12.0),
        ((9.0, 9.0), 20.0),
        ((5.0, 5.0), 17.0),
        ((3.0, 7.0), 15.0),
    ]
    observations, excluded = build_observations(definition, make_nd_table(2, points))
    observation_set = reduce_observations(definition, observations, excluded)
    prepared = prepare_optimization(definition, observation_set)
    result = run_optimization(prepared)

    path = freeze_v2(definition, observation_set, prepared, result, v1.prereg_dir(tmp_path))
    record = read_record(path)
    assert verify_record(record) == ()
    assert record["backend"] == "optimize_nd"
    assert record["objective"]["input_variables"] == ["x0", "x1"]
    assert record["objective"]["search_bounds"] == [[0.0, 10.0], [0.0, 10.0]]
    assert "input_variable" not in record["objective"]
    assert "training_data" not in record
    prediction = record["prediction_at_recommendation"]
    assert "x" not in prediction
    assert list(prediction["condition"]) == ["x0", "x1"]
    assert list(prediction["condition"].values()) == list(result.recommendation.x)
    assert len(record["observations"]["x"][0]) == 2
    assert record["acquisition"]["maximised_by"] == result.config.acquisition
    assert "created_at" not in record["frozen_config"]

    changed = copy.deepcopy(record)
    changed["observations"]["x"][0][1] = 2.0
    assert any("observations do not hash" in p for p in verify_record(changed))
    # The existing one-variable listing skips it, as it skips any record with no scalar x.
    assert list_preregistrations(tmp_path) == []


# ─── The source revision ────────────────────────────────────────────
def test_the_record_says_which_source_revision_produced_it(record: dict[str, Any]) -> None:
    assert record["source_revision"] == {
        "git_commit": "ab" * 20,
        "working_tree": "clean",
        "changed_path_count": 0,
        "changed_paths": [],
        "note": "",
    }
    # Beside it, the three other things a reader needs to rebuild the result.
    assert record["latos_version"] == __version__
    assert len(record["experiment"]["definition_sha256"]) == 64
    assert record["experiment"]["producer"] == "bench-table/1"


def test_the_record_says_what_it_was_computed_in(record: dict[str, Any]) -> None:
    # The same commit on another platform or library version can differ in the
    # last digits of a fitted model, so the record states both.
    environment = record["environment"]
    assert environment == runtime_environment()
    assert environment["python"] == platform.python_version()
    assert environment["platform"] == platform.platform()
    assert environment["numpy"] == np.__version__
    assert environment["scipy"] == scipy.__version__
    assert environment["scikit-learn"] == sklearn.__version__
    note = render_note(record)
    assert f"numpy {np.__version__}, scipy {scipy.__version__}" in note
    assert f"scikit-learn {sklearn.__version__}" in note
    assert f"{platform.python_implementation()} {platform.python_version()} on " in note


def test_a_freeze_records_the_revision_it_is_given_and_does_not_judge_it(
    tmp_path: Path, campaign: Campaign
) -> None:
    dirty = SourceRevision(
        git_commit="cd" * 20, working_tree="dirty", changed_paths=(" M engine.py", "?? new.py")
    )
    path = freeze_v2(*campaign, tmp_path, source=dirty)
    record = read_record(path)
    assert record["source_revision"] == {
        "git_commit": "cd" * 20,
        "working_tree": "dirty",
        "changed_path_count": 2,
        "changed_paths": [" M engine.py", "?? new.py"],
        "note": "",
    }
    # A record that is honest about an uncommitted tree is not a damaged record.
    assert verify_record(record) == ()
    note = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "- Working tree: **DIRTY** (2 path(s) differ from the commit)." in note
    assert "The commit alone cannot be taken as the code that computed this record." in note


def test_a_tree_git_could_not_describe_is_recorded_with_the_reason(campaign: Campaign) -> None:
    unknown = SourceRevision(
        git_commit=None, working_tree="unknown", note="Git is not installed or not on the path"
    )
    record = build_record_v2(*campaign, source=unknown)
    assert record["source_revision"]["git_commit"] is None
    assert verify_record(record) == ()
    note = render_note(record)
    assert "- Git commit: `not available`" in note
    assert "- Working tree: **UNKNOWN** (Git is not installed or not on the path)." in note


def test_two_revisions_give_two_seals(campaign: Campaign) -> None:
    other = replace(SOURCE, git_commit="cd" * 20)
    first = build_record_v2(*campaign)
    second = build_record_v2(*campaign, source=other)
    assert first["seal"]["sha256"] != second["seal"]["sha256"]
    # The data and the definition are the same; only the code that ran differs.
    assert first["observations"]["sha256"] == second["observations"]["sha256"]
    assert first["experiment"]["definition_sha256"] == second["experiment"]["definition_sha256"]


def _drop_source(record: dict[str, Any]) -> None:
    del record["source_revision"]


def _drop_environment(record: dict[str, Any]) -> None:
    del record["environment"]


@pytest.mark.parametrize(
    ("change", "finding"),
    [
        (_drop_source, "does not say which source revision produced it"),
        (_drop_environment, "does not say what environment it was computed in"),
        (_set("source_revision", value="ab" * 20), "does not say which source revision"),
        (_set("source_revision", "working_tree", value="tidy"), "is not one a record can hold"),
        (_set("source_revision", "git_commit", value="HEAD"), "is not a commit hash"),
        (_set("source_revision", "git_commit", value="ab" * 19), "is not a commit hash"),
        (_set("source_revision", "git_commit", value=None), "clean working tree without a commit"),
        (
            _set("source_revision", "changed_path_count", value=3),
            "clean working tree without a commit",
        ),
    ],
)
def test_a_source_statement_that_contradicts_itself_is_caught_even_resealed(
    record: dict[str, Any], change: Callable[[dict[str, Any]], None], finding: str
) -> None:
    changed = copy.deepcopy(record)
    change(changed)
    changed["seal"]["sha256"] = record_seal(changed)  # the file agrees with itself again
    problems = verify_record(changed)
    assert any(finding in problem for problem in problems), problems
    # Only the source statement is wrong, and only it is reported.
    assert all(
        any(
            word in p
            for word in ("source revision", "working", "commit", "changed paths", "environment")
        )
        for p in problems
    ), problems


@pytest.mark.parametrize(
    ("change", "finding"),
    [
        (_set("source_revision", "changed_paths", value=[" M a.py"]), "changed paths listed"),
        (
            _set("source_revision", "working_tree", value="dirty"),
            "dirty working tree with no path that differs",
        ),
        (_set("source_revision", "note", value="tidied"), "clean working tree and lists"),
    ],
)
def test_a_source_statement_that_does_not_add_up_is_caught_even_resealed(
    record: dict[str, Any], change: Callable[[dict[str, Any]], None], finding: str
) -> None:
    changed = copy.deepcopy(record)
    change(changed)
    changed["seal"]["sha256"] = record_seal(changed)
    assert any(finding in problem for problem in verify_record(changed))


def test_an_unknown_tree_must_say_why(campaign: Campaign) -> None:
    unknown = SourceRevision(git_commit=None, working_tree="unknown", note="")
    problems = verify_record(build_record_v2(*campaign, source=unknown))
    assert any("unknown working tree without saying why" in p for p in problems)


# ─── A specimen outside the design bounds ───────────────────────────
def _narrow_campaign() -> Campaign:
    """The example with the search drawn above its first specimen, optimized signed."""
    narrow = make_definition(
        variables=(replace(X, bounds=(0.2, 1.0)), *VARIABLES[1:]),
        objective=ObjectiveDefinition(response="signal", goal=Goal.MAXIMIZE),
    )
    return _campaign(narrow)


def test_a_specimen_outside_the_bounds_is_frozen_whole_and_is_not_fitted(tmp_path: Path) -> None:
    campaign = _narrow_campaign()
    path = freeze_v2(*campaign, tmp_path)
    record = read_record(path)
    assert verify_record(record) == ()

    assert [row["sample_id"] for row in record["specimens"]] == ["b", "c", "d", "e"]
    observations = record["observations"]
    assert observations["y_raw"] == [2.0, 2.1, 2.2, 1.6]
    assert observations["y"] == observations["y_raw"]  # signed: nothing is projected
    assert observations["sample_ids"] == [["b"], ["c"], ["d"], ["e"]]
    assert record["objective"]["search_bounds"] == [0.2, 1.0]
    assert record["objective"]["property"] == "signal"

    excluded = {row["sample_id"]: row for row in record["excluded"]}
    assert list(excluded) == ["a", "late", "unread"]
    kept = excluded["a"]
    assert kept["category"] == "outside_design_bounds"
    assert kept["reasons"] == ["additive_fraction = 0 is outside its bounds [0.2, 1]"]
    assert kept["inputs"]["additive_fraction"] == 0.0
    assert kept["inputs"]["bake_time_min"] == 60.0
    assert kept["metadata"] == {"batch": "b1"}
    assert kept["responses"]["signal"] == {
        "value": -2.6,
        "unit": "a.u.",
        "uncertainty": 0.40,
        "uncertainty_scope": "specimen",
        "provenance": "bench-table/1: invented value",
    }
    assert kept["producer_detail"] == {"repeats": [-3.0, -2.6, -2.2], "ünïcode": "é"}
    assert kept["values"]["signal"] == -2.6
    assert excluded["late"]["category"] == excluded["unread"]["category"] == "rejected"
    assert "responses" not in excluded["late"]
    assert record["exclusion_glossary"] == dict(EXCLUSION_GLOSSARY)

    note = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Specimens kept outside the design bounds (not in the model)" in note
    assert (
        "- **a**: additive_fraction = 0 is outside its bounds [0.2, 1]. signal = -2.6 a.u., "
        "uncertainty 0.4 a.u. (specimen scope; a lower bound)"
    ) in note
    assert "Its only reason to be left out of the fit is that its design value" in note
    assert "## Left out of the model" in note
    assert "- **late**: control bake_time_min: recorded 70 min, expected 60 ± 2" in note
    assert "changes sign" not in note


@pytest.mark.parametrize(
    ("change", "finding"),
    [
        (_set("excluded", 0, "responses", "signal", "value", value=2.6), SEAL),
        (_set("excluded", 0, "category", value="rejected"), SEAL),
        (
            _set("excluded", 0, "producer_detail", "repeats", value=[2.6]),
            "producer detail of 'a' does not hash",
        ),
    ],
)
def test_the_specimen_kept_outside_the_bounds_cannot_be_altered_unseen(
    change: Callable[[dict[str, Any]], None], finding: str
) -> None:
    record = build_record_v2(*_narrow_campaign())
    assert verify_record(record) == ()
    change(record)
    assert any(finding in problem for problem in verify_record(record))


def test_when_nothing_is_rejected_the_note_says_so() -> None:
    narrow = make_definition(variables=(replace(X, bounds=(0.2, 1.0)), *VARIABLES[1:]))
    record = build_record_v2(*_campaign(narrow, ROWS[:5]))
    assert [row["category"] for row in record["excluded"]] == ["outside_design_bounds"]
    note = render_note(record)
    assert "## Specimens kept outside the design bounds (not in the model)" in note
    assert "- Nothing was rejected." in note


def test_a_control_with_no_tolerance_is_written_as_a_requirement() -> None:
    exact = make_definition(
        variables=tuple(replace(v, tolerance=0.0) if v is BAKE else v for v in VARIABLES)
    )
    record = build_record_v2(*_campaign(exact))
    controls = {control["name"]: control for control in record["experiment"]["controls"]}
    assert controls["bake_time_min"]["tolerance"] == 0.0
    note = render_note(record)
    assert "`bake_time_min`: required to equal 60.0 min, with no tolerance; checked" in note
    assert "**late**: control bake_time_min: recorded 70 min, expected exactly 60" in note
