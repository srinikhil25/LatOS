"""The thermal-cycle campaign end to end, on made-up recordings.

Every number here is invented. The recordings are written by
`tests.unit._synthetic_trace`, whose voltage is an exact line in the temperature
difference, so each expected value below is the value that was put in.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latos import campaign_trace
from latos.campaign_trace import (
    DEFINITION,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_UNREADABLE,
    load_campaign,
    main,
    observations_sha256,
    recommend,
    report,
)
from latos.core.exceptions import FitConvergenceError
from latos.experiments import (
    EXCLUDED_OUTSIDE_BOUNDS,
    Goal,
    InsufficientConditions,
    SourceRevision,
    UncertaintyScope,
    VariableKind,
    VariableRole,
    definition_sha256,
    definition_to_dict,
)
from latos.experiments import prereg as experiments_prereg
from latos.experiments.prereg import VersionCollision, read_record, verify_record
from latos.experiments.producers import ite_trace
from latos.optimization import engine
from latos.optimization.prereg import prereg_dir
from latos.optimization.validate import list_preregistrations
from tests.unit._synthetic_trace import (
    BENCH_DEFAULTS,
    SWAPPED,
    boundaries,
    make_recording,
    specimen,
    write_manifest,
)

S = ite_trace.APPARENT_SEEBECK
FOUR = boundaries((180.0,) * 4, (180.0,) * 4)
SECOND = "[HMIM][TFSI]"
LOW, HIGH = 0.239575, 1.0  # the search domain agreed for the campaign
COMMITTED = SourceRevision(git_commit="c0ffee" + "0" * 34, working_tree="clean")
OUTSIDE = f"aliquat_mass_fraction = 0 is outside its bounds [{LOW:g}, 1]"


@pytest.fixture(autouse=True)
def committed_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A committed tree, so no test depends on the state of this repository."""
    monkeypatch.setattr(campaign_trace, "source_revision", lambda: COMMITTED)
    monkeypatch.setattr(
        campaign_trace, "commit_known", lambda commit: commit == COMMITTED.git_commit
    )


def _uncommitted(monkeypatch: pytest.MonkeyPatch, revision: SourceRevision) -> None:
    monkeypatch.setattr(campaign_trace, "source_revision", lambda: revision)


def _mix(first_g: float, second_g: float) -> dict[str, float]:
    return {"Aliquat 336TG": first_g, SECOND: second_g}


def _reverse(name: str) -> dict[str, str]:
    return {"trace_file": name, "polarity_convention": SWAPPED}


def _five_rows(folder: Path) -> list[dict[str, Any]]:
    """Write five recordings shaped like the campaign's first five; return their manifest rows.

    All values invented. Four lie inside the search domain. "pure-second" is at
    zero, outside it.
    """
    # Both pure liquids: four cycles, the first a declared test run.
    make_recording(
        folder / "pure-second.xlsx", switches=FOUR, seebeck_mv_k=[-9.0, -3.0, -2.5, -2.0]
    )
    make_recording(folder / "pure-second-r.xlsx", seebeck_mv_k=[-2.0, -1.9, -1.8], lead_sign=-1)
    make_recording(folder / "pure-first.xlsx", switches=FOUR, seebeck_mv_k=[9.0, 1.5, 1.6, 1.7])
    make_recording(folder / "pure-first-r.xlsx", seebeck_mv_k=[1.7, 1.8, 1.9], lead_sign=-1)
    # One recording with no event marks, so its boundaries are declared.
    unmarked = make_recording(folder / "quarter.xlsx", marks=None, seebeck_mv_k=[2.04, 2.02, 2.00])
    make_recording(folder / "half.xlsx", seebeck_mv_k=[1.9, 2.0, 2.1])
    make_recording(folder / "half-r.xlsx", seebeck_mv_k=[1.9, 1.9, 2.0], lead_sign=-1)
    make_recording(folder / "three-quarters.xlsx", seebeck_mv_k=[2.15, 2.20, 2.25])
    declared = {"on": unmarked[0:6:2], "off": unmarked[1:6:2], "end": unmarked[6]}
    return [
        specimen(
            "pure-second",
            "pure-second.xlsx",
            pure_component=SECOND,
            test_cycles=1,
            reverse=_reverse("pure-second-r.xlsx"),
        ),
        specimen(
            "quarter", "quarter.xlsx", masses_g=_mix(0.06, 0.19), declared_boundaries_s=declared
        ),
        specimen("half", "half.xlsx", masses_g=_mix(0.07, 0.076), reverse=_reverse("half-r.xlsx")),
        specimen("three-quarters", "three-quarters.xlsx", masses_g=_mix(0.19, 0.06)),
        specimen(
            "pure-first",
            "pure-first.xlsx",
            test_cycles=1,
            reverse=_reverse("pure-first-r.xlsx"),
        ),
    ]


def _five(folder: Path) -> Path:
    """The five specimens of `_five_rows`, with their manifest."""
    return write_manifest(folder, _five_rows(folder), defaults=BENCH_DEFAULTS)


# What a person typed into the event column of a recording that had no marks:
# a label at each switch. The logger writes mark numbers there and nothing else.
TYPED_LABELS = tuple(
    f"Cycle {cycle} \N{EN DASH} Heater {phase}"
    for cycle in (1, 2, 3)
    for phase in ("ON (ΔT up)", "OFF (ΔT down)")
)
TYPED_REFUSAL = (
    f"quarter.xlsx: row 13 of the event column holds {TYPED_LABELS[0]!r}, not an event number"
)


def _annotated(folder: Path) -> Path:
    """`_five`, after labels were typed into the event column of the unmarked recording.

    The recording's switching times are declared in the manifest, so nothing
    needs its event column. Its measured columns are written exactly as before.
    """
    manifest = _five(folder)
    typed = list(zip(TYPED_LABELS, boundaries()[:6], strict=True))
    make_recording(folder / "quarter.xlsx", marks=typed, seebeck_mv_k=[2.04, 2.02, 2.00])
    return manifest


def _never_fit(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no model may be fitted here")


def _snapshot(folder: Path) -> dict[str, tuple[int, str]]:
    """Everything under `folder`: when it was last written, and what a file holds.

    Folders are in it too, so one that was made, or one a file was made in and
    taken out of again, shows.
    """
    return {
        str(path.relative_to(folder)): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "a folder",
        )
        for path in sorted(folder.rglob("*"))
    }


def _records(directory: Path) -> list[Path]:
    """The frozen records in `directory`, without their outcome files."""
    return sorted(p for p in directory.glob("prereg_*.json") if ".outcome." not in p.name)


def _with_design(**changes: Any) -> tuple[Any, ...]:
    """The campaign's variables, with `changes` made to the design variable."""
    return tuple(
        replace(v, **changes) if v.role is VariableRole.DESIGN else v for v in DEFINITION.variables
    )


def _redefine(monkeypatch: pytest.MonkeyPatch, **changes: Any) -> None:
    """Run the command under the campaign's definition with `changes` made to it."""
    monkeypatch.setattr(campaign_trace, "DEFINITION", replace(DEFINITION, **changes))
    monkeypatch.setattr(
        campaign_trace,
        "load_campaign",
        lambda path: load_campaign(path, definition=campaign_trace.DEFINITION),
    )


def _renamed(what: str) -> dict[str, Any]:
    """The definition fields that give the design variable, or the response, another name."""
    if what == "the design variable":
        return {"variables": _with_design(name="aliquat_fraction")}
    wanted = DEFINITION.objective.response
    return {
        "responses": tuple(
            replace(r, name="seebeck_cycle") if r.name == wanted else r
            for r in DEFINITION.responses
        ),
        "objective": replace(DEFINITION.objective, response="seebeck_cycle"),
    }


@pytest.fixture(scope="module")
def five(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _five(tmp_path_factory.mktemp("campaign"))


@pytest.fixture
def campaign(five: Path, tmp_path: Path) -> Path:
    """The five specimens in a folder of this test's own, so that it may freeze there.

    A campaign's records are kept beside its manifest, and `five` is shared.
    """
    folder = tmp_path / "campaign"
    shutil.copytree(five.parent, folder)
    return folder / five.name


def _one(folder: Path, **fields: Any) -> campaign_trace.CampaignState:
    """A campaign of the single specimen `run.xlsx`."""
    manifest = write_manifest(
        folder, [specimen("s1", "run.xlsx", **fields)], defaults=BENCH_DEFAULTS
    )
    return load_campaign(manifest)


# ─── The declaration ────────────────────────────────────────────────
def test_the_definition_declares_the_agreed_campaign() -> None:
    assert (DEFINITION.name, DEFINITION.version) == ("ionic-liquid-seebeck-trace", "1.0.0")
    # The producer contract is part of identity; the software version is not.
    assert DEFINITION.producer == "ite-trace/1"
    assert DEFINITION.sample_key == "sample_id"
    assert DEFINITION.specimens_per_condition == 1

    (design,) = DEFINITION.design_variables()
    assert design.name == "aliquat_mass_fraction"
    # The search covers the positive branch only, and nothing below it.
    assert (design.kind, design.unit, design.bounds) == (VariableKind.CONTINUOUS, "1", (LOW, HIGH))
    assert design.precision == 0.01

    response = DEFINITION.objective_response()
    assert (response.name, response.unit) == ("apparent_seebeck_cycle_mv_k", "mV/K")
    assert response.uncertainty_column == "apparent_seebeck_cycle_sem_mv_k"
    assert response.uncertainty_scope is UncertaintyScope.SPECIMEN
    assert response.valid_range == (None, None)  # signed: no sign is ruled out
    objective = DEFINITION.objective
    assert (objective.goal, objective.magnitude) == (Goal.MAXIMIZE, False)
    assert objective.target is None


def test_the_drive_programme_is_declared_as_controls() -> None:
    controls = {
        v.name: (v.kind, v.expected_value, v.tolerance)
        for v in DEFINITION.variables
        if v.role is VariableRole.CONTROL
    }
    phase = (VariableKind.CONTINUOUS, 180.0, 10.0)
    assert controls == {
        "drive_voltage_v": (VariableKind.CONTINUOUS, 0.5, 0.0),
        "on_duration_min_s": phase,
        "on_duration_max_s": phase,
        "off_duration_min_s": phase,
        "off_duration_max_s": phase,
        "cycles_per_specimen": (VariableKind.INTEGER, 3, None),
        "remounted_between_cycles": (VariableKind.CATEGORICAL, "no", None),
        "drops_applied": (VariableKind.INTEGER, 3, None),
    }


def test_the_drive_voltage_is_a_declared_set_point_required_exactly() -> None:
    drive = {v.name: v for v in DEFINITION.variables}["drive_voltage_v"]
    # No tolerance: the manifest either declares the campaign's value or it does
    # not. That is a rule about a declaration, and the wording says so.
    assert (drive.expected_value, drive.tolerance) == (0.5, 0.0)
    assert drive.display_name == "Declared drive set-point (not a measured voltage)"
    assert "declared set-point" in DEFINITION.description
    assert "not a measured quantity" in DEFINITION.description


@pytest.mark.parametrize(
    "changed",
    [
        {"bounds": (0.0, 1.0)},
        {"bounds": (LOW, 0.9)},
        {"precision": 0.02},
        {"precision": None},
    ],
)
def test_the_design_settings_are_part_of_the_definitions_identity(changed: dict[str, Any]) -> None:
    variables = tuple(
        replace(v, **changed) if v.role is VariableRole.DESIGN else v for v in DEFINITION.variables
    )
    assert definition_sha256(replace(DEFINITION, variables=variables)) != definition_sha256(
        DEFINITION
    )


def test_the_objective_and_the_set_point_rule_are_part_of_the_identity() -> None:
    frozen = definition_sha256(DEFINITION)
    magnitude = replace(DEFINITION, objective=replace(DEFINITION.objective, magnitude=True))
    assert definition_sha256(magnitude) != frozen

    for change in ({"tolerance": 0.05}, {"expected_value": 0.6}):
        variables = tuple(
            replace(v, **change) if v.name == "drive_voltage_v" else v for v in DEFINITION.variables
        )
        assert definition_sha256(replace(DEFINITION, variables=variables)) != frozen

    # How the set-point is worded for a reader is not what it is.
    reworded = tuple(
        replace(v, display_name="Drive") if v.name == "drive_voltage_v" else v
        for v in DEFINITION.variables
    )
    assert definition_sha256(replace(DEFINITION, variables=reworded)) == frozen


def test_an_unconfirmed_electrode_is_recorded_and_not_enforced() -> None:
    by_name = {v.name: v for v in DEFINITION.variables}
    electrode = by_name["electrode_material"]
    assert electrode.role is VariableRole.ENVIRONMENT
    assert electrode.expected_value is None
    text = json.dumps(campaign_trace.definition_sha256(DEFINITION)) + DEFINITION.description
    assert "gold" not in text.lower()


def test_every_column_the_definition_reads_is_one_the_producer_emits() -> None:
    read = {v.column for v in DEFINITION.variables if v.column}
    read |= {r.column for r in DEFINITION.responses}
    read |= {r.uncertainty_column for r in DEFINITION.responses if r.uncertainty_column}
    assert read | {DEFINITION.sample_key} == set(ite_trace.COLUMNS)


# ─── One specimen ───────────────────────────────────────────────────
def test_three_cycles_are_one_specimen_at_one_condition(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=[2.0, 2.1, 2.3])
    state = _one(tmp_path)

    assert len(state.table.rows) == 1
    (observation,) = state.observation_set.observations
    (condition,) = state.observation_set.reduced
    assert condition.n_specimens == 1
    assert condition.sample_ids == ("s1",)
    assert condition.y_raw == pytest.approx(6.4 / 3)
    # The cycle SEM arrives as a specimen-scope value, labelled a lower bound.
    value = observation.responses[S]
    assert value.uncertainty == pytest.approx(float(np.std([2.0, 2.1, 2.3], ddof=1)) / math.sqrt(3))
    assert value.uncertainty_scope is UncertaintyScope.SPECIMEN
    assert "lower bound" in value.provenance
    assert condition.sigma_source == "lower_bound_specimen"
    assert state.observation_set.pooled_sd is None  # cycles are repeats, never replicates
    assert state.enough is False


def test_the_signed_value_reaches_the_optimizer_and_no_magnitude_is_taken(tmp_path: Path) -> None:
    # A negative response inside the search domain: it must arrive negative.
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=[-3.0, -2.5, -2.0])
    state = _one(tmp_path)

    assert state.table.rows[0].values[S] == pytest.approx(-2.5)  # the producer
    assert state.observation_set.observations[0].responses[S].value == pytest.approx(-2.5)
    assert state.observation_set.reduced[0].y_raw == pytest.approx(-2.5)  # the reducer
    assert state.prepared.y_raw.tolist() == pytest.approx([-2.5])
    assert state.prepared.y.tolist() == pytest.approx([-2.5])  # and the optimizer's door
    assert state.prepared.objective_aggregation == "identity"
    assert state.prepared.target_name == "apparent_seebeck_cycle_mv_k"
    assert state.prepared.messages == ()  # one sign or two, nothing is projected


# ─── Controls ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("on_s", "off_s", "reason"),
    [
        ((180.0, 180.0, 180.0), (180.0, 180.0, 180.0), None),
        ((170.0, 190.0, 180.0), (190.0, 170.0, 180.0), None),  # the edges are inside
        ((180.0, 180.0, 180.0), (180.0, 191.0, 180.0), "off_duration_max_s: recorded 191 s"),
        ((180.0, 180.0, 180.0), (169.0, 180.0, 180.0), "off_duration_min_s: recorded 169 s"),
        ((180.0, 195.0, 180.0), (180.0, 180.0, 180.0), "on_duration_max_s: recorded 195 s"),
        ((160.0, 180.0, 180.0), (180.0, 180.0, 180.0), "on_duration_min_s: recorded 160 s"),
    ],
)
def test_a_phase_outside_ten_seconds_keeps_the_specimen_out_of_the_model(
    tmp_path: Path, on_s: tuple[float, ...], off_s: tuple[float, ...], reason: str | None
) -> None:
    make_recording(tmp_path / "run.xlsx", switches=boundaries(on_s, off_s))
    state = _one(tmp_path)
    if reason is None:
        assert state.observation_set.excluded == ()
        assert len(state.observation_set.observations) == 1
        return
    assert state.observation_set.observations == ()
    (excluded,) = state.observation_set.excluded
    assert excluded.reasons == (f"control {reason}, expected 180 ± 10",)
    # Kept, with what was measured: left out of the model, not out of the record.
    assert excluded.record.values[S] == pytest.approx(2.0)
    assert f"s1: NOT in the model. control {reason}" in report(state)


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"drive_voltage_v": 0.6}, "control drive_voltage_v: recorded 0.6 V, expected exactly 0.5"),
        (
            {"drive_voltage_v": 0.501},
            "control drive_voltage_v: recorded 0.501 V, expected exactly 0.5",
        ),
        ({"drops_applied": 4}, "control drops_applied: recorded 4, expected 3"),
        (
            {"remounted_between_cycles": "yes"},
            "control remounted_between_cycles: recorded 'yes', expected 'no'",
        ),
    ],
)
def test_a_declared_control_off_its_value_excludes(
    tmp_path: Path, fields: dict[str, Any], reason: str
) -> None:
    make_recording(tmp_path / "run.xlsx")
    (excluded,) = _one(tmp_path, **fields).observation_set.excluded
    assert excluded.reasons == (reason,)


def test_four_cycles_are_excluded_unless_the_first_was_declared_a_test_run(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", switches=FOUR, seebeck_mv_k=[9.0, 2.0, 2.1, 2.3])
    (excluded,) = _one(tmp_path).observation_set.excluded
    assert excluded.reasons == ("control cycles_per_specimen: recorded 4, expected 3",)

    declared = _one(tmp_path, test_cycles=1)
    assert declared.observation_set.excluded == ()
    assert declared.observation_set.reduced[0].y_raw == pytest.approx(6.4 / 3)  # not 3.85


def test_a_refused_specimen_is_excluded_with_the_producers_reason(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    state = _one(tmp_path, polarity_convention=None)
    (excluded,) = state.observation_set.excluded
    (reason,) = excluded.reasons
    assert "polarity_convention is not recorded" in reason
    assert "NOT in the model" in report(state)


def test_an_unrecorded_control_is_noted_and_does_not_exclude(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    manifest = write_manifest(
        tmp_path,
        [specimen("s1", "run.xlsx")],
        defaults={k: v for k, v in BENCH_DEFAULTS.items() if k != "drive_voltage_v"},
    )
    state = load_campaign(manifest)
    (observation,) = state.observation_set.observations
    assert [i.message for i in observation.issues if i.field == "control:drive_voltage_v"] == [
        "control drive_voltage_v not recorded for this specimen"
    ]


@pytest.mark.parametrize(
    "control", ["drive_voltage_v", "drops_applied", "remounted_between_cycles"]
)
def test_an_unrecorded_control_is_kept_unverified_and_is_no_reason_to_refuse(
    control: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Rule R4 of the approved definition: a blank control cell keeps the
    # specimen, flagged "not recorded". Only a recorded deviation excludes.
    rows = _five_rows(tmp_path)
    defaults = {k: v for k, v in BENCH_DEFAULTS.items() if k != control}
    # Declared for every specimen but "half".
    rows = [
        row if row["sample_id"] == "half" else {**row, control: BENCH_DEFAULTS[control]}
        for row in rows
    ]
    manifest = write_manifest(tmp_path, rows, defaults=defaults)
    state = load_campaign(manifest)
    assert state.observation_set.rejected() == ()
    assert "half" in {o.sample_id for o in state.observation_set.observations}
    before = _snapshot(tmp_path)
    assert main(["check", str(manifest)]) == EXIT_OK
    printed = capsys.readouterr().out
    assert f"{control} NOT recorded, so unverified for this specimen" in printed
    assert f"control {control} not recorded for this specimen" in printed  # the warning
    assert "no refusal applies to this data and this source tree" in printed
    assert _snapshot(tmp_path) == before


def test_a_reversed_lead_run_is_optional_and_never_a_reason_to_refuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Diagnostics only, by the approved protocol: absent, or not reversing the sign,
    # it is reported and stops nothing. No threshold is applied to it.
    rows = _five_rows(tmp_path)
    # The leads of "half" were not in fact swapped; "pure-first" declares no such run.
    make_recording(tmp_path / "half-r.xlsx", seebeck_mv_k=[1.9, 1.9, 2.0], lead_sign=1)
    rows = [
        {k: v for k, v in row.items() if k != "reverse"}
        if row["sample_id"] == "pure-first"
        else row
        for row in rows
    ]
    manifest = write_manifest(tmp_path, rows, defaults=BENCH_DEFAULTS)
    assert main(["check", str(manifest)]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "swapping the leads did not reverse the recorded sign (half-r.xlsx" in printed
    assert "no refusal applies to this data and this source tree" in printed


# ─── Five specimens ─────────────────────────────────────────────────
def test_the_four_specimens_inside_the_domain_reach_the_optimizers_door(five: Path) -> None:
    state = load_campaign(five)
    assert [o.sample_id for o in state.observation_set.observations] == [
        "quarter",
        "half",
        "three-quarters",
        "pure-first",
    ]
    prepared = state.prepared
    assert prepared.x[:, 0].tolist() == pytest.approx([0.06 / 0.25, 0.07 / 0.146, 0.19 / 0.25, 1.0])
    assert prepared.y_raw.tolist() == pytest.approx([2.02, 2.0, 2.2, 1.6])
    assert prepared.y.tolist() == prepared.y_raw.tolist()  # signed, handed over unchanged
    assert prepared.direction == "maximize"
    assert prepared.bounds == ((LOW, HIGH),)
    assert prepared.input_names == ("aliquat_mass_fraction",)
    assert prepared.sigma_sources == ("lower_bound_specimen",) * 4
    assert prepared.prediction_noise_source == "lower_bound"
    assert state.enough is True


def test_the_specimen_outside_the_domain_is_preserved_and_not_fitted(five: Path) -> None:
    state = load_campaign(five)
    observation_set = state.observation_set
    assert observation_set.rejected() == ()  # nothing was rejected
    (outside,) = observation_set.outside_bounds()
    assert observation_set.excluded == (outside,)
    assert (outside.sample_id, outside.category) == ("pure-second", EXCLUDED_OUTSIDE_BOUNDS)
    assert outside.reasons == (OUTSIDE,)  # the domain, and nothing else

    # Kept as a specimen is kept: signed value, labelled uncertainty, its recording.
    kept = outside.observation
    assert kept is not None
    assert kept.inputs["aliquat_mass_fraction"] == 0.0
    value = kept.responses[S]
    assert value.value == pytest.approx(-2.5)
    assert value.uncertainty == pytest.approx(0.5 / math.sqrt(3))
    assert value.uncertainty_scope is UncertaintyScope.SPECIMEN
    assert outside.record.values[S] == pytest.approx(-2.5)
    detail = outside.record.producer_detail
    assert detail is not None
    assert [c["purpose"] for c in detail["cycles"]] == [
        "test_run",
        "measurement",
        "measurement",
        "measurement",
    ]

    # And it is in no array the model is fitted to.
    assert 0.0 not in state.prepared.x[:, 0].tolist()
    assert all(v > 0 for v in state.prepared.y.tolist())
    assert all("pure-second" not in c.sample_ids for c in observation_set.reduced)

    (note,) = observation_set.messages
    assert (
        "pure-second (aliquat_mass_fraction = 0; apparent_seebeck_cycle_mv_k = -2.5 mV/K)" in note
    )
    assert "the bounds alone keep them out" in note
    text = report(state)
    assert "Specimens preserved outside the search bounds: 1" in text
    assert "Not rejected, and not outliers." in text
    assert f"    NOT in the fit: {OUTSIDE}" in text
    assert "    signed S = -2.5000 mV/K   cycles: -3.0000, -2.5000, -2.0000" in text
    assert "NOT in the model" not in text


def test_the_edge_of_the_domain_is_inside_it(tmp_path: Path) -> None:
    # Invented masses that land on the lower bound and just under it.
    for name in ("on.xlsx", "under.xlsx"):
        make_recording(tmp_path / name)
    manifest = write_manifest(
        tmp_path,
        [
            specimen("on", "on.xlsx", masses_g=_mix(LOW, 1.0 - LOW)),
            specimen("under", "under.xlsx", masses_g=_mix(0.2395, 0.7605)),
        ],
        defaults=BENCH_DEFAULTS,
    )
    state = load_campaign(manifest)
    assert [o.sample_id for o in state.observation_set.observations] == ["on"]
    (outside,) = state.observation_set.outside_bounds()
    assert outside.sample_id == "under"


def test_a_declared_quality_warning_travels_with_a_specimen_and_excludes_nothing(
    tmp_path: Path,
) -> None:
    note = "strong drift within the session"
    for name in ("in.xlsx", "out.xlsx"):
        make_recording(tmp_path / name, seebeck_mv_k=[1.9, 2.0, 2.1])
    manifest = write_manifest(
        tmp_path,
        [
            specimen("in", "in.xlsx", qc_warnings=[note]),
            specimen("out", "out.xlsx", pure_component=SECOND, qc_warnings=[note]),
        ],
        defaults=BENCH_DEFAULTS,
    )
    state = load_campaign(manifest)
    # Inside the domain: warned, and in the fit all the same.
    (inside,) = state.observation_set.observations
    assert [i.message for i in inside.issues] == [f"in: declared QC warning: {note}"]
    # Outside it: the domain is the only reason; the warning is not one.
    (outside,) = state.observation_set.outside_bounds()
    assert outside.reasons == (OUTSIDE,)
    assert outside.observation is not None
    assert [i.message for i in outside.observation.issues] == [f"out: declared QC warning: {note}"]
    text = report(state)
    assert f"    WARNING: out: declared QC warning: {note}" in text
    assert f"  WARNING: in: declared QC warning: {note}" in text


def test_two_specimens_within_the_precision_are_one_condition(tmp_path: Path) -> None:
    # 0.500, 0.509 and 0.520: the first two are within 0.01, the third is not.
    for name in ("a.xlsx", "b.xlsx", "c.xlsx"):
        make_recording(tmp_path / name)
    manifest = write_manifest(
        tmp_path,
        [
            specimen("a", "a.xlsx", masses_g=_mix(0.500, 0.500)),
            specimen("b", "b.xlsx", masses_g=_mix(0.509, 0.491)),
            specimen("c", "c.xlsx", masses_g=_mix(0.520, 0.480)),
        ],
        defaults=BENCH_DEFAULTS,
    )
    reduced = load_campaign(manifest).observation_set.reduced
    assert [c.sample_ids for c in reduced] == [("a", "b"), ("c",)]


def test_reversed_runs_and_test_cycles_are_reported_and_are_not_observations(five: Path) -> None:
    state = load_campaign(five)
    # Eight recordings, twenty-six cycles, five specimens: four fitted, one kept outside.
    assert len(state.table.rows) == 5
    assert len(state.observation_set.observations) == 4
    assert all(c.n_specimens == 1 for c in state.observation_set.reduced)

    text = report(state)
    assert "Specimens in the fit: 4" in text
    assert "pure-second: cycle 1 (10 to 370 s, S = -9.0000 mV/K) is a declared test run" in text
    assert "pure-first: cycle 1 (10 to 370 s, S = +9.0000 mV/K) is a declared test run" in text
    assert (
        text.count("is a reversed-lead diagnostic, not an observation (recorded sign reverses") == 3
    )
    assert "boundaries: declared; controls within tolerance" in text
    assert "(cycle SEM, 2 d.o.f.; within-specimen lower bound)" in text
    assert text.isascii()


def test_the_report_states_the_domain_and_that_nothing_is_projected(five: Path) -> None:
    state = load_campaign(five)
    assert state.prepared.messages == ()  # no magnitude is taken, so no sign note
    text = report(state)
    assert "Conditions handed to the optimizer as apparent_seebeck_cycle_mv_k (maximize)" in text
    assert f"  search domain: aliquat_mass_fraction in [{LOW:g}, 1]" in text
    assert "  projection: none: the signed values are handed over unchanged" in text
    assert (
        "  4 distinct condition(s); the one-variable optimizer, optimize() needs at least 3" in text
    )
    assert "  x = 0.240000   signed = +2.0200   optimized = +2.0200   sigma = " in text
    assert "x = 0.000000" not in text
    assert "[lower_bound_specimen] A within-specimen uncertainty" in text
    assert "prediction noise = " in text
    assert "[lower_bound]" in text


def test_one_sign_gives_no_sign_warning(tmp_path: Path) -> None:
    for name, slope in (("a.xlsx", 2.0), ("b.xlsx", 2.2), ("c.xlsx", 1.6)):
        make_recording(tmp_path / name, seebeck_mv_k=[slope, slope + 0.1, slope + 0.2])
    manifest = write_manifest(
        tmp_path,
        [
            specimen("a", "a.xlsx", masses_g=_mix(0.1, 0.3)),
            specimen("b", "b.xlsx", masses_g=_mix(0.2, 0.2)),
            specimen("c", "c.xlsx"),
        ],
        defaults=BENCH_DEFAULTS,
    )
    state = load_campaign(manifest)
    assert state.prepared.messages == ()
    assert "Warnings\n  none" in report(state)


# ─── The optimizer ──────────────────────────────────────────────────
def test_the_campaign_runs_the_existing_one_variable_optimizer(
    five: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real = engine.optimize

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("one design variable must not reach optimize_nd")

    monkeypatch.setattr(engine, "optimize", spy)
    monkeypatch.setattr(engine, "optimize_nd", never)
    state = load_campaign(five)
    assert calls == []  # reading and preparing fits nothing
    result = recommend(state)

    ((args, kwargs),) = calls
    assert args[0].tolist() == state.prepared.x[:, 0].tolist()
    assert len(args[0]) == 4  # the specimen outside the domain is not among them
    assert args[1].tolist() == state.prepared.y_raw.tolist()  # signed, as measured
    assert args[1].tolist() == pytest.approx([2.02, 2.0, 2.2, 1.6])
    assert kwargs["point_noise"] is state.prepared.sigma
    assert kwargs["measured_noise"] == state.prepared.prediction_noise
    assert kwargs["bounds"] == (LOW, HIGH)
    assert kwargs["input_name"] == "aliquat_mass_fraction"
    assert kwargs["target_name"] == "apparent_seebeck_cycle_mv_k"
    assert kwargs["direction"] == "maximize"
    assert kwargs["objective_aggregation"] == "identity"
    assert isinstance(result, engine.OptimizationResult)
    assert LOW <= result.recommendation.x <= HIGH
    # Four conditions: the grade is exploratory, so no stop is claimed.
    assert result.reliability is not None
    assert result.reliability.level == "exploratory"
    assert result.converged is False
    assert result.stopping is not None
    assert result.stopping.should_stop is False
    assert result.stopping.data_sufficient is False


def test_too_few_conditions_are_refused_before_any_model(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    with pytest.raises(InsufficientConditions):
        recommend(_one(tmp_path))


# ─── The command ────────────────────────────────────────────────────
READY = (
    "Readiness for `next`\n  no refusal applies to this data and this source tree. A warning "
    "above is not a refusal: read each one before running `next`."
)


def test_check_reports_and_neither_fits_nor_writes(
    five: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("check must not fit a model")

    monkeypatch.setattr(engine, "optimize", never)
    before = sorted(p.name for p in five.parent.iterdir())
    assert main(["check", str(five)]) == 0
    printed = capsys.readouterr().out
    assert "Specimens in the fit: 4" in printed
    assert "Specimens preserved outside the search bounds: 1" in printed
    assert f"Source revision: {COMMITTED.git_commit} (working tree clean)" in printed
    assert "Recommendation" not in printed
    assert printed.rstrip().endswith(READY)
    assert sorted(p.name for p in five.parent.iterdir()) == before
    assert not prereg_dir(five.parent).exists()


def test_check_says_when_the_source_tree_is_not_a_commit(
    five: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dirty = replace(COMMITTED, working_tree="dirty", changed_paths=(" M a.py", "?? b.py"))
    _uncommitted(monkeypatch, dirty)
    # Every specimen can be used, and `next` would still refuse: NOT READY.
    assert main(["check", str(five)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert f"{COMMITTED.git_commit} (working tree DIRTY: 2 path(s) differ)" in printed
    assert "`next` freezes nothing until the source tree is exactly a commit." in printed
    assert "NOT READY" in printed
    assert "could not be used" not in printed


def test_next_freezes_a_record_that_verifies(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["next", str(campaign)]) == 0
    printed = capsys.readouterr().out
    (path,) = _records(prereg_dir(campaign.parent))
    record = read_record(path)

    assert verify_record(record) == ()
    assert f"Frozen: {path}" in printed
    assert f"Source: {COMMITTED.git_commit} (working tree clean)" in printed
    assert f"Seal:   {record['seal']['sha256']}" in printed
    assert "Write the seal in the lab notebook now" in printed
    assert "make aliquat_mass_fraction = " in printed
    assert "changes sign" not in printed

    experiment = record["experiment"]
    assert (experiment["name"], experiment["version"]) == ("ionic-liquid-seebeck-trace", "1.0.0")
    assert experiment["producer"] == "ite-trace/1"
    assert experiment["definition_sha256"] == definition_sha256(DEFINITION)
    assert experiment["definition"]["objective"]["magnitude"] is False
    assert record["source_revision"] == COMMITTED.to_dict()
    assert record["latos_version"]
    assert record["backend"] == "optimize"
    assert record["objective"]["property"] == "apparent_seebeck_cycle_mv_k"
    assert record["objective"]["aggregation"] == "identity"
    assert record["objective"]["search_bounds"] == [LOW, HIGH]
    observations = record["observations"]
    assert observations["y_raw"] == pytest.approx([2.02, 2.0, 2.2, 1.6])  # signed, in the record
    assert observations["y"] == observations["y_raw"]
    assert observations["sigma_sources"] == ["lower_bound_specimen"] * 4
    assert record["prediction_noise"]["source"] == "lower_bound"
    assert list(record["prediction_at_recommendation"]["condition"]) == ["aliquat_mass_fraction"]
    assert record["converged"] is False
    assert record["reliability"]["level"] == "exploratory"

    # The four fitted specimens, no reversed run among them.
    assert [s["sample_id"] for s in record["specimens"]] == [
        "quarter",
        "half",
        "three-quarters",
        "pure-first",
    ]
    assert {s["metadata"]["trace_file"] for s in record["specimens"]}.isdisjoint(
        {"pure-second-r.xlsx", "pure-first-r.xlsx", "half-r.xlsx"}
    )
    assert record["specimens"][0]["metadata"]["boundary_source"] == "declared"
    assert record["specimens"][0]["inputs"]["electrode_material"] is None
    last = record["specimens"][-1]
    assert last["responses"][S]["uncertainty_scope"] == "specimen"
    detail = last["producer_detail"]
    assert detail["contract"] == "ite-trace/1"
    assert detail["cycles"][0]["purpose"] == "test_run"
    assert detail["cycles"][0]["included_in_specimen_value"] is False
    assert detail["reverse_diagnostic"]["recorded_sign_reverses"] is True
    assert len(last["producer_detail_sha256"]) == 64


def test_the_frozen_record_keeps_the_specimen_outside_the_domain_whole(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["next", str(campaign)]) == 0
    (path,) = _records(prereg_dir(campaign.parent))
    record = read_record(path)
    assert verify_record(record) == ()
    observations = record["observations"]

    # Frozen as a specimen is frozen, labelled, and not fitted.
    (kept,) = record["excluded"]
    assert (kept["sample_id"], kept["category"]) == ("pure-second", "outside_design_bounds")
    assert kept["reasons"] == [OUTSIDE]
    assert kept["inputs"]["aliquat_mass_fraction"] == 0.0
    assert kept["responses"][S]["value"] == pytest.approx(-2.5)
    assert kept["responses"][S]["uncertainty_scope"] == "specimen"
    assert kept["producer_detail"]["cycles"][0]["purpose"] == "test_run"
    assert kept["producer_detail"]["reverse_diagnostic"]["recorded_sign_reverses"] is True
    assert len(kept["producer_detail_sha256"]) == 64
    assert "outside_design_bounds" in record["exclusion_glossary"]
    assert "pure-second" not in [i for ids in observations["sample_ids"] for i in ids]

    note = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Specimens kept outside the design bounds (not in the model)" in note
    assert "## Source revision (frozen)" in note
    assert COMMITTED.git_commit is not None
    assert COMMITTED.git_commit in note


@pytest.mark.parametrize(
    "revision",
    [
        replace(COMMITTED, working_tree="dirty", changed_paths=("?? new_module.py",)),
        SourceRevision(git_commit=None, working_tree="unknown", note="not a working tree"),
        SourceRevision(git_commit=None, working_tree="clean"),
    ],
)
def test_next_refuses_before_any_model_unless_the_source_is_exactly_a_commit(
    revision: SourceRevision,
    five: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a recommendation must not be computed from an uncommitted tree")

    monkeypatch.setattr(engine, "optimize", never)
    _uncommitted(monkeypatch, revision)
    assert main(["next", str(five)]) == 1
    printed = capsys.readouterr().out
    assert "Nothing was fitted and nothing was frozen." in printed
    assert revision.describe() in printed
    assert "Recommendation" not in printed
    assert not prereg_dir(five.parent).exists()


def test_verify_accepts_the_frozen_record_and_rejects_an_edited_one(
    campaign: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["next", str(campaign)])
    (path,) = _records(prereg_dir(campaign.parent))
    seal = read_record(path)["seal"]["sha256"]
    capsys.readouterr()

    assert main(["verify", str(path)]) == 0
    assert main(["verify", str(path), "--seal", seal]) == 0
    printed = capsys.readouterr().out
    assert "verifies against its own seal and the seal given" in printed
    assert f"Source revision: {COMMITTED.git_commit} (working tree clean)" in printed
    assert main(["verify", str(path), "--seal", "0" * 64]) == 1

    # Rewritten after a later measurement: the prediction is moved.
    record = json.loads(path.read_text(encoding="utf-8"))
    record["prediction_at_recommendation"]["predicted_mean"] += 1.0
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(record), encoding="utf-8")
    assert main(["verify", str(edited), "--seal", seal]) == 1
    assert "does not match its seal" in capsys.readouterr().out
    assert main(["verify", str(tmp_path / "absent.json")]) == 2
    assert main(["verify", str(campaign)]) == 2  # a manifest is not a record
    capsys.readouterr()
    assert main(["verify", str(tmp_path)]) == EXIT_UNREADABLE  # nor is a folder
    assert f"Could not read the record {tmp_path}" in capsys.readouterr().out


def _frozen(campaign: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """Freeze the campaign once and return its record."""
    assert main(["next", str(campaign)]) == EXIT_OK
    capsys.readouterr()
    (path,) = _records(prereg_dir(campaign.parent))
    return path


def _resealed(path: Path, change: Any) -> None:
    """Edit a record and seal it again, as whoever rewrites one would."""
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    record["seal"]["sha256"] = experiments_prereg.record_seal(record)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_verify_reads_and_changes_nothing(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    before = _snapshot(campaign.parent)
    assert main(["verify", str(path), "--seal", read_record(path)["seal"]["sha256"]]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "Its note is the one it renders" in printed
    assert "gives the values the record holds" in printed
    assert _snapshot(campaign.parent) == before


@pytest.mark.parametrize("how", ["overwritten", "removed"])
def test_verify_says_when_a_recording_is_not_the_one_frozen(
    how: str, campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    recording = campaign.parent / "half.xlsx"
    if how == "overwritten":
        make_recording(recording, seebeck_mv_k=[1.0, 1.0, 1.0])
    else:
        recording.unlink()
    assert main(["verify", str(path)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    said = "is not the one the record was frozen from" if how == "overwritten" else "is not in"
    assert f"  - the recording half.xlsx {said}" in printed


def test_a_reversed_lead_recording_is_compared_too(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    reverse = next(campaign.parent.glob("*-r.xlsx"))
    make_recording(reverse, seebeck_mv_k=[0.5, 0.5, 0.5])
    assert main(["verify", str(path)]) == EXIT_REFUSED
    assert f"the recording {reverse.name} is not the one" in capsys.readouterr().out


def test_a_manifest_extended_after_a_freeze_is_noted_not_refused(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    with campaign.open("a", encoding="utf-8") as manifest:
        manifest.write("\n# the next specimen goes here\n")
    assert main(["verify", str(path)]) == EXIT_OK
    assert f"NOTE: the manifest {campaign.name} has changed since the freeze" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("change", "finding"),
    [
        (
            lambda r: r["specimens"][0]["metadata"].update(trace_sha256="0" * 64),
            "the recording SHA-256 it states is not the one its producer read",
        ),
        (
            lambda r: r["specimens"][0]["producer_detail"].update(contract="ite-trace/2"),
            "its producer detail is not from ite-trace/1",
        ),
        (
            lambda r: r["specimens"][0]["producer_detail"].update(sample_id="another"),
            "its producer detail is for another specimen",
        ),
        (
            lambda r: r["specimens"][0]["producer_detail"]["manifest"].update(sha256="0" * 64),
            "were not all read from one manifest",
        ),
        (
            lambda r: r["specimens"][0]["producer_detail"]["protocol"].update(edited=True),
            "do not all state the same protocol",
        ),
        (
            lambda r: r["source_revision"].update(
                working_tree="dirty", changed_path_count=1, changed_paths=[" M a.py"]
            ),
            "a source tree that was not exactly a commit",
        ),
        (
            lambda r: r["source_revision"].update(git_commit="ab" * 20),
            f"the commit {'ab' * 20} is not in this repository",
        ),
    ],
)
def test_verify_catches_what_a_resealed_record_no_longer_shares_with_its_campaign(
    change: Any, finding: str, campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    _resealed(path, change)
    assert main(["verify", str(path)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert finding in printed, printed
    # The note beside it is still the one the record rendered when it was frozen.
    assert "is not the note this record renders" in printed


def _set_value(record: dict[str, Any], sample_id: str, value: float) -> None:
    """Give one fitted specimen another value, and redo everything derived from it."""
    (row,) = (row for row in record["specimens"] if row["sample_id"] == sample_id)
    row["responses"][ite_trace.APPARENT_SEEBECK]["value"] = value
    block = record["observations"]
    index = next(i for i, ids in enumerate(block["sample_ids"]) if sample_id in ids)
    block["y_raw"][index] = block["y"][index] = value
    block["sha256"] = experiments_prereg.observations_digest_nd(
        block["x"], block["y"], block["y_raw"], sigma=block["sigma"]
    )


@pytest.mark.parametrize(
    ("change", "finding"),
    [
        # A value rewritten consistently passes the record's own checks; its recording
        # read again does not give it.
        (lambda r: _set_value(r, "half", 1.0), "half: its recording, read again"),
        (lambda r: r["acquisition"].update(function="ucb"), "its acquisition is 'ucb'"),
        (lambda r: r["stopping_claim"].update(delta=0.5), "stopping risk delta"),
        (lambda r: r["stopping_claim"].update(epsilon=10.0), "stopping tolerance epsilon"),
        (lambda r: r.update(prior_best=0.0), "its prior best is not the best fitted value"),
        (lambda r: r.pop("reliability"), "holds no reliability grade"),
        (
            lambda r: r.update(
                excluded=[x for x in r["excluded"] if x["sample_id"] != "pure-second"]
            ),
            "does not hold exactly its specimens",
        ),
    ],
)
def test_verify_holds_a_resealed_record_to_its_recordings_and_to_how_next_freezes(
    change: Any, finding: str, campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    _resealed(path, change)
    assert main(["verify", str(path)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert finding in printed, printed


def test_an_edited_note_is_caught_even_with_the_seal(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    seal = read_record(path)["seal"]["sha256"]
    note = path.with_suffix(".md")
    text = note.read_text(encoding="utf-8")
    note.write_text(text.replace("Predicted", "Predicted (edited)"), encoding="utf-8")
    assert main(["verify", str(path), "--seal", seal]) == EXIT_REFUSED
    assert "is not the note this record renders" in capsys.readouterr().out
    note.unlink()
    assert main(["verify", str(path), "--seal", seal]) == EXIT_REFUSED
    assert f"its note {note.name} is not beside it" in capsys.readouterr().out


def test_a_record_of_another_producer_is_not_this_campaigns(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)

    def another(record: dict[str, Any]) -> None:
        record["experiment"]["producer"] = "bench-table/1"

    _resealed(path, another)
    assert main(["verify", str(path)]) == EXIT_REFUSED
    assert "it is not a record of the ite-trace/1 producer" in capsys.readouterr().out


def test_a_version_one_record_of_the_same_data_stops_next(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = load_campaign(campaign)
    directory = prereg_dir(campaign.parent)
    directory.mkdir(parents=True)
    same = {
        "kind": "latos.bo.prereg",
        "training_data": {
            "x": state.prepared.x[:, 0].tolist(),
            "y": state.prepared.y.tolist(),
        },
    }
    (directory / "prereg_20260101T000000Z.json").write_text(json.dumps(same), encoding="utf-8")
    monkeypatch.setattr(engine, "optimize", _never_fit)
    for command in ("check", "next"):
        assert main([command, str(campaign)]) == EXIT_REFUSED
        assert "is a version 1 record whose model saw exactly these conditions" in (
            capsys.readouterr().out
        )

    # One of other data is passed over, as before.
    other = {**same, "training_data": {"x": [0.5], "y": [1.0]}}
    (directory / "prereg_20260101T000000Z.json").write_text(json.dumps(other), encoding="utf-8")
    assert main(["check", str(campaign)]) == EXIT_OK


def test_a_test_run_with_no_slope_is_said_not_crashed() -> None:
    detail = {
        "cycles": [
            {"purpose": "test_run", "on_s": 5.0, "end_s": 370.0, "index": 1, "slope_mv_k": None}
        ]
    }
    (line,) = campaign_trace._left_out_lines("flat", detail)
    assert "cycle 1 (5 to 370 s, no slope could be fitted to it) is a declared test run" in line


def test_a_lock_this_run_made_and_could_not_finish_is_removed(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    real_open = Path.open

    class FullDisk:
        """The lock file is created, and writing into it fails."""

        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> FullDisk:
            return self

        def __exit__(self, *args: Any) -> None:
            self.handle.close()

        def write(self, text: str) -> int:
            raise OSError(28, "No space left on device")

    def failing_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        return FullDisk(handle) if self.name == campaign_trace.FREEZE_LOCK else handle

    monkeypatch.setattr(Path, "open", failing_open)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    assert "the record folder could not be held for this freeze" in printed
    assert not (directory / campaign_trace.FREEZE_LOCK).exists()
    assert _records(directory) == []


def test_a_lock_that_is_no_longer_this_runs_is_left_alone(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    real = campaign_trace.freeze_v2

    def replaced_by_hand(*args: Any, **kwargs: Any) -> Any:
        # Someone deletes this run's lock and another run takes the folder.
        (directory / campaign_trace.FREEZE_LOCK).write_text("another run", encoding="utf-8")
        return real(*args, **kwargs)

    monkeypatch.setattr(campaign_trace, "freeze_v2", replaced_by_hand)
    assert main(["next", str(campaign)]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "no longer holds what this run wrote into it" in printed
    assert (directory / campaign_trace.FREEZE_LOCK).read_text(encoding="utf-8") == "another run"


def test_a_record_folder_that_cannot_be_listed_again_freezes_nothing(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    real_recommend = campaign_trace.recommend
    real_iterdir = Path.iterdir
    fitted: list[bool] = []

    def recommend(state: Any) -> Any:
        fitted.append(True)
        return real_recommend(state)

    def iterdir(self: Path) -> Any:
        if fitted and self == directory:
            raise PermissionError(13, "Permission denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(campaign_trace, "recommend", recommend)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    assert "could not be listed again" in printed
    for text in _NOT_SHOWN:
        assert text not in printed
    monkeypatch.undo()
    assert _records(directory) == []


def test_verify_says_when_the_commit_cannot_be_looked_up(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    monkeypatch.setattr(campaign_trace, "commit_known", lambda commit: None)
    assert main(["verify", str(path)]) == EXIT_OK
    assert "NOTE: whether the recorded commit is in this repository could not be checked" in (
        capsys.readouterr().out
    )


def test_a_record_away_from_its_campaign_is_checked_as_far_as_it_can_be(
    campaign: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    away = tmp_path / "copied"
    away.mkdir()
    copy = away / path.name
    shutil.copy2(path, copy)
    assert main(["verify", str(copy)]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "NOTE: the record is not in a campaign's record folder" in printed
    assert f"NOTE: its note {path.with_suffix('.md').name} is not beside it" in printed


# ─── Freezing: the record folder is held, and an interrupted freeze is found
def test_a_lock_left_in_the_record_folder_stops_check_and_next(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    directory.mkdir(parents=True)
    (directory / campaign_trace.FREEZE_LOCK).write_text(
        "held by `next`, process 4242, since 2026-10-14", encoding="utf-8"
    )
    monkeypatch.setattr(engine, "optimize", _never_fit)
    for command in ("check", "next"):
        assert main([command, str(campaign)]) == EXIT_REFUSED
        printed = capsys.readouterr().out
        assert "says that a `next` is freezing in this folder now" in printed
        assert "(held by `next`, process 4242, since 2026-10-14)" in printed
    assert _records(directory) == []


def test_a_freeze_that_finds_the_folder_held_freezes_nothing_and_shows_nothing(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    real = campaign_trace.recommend

    def another_run_takes_the_folder(state: Any) -> Any:
        result = real(state)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / campaign_trace.FREEZE_LOCK).write_text("another run", encoding="utf-8")
        return result

    monkeypatch.setattr(campaign_trace, "recommend", another_run_takes_the_folder)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    assert "the record folder could not be held for this freeze" in printed
    assert "another `next` is freezing in this folder now" in printed
    for text in _NOT_SHOWN:
        assert text not in printed
    assert _records(directory) == []
    # The other run's lock is not this run's to remove.
    assert (directory / campaign_trace.FREEZE_LOCK).read_text(encoding="utf-8") == "another run"


def test_a_record_frozen_while_the_model_was_fitted_stops_this_freeze(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    real = campaign_trace.recommend

    def another_run_froze_first(state: Any) -> Any:
        result = real(state)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prereg_20260101T000000Z.json").write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(campaign_trace, "recommend", another_run_froze_first)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    assert "while it was being fitted, files appeared in" in printed
    assert "prereg_20260101T000000Z.json" in printed
    for text in _NOT_SHOWN:
        assert text not in printed
    assert [p.name for p in _records(directory)] == ["prereg_20260101T000000Z.json"]
    assert not (directory / campaign_trace.FREEZE_LOCK).exists()  # this run's lock is gone


def test_a_second_next_on_the_same_data_freezes_nothing(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _frozen(campaign, capsys)
    before = _snapshot(campaign.parent)
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["next", str(campaign)]) == EXIT_REFUSED
    assert f"already holds these observations under this definition: {path}" in (
        capsys.readouterr().out
    )
    assert _snapshot(campaign.parent) == before


def test_files_of_an_interrupted_freeze_stop_check_and_next_and_say_what_to_do(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    directory.mkdir(parents=True)
    partial = directory / f".prereg_20260101T000000Z.json{experiments_prereg.PARTIAL_SUFFIX}"
    partial.write_text("{", encoding="utf-8")
    orphan = directory / "prereg_20260101T000000Z.md"
    orphan.write_text("# a note whose record was never published", encoding="utf-8")
    monkeypatch.setattr(engine, "optimize", _never_fit)
    for command in ("check", "next"):
        assert main([command, str(campaign)]) == EXIT_REFUSED
        printed = capsys.readouterr().out
        for left in (partial, orphan):
            assert f"{left} was left by a freeze that did not complete" in printed
        assert "run `verify` on any prereg_*.json there of the same name" in printed
        assert f"move it to {directory.parent / 'interrupted'}" in printed
    assert partial.exists() and orphan.exists()  # nothing is removed for the person


def test_a_dry_run_freezes_nothing_and_says_what_that_costs(
    five: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["next", str(five), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "Nothing was frozen." in printed
    assert "Nothing records that this preview was shown." in printed
    assert "its record will not say that it was seen first" in printed
    assert not prereg_dir(five.parent).exists()


def test_next_freezes_beside_the_manifest_where_the_existing_screens_look(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _five(tmp_path)
    assert main(["next", str(manifest)]) == 0
    (entry,) = list_preregistrations(tmp_path)
    assert entry.property_name == "apparent_seebeck_cycle_mv_k"
    assert entry.input_variable == "aliquat_mass_fraction"
    assert Path(entry.path).parent == prereg_dir(tmp_path)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_next_with_too_few_conditions_freezes_nothing(
    extra: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_recording(tmp_path / "run.xlsx")
    manifest = write_manifest(tmp_path, [specimen("s1", "run.xlsx")], defaults=BENCH_DEFAULTS)
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["next", str(manifest), *extra]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert "1 distinct condition(s); a model needs at least 3." in printed
    assert "  - 1 distinct condition(s) are in the fit, and a model needs at least 3" in printed
    assert not prereg_dir(tmp_path).exists()
    # Every specimen could be used, and a check is still NOT READY.
    assert main(["check", str(manifest)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert "NOT READY" in printed
    assert "could not be used" not in printed


def test_a_missing_or_malformed_manifest_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check", str(tmp_path / "absent.toml")]) == 2
    assert "No such manifest" in capsys.readouterr().out
    manifest = write_manifest(tmp_path, [specimen("s1", "run.xlsx", test_cycle=1)])
    assert main(["next", str(manifest)]) == 2
    assert "unknown key 'test_cycle'" in capsys.readouterr().out
    # A folder where the manifest should be: said, with the status of an unreadable input.
    for command in ("check", "next"):
        assert main([command, str(tmp_path)]) == EXIT_UNREADABLE
        assert str(tmp_path) in capsys.readouterr().out


def test_the_exit_statuses_are_the_documented_ones() -> None:
    assert (EXIT_OK, EXIT_REFUSED, EXIT_UNREADABLE, EXIT_FAILED) == (0, 1, 2, 3)


# ─── A declared specimen that could not be used ─────────────────────
def test_labels_typed_into_the_event_column_refuse_that_recording_and_name_the_cell(
    tmp_path: Path,
) -> None:
    state = load_campaign(_annotated(tmp_path))
    (rejected,) = state.observation_set.rejected()
    assert rejected.sample_id == "quarter"
    assert TYPED_REFUSAL in rejected.reasons
    assert rejected.record.values[S] is None  # no value was taken from it
    # The others are read as before. Three conditions are left, which is enough
    # for a model and is not what the manifest declares.
    assert [o.sample_id for o in state.observation_set.observations] == [
        "half",
        "three-quarters",
        "pure-first",
    ]
    assert state.enough is True


def test_check_fails_when_a_declared_specimen_could_not_be_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _annotated(tmp_path)
    monkeypatch.setattr(engine, "optimize", _never_fit)
    before = _snapshot(tmp_path)
    assert main(["check", str(manifest)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert "Specimens in the fit: 3" in printed
    assert f"  quarter: NOT in the model. {TYPED_REFUSAL}" in printed
    assert "  NOT READY. `next` would fit nothing and freeze nothing, because:" in printed
    assert "  - 1 declared specimen could not be used: quarter." in printed
    assert "A declared specimen is never quietly left out of a fit" in printed
    assert _snapshot(tmp_path) == before  # a check writes nothing, whatever it finds


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_next_does_not_fit_fewer_specimens_than_were_declared(
    extra: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _annotated(tmp_path)
    monkeypatch.setattr(engine, "optimize", _never_fit)
    before = _snapshot(tmp_path)
    assert main(["next", str(manifest), *extra]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert "Nothing was fitted and nothing was frozen. `next` refuses, because:" in printed
    assert "  - 1 declared specimen could not be used: quarter." in printed
    assert "make aliquat_mass_fraction" not in printed
    assert _snapshot(tmp_path) == before
    assert not prereg_dir(tmp_path).exists()


@pytest.mark.parametrize("fault", ["missing", "a folder", "held open"])
def test_a_recording_that_cannot_be_read_is_a_rejected_specimen_and_not_a_crash(
    fault: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _five(tmp_path)
    recording = tmp_path / "half.xlsx"
    reason = "the trace file 'half.xlsx' could not be read"
    if fault == "missing":
        recording.unlink()
        reason = "the trace file 'half.xlsx' was not found beside the manifest"
    elif fault == "a folder":
        recording.unlink()
        recording.mkdir()
    else:
        real = ite_trace.read_hioki_xlsx

        def held_open(path: Path, **options: Any) -> Any:
            if path.name == "half.xlsx":
                raise PermissionError(13, "Permission denied", str(path))
            return real(path, **options)

        monkeypatch.setattr(ite_trace, "read_hioki_xlsx", held_open)
    monkeypatch.setattr(engine, "optimize", _never_fit)

    assert main(["check", str(manifest)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert f"  half: NOT in the model. {reason}" in printed
    assert "Specimens in the fit: 3" in printed  # the others were still read
    assert "  - 1 declared specimen could not be used: half." in printed
    assert main(["next", str(manifest)]) == EXIT_REFUSED
    assert "Nothing was fitted and nothing was frozen." in capsys.readouterr().out
    assert not prereg_dir(tmp_path).exists()


def test_a_specimen_excluded_by_a_control_also_stops_the_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Declared, measured, and outside the drive programme: still a declared
    # specimen that is not in the fit, so the fit is not made without a decision.
    rows = _five_rows(tmp_path)
    rows[2] = {**rows[2], "drops_applied": 4}
    manifest = write_manifest(tmp_path, rows, defaults=BENCH_DEFAULTS)
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["check", str(manifest)]) == EXIT_REFUSED
    assert main(["next", str(manifest)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert "half: NOT in the model. control drops_applied: recorded 4, expected 3" in printed
    assert "  - 1 declared specimen could not be used: half." in printed


def test_a_specimen_outside_the_domain_is_not_a_reason_to_refuse(
    five: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Preserved, not rejected: the fit is exactly the one the manifest declares.
    assert load_campaign(five).observation_set.outside_bounds() != ()
    assert main(["check", str(five)]) == EXIT_OK
    assert "NOT READY" not in capsys.readouterr().out


# ─── One set of observations, one frozen record ─────────────────────
def test_next_refuses_a_second_record_for_the_same_observations(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = campaign.parent
    assert main(["next", str(campaign)]) == EXIT_OK
    (first,) = _records(prereg_dir(folder))
    # The refusal compares the digest the record itself carries.
    assert (
        observations_sha256(load_campaign(campaign)) == read_record(first)["observations"]["sha256"]
    )
    capsys.readouterr()

    monkeypatch.setattr(engine, "optimize", _never_fit)
    before = _snapshot(folder)
    # Again, or as a preview: the first record is found each time.
    for extra in ([], ["--dry-run"]):
        assert main(["next", str(campaign), *extra]) == EXIT_REFUSED
        printed = capsys.readouterr().out
        assert (
            f"  - a frozen record already holds these observations under this definition: {first}"
            in printed
        )
        assert "does not declare exactly" not in printed  # the same data, said once
        assert "make aliquat_mass_fraction" not in printed
    assert _snapshot(folder) == before

    # A check says so too, and is NOT READY: `next` would refuse.
    assert main(["check", str(campaign)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert "NOT READY" in printed
    assert "already holds these observations" in printed


def test_out_is_refused_unless_it_names_the_campaigns_own_record_folder(
    campaign: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A record written anywhere else is one a later run would not find.
    own = prereg_dir(campaign.parent)
    elsewhere = tmp_path / "elsewhere"
    fit = engine.optimize
    monkeypatch.setattr(engine, "optimize", _never_fit)
    before = _snapshot(tmp_path)
    for extra in ([], ["--dry-run"]):
        assert main(["next", str(campaign), "--out", str(elsewhere), *extra]) == EXIT_REFUSED
        printed = capsys.readouterr().out
        assert f"  - --out {elsewhere} is not this campaign's record folder, {own}." in printed
        assert "make aliquat_mass_fraction" not in printed
    assert _snapshot(tmp_path) == before
    assert not elsewhere.exists()

    # Its own folder, named outright, is the same as naming none.
    monkeypatch.setattr(engine, "optimize", fit)
    assert main(["next", str(campaign), "--out", str(own)]) == EXIT_OK
    assert len(_records(own)) == 1


@pytest.mark.parametrize(
    "relabelled", [{"version": "1.0.1"}, {"name": "the-same-under-a-new-name"}]
)
def test_another_version_or_name_does_not_hide_a_record_of_the_same_data(
    relabelled: dict[str, str],
    campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["next", str(campaign)]) == EXIT_OK
    (first,) = _records(prereg_dir(campaign.parent))
    capsys.readouterr()

    _redefine(monkeypatch, **relabelled)
    # A name and a version are labels. The definition's identity is what it was.
    assert definition_sha256(campaign_trace.DEFINITION) == definition_sha256(DEFINITION)
    monkeypatch.setattr(engine, "optimize", _never_fit)
    for extra in ([], ["--dry-run"]):
        assert main(["next", str(campaign), *extra]) == EXIT_REFUSED
        assert f"already holds these observations under this definition: {first}" in (
            capsys.readouterr().out
        )
    assert main(["check", str(campaign)]) == EXIT_REFUSED
    assert "NOT READY" in capsys.readouterr().out
    assert _records(prereg_dir(campaign.parent)) == [first]


@pytest.mark.parametrize("change", [{"bounds": (0.0, 1.0)}, {"precision": 0.001}])
def test_a_changed_definition_under_the_same_version_is_refused_before_any_model(
    change: dict[str, Any],
    campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["next", str(campaign)]) == EXIT_OK
    (first,) = _records(prereg_dir(campaign.parent))
    capsys.readouterr()
    _redefine(monkeypatch, variables=_with_design(**change))
    monkeypatch.setattr(engine, "optimize", _never_fit)
    for extra in ([], ["--dry-run"]):
        assert main(["next", str(campaign), *extra]) == EXIT_REFUSED
        printed = capsys.readouterr().out
        assert (
            "ionic-liquid-seebeck-trace 1.0.0 was frozen earlier under a different definition"
            in printed
        )
        assert first.name in printed
        # Another definition is not "this definition", whatever the observations are.
        assert "already holds these observations" not in printed
    assert _records(prereg_dir(campaign.parent)) == [first]


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param(
            {"version": "1.1.0", "variables": _with_design(precision=0.001)}, id="version"
        ),
        pytest.param(
            {"name": "renamed", "version": "2.0.0", "variables": _with_design(precision=0.001)},
            id="name and version",
        ),
    ],
)
def test_another_definition_does_not_get_past_a_record_that_waits(
    changes: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _five_rows(tmp_path)
    manifest = write_manifest(tmp_path, rows, defaults=BENCH_DEFAULTS)
    assert main(["next", str(manifest)]) == EXIT_OK
    (first,) = _records(prereg_dir(tmp_path))
    capsys.readouterr()

    # A finer precision is another definition, under another version (and name).
    # The same specimens read under it would be a second recommendation from them
    # before the first has been tested, so it waits like anything else.
    _redefine(monkeypatch, **changes)
    assert definition_sha256(campaign_trace.DEFINITION) != definition_sha256(DEFINITION)
    monkeypatch.setattr(engine, "optimize", _never_fit)
    for extra in ([], ["--dry-run"]):
        assert main(["next", str(manifest), *extra]) == EXIT_REFUSED
        printed = capsys.readouterr().out
        assert (
            f"  - {first.name} already holds exactly these specimens, frozen under another "
            f"definition ({definition_sha256(DEFINITION)[:16]}; now: "
            f"{definition_sha256(campaign_trace.DEFINITION)[:16]}), and that record has no "
            "outcome (it was frozen as ionic-liquid-seebeck-trace 1.0.0)." in printed
        )
        assert "make aliquat_mass_fraction" not in printed
    assert main(["check", str(manifest)]) == EXIT_REFUSED
    assert "NOT READY" in capsys.readouterr().out

    # A specimen made since is refused too, and said as what it is.
    make_recording(tmp_path / "sixth.xlsx", seebeck_mv_k=[2.30, 2.35, 2.40])
    sixth = specimen("sixth", "sixth.xlsx", masses_g=_mix(0.15, 0.10))
    write_manifest(tmp_path, [*rows, sixth], defaults=BENCH_DEFAULTS)
    assert main(["next", str(manifest)]) == EXIT_REFUSED
    assert (
        f"  - the manifest does not declare exactly what {first.name} holds (added: sixth)"
        in capsys.readouterr().out
    )
    assert _records(prereg_dir(tmp_path)) == [first]


def test_the_same_recordings_tie_a_record_to_this_campaign_whatever_else_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _five_rows(tmp_path)
    manifest = write_manifest(tmp_path, rows, defaults=BENCH_DEFAULTS)
    assert main(["next", str(manifest)]) == EXIT_OK
    (first,) = _records(prereg_dir(tmp_path))
    capsys.readouterr()

    # Another name, version and definition; every specimen under a new id and read
    # to another value. Only the recordings are the same, and they are enough.
    now = [
        {**row, "polarity_convention": SWAPPED, "sample_id": f"x-{row['sample_id']}"}
        for row in rows
    ]
    write_manifest(tmp_path, now, defaults=BENCH_DEFAULTS)
    _redefine(monkeypatch, name="renamed", version="2.0.0", variables=_with_design(precision=0.001))
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["next", str(manifest)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert f"the manifest does not declare exactly what {first.name} holds (added: " in printed
    assert "(it was frozen as ionic-liquid-seebeck-trace 1.0.0)" in printed
    assert _records(prereg_dir(tmp_path)) == [first]


# What the refusals read of a record: enough to be one, of some other experiment.
OTHER_EXPERIMENT: dict[str, Any] = {
    "schema_version": 2,
    "experiment": {
        "name": "other",
        "version": "1",
        "definition_sha256": "0" * 64,
        "definition": definition_to_dict(replace(DEFINITION, name="other")),
    },
    "observations": {"sha256": "0" * 64},
    "specimens": [],
    "excluded": [],
}


def _other_experiment(**experiment: Any) -> str:
    """`OTHER_EXPERIMENT` as a file's text, with `experiment` fields put in or, for None, out."""
    block = {**OTHER_EXPERIMENT["experiment"], **experiment}
    block = {key: value for key, value in block.items() if value is not None}
    return json.dumps({**OTHER_EXPERIMENT, "experiment": block})


@pytest.mark.parametrize(
    ("content", "blocks"),
    [
        ("{ not json", True),
        (json.dumps({"schema_version": 2}), True),  # says it is a record and has no experiment
        # Says it is a record of another experiment, and lacks what would show that.
        (json.dumps({"schema_version": 2, "experiment": {"name": "other", "version": "1"}}), True),
        (
            json.dumps({"kind": "latos.bo.prereg", "objective": {}}),
            False,
        ),  # version 1: another path
        (json.dumps(OTHER_EXPERIMENT), False),
        # A record cannot be told from another experiment's without its own definition:
        # that is what says which of its fields are the composition and the response.
        (_other_experiment(definition=None), True),
        (_other_experiment(definition="not a block"), True),
        # A definition that names no design variable, or not the response it is about.
        (
            _other_experiment(
                definition={**OTHER_EXPERIMENT["experiment"]["definition"], "variables": []}
            ),
            True,
        ),
        (
            _other_experiment(
                definition={**OTHER_EXPERIMENT["experiment"]["definition"], "responses": []}
            ),
            True,
        ),
        (json.dumps({"schema_version": 2, "experiment": "not a block"}), True),
        ("[]", False),  # JSON, and no record of any kind
        ("42", False),
    ],
)
def test_a_file_in_the_record_folder_that_cannot_be_read_stops_the_run(
    content: str,
    blocks: bool,
    campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = prereg_dir(campaign.parent)
    directory.mkdir(parents=True)
    stray = directory / "prereg_20260101T000000Z.json"
    stray.write_text(content, encoding="utf-8")
    if blocks:
        monkeypatch.setattr(engine, "optimize", _never_fit)
        for extra in ([], ["--dry-run"]):
            assert main(["next", str(campaign), *extra]) == EXIT_REFUSED
            printed = capsys.readouterr().out
            assert f"  - {stray} " in printed
            assert "Nothing was fitted and nothing was frozen." in printed
        assert _records(directory) == [stray]
    else:
        assert main(["next", str(campaign)]) == EXIT_OK
        assert len(_records(directory)) == 2


# ─── After a freeze, nothing else until its outcome is scored ───────
def test_a_specimen_made_after_a_freeze_enters_no_fit_while_that_record_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _five_rows(tmp_path)
    manifest = write_manifest(tmp_path, rows, defaults=BENCH_DEFAULTS)
    assert main(["next", str(manifest)]) == EXIT_OK
    (first,) = _records(prereg_dir(tmp_path))
    make_recording(tmp_path / "sixth.xlsx", seebeck_mv_k=[2.30, 2.35, 2.40])
    sixth = specimen("sixth", "sixth.xlsx", masses_g=_mix(0.15, 0.10))
    write_manifest(tmp_path, [*rows, sixth], defaults=BENCH_DEFAULTS)
    capsys.readouterr()

    monkeypatch.setattr(engine, "optimize", _never_fit)
    outcome = first.with_suffix(".outcome.json")
    refused = (
        f"  - the manifest does not declare exactly what {first.name} holds "
        "(added: sixth), and that record has "
    )
    standing = [
        (None, "no outcome"),
        # The version 1 route takes a bare number: it is not the scoring of this record.
        (
            json.dumps({"kind": "latos.bo.outcome", "prereg": first.name, "measured": 2.35}),
            "an outcome written by the version 1 route, which does not count",
        ),
        ("{ not json", "an outcome file that cannot be read"),
        # Nothing scores a version 2 record yet, so a file that says it does is not believed.
        (
            json.dumps({"kind": "latos.bo.outcome", "schema_version": 2, "prereg": first.name}),
            "an outcome file that nothing can check yet",
        ),
    ]
    for content, beside in standing:
        if content is not None:
            outcome.write_text(content, encoding="utf-8")
        for extra in ([], ["--dry-run"]):
            assert main(["next", str(manifest), *extra]) == EXIT_REFUSED
            printed = capsys.readouterr().out
            assert f"{refused}{beside}." in printed
            assert "nothing scores a version 2 record yet" in printed
            assert "make aliquat_mass_fraction" not in printed
        assert main(["check", str(manifest)]) == EXIT_REFUSED
        assert "NOT READY" in capsys.readouterr().out

    # Nor under another version of the same definition, nor under another name for it,
    # nor when the name and the definition are both changed: the record still holds
    # five of these specimens, so it is this campaign's whatever it is called.
    relabelled: list[dict[str, Any]] = [
        {"version": "1.0.1"},
        {"name": "the-same-under-a-new-name"},
        {"name": "renamed", "version": "2.0.0", "variables": _with_design(precision=0.001)},
    ]
    for changes in relabelled:
        _redefine(monkeypatch, **changes)
        assert main(["next", str(manifest)]) == EXIT_REFUSED
        assert "(added: sixth)" in capsys.readouterr().out
        assert main(["check", str(manifest)]) == EXIT_REFUSED
        assert "NOT READY" in capsys.readouterr().out
    assert _records(prereg_dir(tmp_path)) == [first]


def test_a_specimen_changed_or_dropped_after_a_freeze_stops_the_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _five_rows(tmp_path)
    manifest = write_manifest(tmp_path, rows, defaults=BENCH_DEFAULTS)
    assert main(["next", str(manifest)]) == EXIT_OK
    (first,) = _records(prereg_dir(tmp_path))
    capsys.readouterr()

    monkeypatch.setattr(engine, "optimize", _never_fit)
    make_recording(tmp_path / "half-again.xlsx", seebeck_mv_k=[2.5, 2.6, 2.7])
    half = rows[2]
    cases: list[tuple[dict[str, Any] | None, str]] = [
        # Measured again, under its old id.
        ({**half, "trace_file": "half-again.xlsx"}, "changed: half"),
        # The same recording, declared at another composition.
        ({**half, "masses_g": _mix(0.08, 0.07)}, "changed: half"),
        # Given another id.
        ({**half, "sample_id": "half-b"}, "added: half-b; no longer declared: half"),
        # Taken out of the manifest.
        (None, "no longer declared: half"),
    ]
    for row, departure in cases:
        now = [*rows[:2], *([] if row is None else [row]), *rows[3:]]
        write_manifest(tmp_path, now, defaults=BENCH_DEFAULTS)
        for extra in ([], ["--dry-run"]):
            assert main(["next", str(manifest), *extra]) == EXIT_REFUSED
            printed = capsys.readouterr().out
            assert (
                f"  - the manifest does not declare exactly what {first.name} holds "
                f"({departure}), and that record has no outcome." in printed
            )
            assert "make aliquat_mass_fraction" not in printed
        assert main(["check", str(manifest)]) == EXIT_REFUSED
        assert "NOT READY" in capsys.readouterr().out

    # The leads declared the other way round: the same recording at the same
    # composition, read to another value. Another definition does not let that past.
    write_manifest(
        tmp_path,
        [*rows[:2], {**half, "polarity_convention": SWAPPED}, *rows[3:]],
        defaults=BENCH_DEFAULTS,
    )
    _redefine(monkeypatch, version="1.1.0", variables=_with_design(precision=0.001))
    assert main(["next", str(manifest)]) == EXIT_REFUSED
    assert "(changed: half)" in capsys.readouterr().out
    assert _records(prereg_dir(tmp_path)) == [first]


@pytest.mark.parametrize("renamed", ["the design variable", "the response"])
def test_a_variable_renamed_since_a_freeze_is_the_same_variable(
    renamed: str,
    campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A record is read under the names it was frozen with. Read under the names of
    # the day, every specimen of an older record would look changed.
    assert main(["next", str(campaign)]) == EXIT_OK
    assert len(_records(prereg_dir(campaign.parent))) == 1
    capsys.readouterr()
    _redefine(monkeypatch, version="1.1.0", **_renamed(renamed))
    assert definition_sha256(campaign_trace.DEFINITION) != definition_sha256(DEFINITION)

    # The same specimens under another definition and version. None of them has
    # changed, and the record they are in still waits for its outcome.
    monkeypatch.setattr(engine, "optimize", _never_fit)
    for command in ("check", "next"):
        assert main([command, str(campaign)]) == EXIT_REFUSED
        printed = capsys.readouterr().out
        assert "already holds exactly these specimens, frozen under another" in printed
        assert "changed: " not in printed
    assert len(_records(prereg_dir(campaign.parent))) == 1


@pytest.mark.parametrize("renamed", ["the design variable", "the response"])
def test_the_same_recordings_under_new_names_throughout_are_still_this_campaigns(
    renamed: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _five_rows(tmp_path)
    manifest = write_manifest(tmp_path, rows, defaults=BENCH_DEFAULTS)
    assert main(["next", str(manifest)]) == EXIT_OK
    (first,) = _records(prereg_dir(tmp_path))
    capsys.readouterr()

    # Nothing is measured again. The experiment has another name and version, a
    # variable has another name, and every specimen has another id. What ties the
    # record to this campaign is the data: composition, recording, value.
    again = [{**row, "sample_id": f"x-{row['sample_id']}"} for row in rows]
    write_manifest(tmp_path, again, defaults=BENCH_DEFAULTS)
    _redefine(monkeypatch, name="renamed", version="2.0.0", **_renamed(renamed))
    monkeypatch.setattr(engine, "optimize", _never_fit)
    for extra in ([], ["--dry-run"]):
        assert main(["next", str(manifest), *extra]) == EXIT_REFUSED
        printed = capsys.readouterr().out
        assert (
            f"  - the manifest does not declare exactly what {first.name} holds (added: "
            "x-quarter, x-half, x-three-quarters, x-pure-first, x-pure-second; no longer "
            "declared: quarter, half, three-quarters, pure-first, pure-second), and that "
            "record has no outcome (it was frozen as ionic-liquid-seebeck-trace 1.0.0)."
        ) in printed
    assert main(["check", str(manifest)]) == EXIT_REFUSED
    assert "NOT READY" in capsys.readouterr().out
    assert _records(prereg_dir(tmp_path)) == [first]


@pytest.mark.parametrize(
    ("changes", "new_ids"),
    [
        # Each case leaves one thing, and only one, that makes the record this campaign's.
        pytest.param(
            {"name": "renamed", "version": "2.0.0", "variables": _with_design(precision=0.001)},
            False,
            id="the specimen ids",
        ),
        pytest.param({"name": "renamed"}, True, id="the definition"),
        pytest.param(
            {"version": "1.1.0", "variables": _with_design(precision=0.001)},
            True,
            id="the experiment name",
        ),
    ],
)
def test_one_tie_to_an_earlier_record_is_enough_to_wait_for_it(
    changes: dict[str, Any],
    new_ids: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    departure = (
        "(added: x-quarter, x-half, x-three-quarters, x-pure-first, x-pure-second; no longer "
        "declared: quarter, half, three-quarters, pure-first, pure-second)"
        if new_ids
        else "(changed: quarter, half, three-quarters, pure-first, pure-second)"
    )
    rows = _five_rows(tmp_path)
    manifest = write_manifest(tmp_path, rows, defaults=BENCH_DEFAULTS)
    assert main(["next", str(manifest)]) == EXIT_OK
    (first,) = _records(prereg_dir(tmp_path))
    capsys.readouterr()

    # Every specimen reads to another value (its leads declared the other way round),
    # so no specimen of the record is told by its data.
    now = [
        {
            **row,
            "polarity_convention": SWAPPED,
            "sample_id": f"x-{row['sample_id']}" if new_ids else row["sample_id"],
        }
        for row in rows
    ]
    write_manifest(tmp_path, now, defaults=BENCH_DEFAULTS)
    _redefine(monkeypatch, **changes)
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["next", str(manifest)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert "could not be used" not in printed  # every specimen is accepted: it is the record
    assert f"  - the manifest does not declare exactly what {first.name} holds {departure}" in (
        printed
    )
    assert "(it was frozen as ionic-liquid-seebeck-trace 1.0.0)" in printed
    assert _records(prereg_dir(tmp_path)) == [first]


def test_another_definition_may_read_the_same_specimens_to_other_observations(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A record that states other observations than its specimens give is said to
    # under ITS definition. Under another, other observations are what is
    # expected, so that is not said; the record still waits, and that is said.
    assert main(["next", str(campaign)]) == EXIT_OK
    (first,) = _records(prereg_dir(campaign.parent))
    capsys.readouterr()
    aimed = replace(DEFINITION.objective, goal=Goal.TARGET, target=2.0)
    _redefine(monkeypatch, version="1.1.0", objective=aimed)
    assert (
        observations_sha256(campaign_trace.load_campaign(campaign))
        != (read_record(first)["observations"]["sha256"])
    )
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["next", str(campaign)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert f"{first.name} already holds exactly these specimens" in printed
    assert "states other observations" not in printed
    assert _records(prereg_dir(campaign.parent)) == [first]


def test_the_uncertainty_read_from_a_specimen_is_part_of_what_it_is(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["next", str(campaign)]) == EXIT_OK
    (first,) = _records(prereg_dir(campaign.parent))
    record = json.loads(first.read_text(encoding="utf-8"))
    (half,) = [row for row in record["specimens"] if row["sample_id"] == "half"]
    half["responses"][DEFINITION.objective.response]["uncertainty"] *= 2
    first.write_text(json.dumps(record), encoding="utf-8")
    capsys.readouterr()

    # Under another definition nothing else would show it: the value is the same.
    _redefine(monkeypatch, version="1.1.0", variables=_with_design(precision=0.001))
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["next", str(campaign)]) == EXIT_REFUSED
    assert "(changed: half)" in capsys.readouterr().out
    assert _records(prereg_dir(campaign.parent)) == [first]


def test_a_record_that_no_longer_says_what_it_holds_stops_the_fit(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The same specimens, and a record whose stated observations are not theirs: an
    # edited record, or one frozen before the reduction was changed. Not a duplicate
    # by its own account, and not a reason to freeze these specimens a second time.
    assert main(["next", str(campaign)]) == EXIT_OK
    (first,) = _records(prereg_dir(campaign.parent))
    record = json.loads(first.read_text(encoding="utf-8"))
    record["observations"]["sha256"] = "0" * 64
    first.write_text(json.dumps(record), encoding="utf-8")
    capsys.readouterr()
    monkeypatch.setattr(engine, "optimize", _never_fit)
    for extra in ([], ["--dry-run"]):
        assert main(["next", str(campaign), *extra]) == EXIT_REFUSED
        printed = capsys.readouterr().out
        assert (
            f"  - {first.name} holds these specimens and states other observations than "
            "they give now (0000000000000000; now: " in printed
        )
        assert "does not declare exactly" not in printed
    assert _records(prereg_dir(campaign.parent)) == [first]


@pytest.mark.parametrize("blocked", ["the record folder is a file", "its parent is a file"])
def test_a_record_folder_that_cannot_be_looked_in_stops_the_run(
    blocked: str,
    campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = prereg_dir(campaign.parent)
    if blocked == "the record folder is a file":
        directory.parent.mkdir()
        directory.write_text("not a folder", encoding="utf-8")
        reason = f"the record folder {directory} could not be listed ("
    else:
        directory.parent.write_text("not a folder", encoding="utf-8")
        reason = f"the record folder {directory} cannot be made, because {directory.parent} is"
    monkeypatch.setattr(engine, "optimize", _never_fit)
    before = _snapshot(campaign.parent)
    assert main(["next", str(campaign)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert reason in printed
    assert "Nothing was fitted and nothing was frozen." in printed
    # No specimen is at fault, and `next` would still refuse: NOT READY.
    assert main(["check", str(campaign)]) == EXIT_REFUSED
    printed = capsys.readouterr().out
    assert "NOT READY" in printed
    assert "could not be used" not in printed
    assert _snapshot(campaign.parent) == before


def test_a_record_file_that_cannot_be_opened_stops_the_run(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    held = directory / "prereg_20260101T000000Z.json"
    held.mkdir(parents=True)  # a folder where a record would be: it cannot be read as one
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["next", str(campaign)]) == EXIT_REFUSED
    assert f"  - {held} could not be read (" in capsys.readouterr().out


# ─── Nothing shown as committed before it is ────────────────────────
def test_the_recommendation_is_shown_only_after_the_record_is_frozen_and_verified(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["next", str(campaign)]) == EXIT_OK
    printed = capsys.readouterr().out
    (path,) = _records(prereg_dir(campaign.parent))
    order = [
        f"Frozen: {path}",
        "Seal:   ",
        f"Recommendation, frozen in {path.name}",
        "  make aliquat_mass_fraction = ",
        "Write the seal in the lab notebook now, before the specimen is made.",
    ]
    places = [printed.index(text) for text in order]
    assert places == sorted(places)
    assert printed.count("make aliquat_mass_fraction") == 1
    assert "PREVIEW" not in printed


_NOT_SHOWN = (
    "make aliquat_mass_fraction",
    "  predicted ",
    "95% predictive interval",
    "best fitted condition so far",
    "Recommendation",
    "Frozen:",
    "Seal:",
    "Write the seal",
)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_model_that_cannot_be_fitted_is_said_and_nothing_is_written(
    extra: list[str],
    five: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def no_fit(*args: Any, **kwargs: Any) -> Any:
        raise FitConvergenceError("the fit did not converge")

    monkeypatch.setattr(engine, "optimize", no_fit)
    assert main(["next", str(five), *extra]) == EXIT_FAILED
    captured = capsys.readouterr()
    assert "A model could not be fitted. Nothing was frozen, and there is nothing to show." in (
        captured.out
    )
    assert "  reason: FitConvergenceError: the fit did not converge" in captured.out
    for text in _NOT_SHOWN:
        assert text not in captured.out
    assert "Traceback" not in captured.err  # a failure of a kind a run expects
    assert not prereg_dir(five.parent).exists()


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_result_that_cannot_be_set_out_is_not_called_a_failed_fit(
    extra: list[str],
    five: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("unsupported format")

    monkeypatch.setattr(campaign_trace, "_recommendation_lines", broken)
    assert main(["next", str(five), *extra]) == EXIT_FAILED
    captured = capsys.readouterr()
    assert "A model was fitted and what it suggested could not be set out." in captured.out
    assert "could not be fitted" not in captured.out
    assert "Traceback" in captured.err  # a fault in the program, so its traceback is kept
    for text in _NOT_SHOWN:
        assert text not in captured.out
    assert not prereg_dir(five.parent).exists()  # it failed before anything was written


@pytest.mark.parametrize(
    ("failure", "fault"),
    [
        (VersionCollision("frozen earlier with a different definition"), False),
        (OSError(28, "No space left on device"), False),
        (ValueError("The prepared arrays were not made from this observation set."), False),
        (KeyError("experiment"), True),  # not of a kind a run expects: its traceback is kept
    ],
)
def test_a_freeze_that_fails_shows_no_recommendation(
    failure: Exception,
    fault: bool,
    campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(campaign_trace, "freeze_v2", failing)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    captured = capsys.readouterr()
    printed = captured.out
    assert "A model was fitted and its freeze was NOT confirmed" in printed
    assert f"  reason: {type(failure).__name__}: " in printed
    # The folder is made to hold the freeze lock, and the lock is gone again.
    directory = prereg_dir(campaign.parent)
    assert f"  The attempt created the folder {directory} and left no file in it." in printed
    for text in _NOT_SHOWN:
        assert text not in printed
    assert ("Traceback" in captured.err) is fault
    assert list(directory.iterdir()) == []


def test_a_folder_that_was_there_before_the_failed_attempt_is_not_said_to_be_made(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    directory.mkdir(parents=True)

    def failing(*args: Any, **kwargs: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(campaign_trace, "freeze_v2", failing)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    assert f"  Nothing was written in {directory}." in printed
    assert "created the folder" not in printed
    assert list(directory.iterdir()) == []


def test_files_left_by_a_failed_freeze_are_named_and_older_files_are_not(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)
    directory.mkdir(parents=True)
    older = directory / "prereg_20250101T000000Z.json"
    older.write_text(json.dumps({"kind": "latos.bo.prereg", "objective": {}}), encoding="utf-8")

    def half_written(*args: Any, **kwargs: Any) -> Any:
        (directory / "prereg_20260101T000000Z.json").write_text("{}", encoding="utf-8")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(campaign_trace, "freeze_v2", half_written)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    assert f"  The attempt left these files in {directory}:" in printed
    assert "    prereg_20260101T000000Z.json: not a record: " in printed
    assert older.name not in printed  # it was there before, and this attempt did not leave it
    assert "Nothing was written in" not in printed
    for text in _NOT_SHOWN:
        assert text not in printed


def test_a_note_that_cannot_be_written_leaves_no_record(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The note is made before anything is written, and the record is
    # published last, so a freeze that stops on the note leaves nothing.
    def no_note(record: Any) -> str:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(experiments_prereg, "render_note", no_note)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    directory = prereg_dir(campaign.parent)
    assert (
        "A model was fitted and its freeze was NOT confirmed. A freeze is confirmed only when "
        "the record has been written, read back and verified, and that did not complete."
    ) in printed
    assert list(directory.iterdir()) == []
    assert "left no file in it" in printed
    for text in _NOT_SHOWN:
        assert text not in printed

    # Nothing that could pass for a freeze stops the next run.
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["check", str(campaign)]) == EXIT_OK


def test_a_record_left_by_a_failed_freeze_that_does_not_verify_is_said_so(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = campaign_trace.freeze_v2

    def written_then_failed(*args: Any, **kwargs: Any) -> Any:
        real(*args, **kwargs)
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(campaign_trace, "freeze_v2", written_then_failed)
    monkeypatch.setattr(
        campaign_trace, "verify_record", lambda record, **options: ("it does not match its seal",)
    )
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    (left,) = _records(prereg_dir(campaign.parent))
    assert f"    {left.name}: a version 2 record that does NOT verify" in printed
    assert "verifies against its own seal" not in printed
    assert "was not written" not in printed


def test_a_file_left_by_a_failed_freeze_that_cannot_be_read_is_said_so(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prereg_dir(campaign.parent)

    def left_a_folder(*args: Any, **kwargs: Any) -> Any:
        (directory / "prereg_20260101T000000Z.json").mkdir(parents=True)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(campaign_trace, "freeze_v2", left_a_folder)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    captured = capsys.readouterr()
    assert "    prereg_20260101T000000Z.json: could not be read (" in captured.out
    assert "Traceback" not in captured.err  # the report of a failure does not fail


def test_a_read_back_that_fails_does_not_make_a_whole_record_a_broken_one(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = campaign_trace.read_record
    asked: list[Path] = []

    def held_once(path: Path) -> dict[str, Any]:
        asked.append(path)
        if len(asked) == 1:
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)

    monkeypatch.setattr(campaign_trace, "read_record", held_once)
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    (left,) = _records(prereg_dir(campaign.parent))
    assert "  reason: PermissionError: Permission denied" in printed
    assert "was not written" not in printed
    assert (
        f"    {left.name}: a complete version 2 record, and it verifies against its own seal"
        in printed
    )
    assert f"    {left.with_suffix('.md').name}: " in printed  # its note was published first
    assert "`next` will refuse to fit these observations again" in printed
    assert "Run `verify` on it before deciding what it is." in printed
    for text in _NOT_SHOWN:
        assert text not in printed

    # The next run says the same of that file, and `verify` reads it.
    monkeypatch.setattr(campaign_trace, "read_record", real)
    monkeypatch.setattr(engine, "optimize", _never_fit)
    assert main(["next", str(campaign)]) == EXIT_REFUSED
    assert f"already holds these observations under this definition: {left}" in (
        capsys.readouterr().out
    )
    assert main(["verify", str(left)]) == EXIT_OK


def test_a_record_that_does_not_verify_is_not_presented_as_frozen(
    campaign: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        campaign_trace, "verify_record", lambda record, **options: ("it does not match its seal",)
    )
    assert main(["next", str(campaign)]) == EXIT_FAILED
    printed = capsys.readouterr().out
    (path,) = _records(prereg_dir(campaign.parent))
    assert "A record was written and it does NOT verify." in printed
    assert f"What the model suggested is not shown: {path}" in printed
    assert "  PROBLEM: it does not match its seal" in printed
    # The note beside it states the seal that must not be noted, so it is named.
    note = path.with_suffix(".md")
    assert note.exists()
    assert f"  Its note was written too, and states that record's seal: {note}" in printed
    assert "Do not note its seal and do not make a specimen from it." in printed
    assert "`next` reads it as an earlier record of this campaign" in printed
    for text in _NOT_SHOWN:
        assert text not in printed


def test_the_help_says_which_check_a_dry_run_does_not_make(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("COLUMNS", "200")  # so the line is not broken where it is looked for
    with pytest.raises(SystemExit) as stopped:
        main(["next", "--help"])
    assert stopped.value.code == 0
    assert "make every check except the source-tree one, and print a preview; freeze nothing" in (
        capsys.readouterr().out
    )


def test_a_dry_run_changes_no_file_and_calls_what_it_shows_a_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _five(tmp_path)

    def no_freeze(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dry run must not freeze")

    monkeypatch.setattr(campaign_trace, "freeze_v2", no_freeze)
    # A preview is exempt from the source-revision rule, and from that rule only.
    _uncommitted(monkeypatch, replace(COMMITTED, working_tree="dirty", changed_paths=("?? a.py",)))
    before = _snapshot(tmp_path)
    assert main(["next", str(manifest), "--dry-run"]) == EXIT_OK
    printed = capsys.readouterr().out

    assert _snapshot(tmp_path) == before  # no recording, no manifest, no folder, nothing touched
    assert not prereg_dir(tmp_path).exists()
    heading = printed.index("PREVIEW ONLY. Not frozen, so not a committed prediction")
    assert heading < printed.index("  make aliquat_mass_fraction = ")
    assert "Nothing was frozen." in printed
    # What is said of the source tree is true of a preview too.
    assert "`next` freezes nothing until the source tree is exactly a commit." in printed
    assert "could not be fitted" not in printed
    for text in ("Recommendation", "Frozen:", "Seal:", "Write the seal"):
        assert text not in printed
