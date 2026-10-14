"""The ionic-liquid thermal-cycle campaign: its definition and its bench command.

::

    python -m latos.campaign_trace check  specimens.toml
    python -m latos.campaign_trace next   specimens.toml [--out DIR] [--dry-run]
    python -m latos.campaign_trace verify prereg_<stamp>.json [--seal HEX]

``check`` reads the manifest and its recordings, builds and reduces the
observations, prepares what the optimizer would be handed, and reports, ending
with every reason ``next`` would refuse this data and this source tree. It fits
no model and writes nothing, so it can be run as often as wanted.

``next`` does the same, then asks the existing one-variable optimizer for the
next composition and **freezes** the recommendation as a version 2 record. The
recommendation is shown only after the record has been written, read back and
verified. Until then nothing printed describes one, except the preview of a
``--dry-run``, which says that it is a preview. Before any model is fitted it
refuses when:

* a specimen the manifest declares could not be used, whatever the reason. A
  declared specimen is never quietly left out of a fit;
* a frozen record already holds the same observations under the same
  definition, whatever name and version it was frozen under. One set of
  observations gets one frozen recommendation;
* an earlier record of this campaign is waiting for its outcome. After a freeze
  nothing else is fitted until the outcome has been scored, and a new
  definition, a new version or a new name does not get past that. When the
  manifest no longer declares exactly the specimens that record holds, the
  refusal says which were added, changed, given another id or dropped;
* this experiment name and version were frozen before under another definition;
* a file in the record folder cannot be read, because it may be the record
  that would have stopped the run;
* ``--out`` names any folder but the campaign's own record folder;
* there are too few distinct conditions for a model;
* the source tree is not exactly a Git commit: a prospective recommendation has
  to be reproducible from a precise revision, and one that has been seen cannot
  be unseen.

An earlier record of this campaign is one frozen for this experiment name under
any version, one frozen for this definition under any name, or one that holds
any specimen the manifest declares. A specimen is known by its id or, under
another id, by its recording's SHA-256 or by its composition, recording and
value together; a record is read under the names it was frozen with, so a
variable renamed since is the same variable. Nothing scores a version 2 record
yet, so every earlier record is waiting for its outcome and no file beside one
counts.

A known limitation, kept on purpose: a record of a different experiment kept in
the same record folder is taken for this campaign's when it holds a specimen
under one of the manifest's sample ids, even if the two specimens have nothing
else in common. It then stops ``next`` like any waiting record. Nothing in a
record says reliably which bench it came from, so this fails closed: a sample id
used by two experiments is treated as one specimen until a person moves the
other experiment's records to that experiment's own folder.

Controls, by the approved protocol: a control recorded outside its tolerance
excludes the specimen, which then stops ``next`` as a declared specimen that
could not be used. A control left blank does not (rule R4 of the approved
definition): the specimen is kept and fitted, and is reported as unverified for
that control, because treating a blank more harshly than a control that is
never recorded at all would punish recording.

Optional, by the approved protocol, and so never a reason to refuse: the
environment and metadata columns (the electrode material, humidity, ambient
temperature, batch and dosing notes), and a reversed-lead diagnostic run, which
may be absent, may fail to analyse or may not reverse the sign. Each of those is
reported, as a warning where it applies, and none is judged against a
threshold.

A campaign's frozen records are kept in one folder, ``.latos/prereg/`` beside
its manifest, and that is the only place they are looked for. A record written
anywhere else could not stop a later run, which is why ``--out`` is refused
unless it names that folder. A record that is moved, renamed or deleted by
hand, or a campaign folder copied without its records, is beyond what this
command can see. The seal noted in the lab notebook is what shows that.

``--dry-run`` makes every check above except the last, computes a preview and
freezes nothing: it writes no record and no note, and changes neither the
manifest nor any recording. Nothing records that a preview was shown: a later
``next`` on the same data can freeze the same recommendation, and its record
will not say that it was seen first.

``verify`` recomputes what a frozen record claims about itself, and compares
its seal with one noted at freeze time if that is given.

Exit status
-----------
====  ======================================================================
0     ``check``: ready; no reason ``next`` would refuse applies to this data,
      this record folder and this source tree. ``next``: frozen and
      verified, or with ``--dry-run`` a preview was computed. ``verify``: the
      record verifies.
1     ``check``: NOT READY, with every reason printed: whatever would make
      ``next`` refuse, including a source tree that is not exactly a commit
      and an earlier record that is waiting for its outcome. ``next``:
      refused, with every reason printed; nothing was fitted and no record
      was written. ``verify``: the record does not verify.
2     The manifest, or the record given to ``verify``, could not be read.
3     ``next`` got past every refusal and then failed: the model could not be
      fitted or set out, the freeze was not confirmed (writing the record,
      reading it back and verifying it did not all complete), or the record
      it wrote does not verify. No recommendation was shown. The message
      names every file the attempt left and says what each one is.
====  ======================================================================

With status 3 a traceback is printed as well when the failure is not of a kind
a run expects. Anything else is a fault in the program: Python prints a
traceback and ends with status 1, the number a refusal has. A traceback is
never a refusal.

What this module adds to the generic path
-----------------------------------------
Only the declaration. `DEFINITION` names the design variable and its search
domain, the response and its uncertainty, the objective, and the drive
programme as controls. Everything it is handed to (building, reducing,
dispatching, freezing) is the experiment-agnostic layer in `latos.experiments`,
and the numbers come from the trace producer. No arithmetic on the response
happens here.

The search domain and the sign
------------------------------
The response is signed, and it is optimized signed: the objective asks for the
largest value, not the largest magnitude, and no absolute value is taken
anywhere on the path.

The search covers a declared range of composition, `SEARCH_BOUNDS`, and only
that range. A specimen measured outside it is not thrown away and is not judged:
if it passed every check it is an accepted specimen, kept in the report and in
the frozen record with its signed value, its uncertainty and its warnings, and
labelled as outside the search bounds. It is simply not one of the points the
model is fitted to. Why the domain was drawn where it was is a matter for the
campaign's own notes, not for this file.

The controls
------------
The drive programme is part of what the response means, so a specimen measured
under another programme is excluded, with the reason kept:

* the declared drive set-point. This is what the operator set, as written in
  the manifest. It is required to equal the campaign's value, with no
  tolerance, because a declaration either says that value or does not. Nothing
  here measures the voltage delivered, and nothing here claims it is known
  exactly. A producer that measured it would need a real tolerance, under a new
  definition;
* every accepted heating and cooling phase within a tolerance of its nominal
  length, held through the shortest and the longest phase of each kind;
* three accepted cycles per specimen;
* one mounting, not remounted between cycles;
* three drops applied.

The electrode material is recorded but not controlled, because it is not yet
confirmed; once it is, making it a control is a new definition version.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from latos.core.enums import Severity
from latos.core.exceptions import LatosError
from latos.experiments import (
    ExperimentDefinition,
    Goal,
    ObjectiveDefinition,
    Observation,
    ObservationSet,
    PreparedOptimization,
    RecordTable,
    ResponseDefinition,
    SourceRevision,
    UncertaintyScope,
    VariableDefinition,
    VariableKind,
    VariableRole,
    build_observations,
    definition_sha256,
    min_conditions,
    prepare_optimization,
    reduce_observations,
    run_optimization,
    source_revision,
)
from latos.experiments.prepare import PREDICTION_NOISE_GLOSSARY
from latos.experiments.prereg import (
    PARTIAL_SUFFIX,
    SCHEMA_VERSION,
    RecordError,
    freeze_v2,
    interrupted_freeze_files,
    observations_digest_nd,
    read_record,
    render_note,
    verify_record,
)
from latos.experiments.producers import ite_trace
from latos.experiments.producers.ite_trace import ManifestError, produce_table
from latos.experiments.reduce import SIGMA_GLOSSARY
from latos.experiments.revision import WORKING_TREE_CLEAN, commit_known
from latos.ingestion.hashing import hash_bytes
from latos.optimization.engine import _DEFAULT_DELTA as DEFAULT_DELTA
from latos.optimization.engine import OptimizationResult
from latos.optimization.prereg import prereg_dir
from latos.optimization.validate import outcome_path_for

__all__ = [
    "DEFINITION",
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_UNREADABLE",
    "CampaignState",
    "load_campaign",
    "main",
    "observations_sha256",
    "recommend",
    "report",
    "report_recommendation",
]

EXPERIMENT_NAME = "ionic-liquid-seebeck-trace"
EXPERIMENT_VERSION = "1.0.0"

# What the command returns. See "Exit status" above.
EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_UNREADABLE = 2
EXIT_FAILED = 3

# What stands beside a frozen record as its outcome. None of these means
# scored, and nothing here can: the scoring of a version 2 record is not built,
# and a file cannot be checked against a format that does not exist yet. The
# version 1 route writes this kind with no schema version. It takes a bare
# number and checks neither the unit nor the composition, so what it wrote is
# named for what it is.
_OUTCOME_KIND = "latos.bo.outcome"
_OUTCOME_ABSENT = "no outcome"
_OUTCOME_VERSION_1 = "an outcome written by the version 1 route, which does not count"
_OUTCOME_UNREADABLE = "an outcome file that cannot be read"
_OUTCOME_UNCHECKED = "an outcome file that nothing can check yet"

# Held in the record folder while `next` freezes, so that two runs cannot both
# freeze the same data, and removed when it is done.
FREEZE_LOCK = ".freeze.lock"
# Where files left by a freeze that did not complete may be moved. Nothing in
# Latos reads it.
_INTERRUPTED = "interrupted"

# What `next` freezes with, by way of `run_optimization`, which passes the
# optimizer no acquisition, no stopping tolerance and no risk of its own.
_CAMPAIGN_ACQUISITION = "ei"

# A specimen read again from its recording by the same contract gives the same
# numbers; a library update may move the last bits.
_READ_AGAIN_REL = 1e-9
_READ_AGAIN_ABS = 1e-12

# The search domain: the range of composition this campaign optimizes over,
# decided at the readiness review of 8 Oct 2026. A complete specimen outside it
# is kept in the record and is not one of the fitted points.
SEARCH_BOUNDS = (0.239575, 1.0)

# How close two compositions must be to count as the same condition: two
# specimens this close are reduced together, and a specimen this close to a
# recommended composition counts as having made it.
CONDITION_PRECISION = 0.01

# The drive programme, as agreed for this campaign.
#
# The drive voltage is the operator's declared set-point, not a reading. So the
# control requires the declared value and has no tolerance: that is a statement
# about what must be written in the manifest, not a claim that the voltage
# delivered is known exactly.
DRIVE_SETPOINT_V = 0.5
DRIVE_SETPOINT_TOLERANCE_V = 0.0
PHASE_DURATION_S = 180.0
PHASE_TOLERANCE_S = 10.0
CYCLES_PER_SPECIMEN = 3
DROPS_APPLIED = 3


def _phase_control(column: str) -> VariableDefinition:
    return VariableDefinition(
        name=column,
        role=VariableRole.CONTROL,
        kind=VariableKind.CONTINUOUS,
        unit="s",
        column=column,
        expected_value=PHASE_DURATION_S,
        tolerance=PHASE_TOLERANCE_S,
    )


def _recorded(name: str, role: VariableRole, unit: str = "1") -> VariableDefinition:
    kind = VariableKind.CATEGORICAL if unit == "1" else VariableKind.CONTINUOUS
    return VariableDefinition(name=name, role=role, kind=kind, unit=unit, column=name)


DEFINITION = ExperimentDefinition(
    name=EXPERIMENT_NAME,
    version=EXPERIMENT_VERSION,
    title="Apparent cycle Seebeck coefficient of Aliquat 336TG / [HMIM][TFSI] mixtures on cotton",
    description=(
        "One design variable: the mass fraction of Aliquat 336TG in its mixture with "
        "[HMIM][TFSI]. The response is the apparent cycle Seebeck coefficient, an "
        "operational response to a fixed thermal drive programme and not a steady-state "
        "coefficient. Each specimen is one cotton mounting measured over three accepted "
        "cycles. The response is optimized signed, with no magnitude taken. The search "
        "covers the declared bounds of the design variable only; a complete specimen "
        "outside them is kept in the record as an accepted specimen and is not fitted. "
        "The drive voltage is the operator's declared set-point, required to equal its "
        "expected value; it is not a measured quantity."
    ),
    producer=ite_trace.CONTRACT,
    sample_key=ite_trace.SAMPLE_ID,
    variables=(
        VariableDefinition(
            name=ite_trace.ALIQUAT_MASS_FRACTION,
            role=VariableRole.DESIGN,
            kind=VariableKind.CONTINUOUS,
            unit="1",
            column=ite_trace.ALIQUAT_MASS_FRACTION,
            bounds=SEARCH_BOUNDS,
            precision=CONDITION_PRECISION,
            display_name="Aliquat 336TG mass fraction",
        ),
        VariableDefinition(
            name=ite_trace.DRIVE_VOLTAGE,
            role=VariableRole.CONTROL,
            kind=VariableKind.CONTINUOUS,
            unit="V",
            column=ite_trace.DRIVE_VOLTAGE,
            expected_value=DRIVE_SETPOINT_V,
            tolerance=DRIVE_SETPOINT_TOLERANCE_V,
            display_name="Declared drive set-point (not a measured voltage)",
        ),
        _phase_control(ite_trace.ON_DURATION_MIN),
        _phase_control(ite_trace.ON_DURATION_MAX),
        _phase_control(ite_trace.OFF_DURATION_MIN),
        _phase_control(ite_trace.OFF_DURATION_MAX),
        VariableDefinition(
            name=ite_trace.CYCLES_PER_SPECIMEN,
            role=VariableRole.CONTROL,
            kind=VariableKind.INTEGER,
            unit="1",
            column=ite_trace.CYCLES_PER_SPECIMEN,
            expected_value=CYCLES_PER_SPECIMEN,
        ),
        VariableDefinition(
            name=ite_trace.REMOUNTED,
            role=VariableRole.CONTROL,
            kind=VariableKind.CATEGORICAL,
            unit="1",
            column=ite_trace.REMOUNTED,
            expected_value="no",
        ),
        VariableDefinition(
            name=ite_trace.DROPS_APPLIED,
            role=VariableRole.CONTROL,
            kind=VariableKind.INTEGER,
            unit="1",
            column=ite_trace.DROPS_APPLIED,
            expected_value=DROPS_APPLIED,
        ),
        _recorded(ite_trace.ELECTRODE_MATERIAL, VariableRole.ENVIRONMENT),
        _recorded(ite_trace.PEAK_DELTA_T, VariableRole.ENVIRONMENT, "K"),
        _recorded(ite_trace.MEAN_TEMPERATURE, VariableRole.ENVIRONMENT, "°C"),
        _recorded(ite_trace.RELATIVE_HUMIDITY, VariableRole.ENVIRONMENT, "%"),
        _recorded(ite_trace.AMBIENT_TEMPERATURE, VariableRole.ENVIRONMENT, "°C"),
        _recorded(ite_trace.POLARITY_CONVENTION, VariableRole.METADATA),
        _recorded(ite_trace.BOUNDARY_SOURCE, VariableRole.METADATA),
        _recorded(ite_trace.DOSING_METHOD, VariableRole.METADATA),
        _recorded(ite_trace.NOMINAL_LABEL, VariableRole.METADATA),
        _recorded(ite_trace.MIXTURE_BATCH_ID, VariableRole.METADATA),
        _recorded(ite_trace.TRACE_FILE, VariableRole.METADATA),
        _recorded(ite_trace.TRACE_SHA256, VariableRole.METADATA),
    ),
    responses=(
        ResponseDefinition(
            name=ite_trace.APPARENT_SEEBECK,
            unit="mV/K",
            column=ite_trace.APPARENT_SEEBECK,
            uncertainty_column=ite_trace.APPARENT_SEEBECK_SEM,
            uncertainty_scope=UncertaintyScope.SPECIMEN,
            display_name="Apparent cycle Seebeck coefficient",
        ),
    ),
    objective=ObjectiveDefinition(
        response=ite_trace.APPARENT_SEEBECK,
        goal=Goal.MAXIMIZE,
        magnitude=False,
    ),
    specimens_per_condition=1,
)


@dataclass(frozen=True, slots=True)
class CampaignState:
    """Everything known about the campaign's data, up to the optimizer's door.

    Attributes:
        definition: The experiment the data were read under.
        table: The producer's rows, one per specimen.
        observation_set: Those rows checked, with exclusions, and reduced.
        prepared: What the optimizer would be handed.
    """

    definition: ExperimentDefinition
    table: RecordTable
    observation_set: ObservationSet
    prepared: PreparedOptimization

    @property
    def enough(self) -> bool:
        """Whether there are enough distinct conditions to fit a model."""
        return int(self.prepared.y.size) >= min_conditions(self.prepared.n_dims)


def load_campaign(
    manifest: Path, *, definition: ExperimentDefinition = DEFINITION
) -> CampaignState:
    """Read a manifest and take its specimens as far as the optimizer's door.

    Nothing is fitted and nothing is written.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ManifestError: If the manifest is malformed.
    """
    table = produce_table(manifest)
    observations, excluded = build_observations(definition, table)
    observation_set = reduce_observations(definition, observations, excluded)
    prepared = prepare_optimization(definition, observation_set)
    return CampaignState(definition, table, observation_set, prepared)


def recommend(state: CampaignState, *, seed: int = 0) -> OptimizationResult:
    """Ask the existing one-variable optimizer for the next composition.

    Raises:
        InsufficientConditions: If there are too few distinct conditions.
    """
    result = run_optimization(state.prepared, seed=seed)
    if not isinstance(result, OptimizationResult):  # one design variable, so never
        raise TypeError("this campaign has one design variable and expects the 1-D optimizer")
    return result


def observations_sha256(state: CampaignState) -> str:
    """The digest a frozen record would carry for the observations in `state`.

    It covers what the optimizer would be handed and nothing else: each
    condition's design value, signed value, optimized value and uncertainty.
    With the definition hash it says whether two runs fit the same data.
    """
    prepared = state.prepared
    sigma = None if prepared.sigma is None else prepared.sigma.tolist()
    return observations_digest_nd(
        prepared.x.tolist(), prepared.y.tolist(), prepared.y_raw.tolist(), sigma=sigma
    )


# ─── Reporting ──────────────────────────────────────────────────────
def _detail(row_detail: Any) -> dict[str, Any]:
    return row_detail if isinstance(row_detail, dict) else {}


def _search_domain(definition: ExperimentDefinition) -> str:
    design = definition.design_variables()[0]
    if design.bounds is None:  # a continuous design variable always has bounds
        return design.name
    low, high = design.bounds
    return f"{design.name} in [{low:g}, {high:g}]"


def _left_out_lines(sample_id: str, detail: dict[str, Any]) -> list[str]:
    """What a specimen's recording holds that is not in its value, and why."""
    lines = []
    for cycle in detail.get("cycles", []):
        if cycle["purpose"] == "test_run":
            span = f"{cycle['on_s']:g} to {cycle['end_s']:g} s"
            slope = cycle.get("slope_mv_k")
            said = (
                f"S = {slope:+.4f} mV/K"
                if isinstance(slope, (int, float))
                else "no slope could be fitted to it"
            )
            lines.append(
                f"  {sample_id}: cycle {cycle['index']} ({span}, {said}) is a declared test "
                "run: excluded by protocol, kept in the record"
            )
    reverse = detail.get("reverse_diagnostic")
    if reverse is not None:
        name = reverse.get("source", {}).get("file", "the reversed-lead recording")
        if "errors" in reverse:
            outcome = "could not be analysed"
        else:
            turned = {True: "reverses", False: "does NOT reverse", None: "not compared"}[
                reverse.get("recorded_sign_reverses")
            ]
            outcome = (
                f"recorded sign {turned}; mean {reverse['mean_oriented_slope_mv_k']:+.4f} mV/K "
                "once oriented by its own wiring"
            )
        lines.append(
            f"  {sample_id}: {name} is a reversed-lead diagnostic, not an observation ({outcome})"
        )
    return lines


def _control_state(observation: Observation, definition: ExperimentDefinition) -> str:
    """What an accepted specimen's controls are: all recorded and within tolerance, or not.

    A blank control keeps the specimen (rule R4 of the approved definition), so
    it is said here as unverified rather than as a reason to refuse.
    """
    blank = [
        v.name
        for v in definition.variables
        if v.role is VariableRole.CONTROL and v.column and _blank(observation.inputs.get(v.name))
    ]
    if blank:
        return (
            f"recorded controls within tolerance; {', '.join(blank)} NOT recorded, so unverified "
            "for this specimen (kept, as the protocol says of a blank control)"
        )
    return "controls within tolerance"


def _specimen_block(observation: Observation, definition: ExperimentDefinition) -> list[str]:
    """One accepted specimen: where it is, what it read, how well, and any warning."""
    response = definition.objective.response
    design = definition.design_variables()[0].name
    value = observation.responses[response]
    reduction = _detail(observation.producer_detail).get("reduction") or {}
    cycles = ", ".join(f"{v:+.4f}" for v in reduction.get("cycle_values_mv_k", []))
    uncertainty = (
        "none"
        if value.uncertainty is None
        else f"{value.uncertainty:.4f} (cycle SEM, {reduction.get('degrees_of_freedom')} "
        "d.o.f.; within-specimen lower bound)"
    )
    inputs = observation.inputs
    lines = [
        f"  {observation.sample_id}",
        f"    {design} = {inputs[design]:.6f}",
        f"    signed S = {value.value:+.4f} {value.unit}   cycles: {cycles}",
        f"    uncertainty = {uncertainty}",
        f"    on {inputs[ite_trace.ON_DURATION_MIN]:g} to "
        f"{inputs[ite_trace.ON_DURATION_MAX]:g} s, off "
        f"{inputs[ite_trace.OFF_DURATION_MIN]:g} to {inputs[ite_trace.OFF_DURATION_MAX]:g} s; "
        f"boundaries: {observation.metadata[ite_trace.BOUNDARY_SOURCE]}; "
        + _control_state(observation, definition),
    ]
    lines += [
        f"    {issue.severity.value.upper()}: {issue.message}" for issue in observation.issues
    ]
    return lines


def _specimen_lines(state: CampaignState) -> list[str]:
    definition = state.definition
    observations = state.observation_set.observations
    lines = [f"Specimens in the fit: {len(observations)}"]
    for observation in observations:
        lines += _specimen_block(observation, definition)

    outside = state.observation_set.outside_bounds()
    if outside:
        lines += [
            "",
            f"Specimens preserved outside the search bounds: {len(outside)}",
            "  Accepted specimens: complete, and no recorded control outside its tolerance. "
            "They are not in the fit "
            f"because the search covers {_search_domain(definition)} only. Not rejected, and "
            "not outliers.",
        ]
        for row in outside:
            if row.observation is not None:
                lines += _specimen_block(row.observation, definition)
            lines.append("    NOT in the fit: " + "; ".join(row.reasons))
    return lines


def _excluded_lines(state: CampaignState) -> list[str]:
    lines = ["Excluded data"]
    for row in state.observation_set.rejected():
        label = row.sample_id or row.record.record_id
        lines.append(f"  {label}: NOT in the model. " + "; ".join(row.reasons))
        lines += [
            f"    {issue.severity.value.upper()}: {issue.message}"
            for issue in row.record.issues
            if issue.message not in row.reasons
        ]
    for record in state.table.rows:
        lines += _left_out_lines(record.record_id, _detail(record.producer_detail))
    return lines if len(lines) > 1 else [*lines, "  nothing"]


def _condition_lines(state: CampaignState) -> list[str]:
    prepared = state.prepared
    projection = {
        "identity": "none: the signed values are handed over unchanged",
        "magnitude": "magnitude: the absolute value of each signed value",
        "target_distance": "distance to the target",
    }[prepared.objective_aggregation]
    optimizer = (
        "the one-variable optimizer, optimize()"
        if prepared.n_dims == 1
        else "the multi-variable optimizer, optimize_nd()"
    )
    lines = [
        f"Conditions handed to the optimizer as {prepared.target_name} ({prepared.direction})",
        f"  search domain: {_search_domain(state.definition)}",
        f"  projection: {projection}",
        f"  {prepared.y.size} distinct condition(s); {optimizer} needs at least "
        f"{min_conditions(prepared.n_dims)}",
    ]
    for index, condition in enumerate(state.observation_set.reduced):
        sigma = "none" if prepared.sigma is None else f"{prepared.sigma[index]:.4f}"
        lines.append(
            f"  x = {prepared.x[index, 0]:.6f}   signed = {prepared.y_raw[index]:+.4f}   "
            f"optimized = {prepared.y[index]:+.4f}   sigma = {sigma} [{condition.sigma_source}]"
        )
    for label in sorted(set(prepared.sigma_sources)):
        lines.append(f"  [{label}] {SIGMA_GLOSSARY[label]}")
    noise = "none" if prepared.prediction_noise is None else f"{prepared.prediction_noise:.4f}"
    lines.append(
        f"  prediction noise = {noise} [{prepared.prediction_noise_source}] "
        f"{PREDICTION_NOISE_GLOSSARY[prepared.prediction_noise_source]}"
    )
    return lines


def _warning_lines(state: CampaignState) -> list[str]:
    warnings = [*state.observation_set.messages, *state.prepared.messages]
    for condition in state.observation_set.reduced:
        warnings += [issue.message for issue in condition.issues]
    for row in state.table.rows:
        warnings += [i.message for i in row.issues if i.severity is Severity.WARNING]
    if not warnings:
        return ["Warnings", "  none"]
    return ["Warnings", *(f"  WARNING: {message}" for message in dict.fromkeys(warnings))]


def _revision_lines(revision: SourceRevision | None) -> list[str]:
    if revision is None:
        return []
    lines = [f"Source revision: {revision.describe()}"]
    if not revision.is_clean:
        lines.append("  `next` freezes nothing until the source tree is exactly a commit.")
    return lines


def report(state: CampaignState, revision: SourceRevision | None = None) -> str:
    """The campaign's data as text: specimens, exclusions, conditions, warnings.

    Args:
        state: The campaign, as `load_campaign` returns it.
        revision: The source revision to state in the heading, if it was read.
    """
    definition = state.definition
    lines = [
        f"{definition.title}",
        f"{definition.name} {definition.version}   definition {definition_sha256(definition)}"
        f"   producer contract {definition.producer}",
        *_revision_lines(revision),
        "",
        *_specimen_lines(state),
        "",
        *_excluded_lines(state),
        "",
        *_condition_lines(state),
        "",
        *_warning_lines(state),
    ]
    if not state.enough:
        need = min_conditions(state.prepared.n_dims)
        lines += [
            "",
            f"{state.prepared.y.size} distinct condition(s); a model needs at least {need}.",
        ]
    return "\n".join(lines)


def report_recommendation(
    result: OptimizationResult, state: CampaignState, *, heading: str = "Recommendation"
) -> str:
    """The optimizer's answer as text, in the terms the record freezes.

    Args:
        result: What `recommend` returned.
        state: The campaign it was computed from.
        heading: The first line. It is where a reader learns whether what
            follows is frozen or only a preview, so the caller says which.
    """
    return "\n".join([heading, *_recommendation_lines(result, state)])


def _recommendation_lines(result: OptimizationResult, state: CampaignState) -> list[str]:
    """The optimizer's answer, without a heading to say whether it is frozen."""
    recommendation = result.recommendation
    low, high = recommendation.predictive_interval_95
    name = state.prepared.input_names[0]
    unit = state.definition.objective_response().unit
    lines = [
        f"  make {name} = {recommendation.x:.4f}",
        f"  predicted {result.target_name} = {recommendation.predicted_mean:.4f} {unit}",
        f"  95% predictive interval = [{low:.4f}, {high:.4f}] {unit}   "
        f"(model alone: +/- {recommendation.ci95:.4f} {unit})",
        f"  best fitted condition so far = {result.best_y:.4f} {unit} at {name} = "
        f"{result.best_x:.4f} (specimens outside the search bounds are not fitted)",
        f"  largest expected improvement over the search range = {result.max_ei:.4g} {unit} "
        f"(noise threshold {result.noise_threshold:.4g} {unit})",
    ]
    if result.stopping is not None:
        lines.append(f"  {result.stopping.action}: {result.stopping.reason}")
    if result.reliability is not None:
        lines.append(f"  reliability: {result.reliability.level}. {result.reliability.note}")
    return lines


# ─── What stops a fit ───────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _Frozen:
    """A frozen version 2 record, as far as the refusals need it.

    Attributes:
        path: The record file.
        name: The experiment name it was frozen under.
        version: The experiment version it was frozen under.
        definition_sha256: The identity hash of the definition it was frozen under.
        observations_sha256: The digest of the observations its model was fitted to.
        specimens: Every specimen it holds, fitted or left out, each with the
            mark `_mark` gives it.
        traces: The SHA-256 of every recording those specimens were read from.
        outcome: One of the ``_OUTCOME_`` sentences: what stands beside it.
    """

    path: Path
    name: str
    version: str
    definition_sha256: str
    observations_sha256: str
    specimens: Mapping[str, str]
    traces: frozenset[str]
    outcome: str


def _why(exc: BaseException) -> str:
    """An exception as a short reason, without a traceback."""
    return (getattr(exc, "strerror", None) or str(exc) or type(exc).__name__).strip()


def _record_directory(manifest: Path) -> Path:
    """Where this campaign's frozen records are kept: beside its manifest, and nowhere else."""
    return prereg_dir(manifest.parent)


def _outcome_state(record_path: Path) -> str:
    """What stands beside the record at `record_path` as its outcome, as a sentence.

    No answer means scored. See the ``_OUTCOME_`` sentences.
    """
    path = outcome_path_for(record_path)
    if not path.exists():
        return _OUTCOME_ABSENT
    try:
        outcome = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _OUTCOME_UNREADABLE
    version_1 = (
        isinstance(outcome, dict)
        and outcome.get("kind") == _OUTCOME_KIND
        and "schema_version" not in outcome
    )
    return _OUTCOME_VERSION_1 if version_1 else _OUTCOME_UNCHECKED


def _mark(design: Any, trace: Any, value: Any, uncertainty: Any) -> str:
    """What makes a specimen the same specimen with the same data, as one text.

    Its composition, the fingerprint of its recording, and the value and
    uncertainty read from it. A specimen made again under an old id, a
    recording replaced, and a manifest entry changed so that another value
    comes out each give another mark.
    """
    return json.dumps([design, trace, value, uncertainty])


def _cell(values: Mapping[str, Any], column: str | None) -> Any:
    return None if column is None else values.get(column)


def _declared_marks(state: CampaignState) -> dict[str, str]:
    """Each specimen the manifest declares, with its mark, used or not."""
    design = state.definition.design_variables()[0]
    response = state.definition.objective_response()

    def of(observation: Observation) -> str:
        value = observation.responses.get(response.name)
        return _mark(
            observation.inputs.get(design.name),
            observation.metadata.get(ite_trace.TRACE_SHA256),
            None if value is None else value.value,
            None if value is None else value.uncertainty,
        )

    marks = {
        observation.sample_id: of(observation) for observation in state.observation_set.observations
    }
    for row in state.observation_set.excluded:  # rejected, or outside the search bounds
        label = row.sample_id or row.record.record_id
        if row.observation is not None:
            marks[label] = of(row.observation)
            continue
        values = row.record.values
        marks[label] = _mark(
            _cell(values, design.column),
            values.get(ite_trace.TRACE_SHA256),
            _cell(values, response.column),
            _cell(values, response.uncertainty_column),
        )
    return marks


def _declared_traces(state: CampaignState) -> frozenset[str]:
    """The SHA-256 of every recording a declared specimen was read from."""
    found = [o.metadata.get(ite_trace.TRACE_SHA256) for o in state.observation_set.observations]
    for row in state.observation_set.excluded:
        values = row.observation.metadata if row.observation is not None else row.record.values
        found.append(values.get(ite_trace.TRACE_SHA256))
    return frozenset(str(trace) for trace in found if trace)


def _frozen_traces(record: Mapping[str, Any]) -> frozenset[str]:
    """The SHA-256 of every recording a frozen record's specimens were read from."""
    found = [
        (row["metadata"] if "inputs" in row else row["values"]).get(ite_trace.TRACE_SHA256)
        for row in [*record["specimens"], *record["excluded"]]
    ]
    return frozenset(str(trace) for trace in found if trace)


def _frozen_marks(record: Mapping[str, Any]) -> dict[str, str]:
    """Each specimen a frozen record holds, fitted or left out, with its mark.

    Read the way `_declared_marks` reads the manifest's specimens, so a
    specimen that has not changed gives the same mark on both sides. The
    record is read under the names it was frozen with, which it carries in its
    own definition: a design variable or a response renamed since is the same
    quantity, and its specimens are the same specimens.

    Raises:
        KeyError: If the record's definition names no design variable, or not
            the response its objective is about.
    """
    stored = record["experiment"]["definition"]
    designs = [v for v in stored["variables"] if v["role"] == VariableRole.DESIGN.value]
    wanted = stored["objective"]["response"]
    responses = [r for r in stored["responses"] if r["name"] == wanted]
    if not designs or not responses:
        raise KeyError("its definition names no design variable, or not its objective's response")
    design, response = designs[0], responses[0]
    marks: dict[str, str] = {}
    for row in [*record["specimens"], *record["excluded"]]:
        label = str(row.get("sample_id") or row["record_id"])
        if "inputs" in row:
            value = row["responses"].get(response["name"]) or {}
            marks[label] = _mark(
                row["inputs"].get(design["name"]),
                row["metadata"].get(ite_trace.TRACE_SHA256),
                value.get("value"),
                value.get("uncertainty"),
            )
            continue
        values = row["values"]
        marks[label] = _mark(
            _cell(values, design.get("column")),
            values.get(ite_trace.TRACE_SHA256),
            _cell(values, response.get("column")),
            _cell(values, response.get("uncertainty_column")),
        )
    return marks


def _frozen_records(directory: Path) -> tuple[list[_Frozen], list[str]]:
    """Every version 2 record in `directory`, and each reason one could not be read.

    Records of every experiment are returned: the same observations under the
    same definition are the same data, whatever name and version they were
    frozen under. A file that cannot be read is not passed over. It may be the
    very record that would have stopped the run, so it is reported, and it
    stops the run. A version 1 record is passed over: another command wrote
    it, and it names no experiment.
    """
    found: list[_Frozen] = []
    unreadable: list[str] = []
    try:
        paths = sorted(path for path in directory.iterdir() if path.match("prereg_*.json"))
    except OSError as exc:
        if directory.parent.is_file():
            return found, [
                f"the record folder {directory} cannot be made, because {directory.parent} is a "
                "file, so nothing could be frozen there"
            ]
        if isinstance(exc, FileNotFoundError):
            return found, unreadable  # no record folder yet
        return found, [
            f"the record folder {directory} could not be listed ({_why(exc)}), so an earlier "
            "frozen record of this campaign cannot be ruled out"
        ]
    for path in paths:
        if ".outcome." in path.name:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            unreadable.append(
                f"{path} could not be read ({_why(exc)}), so it cannot be ruled out as an "
                "earlier frozen record of this campaign"
            )
            continue
        if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
            continue
        try:
            experiment = record["experiment"]
            found.append(
                _Frozen(
                    path=path,
                    name=str(experiment["name"]),
                    version=str(experiment["version"]),
                    definition_sha256=str(experiment["definition_sha256"]),
                    observations_sha256=str(record["observations"]["sha256"]),
                    specimens=_frozen_marks(record),
                    traces=_frozen_traces(record),
                    outcome=_outcome_state(path),
                )
            )
        except (KeyError, TypeError, AttributeError) as exc:
            unreadable.append(
                f"{path} says it is a version {SCHEMA_VERSION} record, and a block every such "
                f"record has is missing or is not what it should be ({type(exc).__name__}: "
                f"{exc}), so it cannot be ruled out as an earlier frozen record of this campaign"
            )
    return found, unreadable


def _label(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _unfinished(directory: Path) -> list[str]:
    """What a freeze that did not complete left in the record folder, each as a reason.

    A freeze publishes its record last, so none of these is a record. Each may
    still state a recommendation that was never confirmed, which is why it is
    not passed over: the person who runs this decides what it was.
    """
    try:
        left = interrupted_freeze_files(directory)
    except OSError:
        return []  # the folder could not be listed, and `_frozen_records` says so
    reasons = [
        f"{path} was left by a freeze that did not complete. It is not a record, and nothing "
        "it states was confirmed as frozen. If anyone has opened it, its recommendation has "
        "been seen, and freezing the same data again would give the same one without saying "
        "so. A run stopped at the last step can also have published the record itself beside "
        "it: run `verify` on any prereg_*.json there of the same name before deciding what "
        f"was frozen. Once that is noted, move it to {directory.parent / _INTERRUPTED}, a "
        "folder nothing here reads, and run this again"
        for path in left
    ]
    lock = directory / FREEZE_LOCK
    if lock.exists():
        try:
            held = lock.read_text(encoding="utf-8").strip() or "it says nothing more"
        except OSError as exc:
            held = f"it could not be read: {_why(exc)}"
        reasons.append(
            f"{lock} says that a `next` is freezing in this folder now, or that one was "
            f"stopped before it finished ({held}). If no other `next` is running, delete it "
            "and run this again"
        )
    return reasons


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _version_one_duplicates(state: CampaignState, directory: Path) -> list[str]:
    """A version 1 record in the folder that froze a recommendation from these very data.

    Version 1 records come from another command and name no experiment, so they
    are otherwise passed over. One whose model saw exactly the conditions this
    fit would see has already frozen a recommendation from them.
    """
    if len(state.prepared.input_names) != 1:
        return []  # a version 1 record holds one variable
    now = sorted(zip(state.prepared.x[:, 0].tolist(), state.prepared.y.tolist(), strict=True))
    reasons = []
    try:
        paths = sorted(p for p in directory.iterdir() if p.match("prereg_*.json"))
    except OSError:
        return []  # said by `_frozen_records`
    for path in paths:
        if ".outcome." in path.name:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            training = record["training_data"]
            seen = sorted(zip(training["x"], training["y"], strict=True))
        except (OSError, ValueError, KeyError, TypeError):
            continue  # unreadable files are said by `_frozen_records`; others are not records
        if "schema_version" not in record and seen == now:
            reasons.append(
                f"{path} is a version 1 record whose model saw exactly these conditions, so a "
                "recommendation from these data was frozen already. One set of observations "
                "gets one frozen recommendation"
            )
    return reasons


def _departures(record: _Frozen, declared: Mapping[str, str]) -> list[str]:
    """How the manifest's specimens differ from the ones `record` holds. Empty when they do not."""
    added = [label for label in declared if label not in record.specimens]
    changed = [
        label
        for label, mark in declared.items()
        if label in record.specimens and record.specimens[label] != mark
    ]
    gone = [label for label in record.specimens if label not in declared]
    return [
        f"{name}: {', '.join(labels)}"
        for name, labels in (("added", added), ("changed", changed), ("no longer declared", gone))
        if labels
    ]


def _refusals(
    state: CampaignState,
    revision: SourceRevision,
    directory: Path,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Every reason not to fit a model to `state`, each as a sentence. Empty when none.

    All of them are found, not only the first, so one run says everything that
    has to be put right. Nothing here fits a model or writes a file.

    Args:
        state: The campaign, as `load_campaign` returns it.
        revision: The source revision, as read once for this run.
        directory: The campaign's record folder.
        dry_run: A preview is exempt from the source-revision rule and from
            nothing else.
    """
    reasons: list[str] = []
    definition = state.definition

    rejected = state.observation_set.rejected()
    if rejected:
        names = ", ".join(row.sample_id or row.record.record_id for row in rejected)
        reasons.append(
            f"{_label(len(rejected), 'declared specimen')} could not be used: {names}. "
            'Each is under "Excluded data" with its reason. A declared specimen is never '
            "quietly left out of a fit"
        )

    earlier, unreadable = _frozen_records(directory)
    reasons += unreadable
    reasons += _unfinished(directory)
    reasons += _version_one_duplicates(state, directory)
    identity = definition_sha256(definition)
    digest = observations_sha256(state)
    for record in earlier:
        same_version = (record.name, record.version) == (definition.name, definition.version)
        if same_version and record.definition_sha256 != identity:
            reasons.append(
                f"{definition.name} {definition.version} was frozen earlier under a different "
                f"definition ({record.path.name}: {record.definition_sha256}; now: {identity}). "
                "Bump the version or revert the definition"
            )

    # Every earlier record of this campaign is waiting for its outcome, because
    # nothing can score one yet (see the ``_OUTCOME_`` sentences), and while one
    # waits nothing else is fitted: not other specimens, and not the same
    # specimens read under another definition, version or name.
    declared = _declared_marks(state)
    marks = set(declared.values())
    traces = _declared_traces(state)
    for record in earlier:
        same_definition = record.definition_sha256 == identity
        # A specimen in common is told by its id or, under another id, by its
        # recording or by its whole mark.
        shared = bool(record.traces & traces) or any(
            label in declared or mark in marks for label, mark in record.specimens.items()
        )
        if record.name != definition.name and not same_definition and not shared:
            continue  # another experiment's record
        duplicate = same_definition and record.observations_sha256 == digest
        if duplicate:
            reasons.append(
                f"a frozen record already holds these observations under this definition: "
                f"{record.path} (observations {digest[:16]}). One set of observations gets one "
                "frozen record"
            )
        departures = _departures(record, declared)
        if departures:
            differs = (
                f"the manifest does not declare exactly what {record.path.name} holds "
                f"({'; '.join(departures)})"
            )
        elif duplicate:
            continue  # the same data, said once
        elif same_definition:
            # An edited record, or one frozen before the reduction was changed.
            differs = (
                f"{record.path.name} holds these specimens and states other observations than "
                f"they give now ({record.observations_sha256[:16]}; now: {digest[:16]})"
            )
        else:
            # The same specimens, read under another definition: a second
            # recommendation from them before the first one has been tested.
            differs = (
                f"{record.path.name} already holds exactly these specimens, frozen under "
                f"another definition ({record.definition_sha256[:16]}; now: {identity[:16]})"
            )
        # Said when it is not this experiment name and version: the record may
        # be one the reader does not think of as this campaign's.
        frozen_as = (
            ""
            if (record.name, record.version) == (definition.name, definition.version)
            else f" (it was frozen as {record.name} {record.version})"
        )
        reasons.append(
            f"{differs}, and that record has {record.outcome}{frozen_as}. After a record is "
            "frozen nothing else is fitted until its outcome has been scored, and nothing scores "
            "a version 2 record yet"
        )

    if not state.enough:
        need = min_conditions(state.prepared.n_dims)
        reasons.append(
            f"{state.prepared.y.size} distinct condition(s) are in the fit, and a model needs "
            f"at least {need}"
        )
    if not dry_run and not revision.is_clean:
        # Decided before the model is fitted: something that has been seen
        # cannot be made prospective afterwards by committing the code.
        reasons.append(
            "what is frozen must be reproducible from a precise source revision, and this one "
            f"is not: {revision.describe()}. Commit the code, or restore the tree, and run this "
            "again"
        )
    return reasons


def _readiness_lines(reasons: Sequence[str]) -> list[str]:
    """What `check` says about `next`, from the reasons `next` would give."""
    if not reasons:
        return [
            "Readiness for `next`",
            "  no refusal applies to this data and this source tree. A warning above is not a "
            "refusal: read each one before running `next`.",
        ]
    return [
        "Readiness for `next`",
        "  NOT READY. `next` would fit nothing and freeze nothing, because:",
        *(f"  - {reason}" for reason in reasons),
    ]


# ─── Command line ───────────────────────────────────────────────────
def _load(manifest: Path) -> CampaignState | None:
    try:
        return load_campaign(manifest)
    except FileNotFoundError:
        print(f"No such manifest: {manifest}")
    except ManifestError as exc:
        print(exc)
    except OSError as exc:
        print(f"Could not read the manifest {manifest}: {_why(exc)}")
    return None


def _file_names(directory: Path) -> frozenset[str]:
    try:
        return frozenset(path.name for path in directory.iterdir())
    except OSError:  # no such folder yet
        return frozenset()


def _not_a_record(path: Path) -> str:
    """What a file left by a failed freeze is when it cannot be a record."""
    if path.name.endswith(PARTIAL_SUFFIX):
        return "an unfinished file of this attempt, never published; not a record"
    if path.suffix == ".md":
        return "the note of a record that was not published; not a record"
    return "not a record"


def _what_it_is(path: Path) -> str:
    """A file a failed freeze left, said as what it is."""
    if path.suffix != ".json" or path.name.endswith(PARTIAL_SUFFIX):
        return _not_a_record(path)
    try:
        problems = verify_record(read_record(path))
    except OSError as exc:
        return f"could not be read ({_why(exc)})"
    except ValueError as exc:
        return f"not a record: {_why(exc)}"
    if problems:
        return f"a version {SCHEMA_VERSION} record that does NOT verify"
    return f"a complete version {SCHEMA_VERSION} record, and it verifies against its own seal"


def _freeze_failure(
    exc: BaseException, directory: Path, before: frozenset[str], *, existed: bool
) -> str:
    """What to say when a model was fitted and its freeze was not confirmed.

    A freeze is confirmed when the record has been written, read back and
    verified. Any of the three may be what failed, so the files the attempt
    left are each said as what they are.

    Args:
        exc: What stopped the freeze.
        directory: The record folder.
        before: The names in it before the attempt.
        existed: Whether the folder was there before the attempt.
    """
    left = sorted(_file_names(directory) - before)
    lines = [
        "A model was fitted and its freeze was NOT confirmed. A freeze is confirmed only when the "
        "record has been written, read back and verified, and that did not complete. What the "
        "model suggested is not shown.",
        f"  reason: {type(exc).__name__}: {_why(exc)}",
    ]
    if left:
        lines.append(f"  The attempt left these files in {directory}:")
        lines += [f"    {name}: {_what_it_is(directory / name)}" for name in left]
    if any(name.endswith(".json") for name in left):
        lines += [
            f"  While a .json file there reads as a version {SCHEMA_VERSION} record, or cannot be "
            "read at all, `next` will refuse to fit these observations again. Do not move or "
            "delete it to get past that.",
            "  The seal of a record left this way was never shown. Run `verify` on it before "
            "deciding what it is.",
        ]
    if any(not name.endswith(".json") for name in left):
        lines.append(
            "  A file that is not a record was meant to be removed again and could not be. "
            "Nothing it states was confirmed as frozen. `check` and `next` name each such file "
            f"until it is moved to {directory.parent / _INTERRUPTED}."
        )
    if not left:
        lines.append(
            f"  Nothing was written in {directory}."
            if existed or not directory.is_dir()
            else f"  The attempt created the folder {directory} and left no file in it."
        )
    return "\n".join(lines)


def _failed(said: str, exc: BaseException) -> int:
    """Say that `next` failed before anything was written, and return its status.

    The traceback is kept for a failure that is not of a kind a run expects.
    """
    print()
    print(said)
    print(f"  reason: {type(exc).__name__}: {_why(exc)}")
    _report_fault(exc)
    return EXIT_FAILED


def _report_fault(exc: BaseException) -> None:
    """Print the traceback of a failure that is not of a kind a run expects."""
    if not isinstance(exc, (OSError, ValueError, LatosError)):
        traceback.print_exception(exc)


def _next(manifest: Path, out: Path | None, *, dry_run: bool) -> int:
    state = _load(manifest)
    if state is None:
        return EXIT_UNREADABLE
    revision = source_revision()
    print(report(state, revision))

    directory = _record_directory(manifest)
    listed = _file_names(directory)  # what the refusals are about to see
    reasons = _refusals(state, revision, directory, dry_run=dry_run)
    if out is not None and out.resolve() != directory.resolve():
        reasons.append(
            f"--out {out} is not this campaign's record folder, {directory}. A campaign's "
            "frozen records are kept in that one folder, because a record written anywhere "
            "else could not stop a later run"
        )
    if reasons:
        print()
        print("Nothing was fitted and nothing was frozen. `next` refuses, because:")
        print("\n".join(f"  - {reason}" for reason in reasons))
        return EXIT_REFUSED
    return _fit_and_freeze(state, revision, directory, dry_run=dry_run, listed=listed)


def _fit_and_freeze(
    state: CampaignState,
    revision: SourceRevision,
    directory: Path,
    *,
    dry_run: bool,
    listed: frozenset[str],
) -> int:
    """Fit, then preview or freeze. Only reached when no refusal applies.

    Args:
        state: The campaign.
        revision: The source revision, as read once for this run.
        directory: The record folder.
        dry_run: Preview only.
        listed: The names in the record folder when the refusals were made.
    """
    # Everything that will be shown is made before anything is written, so a
    # failure here leaves no record behind.
    try:
        result = recommend(state)
    except Exception as exc:
        return _failed(
            "A model could not be fitted. Nothing was frozen, and there is nothing to show.", exc
        )
    try:
        shown = _recommendation_lines(result, state)
    except Exception as exc:
        return _failed(
            "A model was fitted and what it suggested could not be set out. Nothing was frozen, "
            "and nothing is shown.",
            exc,
        )
    if dry_run:
        print()
        print("\n".join(["PREVIEW ONLY. Not frozen, so not a committed prediction", *shown]))
        print()
        print(
            "Nothing was frozen. Nothing records that this preview was shown. A later `next` on "
            "the same data can freeze this same recommendation, and its record will not say that "
            "it was seen first."
        )
        return EXIT_OK

    # Hold the record folder for the rest of the run, so two runs cannot both
    # freeze, and look again at what is in it: the refusals were made before
    # the model was fitted, and a record may have been frozen since.
    existed = directory.is_dir()
    lock = directory / FREEZE_LOCK
    # What this run writes into the lock, so that it removes only its own.
    held = (
        f"held by `next`, process {os.getpid()}, since {datetime.now(UTC)}, {os.urandom(8).hex()}"
    )
    created = False
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with lock.open("x", encoding="utf-8") as handle:
            created = True
            handle.write(held)
    except OSError as exc:
        print()
        print(_lock_failure(exc, lock, left=created and not _release(lock, held, written=False)))
        return EXIT_FAILED
    try:
        return _freeze_held(
            state, revision, directory, result, shown, listed=listed, existed=existed
        )
    finally:
        if not _release(lock, held):
            print(
                f"  The freeze lock {lock} could not be removed, or no longer holds what this "
                "run wrote into it. `check` and `next` will say so until it is looked at."
            )


def _release(lock: Path, held: str, *, written: bool = True) -> bool:
    """Remove the lock this run took. False when it could not, or it is no longer this run's.

    Args:
        lock: The lock file.
        held: What this run wrote into it.
        written: Whether the writing completed. A lock this run created and
            could not finish writing is still its own.
    """
    try:
        if written and lock.read_text(encoding="utf-8") != held:
            return False  # another run's, since this one's was removed by hand
        lock.unlink()
    except FileNotFoundError:
        return not written
    except OSError:
        return False
    return True


def _lock_failure(exc: OSError, lock: Path, *, left: bool = False) -> str:
    """What to say when the record folder could not be held for a freeze.

    Args:
        exc: Why.
        lock: The lock file.
        left: Whether this attempt created the lock and could not remove it.
    """
    lines = [
        "A model was fitted and nothing was frozen: the record folder could not be held for "
        "this freeze. What the model suggested is not shown.",
        f"  reason: {type(exc).__name__}: {_why(exc)}",
    ]
    if isinstance(exc, FileExistsError):
        lines.append(
            f"  {lock} says that another `next` is freezing in this folder now, or that one was "
            "stopped before it finished. If no other `next` is running, delete it and run this "
            "again."
        )
    elif left:
        lines.append(
            f"  This attempt created {lock} and could not remove it. `check` and `next` will "
            "refuse until it is deleted."
        )
    return "\n".join(lines)


def _freeze_held(
    state: CampaignState,
    revision: SourceRevision,
    directory: Path,
    result: OptimizationResult,
    shown: list[str],
    *,
    listed: frozenset[str],
    existed: bool,
) -> int:
    """Freeze, read back and verify, with the record folder held. Only then show anything."""
    try:
        now = frozenset(path.name for path in directory.iterdir())
    except OSError as exc:
        print()
        print(
            f"A model was fitted and nothing was frozen: the record folder {directory} could not "
            f"be listed again ({_why(exc)}), so a record frozen meanwhile cannot be ruled out. "
            "What the model suggested is not shown."
        )
        return EXIT_FAILED
    appeared = sorted(now - listed - {FREEZE_LOCK})
    if appeared:
        print()
        print(
            "A model was fitted and nothing was frozen: while it was being fitted, files "
            f"appeared in {directory} that the refusals did not see: {', '.join(appeared)}. "
            "What the model suggested is not shown. Run `next` again, so that they are read."
        )
        return EXIT_FAILED

    # Freeze, read back and verify first. What the model suggested is printed
    # only once a record that verifies is on disk, so nothing on the screen can
    # be taken for a commitment that was not made.
    before = now
    try:
        path = freeze_v2(
            state.definition,
            state.observation_set,
            state.prepared,
            result,
            directory,
            source=revision,
        )
        record = read_record(path)
        problems = verify_record(record)
    except Exception as exc:
        print()
        print(_freeze_failure(exc, directory, before, existed=existed))
        _report_fault(exc)
        return EXIT_FAILED
    if problems:
        print()
        print(
            "A record was written and it does NOT verify. What the model suggested is not "
            f"shown: {path}"
        )
        print("\n".join(f"  PROBLEM: {problem}" for problem in problems))
        note = path.with_suffix(".md")
        if note.exists():
            print(f"  Its note was written too, and states that record's seal: {note}")
        print(
            "  Keep what was written as it is. Do not note its seal and do not make a specimen "
            "from it. While the record is there, `next` reads it as an earlier record of this "
            "campaign."
        )
        return EXIT_FAILED

    print()
    print(f"Frozen: {path}")
    print(f"Source: {revision.describe()}")
    print(f"Seal:   {record['seal']['sha256']}")
    print()
    print("\n".join([f"Recommendation, frozen in {path.name}", *shown]))
    print()
    print("Write the seal in the lab notebook now, before the specimen is made.")
    return EXIT_OK


def _verify(path: Path, seal: str | None) -> int:
    try:
        record = read_record(path)
    except FileNotFoundError:
        print(f"No such record: {path}")
        return EXIT_UNREADABLE
    except RecordError as exc:
        print(exc)
        return EXIT_UNREADABLE
    except OSError as exc:
        print(f"Could not read the record {path}: {_why(exc)}")
        return EXIT_UNREADABLE
    problems = list(verify_record(record, expected_seal=seal))
    notes: list[str] = []
    try:
        found, notes = _campaign_findings(record, path)
        problems += found
    except Exception as exc:
        problems.append(f"the campaign's own checks could not be made: {type(exc).__name__}: {exc}")
    if problems:
        print(f"{path.name} does NOT verify:")
        for problem in problems:
            print(f"  - {problem}")
        for note in notes:
            print(f"  NOTE: {note}")
        return EXIT_REFUSED
    against = "its own seal" if seal is None else "its own seal and the seal given"
    print(f"{path.name} verifies against {against}.")
    print(
        "  Its note is the one it renders, its commit is in this repository as far as Git "
        "could say, and every recording it names that is still beside it is byte for byte "
        "the one it was read from and gives the values the record holds."
    )
    print(f"Seal: {record['seal']['sha256']}")
    source = record["source_revision"]
    print(f"Source revision: {source['git_commit']} (working tree {source['working_tree']})")
    for note in notes:
        print(f"  NOTE: {note}")
    if seal is None:
        print(
            "  Without the seal noted at the freeze, this shows that the record is consistent, "
            "not that it is the one frozen. Pass --seal to compare."
        )
    return EXIT_OK


def _campaign_findings(record: Mapping[str, Any], path: Path) -> tuple[list[str], list[str]]:
    """What a record of this campaign must also agree with, beyond itself.

    Its producer's detail on every row, the recordings and the manifest it was
    read from when they are still beside it, its note, and its commit. Files
    are read and Git is asked read-only; nothing is written.

    Returns:
        The problems, then the notes: what could not be compared, or changed
        in a way a campaign may change after a freeze.
    """
    problems: list[str] = []
    notes: list[str] = []
    if record["experiment"].get("producer") != ite_trace.CONTRACT:
        return [
            f"it is not a record of the {ite_trace.CONTRACT} producer, so it is not a record of "
            "this campaign and none of the campaign's checks applies to it"
        ], notes
    rows = [*record["specimens"], *record["excluded"]]
    problems += _producer_detail_problems(rows)
    problems += _campaign_settings_problems(record)

    source = record["source_revision"]
    if source.get("working_tree") != WORKING_TREE_CLEAN:
        problems.append(
            "the record states a source tree that was not exactly a commit, and `next` freezes "
            "nothing from such a tree"
        )
    commit = source.get("git_commit")
    known = commit_known(commit) if isinstance(commit, str) else None
    if known is False:
        problems.append(f"the commit {commit} is not in this repository")
    elif known is None:
        notes.append("whether the recorded commit is in this repository could not be checked")

    folder = path.resolve().parent
    campaign = folder.parent.parent
    in_campaign = prereg_dir(campaign) == folder
    note = path.with_suffix(".md")
    try:
        written = note.read_text(encoding="utf-8")
    except FileNotFoundError:
        written = None
    except OSError as exc:
        problems.append(f"its note {note.name} could not be read ({_why(exc)})")
        written = ""
    if written is None:
        # A freeze writes the note with the record; a copy elsewhere may not.
        (problems if in_campaign else notes).append(f"its note {note.name} is not beside it")
    elif written and written != render_note(record):
        problems.append(
            f"its note {note.name} is not the note this record renders: it was edited, or "
            "written by another version of Latos. Compare the two before relying on either"
        )

    if not in_campaign:
        notes.append(
            "the record is not in a campaign's record folder, so it was not compared with the "
            "recordings and manifest it was read from"
        )
        return problems, notes
    found, said = _files_against(rows, campaign)
    return problems + found + _manifest_rows(rows, campaign) + _read_again(record, campaign), (
        notes + said
    )


def _manifest_rows(rows: Sequence[Mapping[str, Any]], campaign: Path) -> list[str]:
    """While the manifest is the one frozen, the record holds exactly its specimens."""
    frozen = {
        (detail["manifest"].get("file"), detail["manifest"].get("sha256"))
        for detail in (row.get("producer_detail") or {} for row in rows)
        if isinstance(detail.get("manifest"), dict)
    }
    if len(frozen) != 1:
        return []  # said by `_producer_detail_problems`
    ((name, digest),) = frozen
    path = campaign / str(name)
    if _digest_of(path) != digest:
        return []  # changed or gone since: a note, said by `_files_against`
    declared = sorted(
        str(row.get(ite_trace.SAMPLE_ID, "")).strip() for row in ite_trace.load_manifest(path).rows
    )
    held = sorted(str(row.get("sample_id") or "").strip() for row in rows)
    if declared != held:
        return [
            f"the manifest {name} is the one the record was frozen from, and the record does "
            "not hold exactly its specimens: one was added to the record or taken out of it"
        ]
    return []


def _campaign_settings_problems(record: Mapping[str, Any]) -> list[str]:
    """What `next` always freezes with, and a record of this campaign must state.

    `next` asks for expected improvement, leaves the stopping tolerance at the
    prediction noise and its risk at the optimizer's default, takes the
    optimizer's best fitted value as the baseline, and grades reliability.
    """
    problems = []
    acquisition = record["acquisition"]
    claim = record["stopping_claim"]
    if acquisition.get("function") != _CAMPAIGN_ACQUISITION:
        problems.append(
            f"its acquisition is {acquisition.get('function')!r}; `next` uses "
            f"{_CAMPAIGN_ACQUISITION!r}"
        )
    if claim.get("delta") != DEFAULT_DELTA:
        problems.append(f"its stopping risk delta is not the {DEFAULT_DELTA} `next` uses")
    if claim.get("epsilon") != record["frozen_config"].get("noise_std"):
        problems.append("its stopping tolerance epsilon is not the prediction noise `next` uses")
    if record.get("prior_best") != claim.get("best_measured"):
        problems.append("its prior best is not the best fitted value `next` takes as the baseline")
    if record.get("reliability") is None:
        problems.append("it holds no reliability grade, and `next` always grades")
    return problems


def _read_again(record: Mapping[str, Any], campaign: Path) -> list[str]:
    """Read each frozen specimen again from its recording, as the manifest row it was frozen with.

    The values the producer gives now must be the values the record holds:
    the composition, the recording's SHA-256, the value and its uncertainty.
    Nothing is written.
    """
    stored = record["experiment"]["definition"]
    design = next(v for v in stored["variables"] if v["role"] == VariableRole.DESIGN.value)
    response = next(r for r in stored["responses"] if r["name"] == stored["objective"]["response"])
    problems = []
    for row in [*record["specimens"], *record["excluded"]]:
        label = row.get("sample_id") or row.get("record_id")
        detail = row.get("producer_detail") or {}
        frozen = (detail.get("manifest") or {}).get("row_with_defaults")
        if not isinstance(frozen, dict) or not (detail.get("source") or {}).get("sha256"):
            continue  # refused before its recording was read: nothing to read again
        manifest = ite_trace.Manifest(
            path=campaign / str(detail["manifest"]["file"]),
            sha256=str(detail["manifest"]["sha256"]),
            rows=(frozen,),
        )
        try:
            again = ite_trace._specimen_record(frozen, campaign, manifest).values
        except Exception as exc:
            problems.append(f"{label}: it could not be read again from its recording ({exc})")
            continue
        now = (
            again.get(design["column"]),
            again.get(ite_trace.TRACE_SHA256),
            again.get(response["column"]),
            again.get(response.get("uncertainty_column")),
        )
        if "inputs" in row:
            value = row["responses"].get(response["name"]) or {}
            then = (
                row["inputs"].get(design["name"]),
                row["metadata"].get(ite_trace.TRACE_SHA256),
                value.get("value"),
                value.get("uncertainty"),
            )
        else:
            values = row["values"]
            then = (
                values.get(design["column"]),
                values.get(ite_trace.TRACE_SHA256),
                values.get(response["column"]),
                values.get(response.get("uncertainty_column")),
            )
        if not all(_same_value(a, b) for a, b in zip(now, then, strict=True)):
            problems.append(
                f"{label}: its recording, read again as the row it was frozen with, does not "
                "give the values the record holds"
            )
    return problems


def _same_value(a: Any, b: Any) -> bool:
    """Equal, with numbers compared as numbers to the last bits."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=_READ_AGAIN_REL, abs_tol=_READ_AGAIN_ABS)
    return bool(a == b)


def _producer_detail_problems(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Whether every row's producer detail is this producer's, for this row, from one manifest."""
    problems = []
    manifests: set[tuple[Any, Any]] = set()
    protocols: list[Any] = []
    for row in rows:
        label = row.get("sample_id") or row.get("record_id")
        detail = row.get("producer_detail")
        if not isinstance(detail, dict):
            problems.append(f"{label}: the record holds no producer detail for it")
            continue
        if detail.get("contract") != ite_trace.CONTRACT:
            problems.append(f"{label}: its producer detail is not from {ite_trace.CONTRACT}")
        if row.get("sample_id") is not None and detail.get("sample_id") != row["sample_id"]:
            problems.append(f"{label}: its producer detail is for another specimen")
        values = row["metadata"] if "inputs" in row else row["values"]
        stated = values.get(ite_trace.TRACE_SHA256)
        read = (detail.get("source") or {}).get("sha256")
        if stated is not None and stated != read:
            problems.append(
                f"{label}: the recording SHA-256 it states is not the one its producer read"
            )
        manifest = detail.get("manifest")
        if isinstance(manifest, dict):
            manifests.add((manifest.get("file"), manifest.get("sha256")))
        if "protocol" in detail:
            protocols.append(detail["protocol"])
    if len(manifests) > 1:
        problems.append("the specimens were not all read from one manifest")
    if any(protocol != protocols[0] for protocol in protocols):
        problems.append("the specimens do not all state the same protocol")
    return problems


def _files_against(
    rows: Sequence[Mapping[str, Any]], campaign: Path
) -> tuple[list[str], list[str]]:
    """Compare the recordings and the manifest a record names with the files in `campaign`.

    A recording that is missing or has changed is a problem: the record can no
    longer be taken back to its data. A manifest that has changed is a note,
    because a campaign adds specimens to it after a freeze.
    """
    problems: list[str] = []
    notes: list[str] = []
    recordings: dict[str, str] = {}
    manifests: dict[str, str] = {}
    for row in rows:
        detail = row.get("producer_detail")
        if not isinstance(detail, dict):
            continue
        reverse = detail.get("reverse_diagnostic")
        for read in (detail, reverse if isinstance(reverse, dict) else {}):
            source = read.get("source")
            if isinstance(source, dict) and source.get("file") and source.get("sha256"):
                recordings[str(source["file"])] = str(source["sha256"])
        manifest = detail.get("manifest")
        if isinstance(manifest, dict) and manifest.get("file") and manifest.get("sha256"):
            manifests[str(manifest["file"])] = str(manifest["sha256"])
    for name, digest in sorted(recordings.items()):
        now = _digest_of(campaign / name)
        if now is None:
            problems.append(f"the recording {name} is not in {campaign}, so it cannot be compared")
        elif now != digest:
            problems.append(f"the recording {name} is not the one the record was frozen from")
    for name, digest in sorted(manifests.items()):
        now = _digest_of(campaign / name)
        if now is None:
            notes.append(f"the manifest {name} is not in {campaign}")
        elif now != digest:
            notes.append(
                f"the manifest {name} has changed since the freeze. A campaign adds specimens "
                "to it; `check` and `next` compare its specimens with this record's"
            )
    return problems, notes


def _digest_of(path: Path) -> str | None:
    """The SHA-256 of a file's bytes, or None when it is not a readable file."""
    try:
        return hash_bytes(path.read_bytes())
    except OSError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench command. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m latos.campaign_trace",
        description="The ionic-liquid thermal-cycle campaign: check, recommend and freeze, verify.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="read and report; fit nothing, write nothing")
    check.add_argument("manifest", type=Path, help="the specimen manifest (.toml)")

    following = commands.add_parser("next", help="recommend the next composition and freeze it")
    following.add_argument("manifest", type=Path, help="the specimen manifest (.toml)")
    following.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "accepted only when it names the campaign's own record folder, .latos/prereg/ "
            "beside the manifest; records are kept nowhere else"
        ),
    )
    following.add_argument(
        "--dry-run",
        action="store_true",
        help=("make every check except the source-tree one, and print a preview; freeze nothing"),
    )

    verify = commands.add_parser("verify", help="recompute what a frozen record claims")
    verify.add_argument("record", type=Path, help="the frozen record (.json)")
    verify.add_argument("--seal", default=None, help="the seal noted when it was frozen")

    args = parser.parse_args(argv)
    if args.command == "check":
        state = _load(args.manifest)
        if state is None:
            return EXIT_UNREADABLE
        revision = source_revision()
        print(report(state, revision))
        print()
        reasons = _refusals(state, revision, _record_directory(args.manifest))
        print("\n".join(_readiness_lines(reasons)))
        # The status says whether `next` would go ahead: anything that would
        # make it refuse makes this NOT READY, whatever kind of reason it is.
        return EXIT_REFUSED if reasons else EXIT_OK
    if args.command == "next":
        return _next(args.manifest, args.out, dry_run=args.dry_run)
    return _verify(args.record, args.seal)


if __name__ == "__main__":
    raise SystemExit(main())
