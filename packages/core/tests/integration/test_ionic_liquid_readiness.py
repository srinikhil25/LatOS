"""Readiness check on the real ionic-liquid recordings. Local machines only.

This runs only where the campaign's data folder exists, which is the
researcher's own machine: the folder is gitignored, and so is everything in it.
Everywhere else, CI included, it skips.

It takes the real manifest and recordings through the whole path up to the
optimizer's door, and stops there. **It never fits a model and never freezes
anything.** The first recommendation of a campaign is made once, deliberately,
and not as a side effect of running the tests.

No measured number appears in this file. What Latos derives is compared with
`audit_values.toml`, a file the researcher keeps beside the recordings, and the
search domain is read from the campaign's own definition.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import tomllib
from pathlib import Path
from typing import Any

import pytest

from latos.campaign_trace import DEFINITION, CampaignState, load_campaign, main, report
from latos.core.enums import Severity
from latos.experiments import (
    EXCLUDED_OUTSIDE_BOUNDS,
    Observation,
    UncertaintyScope,
    VariableRole,
    min_conditions,
)
from latos.experiments.producers import ite_trace
from latos.optimization import engine

_DATA = Path(__file__).resolve().parents[4] / "data" / "ionic-liquid-campaign"
_MANIFEST = _DATA / "specimens.toml"
_AUDIT = _DATA / "audit_values.toml"

# The audit values were supplied to four decimal places.
_TOLERANCE_MV_K = 5e-5
# One audit fraction was written to four places and differs from its masses in
# the fourth. The masses are the campaign's rule; this only bounds the gap.
_STATED_FRACTION_TOLERANCE = 5e-4

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_MANIFEST.is_file() and _AUDIT.is_file()),
        reason=f"the campaign's private data is not present at {_DATA}",
    ),
]

S = ite_trace.APPARENT_SEEBECK
X = ite_trace.ALIQUAT_MASS_FRACTION
(_DESIGN,) = DEFINITION.design_variables()
assert _DESIGN.bounds is not None
_LOW, _HIGH = _DESIGN.bounds


def _folder_snapshot() -> dict[str, str]:
    """Every file under the data folder, frozen records included, by content."""
    return {
        path.relative_to(_DATA).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_DATA.rglob("*"))
        if path.is_file()
    }


def _never(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the readiness check must not fit a model")


@pytest.fixture(scope="module")
def expected() -> dict[str, dict[str, Any]]:
    audit = tomllib.loads(_AUDIT.read_text(encoding="utf-8"))
    return {row["sample_id"]: row for row in audit["specimen"]}


@pytest.fixture(scope="module")
def state() -> CampaignState:
    """The campaign up to the optimizer's door, with the optimizer boarded up."""
    patch = pytest.MonkeyPatch()
    patch.setattr(engine, "optimize", _never)
    patch.setattr(engine, "optimize_nd", _never)
    before = _folder_snapshot()
    try:
        loaded = load_campaign(_MANIFEST)
        report(loaded)
    finally:
        patch.undo()
    assert _folder_snapshot() == before, "reading the campaign changed the data folder"
    return loaded


def _complete(state: CampaignState) -> list[Observation]:
    """Every complete specimen: those in the fit, and those kept outside the domain."""
    kept = [row.observation for row in state.observation_set.outside_bounds()]
    return [*state.observation_set.observations, *(o for o in kept if o is not None)]


# ─── Which specimens go where ───────────────────────────────────────
def test_every_specimen_is_complete_and_none_is_rejected(
    state: CampaignState, expected: dict[str, dict[str, Any]]
) -> None:
    assert state.observation_set.rejected() == ()
    assert sorted(o.sample_id for o in _complete(state)) == sorted(expected)
    assert len(state.table.rows) == len(expected) == 5


def test_the_fit_holds_exactly_the_specimens_inside_the_search_domain(
    state: CampaignState, expected: dict[str, dict[str, Any]]
) -> None:
    inside = sorted(o.sample_id for o in _complete(state) if _LOW <= o.inputs[X] <= _HIGH)
    assert sorted(o.sample_id for o in state.observation_set.observations) == inside
    assert len(inside) == 4

    # Four specimens, four conditions: none is within the precision of another.
    reduced = state.observation_set.reduced
    assert len(reduced) == 4
    assert all(condition.n_specimens == 1 for condition in reduced)
    assert _DESIGN.precision is not None
    positions = sorted(float(condition.x[0]) for condition in reduced)
    assert all(b - a > _DESIGN.precision for a, b in itertools.pairwise(positions))
    assert state.enough
    assert state.prepared.n_dims == 1
    assert len(reduced) >= min_conditions(1)


def test_the_specimen_outside_the_domain_is_preserved_whole_and_is_not_fitted(
    state: CampaignState, expected: dict[str, dict[str, Any]]
) -> None:
    (outside,) = state.observation_set.outside_bounds()
    assert outside.category == EXCLUDED_OUTSIDE_BOUNDS
    kept = outside.observation
    assert kept is not None
    assert kept.sample_id in expected
    assert not _LOW <= kept.inputs[X] <= _HIGH

    # The search domain is the whole reason. Nothing else about it is one.
    (reason,) = outside.reasons
    assert reason.startswith(f"{X} = ")
    assert "is outside its bounds" in reason
    # A declared quality note travels with it, as a warning, and excludes nothing.
    notes = [issue for issue in kept.issues if "declared QC warning" in issue.message]
    assert len(notes) == 1
    assert notes[0].severity is Severity.WARNING
    assert "drift" in notes[0].message
    assert not any(issue.severity is Severity.ERROR for issue in kept.issues)

    # Its signed value and uncertainty are kept, and are in no fitted array.
    audit = expected[kept.sample_id]
    value = kept.responses[S]
    assert value.value == pytest.approx(audit["mean_mv_k"], abs=_TOLERANCE_MV_K)
    assert value.uncertainty is not None
    assert value.uncertainty_scope is UncertaintyScope.SPECIMEN
    assert kept.inputs[X] not in state.prepared.x[:, 0].tolist()
    assert all(kept.sample_id not in c.sample_ids for c in state.observation_set.reduced)
    text = report(state)
    assert "Specimens preserved outside the search bounds: 1" in text
    assert "Not rejected, and not outliers." in text
    assert "NOT in the model" not in text


# ─── What each specimen says ────────────────────────────────────────
def test_each_composition_comes_from_its_masses(
    state: CampaignState, expected: dict[str, dict[str, Any]]
) -> None:
    for observation in _complete(state):
        audit = expected[observation.sample_id]
        fraction = observation.inputs[X]
        if "pure_fraction" in audit:
            assert fraction == audit["pure_fraction"]
            continue
        first, second = audit["masses_g"]
        assert fraction == pytest.approx(first / (first + second), abs=1e-12)
        assert abs(fraction - audit["stated_fraction"]) < _STATED_FRACTION_TOLERANCE


def test_each_signed_value_matches_the_audit(
    state: CampaignState, expected: dict[str, dict[str, Any]]
) -> None:
    for observation in _complete(state):
        audit = expected[observation.sample_id]
        detail = observation.producer_detail
        assert detail is not None
        cycles = detail["reduction"]["cycle_values_mv_k"]
        assert cycles == pytest.approx(audit["cycles_mv_k"], abs=_TOLERANCE_MV_K)
        value = observation.responses[S]
        assert value.value == pytest.approx(audit["mean_mv_k"], abs=_TOLERANCE_MV_K)
        # Signed: the audit's sign is the sign that travels.
        assert math.copysign(1.0, value.value) == math.copysign(1.0, audit["mean_mv_k"])

        # The uncertainty is the standard error of the accepted cycles and
        # nothing else, and it says what it is.
        k = len(cycles)
        mean = sum(cycles) / k
        spread = math.sqrt(sum((c - mean) ** 2 for c in cycles) / (k - 1))
        assert value.uncertainty == pytest.approx(spread / math.sqrt(k), rel=1e-9)
        assert value.uncertainty_scope is UncertaintyScope.SPECIMEN
        assert detail["reduction"]["degrees_of_freedom"] == k - 1
        assert "a lower bound" in value.provenance
        assert "not been shown independent or exchangeable" in value.provenance


def test_test_cycles_boundaries_and_reversed_runs_are_as_declared(
    state: CampaignState, expected: dict[str, dict[str, Any]]
) -> None:
    files = {o.metadata[ite_trace.TRACE_FILE] for o in _complete(state)}
    for observation in _complete(state):
        audit = expected[observation.sample_id]
        detail = observation.producer_detail
        assert detail is not None
        purposes = [cycle["purpose"] for cycle in detail["cycles"]]
        assert purposes == ["test_run"] * audit["test_cycles"] + ["measurement"] * 3
        # A test cycle is kept in the record, and kept out of the value.
        assert [cycle["included_in_specimen_value"] for cycle in detail["cycles"]] == [
            purpose == "measurement" for purpose in purposes
        ]
        assert observation.metadata[ite_trace.BOUNDARY_SOURCE] == audit["boundary_source"]
        assert observation.inputs[ite_trace.CYCLES_PER_SPECIMEN] == 3

        if "reverse_file" in audit:
            reverse = detail["reverse_diagnostic"]
            assert reverse["source"]["file"] == audit["reverse_file"]
            assert reverse["recorded_sign_reverses"] is True
            assert audit["reverse_file"] not in files  # a diagnostic, never an observation
        else:
            assert "reverse_diagnostic" not in detail


def test_every_specimen_declares_the_campaigns_drive_set_point(state: CampaignState) -> None:
    drive = next(v for v in DEFINITION.variables if v.name == ite_trace.DRIVE_VOLTAGE)
    assert drive.role is VariableRole.CONTROL
    for record in state.table.rows:
        assert record.values[ite_trace.DRIVE_VOLTAGE] == drive.expected_value
        # A declaration read from the manifest, and it says so: nothing measured it.
        assert "not a measured voltage" in record.provenance[ite_trace.DRIVE_VOLTAGE]


# ─── The optimizer's door ───────────────────────────────────────────
def test_the_prepared_arrays_are_the_signed_values_of_the_positive_branch(
    state: CampaignState, expected: dict[str, dict[str, Any]]
) -> None:
    prepared = state.prepared
    audit = sorted(
        (o.inputs[X], expected[o.sample_id]["mean_mv_k"])
        for o in state.observation_set.observations
    )
    assert prepared.x[:, 0].tolist() == pytest.approx([x for x, _ in audit], abs=1e-12)
    assert prepared.y_raw.tolist() == pytest.approx([v for _, v in audit], abs=_TOLERANCE_MV_K)
    # Signed, and handed over unchanged: no magnitude, no other projection.
    assert prepared.y.tolist() == prepared.y_raw.tolist()
    assert prepared.objective_aggregation == "identity"
    assert prepared.target_name == S
    assert prepared.direction == "maximize"
    assert all(value > 0 for value in prepared.y.tolist())  # the branch the search covers
    assert prepared.messages == ()

    # The search domain is the definition's, exactly, and every point is in it.
    assert prepared.bounds == ((_LOW, _HIGH),)
    assert all(_LOW <= x <= _HIGH for x in prepared.x[:, 0].tolist())
    assert max(prepared.x[:, 0].tolist()) == _HIGH

    # Each uncertainty is one specimen's own, labelled as the lower bound it is.
    assert prepared.sigma_sources == ("lower_bound_specimen",) * 4
    assert prepared.sigma is not None
    by_id = {o.sample_id: o for o in state.observation_set.observations}
    for index, condition in enumerate(state.observation_set.reduced):
        (sample_id,) = condition.sample_ids
        assert prepared.sigma[index] == by_id[sample_id].responses[S].uncertainty
    assert prepared.prediction_noise_source == "lower_bound"
    assert state.observation_set.pooled_sd is None  # nothing is replicated yet


def test_the_check_command_fits_nothing_and_writes_nothing(
    state: CampaignState, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert state.definition is DEFINITION
    monkeypatch.setattr(engine, "optimize", _never)
    monkeypatch.setattr(engine, "optimize_nd", _never)
    before = _folder_snapshot()

    status = main(["check", str(_MANIFEST)])

    # Recordings, manifest and any record frozen on purpose: all as they were.
    assert _folder_snapshot() == before
    printed = capsys.readouterr().out
    # The data are ready. `check` is NOT READY, with status 1, exactly when a
    # reason that is not about the data applies: a source tree that is not a
    # commit, or a record already frozen and waiting for its outcome.
    reasons = [line[4:] for line in printed.splitlines() if line.startswith("  - ")]
    assert status == (1 if reasons else 0)
    assert ("NOT READY" in printed) == bool(reasons)
    for reason in reasons:
        assert reason.startswith("what is frozen must be reproducible from a precise source") or (
            "prereg_" in reason
        ), reason
    for observation in _complete(state):
        assert observation.sample_id in printed
    assert "Specimens in the fit: 4" in printed
    assert "Specimens preserved outside the search bounds: 1" in printed
    assert "projection: none: the signed values are handed over unchanged" in printed
    assert "Source revision: " in printed
    assert "Recommendation" not in printed
