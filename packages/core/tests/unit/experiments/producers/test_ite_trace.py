"""The thermal-cycle trace producer: one row per specimen, signed, with its record."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latos.analysis.thermovoltage.cycles import AUTOMATIC_SETTINGS
from latos.core.enums import Severity
from latos.experiments._canonical import canonical_json
from latos.experiments.producers import ite_trace
from latos.experiments.producers.ite_trace import (
    COLUMNS,
    CONTRACT,
    PROTOCOL,
    SIGNATURE_LIMIT_S,
    ManifestError,
    load_manifest,
    produce_table,
)
from latos.experiments.records import Record, RecordTable, producer_detail_sha256
from tests.unit._synthetic_trace import (
    BENCH_DEFAULTS,
    CHANNELS,
    STRAIGHT,
    SWAPPED,
    boundaries,
    make_recording,
    series,
    specimen,
    write_manifest,
    write_workbook,
)

S = "apparent_seebeck_cycle_mv_k"
SEM = "apparent_seebeck_cycle_sem_mv_k"
X = "aliquat_mass_fraction"
FOUR = boundaries((180.0,) * 4, (180.0,) * 4)


def _table(folder: Path, *specimens: dict[str, Any], **manifest: Any) -> RecordTable:
    manifest.setdefault("defaults", BENCH_DEFAULTS)
    return produce_table(write_manifest(folder, specimens, **manifest))


def _row(folder: Path, **fields: Any) -> Record:
    """The one row of a manifest naming `run.xlsx`, with `fields` on its specimen."""
    (row,) = _table(folder, specimen("s1", "run.xlsx", **fields)).rows
    return row


def _messages(row: Record, severity: Severity) -> list[str]:
    return [issue.message for issue in row.issues if issue.severity is severity]


def _warnings(row: Record) -> list[str]:
    """Warnings, leaving out the one every noise-free made-up recording earns."""
    return [m for m in _messages(row, Severity.WARNING) if "cycles are identical" not in m]


def _refusal(row: Record) -> str:
    """The single reason a refused row gives, having checked that it is refused."""
    assert row.values[S] is None
    assert row.values[SEM] is None
    (message,) = _messages(row, Severity.ERROR)
    return message


# ─── One specimen, one row ──────────────────────────────────────────
def test_three_cycles_of_one_mounting_are_one_row(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=[2.0, 2.1, 2.3])
    table = _table(tmp_path, specimen("s1", "run.xlsx"))

    assert table.producer == CONTRACT == "ite-trace/1"
    assert table.columns == COLUMNS
    (row,) = table.rows
    values = row.values
    sd = float(np.std([2.0, 2.1, 2.3], ddof=1))
    assert values["sample_id"] == row.record_id == "s1"
    assert values[S] == pytest.approx(6.4 / 3)
    assert values[SEM] == pytest.approx(sd / math.sqrt(3))
    assert values["cycles_per_specimen"] == 3
    assert [
        values[f"{phase}_duration_{end}_s"] for phase in ("on", "off") for end in ("min", "max")
    ] == [
        180.0,
        180.0,
        180.0,
        180.0,
    ]
    assert values["boundary_source"] == "event_marks"
    assert values["polarity_convention"] == STRAIGHT
    assert values["trace_file"] == "run.xlsx"
    assert (
        values["trace_sha256"] == hashlib.sha256((tmp_path / "run.xlsx").read_bytes()).hexdigest()
    )
    assert values["peak_delta_t_k"] == pytest.approx(15.0, abs=0.01)
    assert values["mean_temperature_c"] == pytest.approx(25.0)
    assert row.issues == ()

    reduction = row.producer_detail["reduction"]
    assert reduction["cycles_used"] == [1, 2, 3]
    assert reduction["cycle_values_mv_k"] == pytest.approx([2.0, 2.1, 2.3])
    assert (reduction["k"], reduction["degrees_of_freedom"]) == (3, 2)


def test_every_specimen_has_its_row_and_reversed_runs_add_none(tmp_path: Path) -> None:
    for name in ("a.xlsx", "b.xlsx"):
        make_recording(tmp_path / name)
    for name in ("a-r.xlsx", "b-r.xlsx"):
        make_recording(tmp_path / name, lead_sign=-1)
    reverse = {"polarity_convention": SWAPPED}
    table = _table(
        tmp_path,
        specimen("a", "a.xlsx", reverse={"trace_file": "a-r.xlsx", **reverse}),
        specimen("b", "b.xlsx", reverse={"trace_file": "b-r.xlsx", **reverse}),
    )
    # Four recordings, twelve cycles, two specimens: two rows.
    assert [row.record_id for row in table.rows] == ["a", "b"]
    assert [row.values["trace_file"] for row in table.rows] == ["a.xlsx", "b.xlsx"]


# ─── Boundaries ─────────────────────────────────────────────────────
def test_marks_one_to_seven_give_three_measurement_cycles(tmp_path: Path) -> None:
    switches = make_recording(tmp_path / "run.xlsx")
    detail = _row(tmp_path).producer_detail

    assert detail["boundaries"]["source"] == "event_marks"
    assert detail["boundaries"]["times_s"] == switches
    assert [m["number"] for m in detail["source"]["event_marks"]] == [1, 2, 3, 4, 5, 6, 7]
    assert [
        (c["index"], c["purpose"], c["included_in_specimen_value"]) for c in detail["cycles"]
    ] == [
        (1, "measurement", True),
        (2, "measurement", True),
        (3, "measurement", True),
    ]
    assert [(c["on_s"], c["off_s"], c["end_s"]) for c in detail["cycles"]] == [
        (10.0, 190.0, 370.0),
        (370.0, 550.0, 730.0),
        (730.0, 910.0, 1090.0),
    ]


def test_a_declared_test_run_is_excluded_by_protocol_and_kept(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", switches=FOUR, seebeck_mv_k=[9.0, 2.0, 2.1, 2.3])
    row = _row(tmp_path, test_cycles=1)

    # Marks 3 to 5, 5 to 7 and 7 to 9: the first cycle is not in the value.
    assert row.values[S] == pytest.approx(6.4 / 3)
    assert row.values["cycles_per_specimen"] == 3
    assert row.issues == ()  # excluded by declaration, so nothing to warn about
    assert "declared test runs and are excluded by protocol" in row.provenance[S]

    detail = row.producer_detail
    assert detail["reduction"]["cycles_used"] == [2, 3, 4]
    first, *rest = detail["cycles"]
    assert first["purpose"] == "test_run"
    assert first["included_in_specimen_value"] is False
    assert first["exclusion"].endswith("excluded by protocol, not as an outlier")
    assert first["slope_mv_k"] == pytest.approx(9.0)  # kept, not deleted
    assert (first["on_s"], first["end_s"]) == (FOUR[0], FOUR[2])
    assert all(c["purpose"] == "measurement" and c["included_in_specimen_value"] for c in rest)
    assert [(c["on_s"], c["end_s"]) for c in rest] == [
        (FOUR[2], FOUR[4]),
        (FOUR[4], FOUR[6]),
        (FOUR[6], FOUR[8]),
    ]


def test_without_that_declaration_all_four_cycles_are_reported(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", switches=FOUR, seebeck_mv_k=[9.0, 2.0, 2.1, 2.3])
    row = _row(tmp_path)
    # Reported as it is; whether four cycles are acceptable is the definition's call.
    assert row.values[S] == pytest.approx(15.4 / 4)
    assert row.values["cycles_per_specimen"] == 4


def test_test_cycles_cannot_use_up_the_whole_recording(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    assert "3 test cycle(s) are declared" in _refusal(_row(tmp_path, test_cycles=3))


def test_declared_boundaries_are_used_for_a_recording_without_marks(tmp_path: Path) -> None:
    switches = make_recording(tmp_path / "run.xlsx", marks=None, seebeck_mv_k=[2.0, 2.1, 2.3])
    declared = {
        "on": switches[0:6:2],
        "off": switches[1:6:2],
        "end": switches[6],
        "note": "from the bench notes",
    }
    row = _row(tmp_path, declared_boundaries_s=declared)

    assert row.values[S] == pytest.approx(6.4 / 3)
    assert row.values["boundary_source"] == "declared"
    assert row.issues == ()  # in particular, no automatic-segmentation warning
    detail = row.producer_detail["boundaries"]
    assert detail["times_s"] == switches
    assert detail["declared_note"] == "from the bench notes"
    assert "automatic" not in detail
    assert detail["thermal_signature_check"]["status"] == "agrees"


def test_a_declaration_replaces_marks_visibly(tmp_path: Path) -> None:
    switches = boundaries()
    marks = [(number, at + 3.0) for number, at in enumerate(switches, start=1)]
    make_recording(tmp_path / "run.xlsx", marks=marks, tail_s=20.0)
    declared = {"on": switches[0:6:2], "off": switches[1:6:2], "end": switches[6]}
    row = _row(tmp_path, declared_boundaries_s=declared)

    assert row.values["boundary_source"] == "declared"
    detail = row.producer_detail["boundaries"]
    assert detail["times_s"] == switches
    assert detail["replaced_event_marks_s"] == [at + 3.0 for at in switches]  # both sets kept
    (warning,) = _warnings(row)
    assert "declared boundaries are used in place of the recording's event marks" in warning
    assert "Declared minus marked, in s: -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0." in warning


def test_with_neither_marks_nor_a_declaration_the_fallback_warns(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", marks=None)
    row = _row(tmp_path)

    assert row.values["boundary_source"] == "automatic"
    assert row.values[S] == pytest.approx(2.0)  # one exact line, whatever the windows
    (warning,) = _warnings(row)
    assert "estimated from the temperature trace" in warning
    assert "This is a fallback" in warning
    automatic = row.producer_detail["boundaries"]["automatic"]
    assert automatic["settings"] == AUTOMATIC_SETTINGS
    assert automatic["end_is_recording_end"] is True


@pytest.mark.parametrize(
    ("numbers", "message"),
    [
        ([1, 2, 3, 4, 5, 6], "6 event mark(s) cannot describe whole cycles"),
        ([1, 2, 3, 4, 5, 6, 7, 8], "8 event mark(s) cannot describe whole cycles"),
        ([1, 2, 4, 3, 5, 6, 7], "a mark is missing, repeated or out of sequence"),
        ([1, 2, 3, 5, 6, 7, 8], "numbered 1, 2, 3, 5, 6, 7, 8"),
    ],
)
def test_a_wrong_mark_count_or_order_refuses_the_specimen(
    tmp_path: Path, numbers: list[int], message: str
) -> None:
    switches = boundaries((180.0,) * 4, (180.0,) * 4)
    marks = list(zip(numbers, switches, strict=False))
    make_recording(tmp_path / "run.xlsx", switches=switches, marks=marks)
    row = _row(tmp_path)

    reason = _refusal(row)
    assert message in reason
    assert f"Marks found: mark {numbers[0]} at 10 s" in reason  # it shows what it found
    # Still one row, still identified, and nothing measured is claimed.
    assert row.values["sample_id"] == "s1"
    assert row.values[X] == 1.0
    assert row.values["cycles_per_specimen"] is None
    assert row.producer_detail["reduction"] is None


def test_marks_far_from_the_thermal_switches_are_refused(tmp_path: Path) -> None:
    marks = [(number, at + 12.0) for number, at in enumerate(boundaries(), start=1)]
    make_recording(tmp_path / "run.xlsx", marks=marks, tail_s=30.0)
    row = _row(tmp_path)
    reason = _refusal(row)
    assert f"over the {SIGNATURE_LIMIT_S:g} s limit" in reason
    assert "declare the boundaries in the manifest" in reason
    assert row.producer_detail["boundaries"]["thermal_signature_check"]["status"] == "disagrees"


def test_a_declaration_far_from_the_thermal_switches_is_used_with_a_warning(tmp_path: Path) -> None:
    switches = [at + 12.0 for at in boundaries()]
    make_recording(tmp_path / "run.xlsx", switches=boundaries(), marks=None, tail_s=30.0)
    declared = {"on": switches[0:6:2], "off": switches[1:6:2], "end": switches[6]}
    row = _row(tmp_path, declared_boundaries_s=declared)

    assert row.values[S] == pytest.approx(2.0)
    (warning,) = _warnings(row)
    assert "The declaration is used as written." in warning
    assert row.producer_detail["boundaries"]["thermal_signature_check"]["status"] == "disagrees"


def test_the_shortest_and_longest_phase_are_reported_not_judged(tmp_path: Path) -> None:
    switches = boundaries((180.0, 165.0, 180.0), (180.0, 200.0, 180.0))
    make_recording(tmp_path / "run.xlsx", switches=switches)
    row = _row(tmp_path)

    assert row.values[S] == pytest.approx(2.0)  # the producer does not exclude
    assert (row.values["on_duration_min_s"], row.values["on_duration_max_s"]) == (165.0, 180.0)
    assert (row.values["off_duration_min_s"], row.values["off_duration_max_s"]) == (180.0, 200.0)
    recorded = row.producer_detail["controls_recorded"]
    assert recorded["on_durations_s"] == [180.0, 165.0, 180.0]
    assert recorded["off_durations_s"] == [180.0, 200.0, 180.0]
    assert "CONTROL variables" in recorded["judged_by"]


def test_a_test_runs_phases_are_not_among_the_reported_durations(tmp_path: Path) -> None:
    switches = boundaries((180.0,) * 4, (250.0, 180.0, 180.0, 180.0))
    make_recording(tmp_path / "run.xlsx", switches=switches)
    row = _row(tmp_path, test_cycles=1)
    assert row.values["off_duration_max_s"] == 180.0
    assert row.producer_detail["cycles"][0]["off_duration_s"] == 250.0  # kept in the record


def test_a_hole_in_the_samples_is_refused(tmp_path: Path) -> None:
    switches = boundaries()
    time, cold, hot, voltage = series(switches)
    keep = (time < 400.0) | (time >= 460.0)
    marks = [(n, at) for n, at in enumerate(switches, start=1)]
    # Marks are placed by row, so write them where the kept samples put them.
    rows = {n: int(np.argmin(np.abs(time[keep] - at))) for n, at in marks}
    write_workbook(
        tmp_path / "run.xlsx",
        time[keep],
        cold[keep],
        hot[keep],
        voltage[keep],
        marks=[(n, float(rows[n])) for n, _ in marks],
    )
    assert "the recording has a gap of 61 s after 399.0 s" in _refusal(_row(tmp_path))


# ─── Sign ───────────────────────────────────────────────────────────
def test_a_negative_response_stays_negative(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=[-3.0, -2.5, -2.0])
    row = _row(tmp_path)
    assert row.values[S] == pytest.approx(-2.5)  # no magnitude is taken here
    assert row.producer_detail["reduction"]["cycle_values_mv_k"] == pytest.approx(
        [-3.0, -2.5, -2.0]
    )


def test_cycles_of_opposite_sign_reduce_to_their_signed_mean(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=[1.0, -1.0, 0.3])
    assert _row(tmp_path).values[S] == pytest.approx(0.1)  # not 0.77


def test_the_declared_wiring_sets_the_sign(tmp_path: Path) -> None:
    make_recording(tmp_path / "a.xlsx", seebeck_mv_k=2.0)
    make_recording(tmp_path / "b.xlsx", seebeck_mv_k=2.0)
    make_recording(tmp_path / "c.xlsx", seebeck_mv_k=2.0, lead_sign=-1)
    straight, misdeclared, swapped = _table(
        tmp_path,
        specimen("a", "a.xlsx"),
        specimen("b", "b.xlsx", polarity_convention=SWAPPED),
        specimen("c", "c.xlsx", polarity_convention=SWAPPED),
    ).rows
    assert straight.values[S] == pytest.approx(2.0)
    # The same recorded voltage read under the other wiring is the other sign.
    assert misdeclared.values[S] == pytest.approx(-2.0)
    # Leads really swapped, and declared so: the response is what it was.
    assert swapped.values[S] == pytest.approx(2.0)
    assert swapped.producer_detail["polarity"] == {
        "recorded": SWAPPED,
        "convention": SWAPPED,
        "sign": -1,
    }


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        (None, "polarity_convention is not recorded, so the sign of the response is not known"),
        ("", "polarity_convention is not recorded"),
        ("as usual", "does not say which electrode the V+ lead sat on"),
        ("hot and cold", "names both the hot and the cold side"),
    ],
)
def test_wiring_that_is_missing_or_unclear_is_refused(
    tmp_path: Path, declared: str | None, message: str
) -> None:
    make_recording(tmp_path / "run.xlsx")
    assert message in _refusal(_row(tmp_path, polarity_convention=declared))


# ─── Reversed-lead runs ─────────────────────────────────────────────
def test_a_reversed_run_is_a_diagnostic_inside_the_specimen(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=2.0)
    make_recording(tmp_path / "reverse.xlsx", seebeck_mv_k=2.2, lead_sign=-1)
    row = _row(tmp_path, reverse={"trace_file": "reverse.xlsx", "polarity_convention": SWAPPED})

    assert row.values[S] == pytest.approx(2.0)  # the specimen's value is untouched
    assert row.values["cycles_per_specimen"] == 3
    assert row.values["trace_file"] == "run.xlsx"
    assert _warnings(row) == []
    assert _messages(row, Severity.ERROR) == []
    diagnostic = row.producer_detail["reverse_diagnostic"]
    assert diagnostic["role"].endswith("Not a specimen and not an observation.")
    assert diagnostic["source"]["file"] == "reverse.xlsx"
    assert diagnostic["polarity"]["sign"] == -1
    assert diagnostic["mean_recorded_slope_mv_k"] == pytest.approx(-2.2)  # the raw sign turned
    assert diagnostic["forward_mean_recorded_slope_mv_k"] == pytest.approx(2.0)
    assert diagnostic["recorded_sign_reverses"] is True
    assert diagnostic["mean_oriented_slope_mv_k"] == pytest.approx(2.2)
    assert diagnostic["oriented_minus_forward_mv_k"] == pytest.approx(0.2)
    assert len(diagnostic["cycles"]) == 3


def test_a_reversed_run_that_does_not_turn_the_sign_warns(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=2.0)
    make_recording(tmp_path / "reverse.xlsx", seebeck_mv_k=2.2)  # leads not actually swapped
    row = _row(tmp_path, reverse={"trace_file": "reverse.xlsx", "polarity_convention": SWAPPED})

    assert row.values[S] == pytest.approx(2.0)
    assert row.producer_detail["reverse_diagnostic"]["recorded_sign_reverses"] is False
    (warning,) = _warnings(row)
    assert "swapping the leads did not reverse the recorded sign" in warning


def test_a_reversed_run_that_cannot_be_read_costs_only_the_diagnostic(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    row = _row(tmp_path, reverse={"trace_file": "absent.xlsx", "polarity_convention": SWAPPED})
    assert row.values[S] == pytest.approx(2.0)
    assert _messages(row, Severity.ERROR) == []
    (warning,) = _warnings(row)
    assert "the reversed-lead run could not be analysed" in warning
    assert "absent.xlsx" in row.producer_detail["reverse_diagnostic"]["errors"][0]


def test_a_reversed_run_needs_its_own_wiring_declaration(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    make_recording(tmp_path / "reverse.xlsx", lead_sign=-1)
    row = _row(tmp_path, reverse={"trace_file": "reverse.xlsx"})
    (warning,) = _warnings(row)
    assert "polarity_convention is not recorded" in warning  # not inherited from the specimen


# ─── Composition ────────────────────────────────────────────────────
def test_the_fraction_comes_from_the_weighed_masses(tmp_path: Path) -> None:
    make_recording(tmp_path / "IL1_25-IL2_75.xlsx")
    masses = {"Aliquat 336TG": 0.08, "[HMIM][TFSI]": 0.02}
    (row,) = _table(
        tmp_path,
        specimen("s1", "IL1_25-IL2_75.xlsx", masses_g=masses, nominal_label="IL1 25 / IL2 75"),
    ).rows
    # Neither the file name nor the label says 0.8; the masses do.
    assert row.values[X] == pytest.approx(0.08 / 0.10)
    assert row.values["nominal_label"] == "IL1 25 / IL2 75"
    assert row.provenance[X] == (
        "from the weighed masses in the manifest: 0.08 g Aliquat 336TG, 0.02 g [HMIM][TFSI]"
    )
    assert "never used in a calculation" in row.provenance["nominal_label"]
    assert row.producer_detail["composition"] == {
        "variable": X,
        "masses_g": masses,
        "pure_component": None,
        "basis": "weighed masses",
        "value": pytest.approx(0.8),
    }


@pytest.mark.parametrize(("component", "fraction"), [("[HMIM][TFSI]", 0.0), ("Aliquat 336TG", 1.0)])
def test_a_pure_component_is_an_endpoint_by_identity(
    tmp_path: Path, component: str, fraction: float
) -> None:
    make_recording(tmp_path / "run.xlsx")
    row = _row(tmp_path, pure_component=component)
    assert row.values[X] == fraction
    assert type(row.values[X]) is float
    assert row.provenance[X] == f"pure {component}, by identity"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"pure_component": None}, "no composition is declared"),
        (
            {
                "pure_component": "Aliquat 336TG",
                "masses_g": {"Aliquat 336TG": 1.0, "[HMIM][TFSI]": 1.0},
            },
            "both masses_g and pure_component are declared",
        ),
        ({"pure_component": "Aliquat 336"}, "is not one of the two components"),
        ({"masses_g": {"Aliquat 336TG": 0.1}}, "masses_g must give exactly"),
        (
            {"masses_g": {"Aliquat": 0.1, "[HMIM][TFSI]": 0.1}},
            "matched by exact name; found: 'Aliquat'",
        ),
        (
            {"masses_g": {"Aliquat 336TG": 0.1, "[HMIM][TFSI]": 0.1, "water": 0.01}},
            "masses_g must give exactly",
        ),
        ({"masses_g": {"Aliquat 336TG": -0.1, "[HMIM][TFSI]": 0.2}}, "must be zero or more"),
        ({"masses_g": {"Aliquat 336TG": 0.0, "[HMIM][TFSI]": 0.0}}, "not both zero"),
    ],
)
def test_a_composition_that_cannot_be_trusted_gives_no_fraction(
    tmp_path: Path, fields: dict[str, Any], message: str
) -> None:
    make_recording(tmp_path / "IL1_50-IL2_50.xlsx")
    (row,) = _table(tmp_path, specimen("s1", "IL1_50-IL2_50.xlsx", **fields)).rows
    assert row.values[X] is None  # in particular, not 0.5 from the file name
    (error,) = [i for i in row.issues if i.severity is Severity.ERROR]
    assert error.field == X
    assert message in error.message
    assert row.values[S] == pytest.approx(2.0)  # what was measured is still reported


# ─── Channels and units ─────────────────────────────────────────────
def test_channels_are_taken_by_name_and_units_from_the_header(tmp_path: Path) -> None:
    time, cold, hot, voltage = series(boundaries(), seebeck_mv_k=2.0)
    marks = list(enumerate(boundaries(), start=1))
    names = ("T-plate-b", "T-plate-a", "cell")
    write_workbook(tmp_path / "volts.xlsx", time, cold, hot, voltage, marks=marks, names=names)
    write_workbook(
        tmp_path / "millivolts.xlsx",
        time,
        cold,
        hot,
        voltage,
        marks=marks,
        names=names,
        voltage_unit="mV",
    )
    channels = {"cold": "T-plate-b", "hot": "T-plate-a", "voltage": "cell"}
    volts, millivolts = _table(
        tmp_path,
        specimen("v", "volts.xlsx", channels=channels),
        specimen("mv", "millivolts.xlsx", channels=channels),
    ).rows
    assert volts.values[S] == pytest.approx(2.0)
    assert millivolts.values[S] == pytest.approx(2.0)
    assert volts.producer_detail["channels"]["voltage"] == {
        "name": "cell",
        "unit": "V",
        "factor_to_mv": 1000.0,
    }
    assert millivolts.producer_detail["channels"]["voltage"]["factor_to_mv"] == 1.0


def test_no_channel_map_is_a_refusal_not_a_guess(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    (row,) = _table(tmp_path, specimen("s1", "run.xlsx"), defaults={"drops_applied": 3}).rows
    reason = _refusal(row)
    assert "no channel map is declared" in reason
    assert "Recorded channels: CH1-A-1, CH1-A-3, CH1-A-10" in reason


@pytest.mark.parametrize(
    ("channels", "options", "message"),
    [
        ({**CHANNELS, "hot": "CH9"}, {}, "no channel named 'CH9' in run.xlsx; recorded: CH1-A-1"),
        (CHANNELS, {"voltage_unit": "A"}, "is recorded in 'A', which is not a voltage unit"),
        (CHANNELS, {"temperature_unit": "°F"}, "is recorded in '°F', not in degrees Celsius"),
        (
            {"cold": "CH1-A-3", "hot": "CH1-A-1", "voltage": "CH1-A-10"},
            {},
            "does not rise while the drive is on",
        ),
    ],
)
def test_a_channel_or_unit_problem_is_refused(
    tmp_path: Path, channels: dict[str, str], options: dict[str, Any], message: str
) -> None:
    make_recording(tmp_path / "run.xlsx", **options)
    assert message in _refusal(_row(tmp_path, channels=channels))


def test_a_missing_or_unreadable_recording_is_refused_with_its_row_kept(tmp_path: Path) -> None:
    (tmp_path / "broken.xlsx").write_text("not a workbook", encoding="utf-8")
    absent, broken = _table(
        tmp_path, specimen("a", "absent.xlsx"), specimen("b", "broken.xlsx")
    ).rows
    assert "'absent.xlsx' was not found beside the manifest" in _refusal(absent)
    assert "not a readable Excel workbook" in _refusal(broken)
    assert absent.values["trace_sha256"] is None


def test_a_recording_the_system_will_not_open_is_refused_and_the_others_are_still_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_recording(tmp_path / "good.xlsx")
    make_recording(tmp_path / "locked.xlsx")
    (tmp_path / "folder.xlsx").mkdir()
    real = ite_trace.read_hioki_xlsx

    def held_open(path: Path, **options: Any) -> Any:
        if path.name == "locked.xlsx":
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, **options)

    monkeypatch.setattr(ite_trace, "read_hioki_xlsx", held_open)
    good, locked, folder = _table(
        tmp_path,
        specimen("a", "good.xlsx"),
        specimen("b", "locked.xlsx"),
        specimen("c", "folder.xlsx"),
    ).rows
    # One row per specimen, whatever happened to its file.
    assert good.values[S] == pytest.approx(2.0)
    assert _refusal(locked) == "the trace file 'locked.xlsx' could not be read: Permission denied"
    assert "the trace file 'folder.xlsx' could not be read: " in _refusal(folder)
    assert locked.values["trace_sha256"] is None


def test_text_typed_into_the_event_column_refuses_the_recording_whatever_else_is_declared(
    tmp_path: Path,
) -> None:
    # The logger writes mark numbers in that column and nothing else. A label
    # typed there means the file is no longer what the logger exported, and the
    # reader says which cell. Declaring the switching times does not get past it.
    switches = boundaries()
    label = "Cycle 1 \N{EN DASH} Heater ON (ΔT up)"
    make_recording(tmp_path / "run.xlsx", marks=[(label, switches[0])])
    declared = {"on": switches[0:6:2], "off": switches[1:6:2], "end": switches[6]}
    refusal = f"run.xlsx: row 13 of the event column holds {label!r}, not an event number"
    assert _refusal(_row(tmp_path, declared_boundaries_s=declared)) == refusal
    assert _refusal(_row(tmp_path)) == refusal


def test_a_sheet_can_be_named(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", sheet="second run")
    assert _row(tmp_path, sheet="second run").producer_detail["source"]["sheet"] == "second run"
    assert "no sheet named 'V1_H1'" in _refusal(_row(tmp_path, sheet="V1_H1"))


# ─── Uncertainty ────────────────────────────────────────────────────
def test_the_uncertainty_is_the_cycle_standard_error_labelled_as_a_lower_bound(
    tmp_path: Path,
) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=[2.0, 2.1, 2.3])
    row = _row(tmp_path)
    reduction = row.producer_detail["reduction"]
    assert row.values[SEM] == reduction["sem_mv_k"] == reduction["uncertainty_reported_mv_k"]
    assert reduction["sem_mv_k"] == pytest.approx(reduction["sd_mv_k"] / math.sqrt(3))

    basis = row.provenance[SEM]
    assert "divided by sqrt(3), 2 degrees of freedom" in basis
    assert "within-specimen, within-session Type A component and a lower bound" in basis
    assert "not been shown independent or exchangeable" in basis
    stated = PROTOCOL["uncertainty"]
    assert stated["scope"] == "specimen"
    assert stated["is_lower_bound"] is True
    assert "a new mounting of the same mixture" in stated["does_not_cover"]


def test_the_fit_standard_error_of_a_cycle_is_never_the_uncertainty(tmp_path: Path) -> None:
    # Three near-identical loops: every cycle's fit has scatter, the cycles
    # almost none.
    make_recording(tmp_path / "loops.xlsx", lag_mv=1.5)
    # Three exact lines of different slope: no fit scatter, real cycle scatter.
    make_recording(tmp_path / "lines.xlsx", seebeck_mv_k=[2.0, 2.1, 2.3])
    loops, lines = _table(
        tmp_path, specimen("loops", "loops.xlsx"), specimen("lines", "lines.xlsx")
    ).rows

    # Large fit errors, tiny uncertainty: the first is not in the second.
    loop_errors = [c["slope_ols_stderr_mv_k"] for c in loops.producer_detail["cycles"]]
    loop_values = loops.producer_detail["reduction"]["cycle_values_mv_k"]
    assert all(error > 0.001 for error in loop_errors)
    assert loops.values[SEM] == pytest.approx(float(np.std(loop_values, ddof=1)) / math.sqrt(3))
    assert loops.values[SEM] < min(loop_errors) / 100

    # No fit error at all, real uncertainty: nor is the second in the first.
    line_errors = [c["slope_ols_stderr_mv_k"] for c in lines.producer_detail["cycles"]]
    assert all(error < 1e-9 for error in line_errors)
    assert lines.values[SEM] == pytest.approx(float(np.std([2.0, 2.1, 2.3], ddof=1)) / math.sqrt(3))
    assert "slope_ols_stderr_mv_k" not in COLUMNS
    assert PROTOCOL["uncertainty"]["never_used"].startswith("the least-squares standard error")


def test_identical_cycles_report_no_uncertainty(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=2.0)  # one number, three times
    row = _row(tmp_path)
    assert row.values[S] == pytest.approx(2.0)
    assert row.values[SEM] is None
    (warning,) = _messages(row, Severity.WARNING)
    assert "the accepted cycles are identical, so no uncertainty is reported" in warning


def test_one_accepted_cycle_reports_no_uncertainty(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=[9.0, 8.0, 2.0])
    row = _row(tmp_path, test_cycles=2)
    assert row.values[S] == pytest.approx(2.0)
    assert row.values[SEM] is None
    assert row.values["cycles_per_specimen"] == 1
    (warning,) = _messages(row, Severity.WARNING)
    assert "one accepted cycle says nothing about repeatability" in warning
    assert SEM not in row.provenance


# ─── The record ─────────────────────────────────────────────────────
def test_the_detail_is_json_safe_portable_and_reproducible(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=[2.0, 2.1, 2.3])
    make_recording(tmp_path / "reverse.xlsx", lead_sign=-1)
    fields = {"reverse": {"trace_file": "reverse.xlsx", "polarity_convention": SWAPPED}}
    first = _row(tmp_path, **fields)
    second = _row(tmp_path, **fields)

    text = canonical_json(first.producer_detail)
    assert str(tmp_path) not in text
    assert tmp_path.name not in text  # file names only, never where they were
    assert producer_detail_sha256(first.producer_detail) == producer_detail_sha256(
        second.producer_detail
    )

    detail = first.producer_detail
    assert detail["contract"] == CONTRACT
    assert detail["protocol"] == PROTOCOL
    manifest = tmp_path / "specimens.toml"
    assert detail["manifest"]["file"] == "specimens.toml"
    assert detail["manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    # The row as it was read, with the shared declarations applied.
    assert detail["manifest"]["row_with_defaults"]["drive_voltage_v"] == 0.5
    assert detail["manifest"]["row_with_defaults"]["trace_file"] == "run.xlsx"
    assert detail["source"]["sha256"] == first.values["trace_sha256"]
    assert set(detail["cycles"][0]) >= {
        "slope_mv_k",
        "intercept_mv",
        "on_branch_slope_mv_k",
        "off_branch_slope_mv_k",
        "on_duration_s",
        "off_duration_s",
        "included_in_specimen_value",
    }


def test_declared_columns_are_passed_through_and_blanks_stay_blank(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    row = _row(tmp_path, electrode_material="  ", relative_humidity_pct=41, mixture_batch_id="B1")
    values = row.values
    assert values["drive_voltage_v"] == 0.5
    assert values["drops_applied"] == 3
    assert values["remounted_between_cycles"] == "no"
    assert values["dosing_method"] == "pipette"
    assert values["electrode_material"] is None  # blank is not a material
    assert values["relative_humidity_pct"] == 41.0
    assert values["ambient_temperature_c"] is None
    assert values["mixture_batch_id"] == "B1"
    assert row.provenance["drops_applied"] == "declared in the manifest"
    assert "electrode_material" not in row.provenance


def test_the_drive_voltage_is_a_declared_set_point_and_says_so(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    row = _row(tmp_path)
    stated = "the drive set-point declared in the manifest; not a measured voltage"
    assert row.values["drive_voltage_v"] == 0.5
    assert row.provenance["drive_voltage_v"] == stated
    basis = row.producer_detail["controls_recorded"]["basis"]
    assert basis["drive_voltage_v"] == stated
    assert basis["cycle count and phase durations"] == "measured from the cycle boundaries"


def test_a_declared_quality_note_is_a_warning_that_changes_nothing(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx", seebeck_mv_k=[-3.0, -2.5, -2.0])
    plain = _row(tmp_path)
    noted = _row(tmp_path, qc_warnings=["strong drift from cycle to cycle", " second note "])

    assert [(i.field, i.severity, i.message) for i in noted.issues] == [
        (
            "qc_warnings",
            Severity.WARNING,
            "s1: declared QC warning: strong drift from cycle to cycle",
        ),
        ("qc_warnings", Severity.WARNING, "s1: declared QC warning: second note"),
    ]
    # A statement about the specimen: no value moves and nothing is refused.
    assert _messages(noted, Severity.ERROR) == []
    assert noted.values == plain.values
    assert noted.producer_detail["reduction"] == plain.producer_detail["reduction"]
    # Kept in the record exactly as declared.
    declared = noted.producer_detail["manifest"]["row_with_defaults"]["qc_warnings"]
    assert declared == ["strong drift from cycle to cycle", " second note "]
    assert "qc_warnings" not in plain.producer_detail["manifest"]["row_with_defaults"]


def test_nothing_is_written_and_no_recording_is_touched(tmp_path: Path) -> None:
    make_recording(tmp_path / "run.xlsx")
    manifest = write_manifest(tmp_path, [specimen("s1", "run.xlsx")], defaults=BENCH_DEFAULTS)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    produce_table(manifest)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_the_protocol_names_no_electrode_material() -> None:
    # Unconfirmed, so it is recorded per specimen and never stated as a fact.
    text = canonical_json(PROTOCOL).lower()
    assert "gold" not in text
    assert "copper" not in text
    assert "electrode material: recorded per specimen when it is known" in text


# ─── The manifest ───────────────────────────────────────────────────
def _manifest_problems(
    folder: Path, *specimens: dict[str, Any], **manifest: Any
) -> tuple[str, ...]:
    manifest.setdefault("defaults", BENCH_DEFAULTS)
    with pytest.raises(ManifestError) as caught:
        load_manifest(write_manifest(folder, specimens, **manifest))
    return caught.value.problems


def test_a_recording_cannot_be_named_twice(tmp_path: Path) -> None:
    reverse = {"trace_file": "a.xlsx", "polarity_convention": SWAPPED}
    # Named as a reversed-lead run and again as a specimen, the diagnostic
    # would be counted as an observation.
    assert _manifest_problems(
        tmp_path, specimen("a", "a.xlsx"), specimen("b", "b.xlsx", reverse=reverse)
    ) == ("trace file 'a.xlsx' is named 2 times",)
    assert _manifest_problems(tmp_path, specimen("a", "a.xlsx"), specimen("b", "a.xlsx")) == (
        "trace file 'a.xlsx' is named 2 times",
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"test_cycle": 1}, "specimen 1 (s1): unknown key 'test_cycle'"),
        ({"test_cycles": "1"}, "test_cycles must be a whole number, zero or more"),
        ({"test_cycles": -1}, "test_cycles must be a whole number, zero or more"),
        ({"drops_applied": 2.5}, "drops_applied must be a whole number"),
        ({"remounted_between_cycles": "maybe"}, "remounted_between_cycles must be 'yes' or 'no'"),
        ({"drive_voltage_v": "0.5 V"}, "drive_voltage_v must be a number"),
        (
            {"channels": {"cold": "a", "hot": "b"}},
            "channels must name exactly cold, hot and voltage",
        ),
        (
            {"channels": {"cold": "a", "hot": "a", "voltage": "c"}},
            "channels names one recorded channel for two roles",
        ),
        (
            {"declared_boundaries_s": {"on": [1.0], "off": [2.0]}},
            "declared_boundaries_s.end must be a time in s",
        ),
        (
            {"declared_boundaries_s": {"on": [], "off": [2.0], "end": 3.0}},
            "declared_boundaries_s.on must be a list of times in s",
        ),
        (
            {"declared_boundaries_s": {"on": [1.0], "off": [2.0], "end": 3.0, "ends": 4.0}},
            "declared_boundaries_s has an unknown key 'ends'",
        ),
        (
            {"masses_g": {"Aliquat 336TG": "0.1"}},
            "masses_g must be a table of component name to grams",
        ),
        (
            {"reverse": {"polarity_convention": SWAPPED}},
            "specimen 1 (s1), reverse: trace_file is required",
        ),
        (
            {"reverse": {"trace_file": "r.xlsx", "pure_component": "x"}},
            "specimen 1 (s1), reverse: unknown key 'pure_component'",
        ),
        ({"trace_file": " "}, "trace_file is required"),
        ({"sample_id": ""}, "specimen 1: sample_id is required"),
        ({"qc_warnings": "drifts"}, "qc_warnings must be a list of sentences"),
        ({"qc_warnings": ["drifts", " "]}, "qc_warnings must be a list of sentences"),
        ({"qc_warnings": [3]}, "qc_warnings must be a list of sentences"),
    ],
)
def test_a_malformed_specimen_table_is_refused(
    tmp_path: Path, fields: dict[str, Any], message: str
) -> None:
    problems = _manifest_problems(tmp_path, {**specimen("s1", "run.xlsx"), **fields})
    assert any(message in problem for problem in problems), problems


def test_every_manifest_problem_is_reported_at_once(tmp_path: Path) -> None:
    problems = _manifest_problems(
        tmp_path,
        specimen("s1", "a.xlsx", test_cycle=1),
        specimen("s1", "b.xlsx", drops_applied=-1),
        defaults={"polarity_convention": STRAIGHT},
        contract="ite-trace/2",
    )
    assert problems == (
        "contract is 'ite-trace/2'; this producer reads 'ite-trace/1'",
        # The wiring is declared per recording, never inherited.
        "defaults: unknown key 'polarity_convention'",
        "specimen 1 (s1): unknown key 'test_cycle'",
        "specimen 2 (s1): drops_applied must be a whole number, zero or more",
        "sample_id 's1' is used by 2 specimens",
    )


def test_a_manifest_that_is_not_toml_or_has_no_specimens_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "specimens.toml"
    path.write_text("contract = ", encoding="utf-8")
    with pytest.raises(ManifestError, match="is not valid TOML"):
        load_manifest(path)
    path.write_text(f'contract = "{CONTRACT}"\n', encoding="utf-8")
    with pytest.raises(ManifestError, match=r"at least one \[\[specimen\]\] table is required"):
        load_manifest(path)
    with pytest.raises(FileNotFoundError):
        produce_table(tmp_path / "absent.toml")


def test_defaults_apply_to_every_specimen_and_a_specimen_can_override(tmp_path: Path) -> None:
    for name in ("a.xlsx", "b.xlsx"):
        make_recording(tmp_path / name)
    first, second = _table(
        tmp_path, specimen("a", "a.xlsx"), specimen("b", "b.xlsx", drops_applied=4)
    ).rows
    assert (first.values["drops_applied"], second.values["drops_applied"]) == (3, 4)
    assert first.values["drive_voltage_v"] == second.values["drive_voltage_v"] == 0.5


def test_the_module_exports_its_columns() -> None:
    assert len(set(COLUMNS)) == len(COLUMNS) == 24
    assert {
        ite_trace.SAMPLE_ID,
        ite_trace.ALIQUAT_MASS_FRACTION,
        ite_trace.APPARENT_SEEBECK,
    } <= set(COLUMNS)
