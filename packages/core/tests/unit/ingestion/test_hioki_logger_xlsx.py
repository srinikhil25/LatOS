"""Reading a logger recording saved as an Excel workbook."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pytest

from latos.core.exceptions import CorruptedFileError
from latos.ingestion.hioki_logger_xlsx import read_hioki_xlsx
from tests.unit._synthetic_trace import COLD, HOT, VOLTAGE, make_recording, series, write_workbook

HEADER = ["Time", "CH1-A-1[°C]", "CH1-A-3[°C]", "CH1-A-10[V]", "Event"]


def _raw(path: Path, rows: list[list[Any]], *, sheets: tuple[str, ...] = ("V1_H1",)) -> Path:
    """Write rows exactly as given, on every named sheet."""
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name in sheets:
        page = workbook.create_sheet(name)
        for row in rows:
            page.append(row)
    workbook.save(path)
    workbook.close()
    return path


def _rows(*samples: list[Any], header: list[Any] | None = None) -> list[list[Any]]:
    return [["Trigger Time 1/2/2026 09:00:00s"], header or HEADER, *samples]


GOOD = _rows(
    ["1/2/2026 09:00:00.0s", 25.0, 25.5, 0.001, 0],
    ["1/2/2026 09:00:00.2s", 25, 25.6, 0.002, 1],
    ["1/2/2026 09:00:00.4s", 24.9, 25.7, -0.003, 0],
    ["1/2/2026 09:00:00.6s", 24.8, 25.8, 0.004, 2],
)


def test_names_units_values_and_times_are_kept_as_recorded(tmp_path: Path) -> None:
    path = _raw(tmp_path / "run.xlsx", GOOD)
    trace = read_hioki_xlsx(path)

    assert trace.source == path
    assert trace.sheet == "V1_H1"
    assert trace.preamble == ("Trigger Time 1/2/2026 09:00:00s",)
    assert [(c.name, c.unit, c.header) for c in trace.channels] == [
        ("CH1-A-1", "°C", "CH1-A-1[°C]"),
        ("CH1-A-3", "°C", "CH1-A-3[°C]"),
        ("CH1-A-10", "V", "CH1-A-10[V]"),
    ]
    # Volts stay volts: nothing is converted here.
    assert trace.channel("CH1-A-10").values.tolist() == [0.001, 0.002, -0.003, 0.004]
    assert trace.channel("CH1-A-1").values.tolist() == [25.0, 25.0, 24.9, 24.8]
    assert trace.timestamps == tuple(row[0] for row in GOOD[2:])
    assert trace.time_s.tolist() == [0.0, 0.2, 0.4, 0.6]
    assert trace.sample_interval_s == pytest.approx(0.2)
    assert trace.ignored_columns == ()


def test_no_channel_is_given_a_role(tmp_path: Path) -> None:
    trace = read_hioki_xlsx(_raw(tmp_path / "run.xlsx", GOOD))
    for channel in trace.channels:
        assert set(channel.__slots__) == {"name", "unit", "header", "values"}
    with pytest.raises(KeyError, match="recorded: CH1-A-1, CH1-A-3, CH1-A-10"):
        trace.channel("hot")


def test_event_marks_keep_the_numbers_the_logger_wrote(tmp_path: Path) -> None:
    trace = read_hioki_xlsx(_raw(tmp_path / "run.xlsx", GOOD))
    assert trace.has_event_column is True
    assert [(e.number, e.time_s, e.row) for e in trace.events] == [(1, 0.2, 1), (2, 0.6, 3)]


def test_a_recording_without_an_event_column_says_so(tmp_path: Path) -> None:
    rows = _rows(*(row[:4] for row in GOOD[2:]), header=HEADER[:4])
    trace = read_hioki_xlsx(_raw(tmp_path / "run.xlsx", rows))
    assert trace.has_event_column is False
    assert trace.events == ()

    unmarked = _rows(*([*row[:4], 0] for row in GOOD[2:]))
    trace = read_hioki_xlsx(_raw(tmp_path / "unmarked.xlsx", unmarked))
    assert trace.has_event_column is True
    assert trace.events == ()


def test_the_file_is_hashed_and_left_untouched(tmp_path: Path) -> None:
    path = _raw(tmp_path / "run.xlsx", GOOD)
    before = path.read_bytes()
    modified = path.stat().st_mtime_ns
    trace = read_hioki_xlsx(path)
    assert trace.sha256 == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == modified


def test_the_arrays_cannot_be_written_to(tmp_path: Path) -> None:
    trace = read_hioki_xlsx(_raw(tmp_path / "run.xlsx", GOOD))
    for array in (trace.time_s, trace.channels[0].values):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0.0


def test_a_column_that_is_not_a_channel_is_named_not_dropped_unseen(tmp_path: Path) -> None:
    rows = _rows(
        ["1/2/2026 09:00:00.0s", 25.0, 0, 0],
        ["1/2/2026 09:00:01.0s", 25.1, 0, 0],
        header=["Time", "CH1-A-1[°C]", "ALM-CH1", "Event"],
    )
    trace = read_hioki_xlsx(_raw(tmp_path / "run.xlsx", rows))
    assert [c.name for c in trace.channels] == ["CH1-A-1"]
    assert trace.ignored_columns == ("ALM-CH1",)


def test_elapsed_time_survives_midnight(tmp_path: Path) -> None:
    rows = _rows(
        ["1/2/2026 23:59:59.0s", 25.0, 25.5, 0.001, 0],
        ["1/2/2026 23:59:59.5s", 25.0, 25.5, 0.001, 0],
        ["1/3/2026 00:00:00.0s", 25.0, 25.5, 0.001, 0],
        ["1/3/2026 00:00:00.5s", 25.0, 25.5, 0.001, 0],
    )
    trace = read_hioki_xlsx(_raw(tmp_path / "run.xlsx", rows))
    assert trace.time_s.tolist() == [0.0, 0.5, 1.0, 1.5]


def test_a_changed_date_is_read_as_the_documented_limitation_says(tmp_path: Path) -> None:
    # As the ite-trace/1 PROTOCOL states it: a changed date adds nothing while
    # the time of day advances, and exactly one day when it goes back, however
    # many days apart the dates are. Rows days apart are read as closer.
    rows = _rows(
        ["1/2/2026 09:00:00.0s", 25.0, 25.5, 0.001, 0],
        ["1/3/2026 09:00:00.5s", 25.0, 25.5, 0.001, 0],  # a day and 0.5 s later
        ["1/9/2026 08:00:00.0s", 25.0, 25.5, 0.001, 0],  # six days later, clock behind
    )
    trace = read_hioki_xlsx(_raw(tmp_path / "run.xlsx", rows))
    assert trace.time_s.tolist() == [0.0, 0.5, 0.5 + 86400 - 3600.5]


def test_a_workbook_with_several_sheets_needs_one_named(tmp_path: Path) -> None:
    path = _raw(tmp_path / "run.xlsx", GOOD, sheets=("V1_H1", "V2_H1"))
    with pytest.raises(
        CorruptedFileError, match=r"2 sheets \(V1_H1, V2_H1\); name the one to read"
    ):
        read_hioki_xlsx(path)
    assert read_hioki_xlsx(path, sheet="V2_H1").sheet == "V2_H1"
    with pytest.raises(CorruptedFileError, match="no sheet named 'V9'"):
        read_hioki_xlsx(path, sheet="V9")


def _with(index: int, column: int, value: Any) -> list[list[Any]]:
    rows = [list(row) for row in GOOD]
    rows[index][column] = value
    return rows


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (_with(3, 1, None), "row 4 of 'CH1-A-1.°C.' holds None, not a reading"),
        (_with(4, 3, "+OVER"), "row 5 of 'CH1-A-10.V.' holds '.OVER', not a reading"),
        (_with(3, 3, True), "holds True, not a reading"),
        (_with(4, 0, "1/2/2026 09:00:00.2s"), "the time does not advance at row 5"),
        (_with(4, 0, "1/2/2026 08:00:00.0s"), "the time does not advance at row 5"),
        (_with(3, 0, 0.2), "row 4 has time 0.2; expected text such as"),
        (_with(3, 0, "09:00:00.2"), "row 4 has time '09:00:00.2'"),
        (_with(3, 4, 1.5), "row 4 of the event column holds 1.5, not an event number"),
        (_with(3, 4, -1), "row 4 of the event column holds -1, not an event number"),
        (_with(1, 0, "Sample"), "no header row found"),
        (_with(1, 2, "CH1-A-1[°C]"), "names a channel more than once"),
        (
            [GOOD[0], ["Time", "Event"], ["1/2/2026 09:00:00.0s", 0], ["1/2/2026 09:00:01.0s", 0]],
            "the header row names no channel",
        ),
        (GOOD[:3], "fewer than 2 samples"),
        ([*GOOD[:4], [None] * 5, *GOOD[4:]], "row 5 is blank, with samples after it"),
    ],
)
def test_a_layout_that_does_not_match_is_refused(
    tmp_path: Path, rows: list[list[Any]], message: str
) -> None:
    with pytest.raises(CorruptedFileError, match=message):
        read_hioki_xlsx(_raw(tmp_path / "run.xlsx", rows))


def test_trailing_blank_rows_are_padding(tmp_path: Path) -> None:
    trace = read_hioki_xlsx(_raw(tmp_path / "run.xlsx", [*GOOD, [None] * 5, [None] * 5]))
    assert trace.time_s.size == 4


def test_something_that_is_not_a_workbook_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run.xlsx"
    path.write_text("not a workbook", encoding="utf-8")
    with pytest.raises(CorruptedFileError, match="not a readable Excel workbook"):
        read_hioki_xlsx(path)
    with pytest.raises(FileNotFoundError):
        read_hioki_xlsx(tmp_path / "absent.xlsx")


def test_a_whole_synthetic_run_reads_back_as_written(tmp_path: Path) -> None:
    switches = make_recording(tmp_path / "run.xlsx", interval_s=0.5)
    time, cold, hot, voltage = series(switches, interval_s=0.5)
    trace = read_hioki_xlsx(tmp_path / "run.xlsx")

    np.testing.assert_allclose(trace.time_s, time)
    # A workbook stores a float as text, so the last bit may not survive.
    np.testing.assert_allclose(trace.channel(COLD).values, cold, rtol=1e-13)
    np.testing.assert_allclose(trace.channel(HOT).values, hot, rtol=1e-13)
    np.testing.assert_allclose(trace.channel(VOLTAGE).values, voltage / 1000.0)
    assert [e.number for e in trace.events] == [1, 2, 3, 4, 5, 6, 7]
    assert [e.time_s for e in trace.events] == switches


def test_the_header_unit_is_reported_whatever_it_is(tmp_path: Path) -> None:
    time, cold, hot, voltage = series([10.0, 40.0, 70.0])
    write_workbook(tmp_path / "run.xlsx", time, cold, hot, voltage, voltage_unit="mV")
    channel = read_hioki_xlsx(tmp_path / "run.xlsx").channel(VOLTAGE)
    assert channel.unit == "mV"
    np.testing.assert_allclose(channel.values, voltage)  # millivolts, as written
