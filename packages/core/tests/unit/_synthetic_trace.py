"""A made-up thermal-cycle recording, written in the logger's workbook layout.

Nothing here is measured. The temperature difference is a first-order response
to a square-wave drive, and the voltage is an exact straight line in it, one
slope per cycle. So the slope a test gets back is the slope it put in, and any
number that disagrees is the code's doing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl

COLD, HOT, VOLTAGE = "CH1-A-1", "CH1-A-3", "CH1-A-10"
CHANNELS = {"cold": COLD, "hot": HOT, "voltage": VOLTAGE}
STRAIGHT = "V+ on the COLD electrode"
SWAPPED = "V+ on the HOT electrode"
CONTRACT = "ite-trace/1"

SWING_K = 15.0
TAU_S = 20.0
BASE_C = 25.0


def boundaries(
    on_s: Sequence[float] = (180.0, 180.0, 180.0),
    off_s: Sequence[float] = (180.0, 180.0, 180.0),
    *,
    lead_in_s: float = 10.0,
) -> list[float]:
    """Switching times ``on_1, off_1, …, on_n, off_n, end`` for the given phases."""
    times = [lead_in_s]
    for on, off in zip(on_s, off_s, strict=True):
        times.append(times[-1] + on)
        times.append(times[-1] + off)
    return times


def series(
    switches: Sequence[float],
    *,
    seebeck_mv_k: float | Sequence[float] = 2.0,
    offset_mv: float = 5.0,
    interval_s: float = 1.0,
    tail_s: float = 10.0,
    lag_mv: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time, cold and hot temperatures, and V_cold - V_hot in mV.

    Args:
        switches: ``on_1, off_1, …, end``.
        seebeck_mv_k: One slope for every cycle, or one per cycle.
        offset_mv: The intercept of the voltage line.
        interval_s: Sample spacing.
        tail_s: How long the recording runs after the end of the last cycle.
        lag_mv: Adds this much to the voltage while the drive is on and
            subtracts it while it is off, which opens a loop between the
            heating and cooling branches without changing its shape from one
            cycle to the next.
    """
    n_cycles = len(switches) // 2
    slopes = (
        [float(seebeck_mv_k)] * n_cycles
        if isinstance(seebeck_mv_k, (int, float))
        else [float(s) for s in seebeck_mv_k]
    )
    assert len(slopes) == n_cycles
    time = np.arange(0.0, switches[-1] + tail_s + interval_s / 2, interval_s)
    time = np.round(time, 6)

    delta = np.zeros_like(time)
    driven = np.zeros(time.size, dtype=bool)
    level = 0.0
    # Each phase relaxes exactly from where the last one ended.
    for index, start in enumerate(switches):
        stop = switches[index + 1] if index + 1 < len(switches) else math.inf
        on = index % 2 == 0 and index + 1 < len(switches)
        target = SWING_K if on else 0.0
        inside = (time >= start) & (time < stop)
        delta[inside] = target + (level - target) * np.exp(-(time[inside] - start) / TAU_S)
        driven[inside] = on
        if math.isfinite(stop):
            level = target + (level - target) * math.exp(-(stop - start) / TAU_S)

    cycle = np.clip(np.searchsorted(switches[0::2], time, side="right") - 1, 0, n_cycles - 1)
    voltage = np.asarray(slopes)[cycle] * delta + offset_mv
    voltage = voltage + np.where(driven, lag_mv, -lag_mv)
    return time, BASE_C - delta / 2, BASE_C + delta / 2, voltage


def _stamp(seconds: float, *, date: str, start_tenths: int) -> str:
    tenths = start_tenths + round(seconds * 10)
    whole, tenth = divmod(tenths, 10)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{date} {hours:02d}:{minutes:02d}:{secs:02d}.{tenth}s"


def write_workbook(
    path: Path,
    time: np.ndarray,
    cold: np.ndarray,
    hot: np.ndarray,
    voltage_mv: np.ndarray,
    *,
    marks: Sequence[tuple[int | str, float]] | None = None,
    event_column: bool = True,
    lead_sign: int = 1,
    voltage_unit: str = "V",
    names: Sequence[str] = (COLD, HOT, VOLTAGE),
    temperature_unit: str = "°C",
    sheet: str = "V1_H1",
    date: str = "1/2/2026",
    start_clock_s: int = 9 * 3600,
) -> Path:
    """Write the series as the logger software would, and return the path.

    Args:
        path: Where to write.
        time: Sample times in s.
        cold: Cold electrode temperature.
        hot: Hot electrode temperature.
        voltage_mv: V_cold - V_hot in mV. What is written is this times
            `lead_sign`, in `voltage_unit`.
        marks: ``(number, time)`` for each event mark. Text in place of a
            number writes that text, which the logger never does: it is what a
            person typing a label into the column leaves behind.
        event_column: Whether the workbook has an event column at all.
        lead_sign: +1 for the positive lead on the cold electrode, -1 swapped.
        voltage_unit: The unit written in the voltage header.
        names: The three recorded channel names: cold, hot, voltage.
        temperature_unit: The unit written in both temperature headers.
        sheet: The sheet name.
        date: The date text of every time cell.
        start_clock_s: Time of day of the first sample, in s.
    """
    factor = {"V": 1000.0, "mV": 1.0, "uV": 0.001}.get(voltage_unit, 1.0)
    recorded = lead_sign * voltage_mv / factor
    interval = float(time[1] - time[0])
    marked = {round(at / interval): number for number, at in (marks or ())}

    workbook = openpyxl.Workbook()
    page = workbook.active
    page.title = sheet
    page.append([f"Trigger Time {_stamp(0.0, date=date, start_tenths=start_clock_s * 10)[:-3]}s"])
    header = [
        "Time",
        f"{names[0]}[{temperature_unit}]",
        f"{names[1]}[{temperature_unit}]",
        f"{names[2]}[{voltage_unit}]",
    ]
    if event_column:
        header.append("Event")
    page.append(header)
    for index, seconds in enumerate(time):
        row: list[Any] = [
            _stamp(float(seconds), date=date, start_tenths=start_clock_s * 10),
            float(cold[index]),
            float(hot[index]),
            float(recorded[index]),
        ]
        if event_column:
            row.append(marked.get(index, 0))
        page.append(row)
    workbook.save(path)
    workbook.close()
    return path


def make_recording(
    path: Path,
    *,
    switches: Sequence[float] | None = None,
    marks: Sequence[tuple[int | str, float]] | str | None = "all",
    **options: Any,
) -> list[float]:
    """Write one recording and return its switching times.

    Args:
        path: Where to write.
        switches: ``on_1, off_1, …, end``; three 180 s / 180 s cycles if absent.
        marks: ``"all"`` to mark every switch and the end in order, None for a
            recording with no event column, or explicit ``(number, time)``.
        **options: Passed to `series` (the signal) or `write_workbook` (the
            file), whichever takes the name.
    """
    switches = list(switches) if switches is not None else boundaries()
    signal_keys = ("seebeck_mv_k", "offset_mv", "interval_s", "tail_s", "lag_mv")
    signal = {key: options.pop(key) for key in signal_keys if key in options}
    time, cold, hot, voltage = series(switches, **signal)
    if marks == "all":
        marks = [(number, at) for number, at in enumerate(switches, start=1)]
    write_workbook(
        path,
        time,
        cold,
        hot,
        voltage,
        marks=marks if isinstance(marks, list) else None,
        event_column=marks is not None,
        **options,
    )
    return switches


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        inner = ", ".join(
            f"{json.dumps(k, ensure_ascii=False)} = {_toml(v)}" for k, v in value.items()
        )
        return "{ " + inner + " }"
    return "[" + ", ".join(_toml(item) for item in value) + "]"


def write_manifest(
    folder: Path,
    specimens: Sequence[Mapping[str, Any]],
    *,
    defaults: Mapping[str, Any] | None = None,
    contract: str = CONTRACT,
    name: str = "specimens.toml",
) -> Path:
    """Write a specimen manifest beside the recordings and return its path."""
    lines = [f"contract = {_toml(contract)}", ""]
    if defaults is not None:
        lines.append("[defaults]")
        lines += [f"{key} = {_toml(value)}" for key, value in defaults.items()]
        lines.append("")
    for specimen in specimens:
        lines.append("[[specimen]]")
        lines += [f"{key} = {_toml(value)}" for key, value in specimen.items()]
        lines.append("")
    path = folder / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


BENCH_DEFAULTS: dict[str, Any] = {
    "channels": CHANNELS,
    "drive_voltage_v": 0.5,
    "drops_applied": 3,
    "remounted_between_cycles": "no",
    "dosing_method": "pipette",
}


def specimen(sample_id: str, trace_file: str, **fields: Any) -> dict[str, Any]:
    """A manifest row on straight wiring; pure first component unless told otherwise."""
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "trace_file": trace_file,
        "polarity_convention": STRAIGHT,
    }
    if "masses_g" not in fields and "pure_component" not in fields:
        row["pure_component"] = "Aliquat 336TG"
    row.update(fields)
    return {key: value for key, value in row.items() if value is not None}
