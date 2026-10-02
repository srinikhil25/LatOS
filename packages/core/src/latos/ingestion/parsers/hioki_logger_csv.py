"""HIOKI Memory HiLogger text exports — the V(t) and T(t) trace of an i-TE run.

Why this parser exists
----------------------
An ionic thermoelectric cell does not settle quickly. Voltage build-up runs
from several hundred to several thousand seconds, so ΔV/ΔT read at 60 s and at
3000 s are different numbers. A single recorded voltage cannot be audited:
nothing in it says whether the cell had reached steady state, what its time
constant was, or whether it was quietly drifting. The trace answers all three,
which is why `ite_workbook_template.py` asks for `raw_trace_file` and notes
that "a single number cannot be audited".

This parser reads what the instrument actually wrote, so that trace becomes
analysable rather than merely archived.

File format
-----------
Documented in Appendix 3, "Text File (CSV) Internal Format", of the Memory
HiLogger instruction manual. A header block of quoted key/value rows, then a
channel-setting table, then the column header, then the samples::

    "File name","WAVE0001.CSV","V 1.00"
    "Title comment",""
    "Trigger Time","'12-07-01 09:14:03"
    "Ch","Mode","Range","Comment","Scaling","Ratio","Offset"
    "CH-1","Voltage","100mV","cell","Off","-","-"
    "CH-7","Tc","2000 C","hot","Off","-","-"
    "CH-8","Tc","2000 C","cold","Off","-","-"
    "ALM","Alarm","","",
    "Time","CH-1[V]","CH-7[C]","CH-8[C]","ALM-CH1","Event",
    0.000000000E+00,-1.30000E-04, 3.59000E+01, 3.33000E+01,0,0,
    1.000000000E+00,-2.15000E-04, 2.62000E+01, 2.36000E+01,0,0,

Three things about that format are easy to get wrong, and are handled here.

**The unit is in the column header, not the range.** A channel on the 100 mV
range still writes ``[V]``, so reading the numbers as millivolts because the
range says ``100mV`` is wrong by a factor of 1000. The bracketed unit is the
authority and the only thing this parser trusts.

**The delimiter and decimal mark are user-settable.** The instrument's CSV
saving settings choose both, so an export made with comma decimals is
semicolon-separated. The delimiter is sniffed from the column-header row and
numbers are read either way.

**The extension is not fixed.** The same text export is written as ``.TXT``
rather than ``.CSV`` depending on a System-screen setting, so both are
accepted.

Channel roles
-------------
The file records ten interchangeable channels; nothing in it says which
thermocouple was on the hot side. Two signals are used, in order.

The bracketed unit fixes what a column *is*: volts mean the cell, Celsius means
a thermocouple. That works even on a header-suppressed export, where the
channel table is absent entirely.

The channel ``Comment`` field then says which thermocouple is which. Commenting
a channel ``hot``, ``cold`` or ``cell`` on the instrument makes the file
self-describing, and records the assignment at measurement time by the person
who wired it. Failing that, roles fall back to channel order and the parser
says so as a WARNING. A positional guess that goes unreported is how a campaign
ends up with its sign inverted, so it is always reported.

Sign convention
---------------
Voltage is passed through exactly as measured. No minus sign is applied, and
none should be: `ThermovoltageSlopeAnalyzer` documents that it fits the plain
slope, so the voltage handed to it must be V_cold - V_hot. That is fixed at the
terminal block by putting V+ on the cold electrode — rule 2 of the workbook,
not a free choice — and no parser can recover it after the fact.

Validation policy: see `xrd_rigaku_txt.py` — same contract.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["HiokiLoggerCsvParser", "is_hioki_logger_header"]

# How many leading lines to inspect when sniffing. The channel table runs to
# 15 rows under a 3-row key/value header, so the column header can be line 19.
_SNIFF_LINES = 30

# Column header cells look like `CH-1[V]` or `P-3[r/s]`.
_COLUMN_RE = re.compile(r"^\s*(?P<channel>(?:CH|P)-\d+)\s*\[(?P<unit>[^\]]*)\]\s*$")

# Field separators the save dialog can produce, most specific first so a tab wins
# a tie against a comma that only appears inside a quoted comment.
_DELIMITERS = ("\t", ";", ",")

# Clock-time forms the Time column can carry when the export was made with a
# Time Axis Format of Absolute Time or Relative Time rather than Second. Both are
# converted to elapsed seconds so the trace is still usable, rather than being
# rejected for a setting nobody knew mattered.
_ABSOLUTE_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%y-%m-%d %H:%M:%S.%f",
    "%y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
)

# Relative time, as `[d ]h:mm:ss[.fff]`.
_RELATIVE_TIME_RE = re.compile(
    r"^(?:(?P<days>\d+)[dD]?\s+)?(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)$",
)

_SECONDS_PER_DAY = 86400.0
_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_MINUTE = 60.0

# Instrument clock readings carry no timezone. Every other parser here stamps
# naive instrument times as UTC, so this one matches; the unmodified string is
# kept in metadata as the actual record.
_TRIGGER_TIME_FORMATS = ("%y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")
_TRIGGER_TIME_KEYS = frozenset({"trigger time", "start time"})

# Multiplier from a declared unit to millivolts. A unit outside this map is
# refused rather than guessed at.
_VOLT_TO_MV: dict[str, float] = {
    "v": 1000.0,
    "mv": 1.0,
    "uv": 0.001,
    "µv": 0.001,
    "μv": 0.001,
}

# Units meaning degrees Celsius. A difference of two Celsius readings is
# already a difference in kelvin, so no offset is applied anywhere.
_CELSIUS_UNITS = frozenset({"c", "degc", "°c", "℃"})

# Only the analog channels can carry a voltage or a thermocouple. The pulse
# channels are `P-n`, and a pulse count is written `P-1[c]` — which differs from
# a Celsius column, `CH-7[C]`, by letter case alone. Case is far too thin a
# thread to hang a thermocouple assignment on, so role assignment is restricted
# to the analog channels, where the distinction does not arise.
_ANALOG_PREFIX = "CH-"

# Comment substrings that assign a channel to a role. "cold" precedes "cell"
# so the longer, more specific word is tested first.
_ROLE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("hot", "hot"),
    ("cold", "cold"),
    ("cool", "cold"),
    ("cell", "cell"),
    ("sample", "cell"),
)

# Scaling-column values that mean "no scaling applied".
_SCALING_OFF = frozenset({"", "-", "off"})

_MIN_SAMPLES_FOR_INTERVAL = 2


# ─── Record types ───────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _Channel:
    """One row of the channel-setting table."""

    channel: str
    mode: str
    range_text: str
    comment: str
    scaling: str


@dataclass(frozen=True, slots=True)
class _Column:
    """One data column: where it sits, which channel, what unit."""

    index: int
    channel: str
    unit: str


@dataclass(frozen=True, slots=True)
class _Layout:
    """Where the samples start and what the columns mean."""

    header_index: int
    columns: tuple[_Column, ...]
    event_index: int | None


@dataclass(frozen=True, slots=True)
class _Roles:
    """Which channel plays which part, and whether that was guessed."""

    cell: str | None
    hot: str | None
    cold: str | None
    positional: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Samples:
    """The parsed sample block."""

    time_s: np.ndarray
    by_channel: dict[str, np.ndarray]
    event: np.ndarray | None
    malformed: int
    time_axis: str = "seconds"


@dataclass(frozen=True, slots=True)
class _FileRead:
    """A successfully read file, before any interpretation."""

    rows: tuple[tuple[str, ...], ...]
    delimiter: str
    header: dict[str, str]
    trigger_time: datetime | None


# ─── Format sniffing ────────────────────────────────────────────────
def is_hioki_logger_header(lines: list[str]) -> bool:
    """True if `lines` (the first ~30 of a file) are a HIOKI HiLogger export.

    Two accepted shapes, because the key/value header block can be suppressed
    when the data is converted to text:

    * the full export, opening with a quoted ``"File name"`` row and containing
      both a ``"Ch","Mode",...`` channel table and a ``Time`` column header;
    * a header-suppressed export, which has only the ``Time`` column header
      with at least one ``CH-n[unit]`` column.

    Shared with the CasaXPS parser, which uses it as a negative guard: a
    HiLogger sample row opens with two floats and would otherwise be claimed as
    an XPS spectrum.
    """
    return _header_shape(lines) is not None


def _header_shape(lines: list[str]) -> str | None:
    """Classify sniffed lines as ``"full"``, ``"bare"``, or None."""
    found_columns = False
    for line in lines:
        cells = _split_cells(line, _sniff_delimiter(line))
        if (
            cells
            and cells[0].strip().lower() == "time"
            and any(_COLUMN_RE.match(cell) for cell in cells[1:])
        ):
            found_columns = True
    if not found_columns:
        return None
    joined = "".join(lines)
    return "full" if '"File name"' in joined and '"Ch"' in joined else "bare"


def _sniff_delimiter(line: str) -> str:
    """Pick the delimiter by majority on a single line.

    Logger Utility's save dialog offers comma, semicolon and tab separation, the
    last written as ``.TXT``, so all three have to be recognised: a file that
    arrives tab-separated is a legitimate export, not a broken one. Space
    separation is offered too and is deliberately not guessed at, because quoted
    header cells contain spaces and splitting on them would shred the column
    names.
    """
    counts = {d: line.count(d) for d in _DELIMITERS}
    best = max(_DELIMITERS, key=lambda d: counts[d])
    return best if counts[best] else ","


def _split_cells(line: str, delimiter: str) -> list[str]:
    """Split one line into unquoted cells."""
    return [cell.strip().strip('"') for cell in line.split(delimiter)]


# ─── Parser ─────────────────────────────────────────────────────────
class HiokiLoggerCsvParser(BaseParser):
    """Parser for HIOKI Memory HiLogger `.csv` / `.txt` trace exports."""

    name: ClassVar[str] = "hioki-logger-csv"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.THERMOELECTRIC
    supported_extensions: ClassVar[tuple[str, ...]] = (".csv", ".txt")

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """1.0 for a full HiLogger export, 0.8 header-suppressed, else 0.0.

        The full export is unambiguous: nothing else here writes a quoted
        ``"File name"`` row above a ``"Ch","Mode"`` table. A header-suppressed
        export is identified by its ``CH-n[unit]`` columns alone, which is
        strong but not unique, so it scores below a definitive match.
        """
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = [fh.readline() for _ in range(_SNIFF_LINES)]
        except OSError:
            return 0.0
        shape = _header_shape(lines)
        if shape == "full":
            return 1.0
        return 0.8 if shape == "bare" else 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a HiLogger export into time, voltage and temperature arrays."""
        read = _read_file(path)
        if isinstance(read, ValidationIssue):
            return self._empty(read)

        layout = _find_layout(read.rows)
        if layout is None:
            return self._empty(
                _issue("data", Severity.ERROR, "No 'Time' column header row found."),
            )

        samples = _read_samples(read.rows, layout)
        if samples.time_s.size == 0:
            return self._empty(_issue("data", Severity.ERROR, "No sample rows found."))

        issues: list[ValidationIssue] = []
        channels = _read_channel_table(read.rows)
        roles = _assign_roles(channels, layout.columns)
        arrays = _build_arrays(samples, layout.columns, roles, issues)
        _report_roles(roles, issues)
        _check_scaling(channels, roles, issues)
        _report_time_axis(samples, issues)
        if samples.malformed:
            issues.append(
                _issue(
                    "data",
                    Severity.WARNING,
                    f"{samples.malformed} sample row(s) had unreadable numbers and were skipped.",
                ),
            )

        return ParsedData(
            technique=self.technique,
            arrays=arrays,
            metadata=_build_metadata(read, channels, layout, samples, roles),
            instrument="HIOKI Memory HiLogger",
            measured_at=read.trigger_time,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
            features=_build_features(samples, arrays),
        )

    def _empty(self, issue: ValidationIssue) -> ParsedData:
        """A `ParsedData` carrying nothing but the reason it is empty."""
        return ParsedData(
            technique=self.technique,
            arrays={},
            metadata={},
            instrument="HIOKI Memory HiLogger",
            measured_at=None,
            issues=(issue,),
            parser_name=self.name,
            parser_version=self.version,
        )


# ─── Reading ────────────────────────────────────────────────────────
def _read_file(path: Path) -> _FileRead | ValidationIssue:
    """Read every row, sniffing the delimiter from the column-header line."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _issue("file", Severity.ERROR, f"Could not read file: {exc}")

    lines = text.splitlines()
    delimiter = _delimiter_for(lines)
    rows = tuple(
        tuple(cell.strip().strip('"') for cell in row)
        for row in csv.reader(lines, delimiter=delimiter)
    )
    header = _read_kv_header(rows)
    return _FileRead(
        rows=rows,
        delimiter=delimiter,
        header=header,
        trigger_time=_trigger_time(header),
    )


def _delimiter_for(lines: list[str]) -> str:
    """Sniff the delimiter from the column-header line, else the first line."""
    for line in lines[:_SNIFF_LINES]:
        candidate = _sniff_delimiter(line)
        cells = _split_cells(line, candidate)
        if cells and cells[0].strip().lower() == "time":
            return candidate
    return _sniff_delimiter(lines[0]) if lines else ","


def _read_kv_header(rows: tuple[tuple[str, ...], ...]) -> dict[str, str]:
    """Collect the leading ``key,value`` rows that precede the channel table."""
    header: dict[str, str] = {}
    for row in rows:
        if not row or not row[0]:
            continue
        key = row[0].lower()
        if key in {"ch", "time"}:
            break
        if len(row) >= _MIN_SAMPLES_FOR_INTERVAL:
            header[key] = row[1]
    return header


def _trigger_time(header: dict[str, str]) -> datetime | None:
    """Parse the trigger timestamp, stamped UTC because the file states no zone."""
    raw = next((header[k] for k in _TRIGGER_TIME_KEYS if k in header), None)
    if not raw:
        return None
    text = raw.strip().lstrip("'")
    for fmt in _TRIGGER_TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _read_channel_table(rows: tuple[tuple[str, ...], ...]) -> dict[str, _Channel]:
    """Read the ``Ch,Mode,Range,Comment,Scaling,...`` table, if present."""
    table: dict[str, _Channel] = {}
    for row in rows:
        if not row or not _COLUMN_ID_RE.match(row[0]):
            continue
        table[row[0]] = _Channel(
            channel=row[0],
            mode=_cell(row, 1),
            range_text=_cell(row, 2),
            comment=_cell(row, 3),
            scaling=_cell(row, 4),
        )
    return table


_COLUMN_ID_RE = re.compile(r"^(?:CH|P)-\d+$")


def _cell(row: tuple[str, ...], index: int) -> str:
    """Cell `index` of `row`, or an empty string when the row is short."""
    return row[index] if index < len(row) else ""


def _find_layout(rows: tuple[tuple[str, ...], ...]) -> _Layout | None:
    """Locate the ``Time`` header row and decode its column names."""
    for position, row in enumerate(rows):
        if not row or row[0].strip().lower() != "time":
            continue
        columns = []
        event_index = None
        for index, cell in enumerate(row[1:], start=1):
            match = _COLUMN_RE.match(cell)
            if match is not None:
                columns.append(
                    _Column(
                        index=index,
                        channel=match.group("channel"),
                        unit=match.group("unit").strip(),
                    ),
                )
            elif cell.strip().lower() == "event":
                event_index = index
        if columns:
            return _Layout(
                header_index=position,
                columns=tuple(columns),
                event_index=event_index,
            )
    return None


def _read_samples(rows: tuple[tuple[str, ...], ...], layout: _Layout) -> _Samples:
    """Read the sample block below the column header."""
    stamps: list[float] = []
    per_channel: dict[str, list[float]] = {c.channel: [] for c in layout.columns}
    events: list[float] = []
    malformed = 0
    axis = "seconds"

    for row in rows[layout.header_index + 1 :]:
        if not row or not row[0].strip():
            continue
        stamp, form = _parse_time(row[0])
        if stamp is None:
            malformed += 1
            continue
        if form != "seconds":
            axis = form
        stamps.append(stamp)
        for column in layout.columns:
            value = _to_float(_cell(row, column.index))
            per_channel[column.channel].append(float("nan") if value is None else value)
        if layout.event_index is not None:
            marker = _to_float(_cell(row, layout.event_index))
            events.append(0.0 if marker is None else marker)

    time_s = np.asarray(stamps, dtype=np.float64)
    if axis == "absolute" and time_s.size:
        # Absolute timestamps come back as epoch seconds. Downstream wants time
        # from the start of the record, which is what a Second-format export
        # would have written in the first place.
        time_s = time_s - time_s[0]

    return _Samples(
        time_s=time_s,
        by_channel={k: np.asarray(v, dtype=np.float64) for k, v in per_channel.items()},
        event=np.asarray(events, dtype=np.float64) if events else None,
        malformed=malformed,
        time_axis=axis,
    )


def _parse_time(text: str) -> tuple[float | None, str]:
    """Parse one Time cell, whichever Time Axis Format produced it.

    The save dialog offers Second, Point, Absolute Time and Relative Time, and
    only the first is what this parser wants. The other two textual forms are
    converted here rather than rejected, because the setting is easy to miss and
    the trace is otherwise perfectly good.

    Returns the value in seconds (epoch seconds for absolute, rebased by the
    caller) and which form it came from.
    """
    stripped = text.strip()
    if not stripped:
        return None, "seconds"

    numeric = _to_float(stripped)
    if numeric is not None:
        return numeric, "seconds"

    # A Second-format export can carry a unit suffix; the dialog's "Save with a
    # decimal" checkbox is what removes it.
    trimmed = stripped.rstrip("sS").strip()
    numeric = _to_float(trimmed)
    if numeric is not None:
        return numeric, "seconds"

    match = _RELATIVE_TIME_RE.match(stripped)
    if match is not None:
        days = float(match.group("days") or 0.0)
        return (
            days * _SECONDS_PER_DAY
            + float(match.group("h")) * _SECONDS_PER_HOUR
            + float(match.group("m")) * _SECONDS_PER_MINUTE
            + float(match.group("s")),
            "relative",
        )

    for fmt in _ABSOLUTE_TIME_FORMATS:
        try:
            return datetime.strptime(stripped, fmt).replace(tzinfo=UTC).timestamp(), "absolute"
        except ValueError:
            continue
    return None, "seconds"


def _looks_like_sample_index(time_s: np.ndarray) -> bool:
    """True if the Time column could be a Point-format sample index.

    A Point-format export writes 0, 1, 2, 3 ... where seconds belong. At a one
    second recording interval those are the same numbers and nothing is wrong,
    which is why this cannot be an error — but at any other interval a time
    constant fitted against them is wrong by the ratio of the interval, with
    nothing in the file to reveal it. The signature is the only thing available:
    whole numbers, stepping by exactly one, starting at zero.
    """
    if time_s.size < _MIN_SAMPLES_FOR_INTERVAL:
        return False
    finite = time_s[np.isfinite(time_s)]
    if finite.size != time_s.size or float(finite[0]) != 0.0:
        return False
    steps = np.diff(finite)
    return bool(np.all(steps == 1.0))


def _to_float(text: str) -> float | None:
    """Parse a number written with either a dot or a comma decimal mark."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        pass
    try:
        return float(stripped.replace(",", "."))
    except ValueError:
        return None


# ─── Interpretation ─────────────────────────────────────────────────
def _assign_roles(
    channels: dict[str, _Channel],
    columns: tuple[_Column, ...],
) -> _Roles:
    """Work out which column is the cell, the hot side and the cold side.

    The unit decides what a column is; the channel comment decides which
    thermocouple is which. Gaps are filled by channel order and named in
    `positional` so the caller can report the guess.
    """
    analog = [c for c in columns if c.channel.startswith(_ANALOG_PREFIX)]
    volts = [c.channel for c in analog if c.unit.lower() in _VOLT_TO_MV]
    temps = [c.channel for c in analog if c.unit.lower() in _CELSIUS_UNITS]

    from_comment: dict[str, str] = {}
    for channel, record in channels.items():
        role = _role_from_comment(record.comment)
        if role is None or role in from_comment:
            continue
        pool = volts if role == "cell" else temps
        if channel in pool:
            from_comment[role] = channel

    cell = from_comment.get("cell")
    hot = from_comment.get("hot")
    cold = from_comment.get("cold")
    positional: list[str] = []

    if cell is None and volts:
        cell = volts[0]
        positional.append("cell")
    spare = [t for t in temps if t not in {hot, cold}]
    if hot is None and spare:
        hot = spare.pop(0)
        positional.append("hot")
    if cold is None and spare:
        cold = spare.pop(0)
        positional.append("cold")

    return _Roles(cell=cell, hot=hot, cold=cold, positional=tuple(positional))


def _role_from_comment(comment: str) -> str | None:
    """Map a channel comment to a role, or None when it names none."""
    lowered = comment.lower()
    for keyword, role in _ROLE_KEYWORDS:
        if keyword in lowered:
            return role
    return None


def _build_arrays(
    samples: _Samples,
    columns: tuple[_Column, ...],
    roles: _Roles,
    issues: list[ValidationIssue],
) -> dict[str, np.ndarray]:
    """Assemble the arrays downstream analyzers expect.

    Names match `ThermovoltageSlopeAnalyzer`'s raw-ramp form
    (`t_hot_c`, `t_cold_c`, `voltage_mv`) so a parsed trace is analysable with
    no further wiring.
    """
    units = {c.channel: c.unit.lower() for c in columns}
    arrays: dict[str, np.ndarray] = {"time_s": samples.time_s}

    if roles.cell is None:
        issues.append(
            _issue(
                "voltage_mv",
                Severity.ERROR,
                "No voltage column found. Expected a channel whose unit is V, mV or "
                f"uV; saw {sorted(set(units.values()))}.",
            ),
        )
    else:
        scale = _VOLT_TO_MV[units[roles.cell]]
        arrays["voltage_mv"] = samples.by_channel[roles.cell] * scale

    if roles.hot is not None and roles.cold is not None:
        arrays["t_hot_c"] = samples.by_channel[roles.hot]
        arrays["t_cold_c"] = samples.by_channel[roles.cold]
    else:
        issues.append(
            _issue(
                "t_hot_c",
                Severity.WARNING,
                "Fewer than two thermocouple columns found, so no temperature "
                "difference could be formed. The voltage trace is still usable; the "
                "Seebeck slope is not.",
            ),
        )

    if samples.event is not None and bool(np.any(samples.event != 0.0)):
        arrays["event_mark"] = samples.event
    return arrays


def _report_roles(roles: _Roles, issues: list[ValidationIssue]) -> None:
    """Warn when any channel role was assigned by position rather than comment."""
    if not roles.positional:
        return
    named = ", ".join(roles.positional)
    issues.append(
        _issue(
            "channel_roles",
            Severity.WARNING,
            f"Assigned {named} by channel order, not by channel comment. Comment the "
            "channels 'hot', 'cold' and 'cell' on the instrument so the file records "
            "which electrode was which; a positional guess can silently invert the "
            "sign of S.",
        ),
    )


def _report_time_axis(samples: _Samples, issues: list[ValidationIssue]) -> None:
    """Say what the Time column turned out to be when it was not plain seconds.

    Three of the dialog's four Time Axis Formats are not what this parser wants.
    Two of them are textual and were converted on the way in, which is worth
    recording. The third, Point, writes a bare sample index that is
    indistinguishable from seconds at a one second interval and silently wrong at
    any other, so the suspicion is reported rather than resolved.
    """
    if samples.time_axis == "absolute":
        issues.append(
            _issue(
                "time_s",
                Severity.INFO,
                "The Time column held absolute clock times, so the export used a Time "
                "Axis Format of Absolute Time. Converted to seconds from the first "
                "sample.",
            ),
        )
    elif samples.time_axis == "relative":
        issues.append(
            _issue(
                "time_s",
                Severity.INFO,
                "The Time column held formatted elapsed times, so the export used a "
                "Time Axis Format of Relative Time. Converted to seconds.",
            ),
        )
    elif _looks_like_sample_index(samples.time_s):
        issues.append(
            _issue(
                "time_s",
                Severity.INFO,
                "The Time column is whole numbers stepping by exactly one. That is "
                "either a one second recording interval, in which case all is well, or "
                "a Time Axis Format of Point, which writes the sample index instead of "
                "seconds. If the interval was not one second, every time constant from "
                "this file is wrong by the ratio of the interval; re-export with Time "
                "Axis Format set to Second.",
            ),
        )


def _check_scaling(
    channels: dict[str, _Channel],
    roles: _Roles,
    issues: list[ValidationIssue],
) -> None:
    """Error if a channel in use had the instrument's scaling function enabled.

    With scaling on, the recorded numbers are in whatever unit the operator
    configured, so the bracketed column unit no longer describes them and the
    conversion to millivolts is not trustworthy.
    """
    for role, channel in (("cell", roles.cell), ("hot", roles.hot), ("cold", roles.cold)):
        record = channels.get(channel or "")
        if record is None or record.scaling.lower() in _SCALING_OFF:
            continue
        issues.append(
            _issue(
                "scaling",
                Severity.ERROR,
                f"Channel {record.channel} ({role}) was recorded with scaling "
                f"{record.scaling!r}. The values are not raw, so the declared unit "
                "cannot be trusted. Re-record with scaling Off.",
            ),
        )


# ─── Reporting ──────────────────────────────────────────────────────
def _build_metadata(
    read: _FileRead,
    channels: dict[str, _Channel],
    layout: _Layout,
    samples: _Samples,
    roles: _Roles,
) -> dict[str, Any]:
    """Everything worth keeping that is not a numeric array."""
    metadata: dict[str, Any] = {
        "n_points": int(samples.time_s.size),
        "delimiter": read.delimiter,
        "time_axis_format": samples.time_axis,
        "time_column_could_be_sample_index": _looks_like_sample_index(samples.time_s),
        "channel_roles": {
            "cell": roles.cell,
            "hot": roles.hot,
            "cold": roles.cold,
        },
        "channel_roles_positional": list(roles.positional),
        "column_units": {c.channel: c.unit for c in layout.columns},
    }
    for key in ("file name", "title comment"):
        if key in read.header:
            metadata[key.replace(" ", "_")] = read.header[key]
    raw_time = next((read.header[k] for k in _TRIGGER_TIME_KEYS if k in read.header), None)
    if raw_time is not None:
        metadata["trigger_time_text"] = raw_time
        # The instrument writes no timezone. `measured_at` is stamped UTC to
        # match every other parser here; this flag says the zone was assumed,
        # not read, so a later correction knows it is allowed to change it.
        metadata["trigger_time_zone_assumed_utc"] = True
    if channels:
        metadata["channel_table"] = [
            {
                "channel": c.channel,
                "mode": c.mode,
                "range": c.range_text,
                "comment": c.comment,
                "scaling": c.scaling,
            }
            for c in channels.values()
        ]
    if samples.event is not None:
        marks = np.flatnonzero(samples.event != 0.0)
        metadata["event_sample_indices"] = [int(i) for i in marks]
        metadata["event_times_s"] = [float(samples.time_s[i]) for i in marks]
    return metadata


def _build_features(samples: _Samples, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Scalars worth surfacing on the Measurement itself."""
    features: dict[str, float] = {"n_points": float(samples.time_s.size)}
    time_s = samples.time_s
    if time_s.size >= _MIN_SAMPLES_FOR_INTERVAL:
        features["duration_s"] = float(time_s[-1] - time_s[0])
        features["sampling_interval_s"] = float(np.median(np.diff(time_s)))

    voltage = arrays.get("voltage_mv")
    if voltage is not None and voltage.size and bool(np.any(np.isfinite(voltage))):
        features["voltage_mv_final"] = float(voltage[-1])
        features["voltage_mv_max_abs"] = float(np.nanmax(np.abs(voltage)))

    hot, cold = arrays.get("t_hot_c"), arrays.get("t_cold_c")
    if hot is not None and cold is not None and hot.size:
        delta = hot - cold
        if bool(np.any(np.isfinite(delta))):
            features["delta_t_k_final"] = float(delta[-1])
            features["delta_t_k_median"] = float(np.nanmedian(delta))
    return features


def _issue(field: str, severity: Severity, message: str) -> ValidationIssue:
    """Build a `ValidationIssue` with the current timestamp."""
    return ValidationIssue(
        field=field,
        severity=severity,
        message=message,
        detected_at=utc_now(),
    )
