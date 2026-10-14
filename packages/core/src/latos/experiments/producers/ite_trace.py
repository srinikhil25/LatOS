"""Record producer for thermal-cycle traces of an ionic-liquid mixture on cloth.

Contract ``ite-trace/1``.

One specimen is one prepared piece of cloth on one mounting. It is measured by
switching the temperature drive on and off several times while a logger records
both electrode temperatures and the cell voltage. This producer turns a
manifest of such specimens, plus the logger workbooks it names, into a
`RecordTable` with **exactly one row per specimen**, however many cycles the
recording holds and whether or not the specimen could be evaluated.

What the contract fixes
-----------------------
The contract revision is the scientific transformation from a recording to a
reported value. It is not the software version: the same contract can be
re-implemented, and a new release that changes nothing below keeps the
revision. Any change to what follows needs ``ite-trace/2``.

*The response.* ``apparent_seebeck_cycle_mv_k`` is the signed mean, over the
accepted cycles of one specimen, of the whole-cycle least-squares slope of the
recorded voltage against the temperature difference. `analysis.thermovoltage.
cycles` defines the slope, the windows and the reduction, and says what the
quantity is not.

*The sign.* The wiring is declared per specimen and reaches the arithmetic
through `analysis.thermovoltage.slope.resolve_polarity`, the same function the
rest of the package uses. A specimen whose wiring is not declared is refused:
the sign of its response would be a guess. No absolute value is taken anywhere
in this module. A magnitude, if the experiment wants one, is the objective's
business, several steps downstream.

*The uncertainty.* ``apparent_seebeck_cycle_sem_mv_k`` is the standard error of
the mean of the accepted cycles. It is a within-specimen, within-session Type A
component and always a lower bound. With one accepted cycle, or with identical
ones, none is reported. The least-squares standard error of a single cycle's
slope is kept as a diagnostic and is never reported as an uncertainty.

*Composition.* ``aliquat_mass_fraction`` is
m(Aliquat 336TG) / (m(Aliquat 336TG) + m([HMIM][TFSI])), from the weighed
masses in the manifest, the components matched by exact name. A pure component
is declared as such and is 0 or 1 by identity. The value is never taken from a
file name or a nominal label.

*Cycle boundaries.* In order of trust: a declaration in the manifest; the
logger's event marks; an estimate from the temperature trace. The estimate is a
fallback and always warns. A declaration that replaces marks keeps both sets
and warns with the differences. Marks that disagree with the temperature trace
by more than `SIGNATURE_LIMIT_S` are refused, since a mark pressed at the wrong
moment moves the window; a declaration that disagrees is used as written, with
a warning, because the person who ran it has the last word.

*Test cycles.* A manifest may declare that the first cycles of a recording were
deliberate test runs. They are excluded by that declaration, not by any
statistic, and they are kept in the record with their purpose stated.

*Reversed-lead runs.* A specimen may name a second recording of the same
mounting with the voltage leads swapped. It is analysed and kept as a
diagnostic that the sign follows the wiring. It is never a specimen, never a
row, and never an observation.

*Declared quality notes.* A manifest may attach ``qc_warnings`` to a specimen:
sentences written by the researcher about that specimen, such as drift from
cycle to cycle. Each is carried as a WARNING on the row and kept in the record.
It is a statement, not a rule: it changes no value, and no specimen is rejected
or down-weighted because of it.

What is refused, and how
------------------------
A refusal is a row with no value and an ERROR issue saying why, so the specimen
stays visible and the reason travels to the frozen record. Refused: an
unreadable or missing recording; a gap in the samples; a channel that is not in
the file, or a unit that cannot be converted; a missing or uninterpretable
wiring declaration; event marks that are missing, repeated, out of order or
even in number; boundaries that leave the recording or do not increase; a cycle
in which the temperature difference does not rise while the drive is on and
fall while it is off; a composition that is missing, ambiguous or names an
unknown component.

A manifest that is itself malformed (an unknown key, a wrong type, a repeated
sample id or file) is not a specimen problem and raises `ManifestError` for the
whole file. An unknown key is refused rather than ignored because a misspelt
``test_cycles`` would otherwise silently put a test run into the data.

What is not judged here
-----------------------
The drive programme is recorded, not judged. The producer reports the drive
voltage and the dosing as declared, and the cycle count and the shortest and
longest on and off phase as measured. The drive voltage is the operator's
set-point: nothing in this producer measures the voltage delivered, and the
row says so. Whether those are acceptable is the
experiment definition's decision, taken when the row is built. Reporting the
shortest and the longest lets a definition hold every phase to a tolerance
without this module knowing the nominal duration.
"""

from __future__ import annotations

import math
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from latos.analysis.thermovoltage.cycles import (
    AUTOMATIC_SETTINGS,
    BOUNDARY_AUTOMATIC,
    BOUNDARY_DECLARED,
    BOUNDARY_EVENT_MARKS,
    CycleBoundaryError,
    CycleFit,
    automatic_boundaries,
    declared_boundaries,
    event_boundaries,
    fit_cycle,
    reduce_cycles,
    windows_from_boundaries,
)
from latos.analysis.thermovoltage.slope import (
    POLARITY_V_PLUS_ON_COLD,
    POLARITY_V_PLUS_ON_HOT,
    resolve_polarity,
)
from latos.core.enums import Severity
from latos.core.exceptions import CorruptedFileError
from latos.core.models import ValidationIssue, utc_now
from latos.experiments.records import Record, RecordTable, Scalar
from latos.ingestion.hashing import hash_bytes
from latos.ingestion.hioki_logger_xlsx import LoggerChannel, LoggerTrace, read_hioki_xlsx

__all__ = [
    "COLUMNS",
    "COMPONENT_ALIQUAT",
    "COMPONENT_HMIM_TFSI",
    "CONTRACT",
    "PROTOCOL",
    "SIGNATURE_LIMIT_S",
    "Manifest",
    "ManifestError",
    "load_manifest",
    "produce_table",
]

CONTRACT = "ite-trace/1"

# The two liquids, matched by these exact names wherever a manifest names one.
COMPONENT_ALIQUAT = "Aliquat 336TG"
COMPONENT_HMIM_TFSI = "[HMIM][TFSI]"

# ─── Columns ────────────────────────────────────────────────────────
SAMPLE_ID = "sample_id"
ALIQUAT_MASS_FRACTION = "aliquat_mass_fraction"
APPARENT_SEEBECK = "apparent_seebeck_cycle_mv_k"
APPARENT_SEEBECK_SEM = "apparent_seebeck_cycle_sem_mv_k"
DRIVE_VOLTAGE = "drive_voltage_v"
ON_DURATION_MIN = "on_duration_min_s"
ON_DURATION_MAX = "on_duration_max_s"
OFF_DURATION_MIN = "off_duration_min_s"
OFF_DURATION_MAX = "off_duration_max_s"
CYCLES_PER_SPECIMEN = "cycles_per_specimen"
REMOUNTED = "remounted_between_cycles"
DROPS_APPLIED = "drops_applied"
ELECTRODE_MATERIAL = "electrode_material"
PEAK_DELTA_T = "peak_delta_t_k"
MEAN_TEMPERATURE = "mean_temperature_c"
RELATIVE_HUMIDITY = "relative_humidity_pct"
AMBIENT_TEMPERATURE = "ambient_temperature_c"
POLARITY_CONVENTION = "polarity_convention"
BOUNDARY_SOURCE = "boundary_source"
DOSING_METHOD = "dosing_method"
NOMINAL_LABEL = "nominal_label"
MIXTURE_BATCH_ID = "mixture_batch_id"
TRACE_FILE = "trace_file"
TRACE_SHA256 = "trace_sha256"

COLUMNS: tuple[str, ...] = (
    SAMPLE_ID,
    ALIQUAT_MASS_FRACTION,
    APPARENT_SEEBECK,
    APPARENT_SEEBECK_SEM,
    DRIVE_VOLTAGE,
    ON_DURATION_MIN,
    ON_DURATION_MAX,
    OFF_DURATION_MIN,
    OFF_DURATION_MAX,
    CYCLES_PER_SPECIMEN,
    REMOUNTED,
    DROPS_APPLIED,
    ELECTRODE_MATERIAL,
    PEAK_DELTA_T,
    MEAN_TEMPERATURE,
    RELATIVE_HUMIDITY,
    AMBIENT_TEMPERATURE,
    POLARITY_CONVENTION,
    BOUNDARY_SOURCE,
    DOSING_METHOD,
    NOMINAL_LABEL,
    MIXTURE_BATCH_ID,
    TRACE_FILE,
    TRACE_SHA256,
)

# ─── Rules of the contract ──────────────────────────────────────────
# How far an event mark may sit from the switch seen in the temperature trace.
# Provisional: wide enough for a slow press, narrow enough to catch a mark
# pressed during the wrong phase. Its use is recorded with every specimen.
SIGNATURE_LIMIT_S = 10.0

# A step between samples this many times the usual interval is a hole in the
# recording, and a window that spans a hole is not the declared window.
_GAP_FACTOR = 5.0

_CELSIUS_UNITS = ("°C", "C")
_TO_MILLIVOLT = {"V": 1000.0, "mV": 1.0, "µV": 0.001, "μV": 0.001, "uV": 0.001}
_CHANNEL_ROLES = ("cold", "hot", "voltage")

_DECIMALS = 6

# Cycle values closer together than this, relative to their mean, differ only
# by floating-point representation and count as identical.
_IDENTICAL_REL = 1e-9

# What the drive-voltage column is. It is the operator's set-point, so a
# definition can require the declared value; it cannot claim a measured one.
_DRIVE_BASIS = "the drive set-point declared in the manifest; not a measured voltage"

PROTOCOL: dict[str, Any] = {
    "contract": CONTRACT,
    "response": {
        "name": APPARENT_SEEBECK,
        "unit": "mV/K",
        "per_cycle": (
            "ordinary least-squares slope of the signed recorded voltage (mV) against the "
            "temperature difference T_hot - T_cold (K), over every sample of one whole "
            "cycle; free intercept; no time term; no delay correction"
        ),
        "per_specimen": "signed mean of the accepted cycles",
        "is": "an operational response of this drive programme, heating and cooling together",
        "is_not": "a steady-state Seebeck coefficient",
    },
    "sign": (
        "S is the slope of V_cold - V_hot against T_hot - T_cold. With the positive lead on "
        "the cold electrode the recorded voltage is used as it is; with it on the hot "
        "electrode the recorded voltage is negated before fitting. A positive S means the "
        "cold electrode is at the higher potential. No absolute value is taken."
    ),
    "windows": (
        "cycle k is [on_k, on_k+1); the last cycle is [on_n, end); heating branch "
        "[on_k, off_k); cooling branch [off_k, end of the cycle)"
    ),
    "boundaries": {
        "order_of_trust": [BOUNDARY_DECLARED, BOUNDARY_EVENT_MARKS, BOUNDARY_AUTOMATIC],
        "event_marks": (
            "numbered 1, 2, 3, ... in order; odd marks are switch-ons, even marks are "
            "switch-offs, the last mark is the end of the run"
        ),
        "declared": (
            "a declaration replaces the marks; both sets are kept and the specimen warns "
            "with the differences"
        ),
        "automatic": (
            "fallback only, always with a warning: a switch-on is counted when the smoothed "
            "temperature difference climbs past the upper level and a switch-off when it "
            "falls past the lower one; each is timed where the trace left the preceding "
            "trough or crest; the last cycle is closed at the end of the recording"
        ),
        "automatic_settings": dict(AUTOMATIC_SETTINGS),
        "thermal_signature_limit_s": SIGNATURE_LIMIT_S,
        "thermal_signature_rule": (
            "event marks further than the limit from the switch seen in the temperature "
            "trace are refused; a declaration that far off is used as written, with a "
            "warning. The limit is provisional."
        ),
    },
    "test_cycles": (
        "leading cycles declared in the manifest as test runs are excluded by that "
        "declaration, not by any statistic; they are kept in this record"
    ),
    "uncertainty": {
        "reported": "standard deviation of the accepted cycle values (k - 1) / sqrt(k)",
        "scope": "specimen",
        "evaluation": "Type A",
        "is_lower_bound": True,
        "covers": "repeatability of the cycles of one mounting within one session",
        "does_not_cover": (
            "a new mounting of the same mixture (dosing, contact, cloth), day-to-day "
            "changes, and the instrument"
        ),
        "assumption": (
            "the cycles are treated as independent and exchangeable, which has not been "
            "shown for consecutive cycles on one cell"
        ),
        "none_when": "one accepted cycle, or identical accepted cycles",
        "never_used": "the least-squares standard error of a single cycle's slope",
    },
    "diagnostics_never_in_the_value": [
        "heating-branch slope",
        "cooling-branch slope",
        "intercept",
        "least-squares standard error of the slope",
        "R squared",
    ],
    "units": (
        "voltage is converted to mV from the unit in the recorded column header; "
        "temperatures are read in degrees Celsius; channels are named, never assumed by "
        "position"
    ),
    "composition": (
        f"{ALIQUAT_MASS_FRACTION} = m({COMPONENT_ALIQUAT}) / (m({COMPONENT_ALIQUAT}) + "
        f"m({COMPONENT_HMIM_TFSI})), from weighed masses, components matched by exact name; "
        "a pure component is 0 or 1 by identity; never from a file name or nominal label"
    ),
    "reversed_lead_run": (
        "a second recording of the same mounting with the voltage leads swapped; analysed "
        "as a diagnostic; never a specimen and never an observation"
    ),
    "sample_gap_factor": _GAP_FACTOR,
    "bench": [
        "substrate: cotton cloth, 2 cm x 1 cm",
        "stack: plastic wrap, cotton, electrodes, glass slide",
        "dosing: drops from a disposable pipette; the delivered mass is not measured",
        "mixing: 10-15 s; homogeneity is assumed, not verified",
        "one mounting per specimen, not remounted between cycles",
        "electrode material: recorded per specimen when it is known",
    ],
}


# ─── The manifest ───────────────────────────────────────────────────
_RUN_KEYS = frozenset(
    {"trace_file", "sheet", "polarity_convention", "channels", "test_cycles"}
    | {"declared_boundaries_s"}
)
_DECLARED_KEYS = frozenset(
    {
        DRIVE_VOLTAGE,
        DROPS_APPLIED,
        REMOUNTED,
        ELECTRODE_MATERIAL,
        RELATIVE_HUMIDITY,
        AMBIENT_TEMPERATURE,
        DOSING_METHOD,
        MIXTURE_BATCH_ID,
        "notes",
    }
)
_QC_WARNINGS = "qc_warnings"
_SPECIMEN_KEYS = (
    _RUN_KEYS
    | _DECLARED_KEYS
    | {SAMPLE_ID, "masses_g", "pure_component", NOMINAL_LABEL, "reverse", _QC_WARNINGS}
)
# The wiring is not a default: it is declared for each recording, so a new row
# cannot inherit a sign nobody wrote down for it.
_DEFAULT_KEYS = _DECLARED_KEYS | {"channels"}
_BOUNDARY_KEYS = frozenset({"on", "off", "end", "note"})
_TOP_KEYS = frozenset({"contract", "defaults", "specimen"})

_NUMBER_KEYS = (DRIVE_VOLTAGE, RELATIVE_HUMIDITY, AMBIENT_TEMPERATURE)
_TEXT_KEYS = (
    "sheet",
    "polarity_convention",
    "pure_component",
    ELECTRODE_MATERIAL,
    DOSING_METHOD,
    NOMINAL_LABEL,
    MIXTURE_BATCH_ID,
    "notes",
)
_YES_NO = ("yes", "no")


class ManifestError(ValueError):
    """A manifest cannot be used as written.

    Attributes:
        problems: Everything wrong with it, one sentence each.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        listed = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"The specimen manifest is not usable:\n{listed}")


@dataclass(frozen=True, slots=True)
class Manifest:
    """A specimen manifest, read and checked.

    Attributes:
        path: Where it was read from. Recordings are looked for beside it.
        sha256: SHA-256 of the manifest's bytes.
        rows: One mapping per specimen, with the defaults already applied.
    """

    path: Path
    sha256: str
    rows: tuple[Mapping[str, Any], ...]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _boundary_problems(value: Any, where: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{where}: declared_boundaries_s must be a table with on, off and end"]
    problems = [
        f"{where}: declared_boundaries_s has an unknown key {key!r}"
        for key in sorted(set(value) - _BOUNDARY_KEYS)
    ]
    for key in ("on", "off"):
        times = value.get(key)
        if not isinstance(times, list) or not times or not all(_is_number(t) for t in times):
            problems.append(f"{where}: declared_boundaries_s.{key} must be a list of times in s")
    if not _is_number(value.get("end")):
        problems.append(f"{where}: declared_boundaries_s.end must be a time in s")
    if "note" in value and not isinstance(value["note"], str):
        problems.append(f"{where}: declared_boundaries_s.note must be text")
    return problems


def _channel_problems(value: Any, where: str) -> list[str]:
    if not isinstance(value, dict) or set(value) != set(_CHANNEL_ROLES):
        return [f"{where}: channels must name exactly cold, hot and voltage"]
    names = list(value.values())
    if not all(isinstance(name, str) and name.strip() for name in names):
        return [f"{where}: every channel must be given by its recorded name"]
    if len(set(names)) != len(names):
        return [f"{where}: channels names one recorded channel for two roles"]
    return []


def _row_problems(row: Mapping[str, Any], where: str, allowed: frozenset[str]) -> list[str]:
    """Everything structurally wrong with one table of the manifest."""
    problems = [f"{where}: unknown key {key!r}" for key in sorted(set(row) - allowed)]
    for key in _TEXT_KEYS:
        if key in row and not isinstance(row[key], str):
            problems.append(f"{where}: {key} must be text")
    for key in _NUMBER_KEYS:
        if key in row and not _is_number(row[key]):
            problems.append(f"{where}: {key} must be a number")
    for key in ("test_cycles", DROPS_APPLIED):
        if key in row and not _is_count(row[key]):
            problems.append(f"{where}: {key} must be a whole number, zero or more")
    if REMOUNTED in row and not (isinstance(row[REMOUNTED], bool) or row[REMOUNTED] in _YES_NO):
        problems.append(f"{where}: {REMOUNTED} must be 'yes' or 'no'")
    if "channels" in row:
        problems += _channel_problems(row["channels"], where)
    if "declared_boundaries_s" in row:
        problems += _boundary_problems(row["declared_boundaries_s"], where)
    return problems + _specimen_only_problems(row, where)


def _specimen_only_problems(row: Mapping[str, Any], where: str) -> list[str]:
    """Structure problems in the keys only a specimen table may carry."""
    problems = []
    if "masses_g" in row:
        masses = row["masses_g"]
        if not isinstance(masses, dict) or not all(_is_number(m) for m in masses.values()):
            problems.append(f"{where}: masses_g must be a table of component name to grams")
    if _QC_WARNINGS in row:
        notes = row[_QC_WARNINGS]
        if not isinstance(notes, list) or not all(isinstance(n, str) and n.strip() for n in notes):
            problems.append(f"{where}: {_QC_WARNINGS} must be a list of sentences")
    return problems


def _run_file_problems(row: Mapping[str, Any], where: str) -> list[str]:
    name = row.get("trace_file")
    if not isinstance(name, str) or not name.strip():
        return [f"{where}: trace_file is required"]
    return []


def load_manifest(path: Path) -> Manifest:
    """Read a specimen manifest and check its structure.

    Scientific completeness is not checked here: a specimen with no wiring
    declaration is a valid manifest row that the producer will refuse, so that
    the refusal is recorded against that specimen and the others still run.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ManifestError: If it is not valid TOML, was written for another
            contract, or has a structural problem. Every problem is listed.
    """
    path = Path(path)
    raw = path.read_bytes()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ManifestError([f"{path.name} is not valid TOML: {exc}"]) from exc

    problems = [f"unknown top-level key {key!r}" for key in sorted(set(document) - _TOP_KEYS)]
    if document.get("contract") != CONTRACT:
        problems.append(
            f"contract is {document.get('contract')!r}; this producer reads {CONTRACT!r}"
        )
    defaults = document.get("defaults", {})
    if not isinstance(defaults, dict):
        problems.append("defaults must be a table")
        defaults = {}
    problems += _row_problems(defaults, "defaults", _DEFAULT_KEYS)

    specimens = document.get("specimen")
    if not isinstance(specimens, list) or not specimens:
        problems.append("at least one [[specimen]] table is required")
        specimens = []

    rows: list[dict[str, Any]] = []
    for number, specimen in enumerate(specimens, start=1):
        if not isinstance(specimen, dict):
            problems.append(f"specimen {number}: must be a table")
            continue
        sample_id = specimen.get(SAMPLE_ID)
        named = isinstance(sample_id, str) and bool(sample_id.strip())
        where = f"specimen {number}" + (f" ({sample_id})" if named else "")
        if not named:
            problems.append(f"{where}: {SAMPLE_ID} is required")
        problems += _row_problems(specimen, where, _SPECIMEN_KEYS)
        problems += _run_file_problems(specimen, where)
        reverse = specimen.get("reverse")
        if reverse is not None:
            if isinstance(reverse, dict):
                problems += _row_problems(reverse, f"{where}, reverse", _RUN_KEYS)
                problems += _run_file_problems(reverse, f"{where}, reverse")
            else:
                problems.append(f"{where}: reverse must be a table")
        rows.append({**defaults, **specimen})

    ids = Counter(row[SAMPLE_ID].strip() for row in rows if isinstance(row.get(SAMPLE_ID), str))
    problems += [
        f"{SAMPLE_ID} {name!r} is used by {n} specimens" for name, n in ids.items() if n > 1
    ]
    files = Counter(_declared_files(rows))
    # One recording is one run of one mounting. Named twice, it would be counted
    # twice; named once as a specimen and once as a reversed-lead run, the
    # diagnostic would become an observation.
    problems += [f"trace file {name!r} is named {n} times" for name, n in files.items() if n > 1]

    if problems:
        raise ManifestError(problems)
    return Manifest(path=path, sha256=hash_bytes(raw), rows=tuple(rows))


def _declared_files(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        reverse = row.get("reverse")
        for run in (row, reverse if isinstance(reverse, dict) else {}):
            name = run.get("trace_file")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


# ─── One recording ──────────────────────────────────────────────────
class _RefusedError(Exception):
    """A recording cannot give a value. The message says why."""


@dataclass(slots=True)
class _Run:
    """What was established about one recording, however far the analysis got."""

    detail: dict[str, Any] = field(default_factory=dict)
    accepted: list[CycleFit] = field(default_factory=list)
    sign: int = 1
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _rounded(value: float) -> float:
    return round(float(value), _DECIMALS)


def _finite(value: float) -> float | None:
    """`value`, or None when it is not finite: JSON has no NaN."""
    number = float(value)
    return number if math.isfinite(number) else None


def _analyse_run(run: Mapping[str, Any], channels: Any, folder: Path) -> _Run:
    out = _Run()
    try:
        _read_and_fit(out, run, channels, folder)
    except _RefusedError as refusal:
        out.accepted.clear()
        out.errors.append(str(refusal))
    return out


def _read_and_fit(out: _Run, run: Mapping[str, Any], channels: Any, folder: Path) -> None:
    name = str(run["trace_file"]).strip()
    try:
        trace = read_hioki_xlsx(folder / name, sheet=run.get("sheet"))
    except FileNotFoundError:
        raise _RefusedError(f"the trace file {name!r} was not found beside the manifest") from None
    except CorruptedFileError as exc:
        raise _RefusedError(str(exc)) from exc
    except OSError as exc:
        # A folder under that name, a file another program holds open, a disk
        # error. The recording gives no value, and that is said for this
        # specimen so that the others are still read.
        reason = exc.strerror or str(exc) or type(exc).__name__
        raise _RefusedError(f"the trace file {name!r} could not be read: {reason}") from exc

    out.detail["source"] = {
        "file": name,
        "sha256": trace.sha256,
        "sheet": trace.sheet,
        "preamble": list(trace.preamble),
        "first_timestamp": trace.timestamps[0],
        "last_timestamp": trace.timestamps[-1],
        "n_samples": int(trace.time_s.size),
        "sample_interval_s": _rounded(trace.sample_interval_s),
        "recorded_channels": [{"name": c.name, "unit": c.unit} for c in trace.channels],
        "has_event_column": trace.has_event_column,
        "event_marks": [{"number": e.number, "time_s": _rounded(e.time_s)} for e in trace.events],
    }

    time = trace.time_s
    steps = np.diff(time)
    widest = int(np.argmax(steps))
    if steps[widest] > _GAP_FACTOR * trace.sample_interval_s:
        raise _RefusedError(
            f"{name}: the recording has a gap of {steps[widest]:.3g} s after {time[widest]:.1f} s, "
            f"against a sample interval of {trace.sample_interval_s:.3g} s"
        )

    cold, hot, voltage_mv = _channel_arrays(out, trace, channels, name)
    sign = _polarity_sign(out, run, name)
    boundaries = _boundaries(out, run, trace, hot - cold, name)
    try:
        windows = windows_from_boundaries(boundaries)
    except CycleBoundaryError as exc:
        raise _RefusedError(f"{name}: {exc}") from exc

    test_cycles = int(run.get("test_cycles", 0))
    if test_cycles >= len(windows):
        raise _RefusedError(
            f"{name}: {test_cycles} test cycle(s) are declared, but the recording holds "
            f"{len(windows)} cycle(s), so no measurement cycle is left"
        )

    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    for window in windows:
        try:
            fit = fit_cycle(time, cold, hot, voltage_mv, window, polarity_sign=sign)
        except CycleBoundaryError as exc:
            raise _RefusedError(f"{name}: {exc}") from exc
        is_test = window.index <= test_cycles
        entries.append(_cycle_entry(fit, is_test=is_test))
        if not is_test:
            problems += fit.problems
            out.accepted.append(fit)
    out.detail["cycles"] = entries
    if problems:
        raise _RefusedError(f"{name}: " + "; ".join(problems))


def _channel_arrays(
    out: _Run, trace: LoggerTrace, channels: Any, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cold, hot and voltage (mV) series, by the declared channel names."""
    recorded = ", ".join(c.name for c in trace.channels)
    if not isinstance(channels, dict):
        raise _RefusedError(
            f"{name}: no channel map is declared, so nothing says which recorded channel is "
            f"the cold thermocouple, the hot thermocouple and the cell voltage. Recorded "
            f"channels: {recorded}"
        )
    picked: dict[str, LoggerChannel] = {}
    for role in _CHANNEL_ROLES:
        try:
            picked[role] = trace.channel(str(channels[role]))
        except KeyError as exc:
            raise _RefusedError(str(exc.args[0])) from None
    for role in ("cold", "hot"):
        channel = picked[role]
        if channel.unit not in _CELSIUS_UNITS:
            raise _RefusedError(
                f"{name}: the {role} channel {channel.name!r} is recorded in {channel.unit!r}, "
                "not in degrees Celsius"
            )
    voltage = picked["voltage"]
    factor = _TO_MILLIVOLT.get(voltage.unit)
    if factor is None:
        raise _RefusedError(
            f"{name}: the voltage channel {voltage.name!r} is recorded in {voltage.unit!r}, "
            f"which is not a voltage unit this producer converts ({', '.join(_TO_MILLIVOLT)})"
        )
    out.detail["channels"] = {
        "cold": {"name": picked["cold"].name, "unit": picked["cold"].unit},
        "hot": {"name": picked["hot"].name, "unit": picked["hot"].unit},
        "voltage": {"name": voltage.name, "unit": voltage.unit, "factor_to_mv": factor},
    }
    return picked["cold"].values, picked["hot"].values, voltage.values * factor


def _polarity_sign(out: _Run, run: Mapping[str, Any], name: str) -> int:
    recorded = run.get("polarity_convention")
    resolution = resolve_polarity(recorded)
    if resolution.problem is not None:
        raise _RefusedError(f"{name}: {resolution.problem}")
    if resolution.source != "recorded":
        # `resolve_polarity` would assume the usual wiring here. A recording
        # cannot carry its own wiring, so for this producer a blank is a refusal.
        raise _RefusedError(
            f"{name}: polarity_convention is not recorded, so the sign of the response is "
            f"not known. Record it as {POLARITY_V_PLUS_ON_COLD!r} or {POLARITY_V_PLUS_ON_HOT!r}"
        )
    out.sign = resolution.sign
    out.detail["polarity"] = {
        "recorded": recorded,
        "convention": resolution.convention,
        "sign": resolution.sign,
    }
    return resolution.sign


def _boundaries(
    out: _Run, run: Mapping[str, Any], trace: LoggerTrace, delta_t: np.ndarray, name: str
) -> list[float]:
    """The run's switching times, from the most trusted source available."""
    detail: dict[str, Any] = {}
    out.detail["boundaries"] = detail
    declared = run.get("declared_boundaries_s")
    marks = [float(e.time_s) for e in trace.events]

    if declared is not None:
        source = BOUNDARY_DECLARED
        try:
            boundaries = declared_boundaries(declared["on"], declared["off"], declared["end"])
        except CycleBoundaryError as exc:
            raise _RefusedError(f"{name}: {exc}") from exc
        if declared.get("note"):
            detail["declared_note"] = declared["note"]
        if marks:
            detail["replaced_event_marks_s"] = [_rounded(t) for t in marks]
            out.warnings.append(_override_warning(name, boundaries, marks))
    elif marks:
        source = BOUNDARY_EVENT_MARKS
        try:
            boundaries = event_boundaries([e.number for e in trace.events], marks)
        except CycleBoundaryError as exc:
            found = "; ".join(f"mark {e.number} at {e.time_s:g} s" for e in trace.events)
            raise _RefusedError(f"{name}: {exc}. Marks found: {found}") from exc
    else:
        source = BOUNDARY_AUTOMATIC
        try:
            estimate = automatic_boundaries(trace.time_s, delta_t)
        except CycleBoundaryError as exc:
            raise _RefusedError(
                f"{name}: no boundaries are declared, the recording has no event marks, and "
                f"they could not be estimated from the temperature trace: {exc}"
            ) from exc
        boundaries = list(estimate.boundaries)
        detail["automatic"] = {
            "settings": dict(AUTOMATIC_SETTINGS),
            "end_is_recording_end": estimate.end_is_recording_end,
            "swing_k": _rounded(estimate.swing_k),
        }
        out.warnings.append(
            f"{name}: the cycle boundaries were estimated from the temperature trace, "
            "because none are declared and the recording has no event marks. This is a "
            "fallback: its thresholds were checked against one campaign's recordings only, "
            "and it takes the end of the recording as the end of the last cycle. Declare "
            "the boundaries in the manifest to replace it."
        )

    detail["source"] = source
    detail["times_s"] = [_rounded(t) for t in boundaries]
    if source != BOUNDARY_AUTOMATIC:
        _thermal_signature_check(out, detail, boundaries, trace, delta_t, name=name)
    return boundaries


def _override_warning(name: str, declared: Sequence[float], marks: Sequence[float]) -> str:
    opening = f"{name}: declared boundaries are used in place of the recording's event marks"
    if len(declared) != len(marks):
        return f"{opening}; {len(declared)} are declared and {len(marks)} were marked."
    listed = ", ".join(f"{d - m:+.1f}" for d, m in zip(declared, marks, strict=True))
    return f"{opening}. Declared minus marked, in s: {listed}."


def _thermal_signature_check(
    out: _Run,
    detail: dict[str, Any],
    boundaries: Sequence[float],
    trace: LoggerTrace,
    delta_t: np.ndarray,
    *,
    name: str,
) -> None:
    """Compare stated switching times with what the temperature trace shows."""
    source = detail["source"]
    stated = list(boundaries[:-1])  # the end of a run is not a switch
    check: dict[str, Any] = {"limit_s": SIGNATURE_LIMIT_S, "switches_stated": len(stated)}
    detail["thermal_signature_check"] = check
    try:
        estimate = automatic_boundaries(trace.time_s, delta_t)
    except CycleBoundaryError as exc:
        check["status"] = "not available"
        check["reason"] = str(exc)
        out.warnings.append(
            f"{name}: the {source} boundaries could not be compared with the temperature "
            f"trace: {exc}"
        )
        return
    seen = list(estimate.boundaries[:-1] if estimate.end_is_recording_end else estimate.boundaries)
    check["switches_seen"] = len(seen)

    problem: str | None = None
    if len(seen) != len(stated):
        problem = (
            f"{name}: the {source} boundaries give {len(stated)} switch(es), but the "
            f"temperature trace shows {len(seen)}"
        )
    else:
        differences = [s - e for s, e in zip(stated, seen, strict=True)]
        check["stated_minus_seen_s"] = [_rounded(d) for d in differences]
        worst = max(abs(d) for d in differences)
        if worst > SIGNATURE_LIMIT_S:
            problem = (
                f"{name}: the {source} boundaries differ from the switches seen in the "
                f"temperature trace by up to {worst:.1f} s, over the {SIGNATURE_LIMIT_S:g} s limit"
            )
    check["status"] = "agrees" if problem is None else "disagrees"
    if problem is None:
        return
    if source == BOUNDARY_EVENT_MARKS:
        raise _RefusedError(
            f"{problem}. If the marks are right, declare the boundaries in the manifest."
        )
    out.warnings.append(f"{problem}. The declaration is used as written.")


def _cycle_entry(fit: CycleFit, *, is_test: bool) -> dict[str, Any]:
    window = fit.window
    entry: dict[str, Any] = {
        "index": window.index,
        "purpose": "test_run" if is_test else "measurement",
        "included_in_specimen_value": not is_test and not fit.problems,
        "on_s": _rounded(window.on_s),
        "off_s": _rounded(window.off_s),
        "end_s": _rounded(window.end_s),
        "on_duration_s": _rounded(window.on_duration_s),
        "off_duration_s": _rounded(window.off_duration_s),
        "n_samples": fit.n_samples,
        "slope_mv_k": _finite(fit.slope_mv_k),
        "intercept_mv": _finite(fit.intercept_mv),
        "on_branch_slope_mv_k": _finite(fit.on_slope_mv_k),
        "off_branch_slope_mv_k": _finite(fit.off_slope_mv_k),
        "slope_ols_stderr_mv_k": _finite(fit.slope_stderr_mv_k),
        "r_squared": _finite(fit.r_squared),
        "delta_t_min_k": _rounded(fit.delta_t_min_k),
        "delta_t_max_k": _rounded(fit.delta_t_max_k),
        "mean_temperature_c": _rounded(fit.mean_temperature),
        "problems": list(fit.problems),
    }
    if is_test:
        entry["exclusion"] = (
            "declared in the manifest as a test run: excluded by protocol, not as an outlier"
        )
    return entry


# ─── One specimen ───────────────────────────────────────────────────
def _issue(column: str, severity: Severity, message: str) -> ValidationIssue:
    return ValidationIssue(field=column, severity=severity, message=message, detected_at=utc_now())


def _text(value: Any) -> str | None:
    """A declared text value, or None when it is blank."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _masses_problem(masses: Mapping[str, Any]) -> str | None:
    """Why the declared masses cannot give a fraction, or None when they can."""
    if set(masses) != {COMPONENT_ALIQUAT, COMPONENT_HMIM_TFSI}:
        found = ", ".join(repr(key) for key in sorted(masses))
        return (
            f"masses_g must give exactly {COMPONENT_ALIQUAT!r} and {COMPONENT_HMIM_TFSI!r}, "
            f"matched by exact name; found: {found}"
        )
    if min(masses.values()) < 0 or sum(masses.values()) <= 0:
        return "masses_g must be zero or more, and not both zero"
    return None


def _composition_problem(masses: Any, pure: Any) -> str | None:
    """Why the declared composition cannot be used, or None when it can."""
    components = (COMPONENT_ALIQUAT, COMPONENT_HMIM_TFSI)
    if masses is not None and pure is not None:
        return "both masses_g and pure_component are declared; give one of them"
    if isinstance(masses, dict):
        return _masses_problem(masses)
    if pure is None:
        return (
            "no composition is declared. Give the weighed masses_g of both components, or "
            "pure_component. The file name and the nominal label are never used"
        )
    if pure not in components:
        return (
            f"pure_component {pure!r} is not one of the two components, which are matched "
            f"by exact name: {COMPONENT_ALIQUAT!r}, {COMPONENT_HMIM_TFSI!r}"
        )
    return None


def _composition(row: Mapping[str, Any]) -> tuple[float | None, dict[str, Any], str, str | None]:
    """The design value, its record, its provenance, and why it is missing."""
    masses = row.get("masses_g")
    pure = row.get("pure_component")
    detail: dict[str, Any] = {
        "variable": ALIQUAT_MASS_FRACTION,
        "masses_g": dict(masses) if isinstance(masses, dict) else None,
        "pure_component": pure,
        "value": None,
    }
    problem = _composition_problem(masses, pure)
    if problem is not None:
        return None, detail, "", problem

    if isinstance(masses, dict):
        aliquat = float(masses[COMPONENT_ALIQUAT])
        other = float(masses[COMPONENT_HMIM_TFSI])
        value = aliquat / (aliquat + other)
        detail["basis"] = "weighed masses"
        provenance = (
            f"from the weighed masses in the manifest: {aliquat:g} g {COMPONENT_ALIQUAT}, "
            f"{other:g} g {COMPONENT_HMIM_TFSI}"
        )
    else:
        value = 1.0 if pure == COMPONENT_ALIQUAT else 0.0
        detail["basis"] = "pure component, by identity"
        provenance = f"pure {pure}, by identity"
    detail["value"] = value
    return value, detail, provenance, None


def _yes_no(value: Any) -> str | None:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value if value in _YES_NO else None


def _reverse_diagnostic(
    reverse: Mapping[str, Any],
    row: Mapping[str, Any],
    folder: Path,
    forward: _Run,
) -> tuple[dict[str, Any], list[str]]:
    """Analyse the reversed-lead run of a specimen. Never a row of its own."""
    name = str(reverse["trace_file"]).strip()
    run = _analyse_run(reverse, reverse.get("channels", row.get("channels")), folder)
    detail = dict(run.detail)
    detail["role"] = (
        "diagnostic: the same mounting recorded again with the voltage leads swapped. "
        "Not a specimen and not an observation."
    )
    warnings = [f"reversed-lead run: {w}" for w in run.warnings]
    if run.errors:
        detail["errors"] = list(run.errors)
        warnings.append(
            "the reversed-lead run could not be analysed, so it shows nothing about the "
            "sign: " + "; ".join(run.errors)
        )
        return detail, warnings

    oriented = float(np.mean([fit.slope_mv_k for fit in run.accepted]))
    # Orienting multiplies by the sign, and the sign is its own inverse.
    recorded = oriented * run.sign
    detail["mean_oriented_slope_mv_k"] = oriented
    detail["mean_recorded_slope_mv_k"] = recorded
    if forward.accepted:
        forward_oriented = float(np.mean([fit.slope_mv_k for fit in forward.accepted]))
        forward_recorded = forward_oriented * forward.sign
        reverses = recorded * forward_recorded < 0
        detail["forward_mean_recorded_slope_mv_k"] = forward_recorded
        detail["recorded_sign_reverses"] = reverses
        detail["oriented_minus_forward_mv_k"] = oriented - forward_oriented
        if not reverses:
            warnings.append(
                f"swapping the leads did not reverse the recorded sign ({name}: "
                f"{recorded:+.4g} mV/K recorded, against {forward_recorded:+.4g} mV/K), so "
                "the recorded voltage is not following the wiring"
            )
    return detail, warnings


@dataclass(slots=True)
class _Columns:
    """The parts of one row, gathered as they are worked out."""

    values: dict[str, Scalar] = field(default_factory=lambda: dict.fromkeys(COLUMNS))
    provenance: dict[str, str] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)


def _measured_columns(
    columns: _Columns, run: _Run, row: Mapping[str, Any], sample_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill the columns that come from the accepted cycles.

    Returns:
        The reduction as it goes into the record, and the measured part of the
        drive programme.
    """
    fits = run.accepted
    values, provenance = columns.values, columns.provenance
    file_name = str(row["trace_file"]).strip()
    boundary_source = run.detail["boundaries"]["source"]
    test_cycles = int(row.get("test_cycles", 0))

    reduction = reduce_cycles([fit.slope_mv_k for fit in fits])
    # Identical cycles are not evidence of perfect repeatability: they are one
    # number computed several times.
    identical = reduction.sd is not None and reduction.sd <= _IDENTICAL_REL * abs(reduction.mean)
    sem = None if identical else reduction.sem
    on = [_rounded(fit.window.on_duration_s) for fit in fits]
    off = [_rounded(fit.window.off_duration_s) for fit in fits]

    values[APPARENT_SEEBECK] = reduction.mean
    values[APPARENT_SEEBECK_SEM] = sem
    values[ON_DURATION_MIN], values[ON_DURATION_MAX] = min(on), max(on)
    values[OFF_DURATION_MIN], values[OFF_DURATION_MAX] = min(off), max(off)
    values[CYCLES_PER_SPECIMEN] = reduction.k
    values[PEAK_DELTA_T] = _rounded(max(fit.delta_t_max_k for fit in fits))
    values[MEAN_TEMPERATURE] = _rounded(float(np.mean([fit.mean_temperature for fit in fits])))

    indices = ", ".join(str(fit.window.index) for fit in fits)
    excluded = (
        f"; the first {test_cycles} cycle(s) were declared test runs and are excluded by protocol"
        if test_cycles
        else ""
    )
    provenance[APPARENT_SEEBECK] = (
        f"signed mean of {reduction.k} whole-cycle least-squares slope(s), cycle(s) {indices} "
        f"of {file_name}; boundaries: {boundary_source}{excluded}"
    )
    if sem is not None:
        provenance[APPARENT_SEEBECK_SEM] = (
            f"standard deviation of the {reduction.k} accepted cycle values divided by "
            f"sqrt({reduction.k}), {reduction.dof} degrees of freedom: a within-specimen, "
            "within-session Type A component and a lower bound; consecutive cycles have not "
            "been shown independent or exchangeable"
        )
    else:
        why = (
            "one accepted cycle says nothing about repeatability"
            if reduction.k == 1
            else "the accepted cycles are identical"
        )
        columns.issues.append(
            _issue(
                APPARENT_SEEBECK_SEM,
                Severity.WARNING,
                f"{sample_id}: {why}, so no uncertainty is reported",
            )
        )
    for column in (ON_DURATION_MIN, ON_DURATION_MAX, OFF_DURATION_MIN, OFF_DURATION_MAX):
        provenance[column] = (
            f"measured from the cycle boundaries ({boundary_source}), over the accepted cycles"
        )
    provenance[CYCLES_PER_SPECIMEN] = "counted from the cycle boundaries, test cycles excluded"
    provenance[PEAK_DELTA_T] = "largest T_hot - T_cold over the accepted cycles"
    provenance[MEAN_TEMPERATURE] = "mean electrode temperature over the accepted cycles"

    reduction_detail = {
        "cycles_used": [fit.window.index for fit in fits],
        "cycle_values_mv_k": [fit.slope_mv_k for fit in fits],
        "k": reduction.k,
        "mean_mv_k": reduction.mean,
        "sd_mv_k": reduction.sd,
        "sem_mv_k": reduction.sem,
        "degrees_of_freedom": reduction.dof,
        "uncertainty_reported_mv_k": sem,
    }
    programme = {CYCLES_PER_SPECIMEN: reduction.k, "on_durations_s": on, "off_durations_s": off}
    return reduction_detail, programme


def _declared_columns(columns: _Columns, row: Mapping[str, Any]) -> None:
    """Fill the columns the manifest declares, exactly as declared."""
    values = columns.values
    for column in _NUMBER_KEYS:
        declared = row.get(column)
        values[column] = float(declared) if declared is not None else None
    values[DROPS_APPLIED] = row.get(DROPS_APPLIED)
    values[REMOUNTED] = _yes_no(row.get(REMOUNTED))
    for column in (ELECTRODE_MATERIAL, DOSING_METHOD, NOMINAL_LABEL, MIXTURE_BATCH_ID):
        values[column] = _text(row.get(column))
    for column in (*_NUMBER_KEYS, DROPS_APPLIED, REMOUNTED, ELECTRODE_MATERIAL, DOSING_METHOD):
        if values[column] is not None:
            columns.provenance[column] = "declared in the manifest"
    if values[DRIVE_VOLTAGE] is not None:
        # The operator's set-point. Nothing here measured the voltage delivered.
        columns.provenance[DRIVE_VOLTAGE] = _DRIVE_BASIS
    for column in (NOMINAL_LABEL, MIXTURE_BATCH_ID):
        if values[column] is not None:
            columns.provenance[column] = "declared in the manifest; never used in a calculation"


def _specimen_record(row: Mapping[str, Any], folder: Path, manifest: Manifest) -> Record:
    sample_id = str(row[SAMPLE_ID]).strip()
    columns = _Columns()
    values = columns.values

    fraction, composition, fraction_provenance, fraction_problem = _composition(row)
    values[SAMPLE_ID] = sample_id
    values[ALIQUAT_MASS_FRACTION] = fraction
    if fraction_problem is None:
        columns.provenance[ALIQUAT_MASS_FRACTION] = fraction_provenance
    else:
        columns.issues.append(
            _issue(ALIQUAT_MASS_FRACTION, Severity.ERROR, f"{sample_id}: {fraction_problem}")
        )

    run = _analyse_run(row, row.get("channels"), folder)
    columns.issues += [_issue(APPARENT_SEEBECK, Severity.ERROR, m) for m in run.errors]
    columns.issues += [_issue(APPARENT_SEEBECK, Severity.WARNING, m) for m in run.warnings]

    _declared_columns(columns, row)
    controls: dict[str, Any] = {
        DRIVE_VOLTAGE: values[DRIVE_VOLTAGE],
        DROPS_APPLIED: values[DROPS_APPLIED],
        REMOUNTED: values[REMOUNTED],
        "basis": {
            DRIVE_VOLTAGE: _DRIVE_BASIS,
            DROPS_APPLIED: "declared in the manifest",
            REMOUNTED: "declared in the manifest",
            "cycle count and phase durations": "measured from the cycle boundaries",
        },
        "judged_by": "the CONTROL variables of the experiment definition, when the row is built",
    }
    reduction_detail: dict[str, Any] | None = None
    if run.accepted:
        reduction_detail, programme = _measured_columns(columns, run, row, sample_id)
        controls.update(programme)

    # A declared quality note travels with the specimen as a warning. It is the
    # researcher's statement and changes no value and excludes nothing.
    columns.issues += [
        _issue(_QC_WARNINGS, Severity.WARNING, f"{sample_id}: declared QC warning: {note.strip()}")
        for note in row.get(_QC_WARNINGS, ())
    ]

    values[POLARITY_CONVENTION] = run.detail.get("polarity", {}).get("convention")
    values[BOUNDARY_SOURCE] = run.detail.get("boundaries", {}).get("source")
    values[TRACE_FILE] = str(row["trace_file"]).strip()
    values[TRACE_SHA256] = run.detail.get("source", {}).get("sha256")

    detail: dict[str, Any] = {
        "contract": CONTRACT,
        "sample_id": sample_id,
        "composition": composition,
        **run.detail,
        "reduction": reduction_detail,
        "controls_recorded": controls,
        "manifest": {
            "file": manifest.path.name,
            "sha256": manifest.sha256,
            "row_with_defaults": dict(row),
        },
        "protocol": PROTOCOL,
    }

    reverse = row.get("reverse")
    if isinstance(reverse, dict):
        diagnostic, warnings = _reverse_diagnostic(reverse, row, folder, run)
        detail["reverse_diagnostic"] = diagnostic
        columns.issues += [
            _issue(APPARENT_SEEBECK, Severity.WARNING, f"{sample_id}: {message}")
            for message in warnings
        ]

    return Record(
        record_id=sample_id,
        values=values,
        provenance=columns.provenance,
        issues=tuple(columns.issues),
        producer_detail=detail,
    )


def produce_table(manifest_path: Path) -> RecordTable:
    """Turn a specimen manifest and its recordings into a record table.

    Args:
        manifest_path: The TOML manifest. Each recording it names is looked for
            in the same folder. Nothing is written anywhere.

    Returns:
        A table under `CONTRACT` with one row per specimen in the manifest, in
        manifest order. A specimen that could not be evaluated still has its
        row, with no value and an ERROR issue saying why.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ManifestError: If the manifest itself is malformed.
    """
    manifest = load_manifest(manifest_path)
    folder = manifest.path.parent
    rows = tuple(_specimen_record(row, folder, manifest) for row in manifest.rows)
    return RecordTable(producer=CONTRACT, columns=COLUMNS, rows=rows)
