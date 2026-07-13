"""Oscilloscope-waveform parser for Tektronix `.csv` exports (shock / drop test).

File format
-----------
Tektronix TBS-series scopes export a single channel as CSV: a key/value
metadata header, then a ``TIME,CH1`` column header, then the waveform
samples::

    Model,TBS1052C
    Firmware Version,v2015-08-04-...
    <blank>
    Point Format,Y
    Horizontal Units,S
    Sample Interval,0.00016
    Record Length,2000
    Vertical Units,V
    Vertical Scale,0.5
    Label,
    TIME,CH1
    -1.600e-01,0
    -1.598e-01,0
    ...

Used here for a **shock / drop test**: a ball is dropped onto a sample and
a force sensor's voltage is captured. The *peak* of the waveform is the
transmitted shock; converting it to force needs the sensor's voltage->force
calibration, which the experimenter keeps alongside the raw data. This
parser therefore stores the voltage waveform and the peak voltage — the
force calibration is applied downstream.

This is the reason a generic `.csv` sniffer is dangerous: the CasaXPS
parser will happily claim any `.csv` with numeric pairs, so a Tektronix
waveform gets mislabelled as an XPS spectrum. This parser claims the
Tektronix signature with confidence 1.0, and the CasaXPS parser now
rejects it explicitly.

Validation policy: see `xrd_rigaku_txt.py` — same contract.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["ShockTektronixCsvParser", "is_tektronix_scope_header"]

# How many leading lines to inspect when sniffing the format.
_SNIFF_LINES = 20

# A metadata or sample row needs at least this many comma-separated columns.
_MIN_COLUMNS = 2

# Header key -> normalized metadata key. Only keys we care to surface.
_META_KEYS: dict[str, str] = {
    "model": "instrument_model",
    "sample interval": "sample_interval_s",
    "record length": "record_length",
    "horizontal units": "horizontal_units",
    "vertical units": "vertical_units",
    "vertical scale": "vertical_scale",
}


def is_tektronix_scope_header(lines: list[str]) -> bool:
    """True if `lines` (the first ~20 of a file) are a Tektronix scope export.

    Requires all three structural markers, so it never fires on a CasaXPS
    region file or an arbitrary two-column CSV:

    * the first non-empty line starts with ``Model,`` (Tektronix header);
    * a ``TIME,`` column-header line is present;
    * both ``Horizontal Units`` and ``Vertical Units`` keys appear.

    Shared with the CasaXPS parser (which uses it as a negative guard).
    """
    first = next((ln for ln in lines if ln.strip()), "")
    if not first.startswith("Model,"):
        return False
    joined = "".join(lines)
    if "Horizontal Units" not in joined or "Vertical Units" not in joined:
        return False
    return any(ln.strip().upper().startswith("TIME,") for ln in lines)


class ShockTektronixCsvParser(BaseParser):
    """Parser for Tektronix oscilloscope `.csv` waveform exports."""

    name: ClassVar[str] = "shock-tektronix-csv"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.SHOCK
    supported_extensions: ClassVar[tuple[str, ...]] = (".csv",)

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """1.0 for a Tektronix scope CSV, 0.0 otherwise.

        Keyed on the distinctive header (``Model,`` + ``TIME,`` + unit
        keys), so it is unambiguous and never competes with real XPS
        `.csv` files.
        """
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = [fh.readline() for _ in range(_SNIFF_LINES)]
        except OSError:
            return 0.0
        return 1.0 if is_tektronix_scope_header(lines) else 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a Tektronix scope CSV into a `ParsedData` waveform."""
        issues: list[ValidationIssue] = []
        metadata: dict[str, Any] = {}
        time_s: list[float] = []
        voltage: list[float] = []

        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                in_data = False
                for row in csv.reader(fh):
                    if not row:
                        continue
                    if not in_data:
                        # The `TIME,CH1` line marks the start of the samples.
                        if row[0].strip().upper() == "TIME":
                            in_data = True
                            continue
                        _record_metadata(row, metadata)
                        continue
                    pair = _parse_sample(row)
                    if pair is not None:
                        time_s.append(pair[0])
                        voltage.append(pair[1])
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    field="file",
                    severity=Severity.ERROR,
                    message=f"Could not read file: {exc}",
                    detected_at=utc_now(),
                ),
            )

        if not voltage:
            issues.append(
                ValidationIssue(
                    field="data",
                    severity=Severity.ERROR,
                    message="No waveform samples found in file.",
                    detected_at=utc_now(),
                ),
            )

        arrays: dict[str, np.ndarray] = (
            {
                "time_s": np.asarray(time_s, dtype=np.float64),
                "voltage_v": np.asarray(voltage, dtype=np.float64),
            }
            if voltage
            else {}
        )

        features: dict[str, float] = {}
        if voltage:
            varr = np.asarray(voltage, dtype=np.float64)
            peak_idx = int(np.argmax(np.abs(varr)))
            # Peak transmitted shock, as a voltage magnitude. Force [N]
            # follows once the sensor's V->N calibration is applied.
            features["peak_voltage_v"] = float(abs(varr[peak_idx]))
            if time_s:
                features["peak_time_ms"] = float(time_s[peak_idx] * 1000.0)

        metadata["n_points"] = len(voltage)

        return ParsedData(
            technique=self.technique,
            arrays=arrays,
            metadata=metadata,
            instrument=str(metadata.get("instrument_model") or "Tektronix oscilloscope"),
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
            features=features,
        )


# ─── Module-level helpers ───────────────────────────────────────────
def _record_metadata(row: list[str], metadata: dict[str, Any]) -> None:
    """Store a header ``key,value`` row under a normalized metadata key."""
    if len(row) < _MIN_COLUMNS or not row[0].strip():
        return
    key = _META_KEYS.get(row[0].strip().lower())
    if key is None:
        return
    metadata[key] = _coerce(row[1].strip())


def _parse_sample(row: list[str]) -> tuple[float, float] | None:
    """Parse a ``time,value`` sample row, or None if it isn't numeric."""
    if len(row) < _MIN_COLUMNS:
        return None
    try:
        return float(row[0].strip()), float(row[1].strip())
    except ValueError:
        return None


def _coerce(value: str) -> Any:
    """Coerce a header value to float when it looks numeric, else keep the string."""
    try:
        return float(value)
    except ValueError:
        return value
