"""XRD parser for Bruker RAW4 text exports (DIFFRAC.EVA "save as text").

File format
-----------
INI-style sections under a ``;RAW4.00`` magic line, then a comma-separated data
block::

    ;RAW4.00
    [RawHeader]
    Date=06/17/2026
    Time=12:11:21
    [VarInfo]
    Type=SAMPLEID
    Value=Commander Sample ID
    [HardwareConfiguration]
    Anode=Cu
    [RangeHeader]
    ActuallyUsedLambda=1.5418
    Start=3
    Increment=0.0204724
    Steps=587
    Time=192
    [Data]
         Angle,       PSD,
             3,      2875,
       3.02047,      2791,

Two structural traps, both handled here:

* **Keys repeat across sections with different meanings.** ``Time`` is a clock
  time in ``[RawHeader]`` and a per-step counting duration in ``[RangeHeader]``.
  Parsing the file as one flat key/value map silently takes whichever comes
  last, so sections are tracked and keys are read section-scoped.
* **``[VarInfo]`` appears many times**, each block a ``Type``/``Value`` pair
  carrying the sample id, operator, comment and creator. They are collected
  into a mapping rather than overwriting one another.

Data rows carry a trailing comma, so splitting yields an empty final field.

Largest visible d-spacing
-------------------------
``max_d_spacing_nm`` is derived from the START angle via Bragg's law. It answers
a question that is easy to get wrong when a scan is reused for a purpose it was
not planned for: a reflection at larger d than this was never in range, and its
absence from the pattern means nothing. For a layered material whose basal
spacing sits at low angle, that distinction decides whether a "phase absent"
claim is evidence or an artefact of the scan window.

Validation policy
-----------------
Never raises. Problems surface as `ValidationIssue`s:

* no data points                       -> ERROR   (empty or truncated export)
* wavelength missing                   -> WARNING (no d-spacing can be computed)
* non-monotonic 2theta                 -> WARNING (merged or corrupted file)
* point count disagrees with `Steps`   -> WARNING (truncated acquisition)
* timestamp has no timezone            -> INFO    (UTC attached; really local)
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["BrukerRaw4TxtParser", "max_visible_d_spacing_nm"]

_MAGIC = ";RAW4"
_SECTION_RE = re.compile(r"^\[([^\]]+)\]\s*$")
_KEY_RE = re.compile(r"^([^=]+)=(.*)$")

# Sections whose keys we read, scoped so repeated key names cannot collide.
_RAW_HEADER = "RawHeader"
_RANGE_HEADER = "RangeHeader"
_HARDWARE = "HardwareConfiguration"
_VAR_INFO = "VarInfo"
_DATA = "Data"

# A data row is "<angle>, <counts>," — the trailing comma yields a third,
# empty field when split.
_MIN_DATA_FIELDS = 2

_SNIFF_BYTES = 256

# Bragg's law has no solution outside this open interval.
_MAX_TWO_THETA_DEG = 180.0


def max_visible_d_spacing_nm(start_two_theta_deg: float, wavelength_angstrom: float) -> float:
    """Largest d-spacing a scan beginning at `start_two_theta_deg` can reach.

    Bragg's law, evaluated at the start of the scan window. A reflection at
    larger d than this simply was not scanned, so its absence is not evidence
    of a phase being absent.

    Raises:
        ValueError: If the angle is outside (0, 180) or the wavelength is not
            positive.
    """
    if not 0.0 < start_two_theta_deg < _MAX_TWO_THETA_DEG:
        raise ValueError(f"2theta must be in (0, 180) degrees, got {start_two_theta_deg!r}")
    if wavelength_angstrom <= 0:
        raise ValueError(f"wavelength must be positive, got {wavelength_angstrom!r}")
    theta = np.radians(start_two_theta_deg / 2.0)
    return float(wavelength_angstrom / (2.0 * np.sin(theta)) / 10.0)


def _issue(message: str, severity: Severity, field: str = "xrd") -> ValidationIssue:
    return ValidationIssue(field=field, message=message, severity=severity, detected_at=utc_now())


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class BrukerRaw4TxtParser(BaseParser):
    """Bruker DIFFRAC RAW4 text export."""

    name: ClassVar[str] = "bruker-raw4-txt"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.XRD
    supported_extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def can_parse(self, path: Path) -> float:
        """Confidence from the `;RAW4` magic line, which is unambiguous."""
        try:
            head = path.read_bytes()[:_SNIFF_BYTES]
        except OSError:
            return 0.0
        first = head.decode("utf-8", errors="replace").lstrip().splitlines()[:1]
        if first and first[0].strip().upper().startswith(_MAGIC):
            return 1.0
        return 0.0

    @staticmethod
    def _read_sections(
        lines: list[str],
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], list[str]]:
        """Split into section-scoped keys, VarInfo pairs, and raw data lines.

        Returns ``(sections, var_info, data_lines)``. `sections` keeps only the
        FIRST value seen for a key within a section, so a repeated block cannot
        overwrite the one that was read.
        """
        sections: dict[str, dict[str, str]] = {}
        var_info: dict[str, str] = {}
        data_lines: list[str] = []
        current = ""
        pending_type: str | None = None

        for raw in lines:
            line = raw.rstrip("\r")
            section = _SECTION_RE.match(line.strip())
            if section:
                current = section.group(1).strip()
                if current == _VAR_INFO:
                    pending_type = None
                continue
            if current == _DATA:
                data_lines.append(line)
                continue
            key_value = _KEY_RE.match(line.strip())
            if not key_value:
                continue
            key, value = key_value.group(1).strip(), key_value.group(2).strip()
            if current == _VAR_INFO:
                # Blocks are Type=... then Value=...; pair them up.
                if key == "Type":
                    pending_type = value
                elif key == "Value" and pending_type:
                    var_info[pending_type] = value
                    pending_type = None
                continue
            sections.setdefault(current, {}).setdefault(key, value)
        return sections, var_info, data_lines

    def parse(self, path: Path) -> ParsedData:
        """Read one RAW4 text export into 2theta and intensity arrays."""
        issues: list[ValidationIssue] = []
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            return self._empty(f"Could not read the file: {exc}")

        sections, var_info, data_lines = self._read_sections(lines)

        two_theta: list[float] = []
        counts: list[float] = []
        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < _MIN_DATA_FIELDS:
                continue
            angle, intensity = _to_float(parts[0]), _to_float(parts[1])
            if angle is None or intensity is None:
                continue  # the "Angle, PSD," caption row lands here
            two_theta.append(angle)
            counts.append(intensity)

        if not two_theta:
            return self._empty(
                "No data points — the export is empty or was truncated.", extra_issues=issues
            )

        angles = np.asarray(two_theta, dtype=float)
        intensities = np.asarray(counts, dtype=float)

        range_header = sections.get(_RANGE_HEADER, {})
        wavelength = _to_float(range_header.get("ActuallyUsedLambda"))
        metadata: dict[str, Any] = {
            "n_points": int(angles.size),
            "two_theta_start_deg": float(angles.min()),
            "two_theta_end_deg": float(angles.max()),
            "step_deg": _to_float(range_header.get("Increment")),
            "anode": sections.get(_HARDWARE, {}).get("Anode"),
            "scan_mode": range_header.get("ScanMode"),
            "generator_current_ma": _to_float(range_header.get("GeneratorCurrent")),
            "generator_voltage_kv": _to_float(range_header.get("GeneratorVoltage")),
            # Time in [RangeHeader] is a counting duration, NOT a clock time -
            # the clock lives in [RawHeader] under the same key name.
            "range_duration_s": _to_float(range_header.get("Time")),
        }
        if wavelength is not None:
            metadata["wavelength_angstrom"] = wavelength
            metadata["max_d_spacing_nm"] = max_visible_d_spacing_nm(float(angles.min()), wavelength)
        else:
            issues.append(
                _issue(
                    "No wavelength in the export, so no d-spacing can be computed from "
                    "these angles.",
                    Severity.WARNING,
                )
            )
        for key, name in (
            ("SAMPLEID", "sample_id"),
            ("USER", "operator"),
            ("COMMENT", "comment"),
            ("CREATOR", "creator"),
        ):
            if var_info.get(key):
                metadata[name] = var_info[key]

        issues.extend(self._quality_checks(angles, range_header))
        measured_at, timestamp_issue = self._read_timestamp(sections.get(_RAW_HEADER, {}))
        if timestamp_issue:
            issues.append(timestamp_issue)

        return ParsedData(
            technique=Technique.XRD,
            arrays={"two_theta_deg": angles, "intensity": intensities},
            metadata=metadata,
            instrument=f"Bruker ({metadata.get('anode') or 'unknown'} anode)",
            measured_at=measured_at,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )

    @staticmethod
    def _quality_checks(angles: np.ndarray, range_header: dict[str, str]) -> list[ValidationIssue]:
        out: list[ValidationIssue] = []
        if angles.size > 1 and not np.all(np.diff(angles) > 0):
            out.append(
                _issue(
                    "2theta is not strictly increasing — the file may be two scans "
                    "concatenated, or corrupted.",
                    Severity.WARNING,
                )
            )
        declared = _to_float(range_header.get("Steps"))
        if declared is not None and abs(declared - angles.size) > 1:
            out.append(
                _issue(
                    f"The header declares {int(declared)} steps but {angles.size} points are "
                    "present — the acquisition was cut short or the file is truncated.",
                    Severity.WARNING,
                )
            )
        return out

    @staticmethod
    def _read_timestamp(
        raw_header: dict[str, str],
    ) -> tuple[datetime | None, ValidationIssue | None]:
        """Combine the `[RawHeader]` Date and Time, attaching UTC with a caveat."""
        date_text, time_text = raw_header.get("Date"), raw_header.get("Time")
        if not date_text or not time_text:
            return None, None
        for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                stamp = datetime.strptime(f"{date_text} {time_text}", fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
            return stamp, _issue(
                "Bruker timestamps carry no timezone; UTC was attached for consistency. "
                "Treat measured_at as approximate — it is local time at the instrument.",
                Severity.INFO,
                field="measured_at",
            )
        return None, None

    def _empty(
        self,
        message: str,
        *,
        extra_issues: list[ValidationIssue] | None = None,
    ) -> ParsedData:
        issues = list(extra_issues or [])
        issues.append(_issue(message, Severity.ERROR))
        return ParsedData(
            technique=Technique.XRD,
            arrays={},
            metadata={},
            instrument="Bruker",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )
