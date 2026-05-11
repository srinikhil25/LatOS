"""EDS parser for Bruker `.spx` (TRTSpectrum) files.

File format
-----------
Despite the name, `.spx` is XML — Bruker's "TRTSpectrum" format is just a
WINDOWS-1252 encoded XML document. Structure relevant to us::

    <TRTSpectrum>
      <ClassInstance Type="TRTSpectrum">
        <ClassInstance Type="TRTSpectrumHeader">
          <ChannelCount>4096</ChannelCount>
          <CalibAbs>-4.8013995E-1</CalibAbs>
          <CalibLin>5.0025E-3</CalibLin>
          <Date>2.12.2022</Date>     <!-- DD.MM.YYYY (locale-dependent) -->
          <Time>18:0:24</Time>       <!-- 24-hour, no timezone -->
        </ClassInstance>
        <Channels>0,0,0,...,1,1,2,1,3</Channels>   <!-- comma-separated counts -->
      </ClassInstance>
      ...
    </TRTSpectrum>

Energy axis is computed: `energy[i] = CalibAbs + CalibLin * i` (keV).

Date format note: the Date field is often DD.MM.YYYY (German/European
locale of the instrument software). We try both DD.MM.YYYY and the
DD.M.YYYY single-digit-month variant. The Time field has no timezone, so
even when we successfully parse it we ATTACH UTC tzinfo with a warning —
better than naive (which `ParsedData` rejects), but the user should
know this is approximate.

Validation policy: see `xrd_rigaku_txt.py` — same contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from xml.etree import ElementTree as ET

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["BrukerSpxParser"]

# Sniff bytes — enough to see the <TRTSpectrum> root element.
_SNIFF_BYTES = 512

# Bruker uses Windows-1252 encoding in the XML header. Reading anything
# else risks a UnicodeDecodeError on instrument names with umlauts.
_BRUKER_ENCODING = "windows-1252"

# Date formats we try, in order. DD.MM.YYYY is the locale we've seen;
# zero-padded variants and ISO are accepted as fallbacks.
_DATE_FORMATS = ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d")
# Time formats. Bruker often emits non-zero-padded values (`18:0:24`).
_TIME_FORMATS = ("%H:%M:%S",)


class BrukerSpxParser(BaseParser):
    """Parser for Bruker EDS `.spx` (TRTSpectrum XML) files."""

    name: ClassVar[str] = "bruker-eds-spx"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.EDS
    supported_extensions: ClassVar[tuple[str, ...]] = (".spx",)

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """Confidence 1.0 if file head contains `<TRTSpectrum>`, else 0.0."""
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("rb") as fh:
                head = fh.read(_SNIFF_BYTES)
        except OSError:
            return 0.0
        return 1.0 if b"<TRTSpectrum>" in head else 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a Bruker `.spx` into a `ParsedData`."""
        issues: list[ValidationIssue] = []

        try:
            # Bruker's XML declaration claims WINDOWS-1252 — read with that
            # codec to avoid mojibake on instrument-name accents.
            with path.open("r", encoding=_BRUKER_ENCODING, errors="replace") as fh:
                root = ET.fromstring(fh.read())
        except (ET.ParseError, OSError) as exc:
            issues.append(
                ValidationIssue(
                    field="xml",
                    severity=Severity.ERROR,
                    message=f"Could not parse SPX file: {exc}",
                    detected_at=utc_now(),
                ),
            )
            return self._empty_result(issues)

        intensity = _extract_channels(root, issues)
        calib_abs = _find_float(root, "CalibAbs")
        calib_lin = _find_float(root, "CalibLin")
        channel_count = _find_int(root, "ChannelCount")
        primary_energy_kv = _find_float(root, "PrimaryEnergy")

        # Synthesize energy axis. Without calibration values we fall back
        # to channel indices and warn — the file is still useful for
        # diagnostics but unusable for element ID until recalibrated.
        if intensity and calib_abs is not None and calib_lin is not None:
            energy = np.arange(len(intensity), dtype=np.float64) * calib_lin + calib_abs
        elif intensity:
            issues.append(
                ValidationIssue(
                    field="calibration",
                    severity=Severity.WARNING,
                    message="CalibAbs/CalibLin missing; using channel indices as x-axis.",
                    detected_at=utc_now(),
                ),
            )
            energy = np.arange(len(intensity), dtype=np.float64)
        else:
            energy = np.array([], dtype=np.float64)
            issues.append(
                ValidationIssue(
                    field="data",
                    severity=Severity.ERROR,
                    message="No <Channels> data found.",
                    detected_at=utc_now(),
                ),
            )

        measured_at = _extract_measured_at(root, issues)

        arrays: dict[str, np.ndarray] = (
            {
                "energy_kev": energy,
                "intensity": np.asarray(intensity, dtype=np.float64),
            }
            if intensity
            else {}
        )

        metadata: dict[str, Any] = {
            "calib_abs": calib_abs,
            "calib_lin": calib_lin,
            "channel_count": channel_count,
            "primary_energy_kv": primary_energy_kv,
            "n_points": len(intensity),
        }

        return ParsedData(
            technique=self.technique,
            arrays=arrays,
            metadata=metadata,
            instrument="Bruker EDS (.spx)",
            measured_at=measured_at,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )

    def _empty_result(self, issues: list[ValidationIssue]) -> ParsedData:
        """Build a minimal ParsedData when parsing failed early."""
        return ParsedData(
            technique=self.technique,
            arrays={},
            metadata={},
            instrument="Bruker EDS (.spx)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )


# ─── Module-level helpers ───────────────────────────────────────────
def _find_float(root: ET.Element, tag: str) -> float | None:
    """First descendant with `tag`, parsed as float; None on miss/parse error."""
    for el in root.iter(tag):
        if el.text is None:
            continue
        try:
            return float(el.text.strip())
        except ValueError:
            return None
    return None


def _find_int(root: ET.Element, tag: str) -> int | None:
    """First descendant with `tag`, parsed as int; None on miss/parse error."""
    for el in root.iter(tag):
        if el.text is None:
            continue
        try:
            return int(el.text.strip())
        except ValueError:
            return None
    return None


def _extract_channels(root: ET.Element, issues: list[ValidationIssue]) -> list[float]:
    """Pull comma-separated counts out of the `<Channels>` element."""
    el = root.find(".//Channels")
    if el is None or not el.text:
        return []
    text = el.text.strip().rstrip(",")
    try:
        return [float(v) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        issues.append(
            ValidationIssue(
                field="channels",
                severity=Severity.ERROR,
                message=f"Could not parse <Channels> as comma-separated floats: {exc}",
                detected_at=utc_now(),
            ),
        )
        return []


def _extract_measured_at(
    root: ET.Element,
    issues: list[ValidationIssue],
) -> datetime | None:
    """Combine `<Date>` and `<Time>` into a tz-aware datetime, or None."""
    date_text = _first_text(root, "Date")
    time_text = _first_text(root, "Time")
    if date_text is None or time_text is None:
        return None

    parsed_date: datetime | None = None
    for fmt in _DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date_text, fmt)
            break
        except ValueError:
            continue
    if parsed_date is None:
        issues.append(
            ValidationIssue(
                field="date",
                severity=Severity.WARNING,
                message=f"Could not parse <Date>: {date_text!r}",
                detected_at=utc_now(),
            ),
        )
        return None

    parsed_time: datetime | None = None
    for fmt in _TIME_FORMATS:
        try:
            parsed_time = datetime.strptime(time_text, fmt)
            break
        except ValueError:
            continue
    if parsed_time is None:
        issues.append(
            ValidationIssue(
                field="time",
                severity=Severity.WARNING,
                message=f"Could not parse <Time>: {time_text!r}",
                detected_at=utc_now(),
            ),
        )
        return None

    # Bruker stamps lack a timezone. We attach UTC and warn — better than
    # naive (which `ParsedData` rejects), but the user should know.
    issues.append(
        ValidationIssue(
            field="measured_at",
            severity=Severity.WARNING,
            message=(
                "Bruker SPX timestamps lack timezone info; attached UTC for consistency. "
                "Treat measured_at as approximate — it is local time at the instrument."
            ),
            detected_at=utc_now(),
        ),
    )
    return datetime(
        year=parsed_date.year,
        month=parsed_date.month,
        day=parsed_date.day,
        hour=parsed_time.hour,
        minute=parsed_time.minute,
        second=parsed_time.second,
        tzinfo=UTC,
    )


def _first_text(root: ET.Element, tag: str) -> str | None:
    """First descendant element's stripped text, or None."""
    for el in root.iter(tag):
        if el.text is not None and el.text.strip():
            return el.text.strip()
    return None
