"""XPS parser for CasaXPS-exported `.csv` region files.

File format
-----------
Plain CSV. CasaXPS prepends a few non-data lines that vary by export
options — typically a region count, a blank line, the region label
("C1s", "Cu2p", "Se3d", ...), and another integer. Then the data rows::

    3
    <blank>
    C1s
    1
    296.0658,6568.7500
    295.9658,6590.0000
    ...

We don't try to interpret the header numerically (CasaXPS exports vary
across versions). Instead we skip every leading line until we hit one
that parses as `<float>,<float>` and keep going from there. The element
label (a non-numeric token like "C1s" before the data) is captured as
the region name.

Validation policy: see `xrd_rigaku_txt.py` — same contract.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["CasaXpsCsvParser"]

# How many lines to inspect when sniffing the format. The header section
# is up to ~5 lines, then we want to see at least a few numeric data rows.
_SNIFF_LINES = 20

# Minimum data-row count required for a confident match.
_MIN_DATA_ROWS_FOR_MATCH = 3

# A region label like "C1s", "Cu2p", "O 1s", "Se 3d" — element symbol
# followed by an orbital. Matched loosely: any non-numeric token between
# 2 and 8 chars is treated as a candidate label.
_REGION_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 ]{1,7}$")

# A data row needs at least these many comma-separated columns
# ("binding_energy,intensity").
_MIN_COLUMNS = 2


class CasaXpsCsvParser(BaseParser):
    """Parser for CasaXPS-exported `.csv` region files."""

    name: ClassVar[str] = "casaxps-csv"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.XPS
    supported_extensions: ClassVar[tuple[str, ...]] = (".csv",)

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """`.csv` is generic — confidence relies on file structure.

        Returns 1.0 only if we see >=3 consecutive `<float>,<float>` rows
        after a short header. Otherwise 0.0. We deliberately don't have a
        0.5 tier: any random CSV could have two numeric columns, but
        CasaXPS exports specifically have leading non-numeric lines
        followed by a long run of numeric pairs.
        """
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = [fh.readline() for _ in range(_SNIFF_LINES)]
        except OSError:
            return 0.0

        consecutive = 0
        max_consecutive = 0
        for line in lines:
            if _looks_like_xps_data_row(line):
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        if max_consecutive >= _MIN_DATA_ROWS_FOR_MATCH:
            return 1.0
        return 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a CasaXPS `.csv` into a `ParsedData`."""
        issues: list[ValidationIssue] = []
        binding_energy: list[float] = []
        intensity: list[float] = []
        region_label: str | None = None
        malformed = 0

        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.reader(fh)
                in_data = False
                for row in reader:
                    if not in_data:
                        # Look for the region label in the leading header.
                        label = _extract_region_label(row)
                        if label is not None:
                            region_label = label
                        # Try to parse as data; the first parseable row
                        # flips us into data mode.
                        pair = _parse_data_pair(row)
                        if pair is None:
                            continue
                        in_data = True
                        binding_energy.append(pair[0])
                        intensity.append(pair[1])
                    else:
                        pair = _parse_data_pair(row)
                        if pair is None:
                            malformed += 1
                            continue
                        binding_energy.append(pair[0])
                        intensity.append(pair[1])
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    field="file",
                    severity=Severity.ERROR,
                    message=f"Could not read file: {exc}",
                    detected_at=utc_now(),
                ),
            )

        if malformed > 0:
            issues.append(
                ValidationIssue(
                    field="data",
                    severity=Severity.WARNING,
                    message=f"{malformed} data row(s) could not be parsed; skipped.",
                    detected_at=utc_now(),
                ),
            )

        if not binding_energy:
            issues.append(
                ValidationIssue(
                    field="data",
                    severity=Severity.ERROR,
                    message="No data points found in file.",
                    detected_at=utc_now(),
                ),
            )

        # XPS BE typically scans high-to-low — we don't enforce direction
        # but we do flag if it's neither monotonically increasing nor
        # decreasing (probably corrupted).
        if len(binding_energy) > 1 and not _is_monotonic(binding_energy):
            issues.append(
                ValidationIssue(
                    field="binding_energy",
                    severity=Severity.WARNING,
                    message="Binding energy is not monotonic in either direction.",
                    detected_at=utc_now(),
                ),
            )

        arrays: dict[str, np.ndarray] = (
            {
                "binding_energy": np.asarray(binding_energy, dtype=np.float64),
                "intensity": np.asarray(intensity, dtype=np.float64),
            }
            if binding_energy
            else {}
        )

        metadata: dict[str, Any] = {
            "region": region_label,
            # Filename stem is often the most reliable region label
            # (CasaXPS users typically name files "C 1s.csv", etc.).
            "filename_stem": path.stem,
            "n_points": len(binding_energy),
            "be_start": binding_energy[0] if binding_energy else None,
            "be_end": binding_energy[-1] if binding_energy else None,
        }

        return ParsedData(
            technique=self.technique,
            arrays=arrays,
            metadata=metadata,
            instrument=None,
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )


# ─── Module-level helpers ───────────────────────────────────────────
def _looks_like_xps_data_row(line: str) -> bool:
    """True if `line` looks like an XPS data row (`<float>,<float>`)."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < _MIN_COLUMNS:
        return False
    try:
        float(parts[0])
        float(parts[1])
    except ValueError:
        return False
    return True


def _parse_data_pair(row: list[str]) -> tuple[float, float] | None:
    """Try to parse a CSV row as a (binding_energy, intensity) pair.

    Returns None if the row doesn't look like data — leading header lines,
    empty rows, or anything else non-numeric.
    """
    if len(row) < _MIN_COLUMNS:
        return None
    try:
        return float(row[0].strip()), float(row[1].strip())
    except ValueError:
        return None


def _extract_region_label(row: list[str]) -> str | None:
    """Pull a candidate region label (`C1s`, `Cu2p`, ...) from a header row.

    Returns None if the row doesn't look like a region label.
    """
    if len(row) != 1:
        return None
    cell = row[0].strip()
    if not cell:
        return None
    # Reject pure numbers (those are region count / index).
    try:
        float(cell)
        return None
    except ValueError:
        pass
    if _REGION_LABEL_RE.match(cell):
        return cell
    return None


def _is_monotonic(values: list[float]) -> bool:
    """True if `values` is monotonically non-decreasing OR non-increasing."""
    if len(values) < _MIN_COLUMNS:
        return True
    increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    decreasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    return increasing or decreasing
