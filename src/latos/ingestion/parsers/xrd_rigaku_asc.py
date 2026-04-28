"""XRD parser for Rigaku ASCII (`.ASC`) two-column exports.

File format
-----------
Plain ASCII. NO header. Every line is a (2θ, intensity) pair separated by
whitespace, in scientific notation. Example::

     5.01086644000000E+0000   0.00000000000000E+0000
     5.03286644000000E+0000  -3.99035911560059E+0001
     5.05486644000000E+0000  -7.47570419311523E+0001

Note that intensities can be negative — `.ASC` files often hold smoothed
or background-subtracted curves, not raw counts. We don't reject these,
but we emit a warning if a substantial fraction of intensities is below
zero, since downstream phase-ID assumes counts >= 0.

Why this is a different parser from `RigakuXrdTxtParser`
--------------------------------------------------------
Same vendor, completely different format: `.txt` has a `;Key = Value`
header section; `.ASC` has no header at all. The two share no parsing
code. Each gets its own `BaseParser` subclass.

Validation policy: see `xrd_rigaku_txt.py` — same contract. Never raise,
emit `ValidationIssue`s for malformed rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["RigakuXrdAscParser"]

# How many lines to inspect when sniffing the format. We need enough to
# confirm "two whitespace-separated floats per line, no header".
_SNIFF_LINES = 5

# Threshold above which a "many negative intensities" warning fires.
# Raw XRD counts can never be negative; even on noisy detectors fewer than
# ~5% of values go below zero. 10% means the file has clearly been
# baseline-corrected or smoothed — researchers should know that before
# treating it as raw counts in downstream analysis.
_NEG_INTENSITY_WARN_FRAC = 0.10

# Each data row must contain at least this many whitespace-separated columns.
_MIN_COLUMNS = 2


class RigakuXrdAscParser(BaseParser):
    """Parser for Rigaku ASCII `.ASC` two-column XRD exports."""

    name: ClassVar[str] = "rigaku-xrd-asc"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.XRD
    # `.asc` (lowercase) is the canonical match key — `_extension_matches`
    # is case-insensitive, so `.ASC` files match too.
    supported_extensions: ClassVar[tuple[str, ...]] = (".asc",)

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """Confidence based on the first few rows being parseable as float pairs.

        `.asc` is generic (audio software, AutoCAD, etc. also use it), so
        we cannot rely on the extension alone. Confidence levels:
            1.0 — first 5 rows all parse as `<float> <float>`
            0.7 — most rows parse, a couple don't (noisy file but recognizable)
            0.0 — extension wrong, file unreadable, or rows aren't numeric pairs
        """
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = []
                for _ in range(_SNIFF_LINES):
                    line = fh.readline()
                    if not line:
                        break
                    lines.append(line)
        except OSError:
            return 0.0

        if not lines:
            return 0.0

        ok = 0
        for line in lines:
            parts = line.split()
            if len(parts) < _MIN_COLUMNS:
                continue
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                continue
            ok += 1

        if ok == len(lines):
            return 1.0
        # Intermediate tier requires at least one valid row AND most rows
        # valid. The `ok > 0` guard prevents single-line garbage files
        # (where len-1 == 0) from scoring 0.7.
        if ok > 0 and ok >= len(lines) - 1:
            return 0.7
        return 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a Rigaku `.ASC` file into a `ParsedData`."""
        issues: list[ValidationIssue] = []
        two_theta: list[float] = []
        intensity: list[float] = []
        malformed = 0

        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if len(parts) < _MIN_COLUMNS:
                        malformed += 1
                        continue
                    # Parse both columns BEFORE appending either; see
                    # `xrd_rigaku_txt.py` for why this matters.
                    try:
                        tt = float(parts[0])
                        inten = float(parts[1])
                    except ValueError:
                        malformed += 1
                        continue
                    two_theta.append(tt)
                    intensity.append(inten)
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

        if not two_theta:
            issues.append(
                ValidationIssue(
                    field="data",
                    severity=Severity.ERROR,
                    message="No data points found in file.",
                    detected_at=utc_now(),
                ),
            )

        # Negative-intensity sanity check. Background-subtracted curves
        # commonly have many negatives; a clean raw scan should have ~0.
        if intensity:
            neg_frac = sum(1 for v in intensity if v < 0) / len(intensity)
            if neg_frac >= _NEG_INTENSITY_WARN_FRAC:
                issues.append(
                    ValidationIssue(
                        field="intensity",
                        severity=Severity.WARNING,
                        message=(
                            f"{neg_frac:.0%} of intensity values are negative - "
                            "file likely contains a background-subtracted or smoothed "
                            "curve, not raw counts."
                        ),
                        detected_at=utc_now(),
                    ),
                )

        # Monotonic check. `.ASC` is two columns with no header to confirm
        # ordering, so we infer it.
        if len(two_theta) > 1 and not _is_monotonic_increasing(two_theta):
            issues.append(
                ValidationIssue(
                    field="two_theta",
                    severity=Severity.WARNING,
                    message="2θ values are not monotonically increasing.",
                    detected_at=utc_now(),
                ),
            )

        arrays: dict[str, np.ndarray] = (
            {
                "two_theta": np.asarray(two_theta, dtype=np.float64),
                "intensity": np.asarray(intensity, dtype=np.float64),
            }
            if two_theta
            else {}
        )

        metadata: dict[str, Any] = {
            "n_points": len(two_theta),
            "scan_start_deg": two_theta[0] if two_theta else None,
            "scan_finish_deg": two_theta[-1] if two_theta else None,
        }

        return ParsedData(
            technique=self.technique,
            arrays=arrays,
            metadata=metadata,
            instrument=None,  # No instrument info in the file.
            measured_at=None,  # No timestamp.
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )


def _is_monotonic_increasing(values: list[float]) -> bool:
    """True if `values` is non-strictly increasing (allows duplicates)."""
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))
