"""XRD parser for Rigaku Ultima-series `.txt` exports.

File format
-----------
Plain ASCII. Header section first — each line `;Key = Value`. Data section
after the header — each line `<two_theta> <intensity>` separated by
whitespace. No section delimiter; the data section starts at the first
line that doesn't begin with `;`.

Example header lines::

    ;SampleName         = bs3a.raw
    ;KAlpha1            = 1.54056
    ;Target             = Cu
    ;Start              = 5
    ;Finish             = 120
    ;Width              = 0.1

Example data lines::

    5.0000 208
    5.1000 214

Why this is a separate parser from the other XRD formats
--------------------------------------------------------
PANalytical exports XML (`.xrdml`), Rigaku's own ASCII export uses `.ASC`
with a different header style, and `.xy` files have no header at all.
Each gets its own `BaseParser` subclass with its own `can_parse()`
sniffer; the dispatcher (Stage 1C.5) picks whichever scores highest.

Validation policy
-----------------
This parser never raises. Failures are returned via `ValidationIssue`s on
the `ParsedData`:
- No data points at all → `Severity.ERROR`
- Wavelength missing → `Severity.WARNING` (downstream phase ID needs it)
- Non-monotonic 2θ → `Severity.WARNING` (suggests merged/corrupted file)
- Lines that fail to parse as `<float> <float>` → counted, single
  `Severity.WARNING` summarizing the count
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["RigakuXrdTxtParser"]


# Header line: `;Key = Value` with arbitrary whitespace around `=`.
# Key allows letters, digits, dots — Rigaku uses both alphanumeric keys
# (`SampleName`) and dotted ones (`SlitName0`, `KAlpha1`).
_HEADER_RE = re.compile(r"^;([\w.]+)\s*=\s*(.*)$")

# Header keys we treat as "this is definitely a Rigaku XRD txt file" for
# `can_parse` confidence scoring. SampleName + KAlpha1 together are highly
# specific to Rigaku's Ultima series — no other lab software in our
# experience emits exactly this combination.
_RIGAKU_SIGNATURE_KEYS = (b";SampleName", b";KAlpha1")

# How many bytes to read from the file head when sniffing for the format.
# Rigaku headers are typically ~700 bytes; 4 KB is generous and still
# negligible to read.
_SNIFF_BYTES = 4096

# Data rows must have at least this many whitespace-separated columns.
# Rigaku exports two: 2θ and intensity. Anything fewer is malformed.
_MIN_COLUMNS = 2


class RigakuXrdTxtParser(BaseParser):
    """Parser for Rigaku Ultima-series XRD `.txt` exports."""

    name: ClassVar[str] = "rigaku-xrd-txt"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.XRD
    supported_extensions: ClassVar[tuple[str, ...]] = (".txt",)

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """Return confidence in [0, 1] that this is a Rigaku XRD `.txt` file.

        Reads only the first ~4 KB of the file. Cheap enough to call on
        every `.txt` in a folder during dispatch.

        Confidence levels:
            1.0 — both `;SampleName` and `;KAlpha1` present (definitive)
            0.7 — one of the two present (very likely)
            0.0 — wrong extension or no Rigaku signature found

        Never raises: unreadable files return 0.0 silently — dispatch
        will simply pick a different parser or flag the file as unknown.
        """
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("rb") as fh:
                head = fh.read(_SNIFF_BYTES)
        except OSError:
            return 0.0

        hits = sum(1 for sig in _RIGAKU_SIGNATURE_KEYS if sig in head)
        if hits == len(_RIGAKU_SIGNATURE_KEYS):
            return 1.0
        if hits > 0:
            return 0.7
        return 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a Rigaku XRD `.txt` file into a `ParsedData`.

        Returns a `ParsedData` even on errors — failures are reported as
        `ValidationIssue`s on the result, not exceptions.
        """
        issues: list[ValidationIssue] = []
        raw_headers, two_theta, intensity, malformed_rows = self._read_file(path, issues)

        if malformed_rows > 0:
            issues.append(
                ValidationIssue(
                    field="data",
                    severity=Severity.WARNING,
                    message=f"{malformed_rows} data row(s) could not be parsed; skipped.",
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

        # Monotonic-2θ check. Diffractometers always scan in one direction
        # within a single file; non-monotonic data means the file was
        # merged from multiple scans or the data section is corrupted.
        if len(two_theta) > 1 and not _is_monotonic_increasing(two_theta):
            issues.append(
                ValidationIssue(
                    field="two_theta",
                    severity=Severity.WARNING,
                    message="2θ values are not monotonically increasing.",
                    detected_at=utc_now(),
                ),
            )

        metadata, extra_issues = self._build_metadata(raw_headers, len(two_theta))
        issues.extend(extra_issues)

        # Empty arrays only when nothing parsed. Same-length is automatic
        # because we append (theta, intensity) as pairs.
        arrays: dict[str, np.ndarray] = (
            {
                "two_theta": np.asarray(two_theta, dtype=np.float64),
                "intensity": np.asarray(intensity, dtype=np.float64),
            }
            if two_theta
            else {}
        )

        instrument = raw_headers.get("Gonio") or None

        return ParsedData(
            technique=self.technique,
            arrays=arrays,
            metadata=metadata,
            instrument=instrument,
            measured_at=None,  # Rigaku .txt has no acquisition timestamp.
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )

    # ─── Internals ───────────────────────────────────────────────────
    @staticmethod
    def _read_file(
        path: Path,
        issues: list[ValidationIssue],
    ) -> tuple[dict[str, str], list[float], list[float], int]:
        """Read the file once, separating headers from data rows.

        Args:
            path: File to read.
            issues: List to append fatal-read issues to (e.g. file unreadable).

        Returns:
            (headers, two_theta, intensity, malformed_rows)
        """
        headers: dict[str, str] = {}
        two_theta: list[float] = []
        intensity: list[float] = []
        malformed = 0

        try:
            # `errors="replace"` so a stray non-UTF-8 byte in a header
            # comment doesn't kill the whole parse. The header regex will
            # simply skip lines with replacement chars.
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith(";"):
                        match = _HEADER_RE.match(line)
                        if match:
                            headers[match.group(1)] = match.group(2).strip()
                        continue
                    parts = line.split()
                    if len(parts) < _MIN_COLUMNS:
                        malformed += 1
                        continue
                    # Parse both columns BEFORE appending either — otherwise a
                    # malformed second column leaves an orphan two_theta and
                    # the (two_theta, intensity) arrays end up with mismatched
                    # lengths, which fails ParsedData's same-length invariant.
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

        return headers, two_theta, intensity, malformed

    @staticmethod
    def _build_metadata(
        raw: dict[str, str],
        n_points: int,
    ) -> tuple[dict[str, Any], list[ValidationIssue]]:
        """Pluck the headers we care about into a JSON-safe metadata dict.

        Returns the metadata dict and a list of ValidationIssues for any
        problems detected (missing optional fields, non-numeric values
        where numbers were expected).
        """
        issues: list[ValidationIssue] = []

        def to_float(key: str, *, required: bool = False) -> float | None:
            """Coerce `raw[key]` to float, recording an issue on failure."""
            val = raw.get(key)
            if val is None or val == "":
                if required:
                    issues.append(
                        ValidationIssue(
                            field=key,
                            severity=Severity.WARNING,
                            message=f"Missing required header `{key}`.",
                            detected_at=utc_now(),
                        ),
                    )
                return None
            try:
                return float(val)
            except ValueError:
                issues.append(
                    ValidationIssue(
                        field=key,
                        severity=Severity.WARNING,
                        message=f"Header `{key}` is not numeric: {val!r}",
                        detected_at=utc_now(),
                    ),
                )
                return None

        # Slit names. Rigaku numbers them SlitName0..SlitName7. We collect
        # whichever are present and store them as a list, in numeric order.
        slits = [raw[k] for k in sorted(raw) if k.startswith("SlitName") and raw[k]]

        metadata: dict[str, Any] = {
            "sample_name": raw.get("SampleName") or None,
            "target": raw.get("Target") or None,
            "wavelength_ka1": to_float("KAlpha1", required=True),
            "wavelength_ka2": to_float("KAlpha2"),
            # Rigaku misspells "KBeta" as "KBata" in their export. Look for
            # both for forward-compatibility if they ever fix it.
            "wavelength_kbeta": to_float("KBeta") or to_float("KBata"),
            "voltage_kv": to_float("KV"),
            "current_ma": to_float("mA"),
            "axis": raw.get("AxisName") or None,
            "monochromator": raw.get("IncidentMonochro") or None,
            "attachment": raw.get("Attachment") or None,
            "counter": raw.get("Counter") or None,
            "scan_start_deg": to_float("Start"),
            "scan_finish_deg": to_float("Finish"),
            "scan_step_deg": to_float("Width"),
            "scan_speed": raw.get("Speed") or None,
            "x_unit": raw.get("Xunit") or None,
            "y_unit": raw.get("Yunit") or None,
            "n_points": n_points,
            "slits": slits,
            "operator": raw.get("Operator") or None,
            "memo": raw.get("Memo") or None,
            "comment": raw.get("Comment") or None,
        }

        return metadata, issues


def _is_monotonic_increasing(values: list[float]) -> bool:
    """True if `values` is non-strictly increasing (allows duplicates)."""
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))
