"""Hall-effect parser for `.xls` workbooks.

File format
-----------
Old binary Excel (`.xls`). The user's Hall instrument exports a single
sheet with a large block of empty rows at the top, then a header row of
column names, then exactly one data row of values — a single
(temperature, sheet resistance, mobility, ...) measurement at one
temperature point.

Example header (truncated)::

    Temperature (°C) | Sheet resistance (Ω) | Resistivity (Ω cm) |
    Conductivity (1/(Ω cm)) | CCC Bulk (1/cm³) | Mobility (cm²/(V s)) | ...

Hall measurements at a single temperature have no "spectrum" to plot —
every column is a scalar. We therefore put EVERY value into `metadata`
(a JSON-safe flat dict) and leave `arrays` empty. This matches the
metadata-only `ParsedData` shape used for `.tif` files in Stage 5.

Multiple-temperature sweeps would have multiple data rows; we'd then
return arrays keyed by column. Stage 1C only sees single-row exports;
multi-row support is deferred until we encounter a fixture that has
them.

Validation policy: see `xrd_rigaku_txt.py` — same contract.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, ClassVar

import xlrd

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["HallXlsParser"]

# Words/phrases in the header row that indicate a Hall-effect file.
# Any one is sufficient — these are very specific to Hall measurements.
_HALL_HEADER_KEYWORDS = (
    "Hall coefficient",
    "Sheet resistance",
    "Mobility",
    "Resistivity",
)

# Regex for normalizing column names into JSON-safe keys.
# Strips units in parentheses, lowercases, replaces non-word chars with _.
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


class HallXlsParser(BaseParser):
    """Parser for Hall-effect `.xls` workbooks (single-temperature export)."""

    name: ClassVar[str] = "hall-xls"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.HALL
    supported_extensions: ClassVar[tuple[str, ...]] = (".xls",)

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """Confidence based on header row containing Hall-specific keywords."""
        if not self._extension_matches(path):
            return 0.0
        try:
            wb = xlrd.open_workbook(str(path))
        except Exception:
            return 0.0
        try:
            sheet = wb.sheet_by_index(0)
            for r in range(sheet.nrows):
                row_text = " ".join(str(sheet.cell_value(r, c)) for c in range(sheet.ncols))
                if any(kw in row_text for kw in _HALL_HEADER_KEYWORDS):
                    return 1.0
            return 0.0
        finally:
            # xlrd.Book has no close(); release_resources is the safe analog.
            wb.release_resources()

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a Hall `.xls` into a `ParsedData` (metadata-only)."""
        issues: list[ValidationIssue] = []
        metadata: dict[str, Any] = {}

        try:
            wb = xlrd.open_workbook(str(path))
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    field="file",
                    severity=Severity.ERROR,
                    message=f"Could not open workbook: {exc}",
                    detected_at=utc_now(),
                ),
            )
            return self._empty_result(issues)

        try:
            sheet = wb.sheet_by_index(0)
            header_row, data_row = _find_header_and_data(sheet)
            if header_row is None or data_row is None:
                issues.append(
                    ValidationIssue(
                        field="data",
                        severity=Severity.ERROR,
                        message=(
                            "Could not locate Hall header row (looking for "
                            "Hall/Mobility/Resistivity keywords) or its data row."
                        ),
                        detected_at=utc_now(),
                    ),
                )
            else:
                metadata = _build_metadata(sheet, header_row, data_row, issues)
        finally:
            wb.release_resources()

        return ParsedData(
            technique=self.technique,
            arrays={},  # Hall is scalar at a single temperature.
            metadata=metadata,
            instrument=None,
            measured_at=None,
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
            instrument=None,
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )


# ─── Module-level helpers ───────────────────────────────────────────
def _find_header_and_data(sheet: Any) -> tuple[int | None, int | None]:
    """Locate the header row (row containing Hall keywords) and the next data row."""
    header_row = None
    for r in range(sheet.nrows):
        row_text = " ".join(str(sheet.cell_value(r, c)) for c in range(sheet.ncols))
        if any(kw in row_text for kw in _HALL_HEADER_KEYWORDS):
            header_row = r
            break
    if header_row is None:
        return None, None

    # Data row is the first non-empty row AFTER the header.
    for r in range(header_row + 1, sheet.nrows):
        if any(str(sheet.cell_value(r, c)).strip() for c in range(sheet.ncols)):
            return header_row, r
    return header_row, None


def _build_metadata(
    sheet: Any,
    header_row: int,
    data_row: int,
    issues: list[ValidationIssue],
) -> dict[str, Any]:
    """Pair each column header with its data-row value into a flat dict."""
    metadata: dict[str, Any] = {}
    for c in range(sheet.ncols):
        raw_label = str(sheet.cell_value(header_row, c)).strip()
        if not raw_label:
            continue
        key = _normalize_column_name(raw_label)
        if not key:
            continue
        value = sheet.cell_value(data_row, c)
        # Coerce to JSON-safe scalar. xlrd returns floats for most numerics
        # but can return empty string for blank cells.
        coerced = _coerce_cell(value)
        if coerced is _UNCOERCED:
            issues.append(
                ValidationIssue(
                    field=key,
                    severity=Severity.WARNING,
                    message=f"Column {raw_label!r} value not JSON-coercible: {value!r}",
                    detected_at=utc_now(),
                ),
            )
            continue
        metadata[key] = coerced
        # Stash the original (with units) under a sister key for display.
        metadata[f"{key}__label"] = raw_label
    return metadata


def _normalize_column_name(label: str) -> str:
    """Lowercase and snake-case a column header.

    Examples:
        "Temperature (°C)"  → "temperature_c"
        "Sheet resistance (Ω)" → "sheet_resistance"
        "Mobility (cm²/(V s))" → "mobility_cm2_v_s"
    """
    cleaned = _NORMALIZE_RE.sub("_", label.lower()).strip("_")
    return cleaned


# Sentinel for "we couldn't coerce this cell to a JSON-safe value".
# Using a unique object so `is` comparison never collides with real data.
_UNCOERCED = object()


def _coerce_cell(value: Any) -> Any:
    """Coerce an xlrd cell value into a JSON-safe scalar, or `_UNCOERCED`."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        # xlrd returns NaN/Inf as floats; those aren't JSON-safe.
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return _UNCOERCED
