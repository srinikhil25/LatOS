"""UV-DRS parser for raw `.txt` reflectance exports.

Format
------
A small CSV-ish text file: a quoted title line carrying the sample name,
a quoted header, then ``wavelength, reflectance%`` rows::

    "CS-1 - RawData"
    "Wavelength nm.","R%"
    200.00,9.645
    201.00,9.769
    ...

This is *raw* diffuse reflectance in percent — exactly what the Tauc
band-gap analyzer expects (it auto-scales percent → fraction). The
sample name is recovered from the title (``"CS-1 - RawData"`` → ``CS-1``)
and emitted as ``metadata["sample_name"]`` so UV-DRS lines up with the
same sample's other measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["UvDrsTxtParser"]

_SNIFF_LINES = 4
_MIN_COLUMNS = 2
_WAVELENGTH_NM_MIN = 100.0
_WAVELENGTH_NM_MAX = 2600.0


def _sample_from_title(title: str) -> str | None:
    """``'"CS-1 - RawData"'`` → ``'CS-1'`` (drop quotes + the ' - RawData' tag)."""
    cleaned = title.strip().strip('"').strip()
    if not cleaned:
        return None
    head = cleaned.split(" - ")[0].strip()
    return head or cleaned


def _parse_row(line: str) -> tuple[float, float] | None:
    parts = line.replace('"', "").split(",")
    if len(parts) < _MIN_COLUMNS:
        return None
    try:
        wl, refl = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if not (_WAVELENGTH_NM_MIN <= wl <= _WAVELENGTH_NM_MAX):
        return None
    return wl, refl


class UvDrsTxtParser(BaseParser):
    """Parser for raw UV-DRS `.txt` (wavelength, R%) exports."""

    name: ClassVar[str] = "uvdrs-txt"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.UV_DRS
    supported_extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def can_parse(self, path: Path) -> float:
        """1.0 when a header line names Wavelength and R% / reflectance."""
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                head = "".join(next(fh, "") for _ in range(_SNIFF_LINES)).lower()
        except OSError:
            return 0.0
        return 1.0 if "wavelength" in head and ("r%" in head or "reflectance" in head) else 0.0

    def parse(self, path: Path) -> ParsedData:
        """Parse (wavelength, reflectance) rows + the sample name from the title."""
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            return self._empty([
                ValidationIssue(
                    field="file", severity=Severity.ERROR,
                    message=f"Could not read file: {exc}", detected_at=utc_now(),
                ),
            ])

        sample_name = _sample_from_title(lines[0]) if lines else None
        wavelength: list[float] = []
        reflectance: list[float] = []
        for line in lines:
            row = _parse_row(line)
            if row is not None:
                wavelength.append(row[0])
                reflectance.append(row[1])

        if not wavelength:
            return self._empty([
                ValidationIssue(
                    field="data", severity=Severity.ERROR,
                    message="No (wavelength, reflectance) rows found.", detected_at=utc_now(),
                ),
            ])

        metadata: dict[str, Any] = {
            "n_points": len(wavelength),
            "wavelength_min_nm": min(wavelength),
            "wavelength_max_nm": max(wavelength),
        }
        if sample_name:
            metadata["sample_name"] = sample_name
        return ParsedData(
            technique=self.technique,
            arrays={
                "wavelength": np.asarray(wavelength, dtype=np.float64),
                "reflectance": np.asarray(reflectance, dtype=np.float64),
            },
            metadata=metadata,
            instrument="UV-DRS (txt export)",
            measured_at=None,
            issues=(),
            parser_name=self.name,
            parser_version=self.version,
        )

    def _empty(self, issues: list[ValidationIssue]) -> ParsedData:
        return ParsedData(
            technique=self.technique,
            arrays={},
            metadata={},
            instrument="UV-DRS (txt export)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )
