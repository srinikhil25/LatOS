"""EDS parser for EMSA/MAS `.emsa` spectral files (JEOL and others).

Format
------
A plain-text, well-specified standard (EMSA/MAS Spectral Data File). A
header of ``#KEYWORD : value`` lines, then ``#SPECTRUM : DATA BEGINS
HERE``, then the data, then ``#ENDOFDATA``::

    #FORMAT      : EMSA/MAS Spectral Data File
    #XUNITS      : Energy (eV)
    #XPERCHAN    : 10.
    #OFFSET      : 0.
    #DATATYPE    : Y        (Y = one intensity per channel; XY = x,y pairs)
    #SPECTRUM    : DATA BEGINS HERE
    0.000,
    12.000,
    ...
    #ENDOFDATA

For ``DATATYPE Y`` the energy axis is synthesized as
``energy[i] = OFFSET + i · XPERCHAN`` and converted to keV (EDS
convention), matching the Bruker `.spx` parser's ``energy_kev`` /
``intensity`` arrays so both EDS formats are interchangeable downstream.
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

__all__ = ["EdsEmsaParser"]

_KEYWORD_RE = re.compile(r"^#\s*([A-Za-z0-9]+)\s*:?\s*(.*)$")
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_EV_TO_KEV = 1e-3


def _to_float(value: str, default: float) -> float:
    m = _NUMBER_RE.search(value)
    return float(m.group()) if m else default


class EdsEmsaParser(BaseParser):
    """Parser for EMSA/MAS `.emsa` EDS spectra."""

    name: ClassVar[str] = "eds-emsa"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.EDS
    supported_extensions: ClassVar[tuple[str, ...]] = (".emsa",)

    def can_parse(self, path: Path) -> float:
        """1.0 when the file opens with an EMSA/MAS format header."""
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                head = fh.read(400).upper()
        except OSError:
            return 0.0
        return 1.0 if "EMSA" in head and "#FORMAT" in head else 0.0

    def parse(self, path: Path) -> ParsedData:
        """Parse the EMSA header + spectrum into energy_kev + intensity."""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return self._empty([
                ValidationIssue(
                    field="file", severity=Severity.ERROR,
                    message=f"Could not read file: {exc}", detected_at=utc_now(),
                ),
            ])

        keywords: dict[str, str] = {}
        data_tokens: list[float] = []
        in_data = False
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = _KEYWORD_RE.match(line)
                key = m.group(1).upper() if m else ""
                if key == "SPECTRUM":
                    in_data = True
                    continue
                if key == "ENDOFDATA":
                    break
                if not in_data and m:
                    keywords[key] = m.group(2).strip()
                continue
            if in_data:
                data_tokens.extend(float(t) for t in _NUMBER_RE.findall(line))

        if not data_tokens:
            return self._empty([
                ValidationIssue(
                    field="data", severity=Severity.ERROR,
                    message="No spectrum data found after #SPECTRUM.", detected_at=utc_now(),
                ),
            ])

        datatype = keywords.get("DATATYPE", "Y").upper()
        if datatype == "XY":
            energy_raw = np.asarray(data_tokens[0::2], dtype=np.float64)
            intensity = np.asarray(data_tokens[1::2], dtype=np.float64)
            n = min(energy_raw.size, intensity.size)
            energy_raw, intensity = energy_raw[:n], intensity[:n]
        else:  # DATATYPE Y — synthesize the energy axis.
            intensity = np.asarray(data_tokens, dtype=np.float64)
            xperchan = _to_float(keywords.get("XPERCHAN", "1"), 1.0)
            offset = _to_float(keywords.get("OFFSET", "0"), 0.0)
            energy_raw = offset + np.arange(intensity.size, dtype=np.float64) * xperchan

        # Normalize the energy axis to keV (EDS convention).
        xunits = keywords.get("XUNITS", "").lower()
        energy_kev = energy_raw if "kev" in xunits else energy_raw * _EV_TO_KEV

        metadata: dict[str, Any] = {
            "title": keywords.get("TITLE", "") or None,
            "beam_kv": _to_float(keywords.get("BEAMKV", ""), 0.0) or None,
            "live_time_s": _to_float(keywords.get("LIVETIME", ""), 0.0) or None,
            "n_points": int(intensity.size),
            "energy_units": "keV",
        }
        return ParsedData(
            technique=self.technique,
            arrays={"energy_kev": energy_kev, "intensity": intensity},
            metadata=metadata,
            instrument="EDS (EMSA/MAS)",
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
            instrument="EDS (EMSA/MAS)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )
