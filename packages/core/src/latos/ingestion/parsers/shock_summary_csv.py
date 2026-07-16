"""Shock-summary CSV parser: per-composition peak forces from a drop test.

Companion to ``shock_tektronix_csv.py``. That parser reads one raw
oscilloscope waveform (a single drop). This one reads a small
per-composition **summary**: the peak transmitted force/voltage for each
replicate drop of one composite, plus the composition (ionic-liquid and
acrylic-particle masses and fractions).

It exists for a real collaborator situation: the peak values arrived in
an aggregate Excel table rather than as raw traces for every sample.
Splitting that table into one file per composition and parsing it here
lets Latos treat each composition as a sample and run Bayesian
optimization over the particle loading. No waveform is stored — these
are peaks only, so nothing is fabricated.

File format (self-describing; produced by our split step, not an
instrument)::

    Latos Shock Summary,1
    Ionic Liquid Mass g,1.0033
    Acrylic Particle Mass g,0.6736
    Particle Mass Fraction wt%,40.169
    Particle Volume Fraction vol%,43.840
    Peak Force Calibration N per V,50
    Replicate,Peak Voltage V,Peak Force N
    1,1.12,56
    2,1.48,74
    3,1.36,68

The representative transmitted shock for a composition is the MEAN of the
replicate peaks; the scatter (sample standard deviation) is surfaced as a
feature so downstream reliability can weigh a noisy composition.
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

__all__ = ["ShockSummaryCsvParser", "is_shock_summary_header"]

# First-line signature that identifies our split files unambiguously.
_SIGNATURE = "latos shock summary"

# A replicate data row needs replicate, voltage, force.
_MIN_DATA_COLUMNS = 3

# Header ``key,value`` rows we surface as normalized metadata keys.
_META_KEYS: dict[str, str] = {
    "ionic liquid mass g": "ionic_liquid_mass_g",
    "acrylic particle mass g": "acrylic_particle_mass_g",
    "particle mass fraction wt%": "particle_wt_pct",
    "particle volume fraction vol%": "particle_vol_pct",
    "peak force calibration n per v": "force_calibration_n_per_v",
}

# Composition fields also promoted to features so the optimizer can use
# the loading as its input axis and the fraction shows in the feature table.
_FEATURE_META = ("particle_vol_pct", "particle_wt_pct")


def is_shock_summary_header(first_line: str) -> bool:
    """True if the first line carries the Latos shock-summary signature."""
    return first_line.strip().lower().startswith(_SIGNATURE)


class ShockSummaryCsvParser(BaseParser):
    """Parser for per-composition shock peak-force summary files."""

    name: ClassVar[str] = "shock-summary-csv"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.SHOCK
    supported_extensions: ClassVar[tuple[str, ...]] = (".csv",)

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """1.0 for a Latos shock-summary CSV, 0.0 otherwise.

        Keyed on the distinctive first-line signature, so it never
        competes with a Tektronix waveform or a CasaXPS region file.
        """
        if path.suffix.lower() not in self.supported_extensions:
            return 0.0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                first = fh.readline()
        except OSError:
            return 0.0
        return 1.0 if is_shock_summary_header(first) else 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a shock-summary CSV into a scalar (peak-only) ParsedData."""
        issues: list[ValidationIssue] = []
        metadata: dict[str, Any] = {}
        forces: list[float] = []
        voltages: list[float] = []

        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                in_data = False
                for row in csv.reader(fh):
                    if not row or not row[0].strip():
                        continue
                    if not in_data:
                        if row[0].strip().lower() == "replicate":
                            in_data = True
                            continue
                        key = _META_KEYS.get(row[0].strip().lower())
                        if key is not None and len(row) >= 2:  # noqa: PLR2004
                            metadata[key] = _coerce(row[1].strip())
                        continue
                    parsed = _row_floats(row)
                    if parsed is not None:
                        voltages.append(parsed[1])
                        forces.append(parsed[2])
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    field="file",
                    severity=Severity.ERROR,
                    message=f"Could not read file: {exc}",
                    detected_at=utc_now(),
                ),
            )

        if not forces:
            issues.append(
                ValidationIssue(
                    field="data",
                    severity=Severity.ERROR,
                    message="No replicate peak rows found in shock summary.",
                    detected_at=utc_now(),
                ),
            )

        features: dict[str, float] = {}
        if forces:
            farr = np.asarray(forces, dtype=np.float64)
            varr = np.asarray(voltages, dtype=np.float64)
            features["peak_force_n"] = float(farr.mean())
            features["peak_voltage_v"] = float(varr.mean())
            if farr.size > 1:
                features["peak_force_sd_n"] = float(farr.std(ddof=1))
            for key in _FEATURE_META:
                value = metadata.get(key)
                if isinstance(value, (int, float)):
                    features[key] = float(value)

        metadata["n_replicates"] = len(forces)

        return ParsedData(
            technique=self.technique,
            arrays={},
            metadata=metadata,
            instrument="drop test (peak summary)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
            features=features,
        )


# ─── Module-level helpers ───────────────────────────────────────────
def _row_floats(row: list[str]) -> tuple[float, float, float] | None:
    """Parse a ``replicate,voltage,force`` row, or None if non-numeric."""
    if len(row) < _MIN_DATA_COLUMNS:
        return None
    try:
        return float(row[0].strip()), float(row[1].strip()), float(row[2].strip())
    except ValueError:
        return None


def _coerce(value: str) -> Any:
    """Coerce a header value to float when numeric, else keep the string."""
    try:
        return float(value)
    except ValueError:
        return value
