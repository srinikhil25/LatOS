"""Enums shared across the Latos domain layer.

Defined separately from models to keep imports light and prevent cycles.
"""

from __future__ import annotations

from enum import StrEnum


class Technique(StrEnum):
    """Characterization techniques supported by Latos.

    Values are short lowercase identifiers, used in DB rows and config files.
    """

    XRD = "xrd"
    XPS = "xps"
    UV_DRS = "uv_drs"
    HALL = "hall"
    THERMOELECTRIC = "thermoelectric"
    EDS = "eds"
    TEM = "tem"
    SEM = "sem"
    STEM = "stem"
    RAMAN = "raman"
    SHOCK = "shock"
    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        """Human-readable label for UI."""
        return _TECHNIQUE_DISPLAY[self]


_TECHNIQUE_DISPLAY: dict[Technique, str] = {
    Technique.XRD: "X-Ray Diffraction",
    Technique.XPS: "X-Ray Photoelectron Spectroscopy",
    Technique.UV_DRS: "UV-Vis Diffuse Reflectance",
    Technique.HALL: "Hall Effect",
    Technique.THERMOELECTRIC: "Thermoelectric Properties",
    Technique.EDS: "Energy-Dispersive X-Ray Spectroscopy",
    Technique.TEM: "Transmission Electron Microscopy",
    Technique.SEM: "Scanning Electron Microscopy",
    Technique.STEM: "Scanning Transmission Electron Microscopy",
    Technique.RAMAN: "Raman Spectroscopy",
    Technique.SHOCK: "Mechanical Shock (drop test)",
    Technique.UNKNOWN: "Unknown",
}


class FileRole(StrEnum):
    """Role a file plays in a measurement."""

    RAW = "raw"  # original instrument output
    PROCESSED = "processed"  # cleaned, smoothed, baseline-subtracted
    DERIVED = "derived"  # computed (Tauc plot data, fit results, etc.)
    METADATA = "metadata"  # JSON/XML auxiliary metadata


class Severity(StrEnum):
    """Severity level for validation issues and log records."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def order(self) -> int:
        """Numeric priority — higher means more severe."""
        return _SEVERITY_ORDER[self]


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.ERROR: 2,
}


class ReviewStatus(StrEnum):
    """Whether a project's auto-categorization has been human-verified.

    After ingestion a project is `NEEDS_REVIEW`: Latos has guessed the
    sample identity and technique of every file, but those guesses can
    be wrong (messy folder names, doped-sample over-merging, …). The
    downstream pipeline — analysis, correlation, optimization — requires
    `CONFIRMED`, because property predictions are only trustworthy when
    a person has signed off on which file is which sample.
    """

    NEEDS_REVIEW = "needs_review"
    CONFIRMED = "confirmed"
