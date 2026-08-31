"""Raman parser for Renishaw WiRE ASCII exports.

File format
-----------
Two tab-separated columns under a comment header, CRLF line endings::

    #Wave		#Intensity
    3200.263672	1003.922974
    3198.679688	983.402527

Renishaw writes the header with a DOUBLE tab, and exports wavenumber in
DESCENDING order. Arrays are published ascending here, matching every other
spectroscopy parser in Latos, so downstream code never has to ask.

This reads the ASCII export, not the native ``.wdf`` binary that sits beside it.

What the file does NOT contain
------------------------------
Nothing about the acquisition: no laser wavelength, no power, no objective, no
exposure or accumulation count. Those live only in the operator's file naming
(``...50x1_`` for a 50x objective at 1% power). The consequence is worth stating
plainly, because it is easy to forget when two spectra are plotted together:

    **Absolute intensities are not comparable between files.**

Replicate spectra of one sample, taken minutes apart, have been observed to
differ by a factor of two in absolute counts while agreeing to a few percent on
band RATIOS. Comparisons must therefore be built from ratios, normalised
shapes, or significance against a blank - never from raw peak heights across
files. A `Severity.INFO` issue says so on every parse.

Validation policy
-----------------
Never raises. Problems surface as `ValidationIssue`s:

* no data points                -> ERROR   (empty or malformed export)
* fewer than 50 points          -> WARNING (too sparse to fit a band)
* non-monotonic wavenumber      -> WARNING (merged or corrupted file)
* detector saturation           -> WARNING (clipped peaks; intensities invalid)
* cosmic-ray spikes             -> WARNING (single-point outliers, with a count)
* acquisition settings absent   -> INFO    (always; see above)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["RenishawRamanTxtParser", "find_cosmic_rays"]

# The header Renishaw writes. Matched loosely: the tab count has been seen to
# vary between WiRE versions, so the two tokens are checked rather than the
# exact string.
_HEADER_TOKENS = ("#wave", "#intensity")

# Below this a spectrum is too sparse to fit a band against.
_MIN_POINTS = 50

# Cosmic-ray spikes are 1-2 points wide and tower over the local baseline.
# A spike must exceed this many robust deviations above its local median.
_SPIKE_SIGMA = 12.0
_SPIKE_WINDOW = 7

# Detector saturation: this fraction of points sharing the maximum value means
# the peak was clipped rather than measured.
_SATURATION_FRACTION = 0.002

_SNIFF_BYTES = 512

# A spike-detection window must span a point and neighbours either side.
_MIN_WINDOW = 3

# A data row is a (wavenumber, intensity) pair.
_MIN_DATA_COLUMNS = 2

# Positions listed in the cosmic-ray message before it is truncated.
_MAX_LISTED_SPIKES = 5


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation, scaled to be comparable with a std dev."""
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def find_cosmic_rays(
    intensity: np.ndarray,
    *,
    sigma: float = _SPIKE_SIGMA,
    window: int = _SPIKE_WINDOW,
) -> np.ndarray:
    """Indices of single-point cosmic-ray spikes.

    A cosmic ray is a detector event, not light from the sample: it lands on
    one or two pixels and is far above the local baseline. Real Raman bands are
    many points wide, so width is what separates them. Each point is compared
    against the median of its neighbourhood, scaled by the robust spread of the
    whole spectrum, and only points that also rise well above BOTH immediate
    neighbours are reported - which is what excludes the summit of a genuine
    band, where the neighbours are nearly as high.

    Returns an empty array for a spectrum shorter than the window.

    Raises:
        ValueError: If `window` is not an odd number of at least 3.
    """
    if window < _MIN_WINDOW or window % 2 == 0:
        raise ValueError(f"window must be an odd number >= 3, got {window!r}")
    y = np.asarray(intensity, dtype=float)
    if y.size < window:
        return np.array([], dtype=int)

    half = window // 2
    padded = np.pad(y, half, mode="edge")
    local_median = np.array([np.median(padded[i : i + window]) for i in range(y.size)])
    spread = _mad(y - local_median)
    if spread <= 0:
        return np.array([], dtype=int)

    excess = (y - local_median) / spread
    candidates = np.where(excess > sigma)[0]
    # Keep only points that also stand well clear of both neighbours: a band
    # apex sits on a shoulder, a cosmic ray does not.
    keep = [
        i
        for i in candidates
        if 0 < i < y.size - 1 and y[i] > 2.0 * max(y[i - 1], y[i + 1]) - local_median[i]
    ]
    return np.asarray(keep, dtype=int)


def _issue(message: str, severity: Severity, field: str = "raman") -> ValidationIssue:
    return ValidationIssue(field=field, message=message, severity=severity, detected_at=utc_now())


class RenishawRamanTxtParser(BaseParser):
    """Renishaw WiRE two-column ASCII spectrum export."""

    name: ClassVar[str] = "renishaw-raman-txt"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.RAMAN
    supported_extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def can_parse(self, path: Path) -> float:
        """Confidence from the `#Wave` / `#Intensity` header."""
        try:
            head = path.read_bytes()[:_SNIFF_BYTES]
        except OSError:
            return 0.0
        first = head.decode("utf-8", errors="replace").splitlines()[:1]
        if not first:
            return 0.0
        lowered = first[0].lower()
        if all(token in lowered for token in _HEADER_TOKENS):
            return 1.0
        return 0.0

    def parse(self, path: Path) -> ParsedData:
        """Read one spectrum into ascending wavenumber and intensity arrays."""
        issues: list[ValidationIssue] = []
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            return self._empty(f"Could not read the file: {exc}")

        wavenumber: list[float] = []
        intensity: list[float] = []
        skipped = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", "\t").split()
            if len(parts) < _MIN_DATA_COLUMNS:
                skipped += 1
                continue
            try:
                wavenumber.append(float(parts[0]))
                intensity.append(float(parts[1]))
            except ValueError:
                skipped += 1

        if skipped:
            issues.append(
                _issue(f"{skipped} line(s) could not be read as a number pair.", Severity.WARNING)
            )
        if not wavenumber:
            return self._empty(
                "No data points — the export is empty or malformed.", extra_issues=issues
            )

        raw_wavenumber = np.asarray(wavenumber, dtype=float)
        raw_intensity = np.asarray(intensity, dtype=float)

        # Renishaw writes descending; publish ascending like every other
        # spectroscopy parser here, so callers never have to check.
        descending = bool(raw_wavenumber.size > 1 and raw_wavenumber[0] > raw_wavenumber[-1])
        order = np.argsort(raw_wavenumber)
        shifts = raw_wavenumber[order]
        counts = raw_intensity[order]

        issues.extend(self._quality_checks(shifts, counts))
        issues.append(
            _issue(
                "The export carries no acquisition settings — no laser wavelength, power, "
                "objective or exposure. Absolute intensities are therefore NOT comparable "
                "between files; use band ratios, normalised shapes, or significance against "
                "a blank instead.",
                Severity.INFO,
            )
        )

        steps = np.diff(shifts)
        metadata: dict[str, Any] = {
            "n_points": int(shifts.size),
            "wavenumber_min_cm1": float(shifts.min()),
            "wavenumber_max_cm1": float(shifts.max()),
            "median_step_cm1": float(np.median(steps)) if steps.size else 0.0,
            "stored_descending": descending,
        }

        return ParsedData(
            technique=Technique.RAMAN,
            arrays={"raman_shift_cm1": shifts, "intensity": counts},
            metadata=metadata,
            instrument="Renishaw (WiRE ASCII export)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )

    @staticmethod
    def _quality_checks(shifts: np.ndarray, counts: np.ndarray) -> list[ValidationIssue]:
        out: list[ValidationIssue] = []

        if shifts.size < _MIN_POINTS:
            out.append(
                _issue(
                    f"Only {shifts.size} points — too sparse to fit a band against.",
                    Severity.WARNING,
                )
            )

        if shifts.size > 1 and not np.all(np.diff(shifts) > 0):
            out.append(
                _issue(
                    "Wavenumber is not strictly increasing after sorting, so the axis "
                    "contains duplicates. The file may be two spectra concatenated.",
                    Severity.WARNING,
                )
            )

        if counts.size:
            top = float(counts.max())
            at_top = int(np.sum(counts >= top * (1 - 1e-9)))
            if at_top > max(2, _SATURATION_FRACTION * counts.size):
                out.append(
                    _issue(
                        f"{at_top} points sit at the maximum value ({top:.0f}) — the detector "
                        "saturated and those peaks are clipped, not measured.",
                        Severity.WARNING,
                    )
                )

        spikes = find_cosmic_rays(counts)
        if spikes.size:
            where = ", ".join(f"{shifts[i]:.0f}" for i in spikes[:_MAX_LISTED_SPIKES])
            more = "" if spikes.size <= _MAX_LISTED_SPIKES else ", ..."
            out.append(
                _issue(
                    f"{spikes.size} cosmic-ray spike(s) detected at {where}{more} cm-1. "
                    "These are detector events, not sample signal; remove them before "
                    "fitting or integrating.",
                    Severity.WARNING,
                )
            )
        return out

    def _empty(
        self,
        message: str,
        *,
        extra_issues: list[ValidationIssue] | None = None,
    ) -> ParsedData:
        issues = list(extra_issues or [])
        issues.append(_issue(message, Severity.ERROR))
        return ParsedData(
            technique=Technique.RAMAN,
            arrays={},
            metadata={},
            instrument="Renishaw (WiRE ASCII export)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )
