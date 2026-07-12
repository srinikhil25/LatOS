"""EDS composition analyzer — elements from characteristic X-ray peaks.

EDS (energy-dispersive X-ray spectroscopy) identifies elements by the
energies of their characteristic X-ray lines. This analyzer:

1. Detects peaks in the energy–intensity spectrum.
2. Matches each peak to the nearest known characteristic line
   (Kα / Lα / Mα) within a tolerance → the element.
3. Reports the elements found and a *semi-quantitative* relative
   composition (normalized peak intensities).

Honesty note: true at% requires standards and a ZAF / Cliff–Lorimer
k-factor correction. Without those, the composition here is a relative
peak-intensity estimate — good for "what's present and roughly how
much", not for publication-grade quantification. The analyzer says so
in an INFO issue.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

import numpy as np
from scipy.signal import find_peaks

from latos.analysis.base_analyzer import AnalyzerInputs, AnalyzerOutput, BaseAnalyzer
from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now

__all__ = ["EdsCompositionAnalyzer"]

# Principal characteristic X-ray line energies (keV). Curated for the
# elements common in inorganic / thermoelectric materials plus typical
# contaminants. Each element lists its most intense EDS line(s); matching
# to the nearest within tolerance and grouping by element then naturally
# combines an element's K and L families.
_LINES: tuple[tuple[str, str, float], ...] = (
    ("C", "Ka", 0.277),
    ("N", "Ka", 0.392),
    ("O", "Ka", 0.525),
    ("F", "Ka", 0.677),
    ("Na", "Ka", 1.041),
    ("Mg", "Ka", 1.254),
    ("Al", "Ka", 1.486),
    ("Si", "Ka", 1.740),
    ("P", "Ka", 2.013),
    ("S", "Ka", 2.307),
    ("Cl", "Ka", 2.622),
    ("K", "Ka", 3.313),
    ("Ca", "Ka", 3.690),
    ("Ti", "Ka", 4.508),
    ("Cr", "Ka", 5.411),
    ("Mn", "Ka", 5.894),
    ("Fe", "Ka", 6.398),
    ("Co", "Ka", 6.924),
    ("Ni", "Ka", 7.471),
    ("Cu", "La", 0.930),
    ("Cu", "Ka", 8.040),
    ("Zn", "Ka", 8.630),
    ("Ga", "Ka", 9.241),
    ("Ge", "Ka", 9.874),
    ("Se", "La", 1.379),
    ("Se", "Ka", 11.220),
    ("Br", "La", 1.480),
    ("Mo", "La", 2.293),
    ("Ag", "La", 2.984),
    ("Cd", "La", 3.133),
    ("Sn", "La", 3.444),
    ("Sb", "La", 3.604),
    ("Te", "La", 3.769),
    ("I", "La", 3.937),
    ("Cs", "La", 4.286),
    ("Ba", "La", 4.466),
    ("W", "Ma", 1.775),
    ("W", "La", 8.398),
    ("Au", "Ma", 2.123),
    ("Au", "La", 9.713),
    ("Pb", "Ma", 2.342),
    ("Pb", "La", 10.551),
    ("Bi", "Ma", 2.419),
    ("Bi", "La", 10.839),
)

# Below this energy the zero-strobe / noise peak dominates — ignore it.
_MIN_LINE_KEV = 0.20


class EdsCompositionAnalyzer(BaseAnalyzer):
    """Element identification + semi-quantitative composition from EDS."""

    name: ClassVar[str] = "eds-composition"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.EDS,)
    default_params: ClassVar[dict[str, Any]] = {
        # Peak must reach this fraction of the max intensity to count.
        "min_peak_frac": 0.03,
        # A peak matches an element line within this energy window (keV).
        "match_tolerance_kev": 0.10,
    }

    def accepts(self, measurement: Any) -> bool:
        """Accept any EDS measurement that has at least one source file."""
        return len(measurement.files) > 0

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        """Detect peaks, match elements, estimate relative composition."""
        energy = inputs.arrays.get("energy_kev")
        intensity = inputs.arrays.get("intensity")
        if energy is None or intensity is None or energy.size == 0:
            return _error("Missing energy_kev / intensity arrays — cannot analyze EDS.")

        energy = np.asarray(energy, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)
        i_max = float(np.max(intensity))
        if i_max <= 0:
            return _error("EDS spectrum is empty (all-zero intensity).")

        params = inputs.params
        min_frac = float(params.get("min_peak_frac", 0.03))
        tol = float(params.get("match_tolerance_kev", 0.10))

        peak_idx, _ = find_peaks(intensity, prominence=min_frac * i_max)

        per_element: dict[str, float] = defaultdict(float)
        matched: list[tuple[float, str, str]] = []  # (energy, element, line)
        for p in peak_idx:
            e_kev = float(energy[p])
            if e_kev < _MIN_LINE_KEV:
                continue
            best = _nearest_line(e_kev, tol)
            if best is not None:
                element, line = best
                per_element[element] += float(intensity[p])
                matched.append((round(e_kev, 3), element, line))

        if not per_element:
            return _error("No characteristic peaks matched a known element line.")

        total = sum(per_element.values())
        ranked = sorted(per_element.items(), key=lambda kv: kv[1], reverse=True)
        outputs: dict[str, Any] = {
            "n_elements": len(ranked),
            "elements": [el for el, _ in ranked],
            "composition_rel_pct": [f"{el}: {100 * v / total:.1f}" for el, v in ranked],
            "matched_peaks_kev": [m[0] for m in sorted(matched)],
        }
        issues = (
            ValidationIssue(
                field="composition",
                severity=Severity.INFO,
                message=(
                    "Composition is semi-quantitative (relative peak intensity); "
                    "true at% needs standards + ZAF/Cliff-Lorimer k-factors."
                ),
                detected_at=utc_now(),
            ),
        )
        return AnalyzerOutput(outputs=outputs, derived_arrays={}, issues=issues)


def _nearest_line(e_kev: float, tol: float) -> tuple[str, str] | None:
    """Nearest (element, line) within `tol` keV of `e_kev`, or None."""
    best: tuple[str, str] | None = None
    best_d = tol
    for element, line, line_e in _LINES:
        d = abs(line_e - e_kev)
        if d <= best_d:
            best_d = d
            best = (element, line)
    return best


def _error(message: str) -> AnalyzerOutput:
    return AnalyzerOutput(
        outputs={},
        derived_arrays={},
        issues=(
            ValidationIssue(
                field="analyze",
                severity=Severity.ERROR,
                message=message,
                detected_at=utc_now(),
            ),
        ),
    )
