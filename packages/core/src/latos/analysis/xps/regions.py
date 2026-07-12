"""XPS region analyzer — peak binding energies from a CasaXPS region export.

A CasaXPS CSV region ("Cu 2p", "Se 3d", …) is a binding-energy vs
intensity trace. This analyzer reports the *apex-level* facts a
researcher checks first:

- the main peak positions (binding energies) and rough FWHMs,
- the C 1s offset from the 284.8 eV adventitious-carbon reference when
  the region IS C 1s — the standard charge-calibration hint.

Honesty note: apex positions are read directly off the trace. Chemical-
state assignment needs proper background subtraction (Shirley), peak
deconvolution, and charge referencing — this analyzer deliberately does
NOT claim oxidation states. It says so in an INFO issue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from scipy.signal import find_peaks, peak_widths

from latos.analysis.base_analyzer import AnalyzerInputs, AnalyzerOutput, BaseAnalyzer
from latos.core.enums import Severity, Technique
from latos.core.models import Measurement, ValidationIssue, utc_now

__all__ = ["XpsRegionsAnalyzer"]

# Adventitious-carbon C 1s reference for charge correction (eV).
_C1S_REFERENCE_EV = 284.8

# How many peaks to report (strongest first) — a survey region can have
# many; the top few carry the chemistry.
_MAX_REPORTED_PEAKS = 6


class XpsRegionsAnalyzer(BaseAnalyzer):
    """Apex-level peak positions for one XPS region trace."""

    name: ClassVar[str] = "xps-regions"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.XPS,)
    default_params: ClassVar[dict[str, Any]] = {
        # Peak prominence as a fraction of the region's intensity span.
        "min_peak_frac": 0.05,
    }

    def accepts(self, measurement: Measurement) -> bool:
        """Accept any XPS region that has a source file."""
        return len(measurement.files) > 0

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        """Find apex peaks in the region and report their binding energies."""
        be = inputs.arrays.get("binding_energy")
        intensity = inputs.arrays.get("intensity")
        if be is None or intensity is None or be.size == 0:
            return _error("Missing binding_energy / intensity arrays — cannot analyze XPS region.")

        be = np.asarray(be, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)

        # XPS traces are conventionally recorded high→low BE; sort
        # ascending so index math and widths are well-defined.
        order = np.argsort(be)
        be, intensity = be[order], intensity[order]

        span = float(np.max(intensity) - np.min(intensity))
        if span <= 0:
            return _error("XPS region is flat (no intensity variation).")

        min_frac = float(inputs.params.get("min_peak_frac", 0.05))
        peak_idx, _ = find_peaks(intensity, prominence=min_frac * span)
        if peak_idx.size == 0:
            return _error(
                "No peaks detected in this region — check the trace or lower min_peak_frac.",
            )

        # FWHM in eV from half-height widths × the median BE step.
        widths_samples = peak_widths(intensity, peak_idx, rel_height=0.5)[0]
        step_ev = float(np.median(np.abs(np.diff(be)))) if be.size > 1 else 0.0

        # Strongest first, cap the list.
        strongest = np.argsort(intensity[peak_idx])[::-1][:_MAX_REPORTED_PEAKS]
        idx = peak_idx[strongest]
        peak_bes = [round(float(b), 2) for b in be[idx]]
        peak_fwhms = [round(float(w) * step_ev, 2) for w in widths_samples[strongest]]

        region = _region_label(inputs.measurement)
        outputs: dict[str, Any] = {
            "region": region,
            "n_peaks": int(peak_idx.size),
            "peak_binding_energies_ev": peak_bes,
            "peak_fwhms_ev": peak_fwhms,
            "main_peak_be_ev": peak_bes[0],
            "be_range_ev": [round(float(be[0]), 1), round(float(be[-1]), 1)],
        }

        issues: list[ValidationIssue] = [
            ValidationIssue(
                field="peaks",
                severity=Severity.INFO,
                message=(
                    "Apex positions only — chemical-state assignment needs Shirley "
                    "background subtraction, deconvolution, and charge referencing."
                ),
                detected_at=utc_now(),
            ),
        ]

        # C 1s region → report the offset from adventitious carbon, the
        # standard charge-correction reference for the whole sample set.
        if _is_c1s(region):
            offset = round(peak_bes[0] - _C1S_REFERENCE_EV, 2)
            outputs["charge_offset_vs_c1s_284p8_ev"] = offset
            issues.append(
                ValidationIssue(
                    field="charge_reference",
                    severity=Severity.INFO,
                    message=(
                        f"C 1s apex at {peak_bes[0]:.2f} eV → {offset:+.2f} eV vs the "
                        "284.8 eV adventitious-carbon reference; apply as the charge "
                        "correction for this sample's other regions."
                    ),
                    detected_at=utc_now(),
                )
            )

        return AnalyzerOutput(outputs=outputs, derived_arrays={}, issues=tuple(issues))


def _region_label(measurement: Measurement) -> str:
    """Region name from the source file stem (e.g. 'Cu 2p.csv' → 'Cu 2p')."""
    try:
        return Path(measurement.files[0].path).stem.strip()
    except Exception:  # display label only, never fatal
        return "unknown region"


def _is_c1s(region: str) -> bool:
    return region.lower().replace(" ", "") in {"c1s", "carbon1s"}


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
