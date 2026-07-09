"""Tests for `latos.analysis.eds.composition.EdsCompositionAnalyzer`."""

from __future__ import annotations

import numpy as np
import pytest

from latos.analysis.base_analyzer import AnalyzerInputs
from latos.analysis.eds.composition import EdsCompositionAnalyzer
from latos.core.enums import Severity, Technique


def _gauss(x, center, height, width=0.08):
    return height * np.exp(-((x - center) ** 2) / (2 * width**2))


def _spectrum(*peaks: tuple[float, float]) -> dict[str, np.ndarray]:
    """Synthetic EDS: energy 0–20 keV with Gaussian peaks at (energy, height)."""
    energy = np.linspace(0.0, 20.0, 2000)
    intensity = np.full_like(energy, 5.0)  # flat baseline
    for center, height in peaks:
        intensity += _gauss(energy, center, height)
    return {"energy_kev": energy, "intensity": intensity}


def _measure_stub():
    class _M:
        files = (object(),)  # accepts() only checks len(files) > 0

    return _M()


def _run(arrays):
    a = EdsCompositionAnalyzer()
    inputs = AnalyzerInputs(measurement=_measure_stub(), arrays=arrays, params=a.merge_params(None))
    return a.analyze(inputs)


class TestMetadata:
    def test_accepts_eds_only(self):
        assert Technique.EDS in EdsCompositionAnalyzer().accepts_techniques


class TestIdentification:
    def test_identifies_cu_and_se(self):
        # Cu Kα 8.04, Se Kα 11.22.
        out = _run(_spectrum((8.04, 1000.0), (11.22, 600.0)))
        assert "Cu" in out.outputs["elements"]
        assert "Se" in out.outputs["elements"]
        # Cu is the stronger peak → ranked first.
        assert out.outputs["elements"][0] == "Cu"

    def test_identifies_dopant_elements(self):
        # Cs Lα 4.286, Bi Lα 10.839, I Lα 3.937.
        out = _run(_spectrum((4.286, 500.0), (10.839, 400.0), (3.937, 450.0)))
        assert {"Cs", "Bi", "I"} <= set(out.outputs["elements"])

    def test_composition_sums_to_100(self):
        out = _run(_spectrum((8.04, 1000.0), (11.22, 500.0)))
        pcts = [float(s.split(": ")[1]) for s in out.outputs["composition_rel_pct"]]
        assert sum(pcts) == pytest.approx(100.0, abs=0.2)

    def test_semiquant_caveat_emitted(self):
        out = _run(_spectrum((8.04, 1000.0)))
        assert any(
            i.severity is Severity.INFO and "semi-quantitative" in i.message
            for i in out.issues
        )


class TestGuards:
    def test_zero_spectrum_errors(self):
        energy = np.linspace(0, 20, 100)
        out = _run({"energy_kev": energy, "intensity": np.zeros_like(energy)})
        assert out.outputs == {}
        assert any(i.severity is Severity.ERROR for i in out.issues)

    def test_no_matching_peaks_errors(self):
        # A peak at 15.5 keV matches no curated line within tolerance.
        out = _run(_spectrum((15.5, 1000.0)))
        assert out.outputs == {}

    def test_ignores_zero_strobe_peak(self):
        # A huge peak at ~0 keV (noise/strobe) must not be identified.
        out = _run(_spectrum((0.05, 5000.0), (8.04, 1000.0)))
        assert out.outputs["elements"] == ["Cu"]
