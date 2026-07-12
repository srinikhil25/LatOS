"""Tests for `latos.analysis.xps.regions.XpsRegionsAnalyzer`."""

from __future__ import annotations

import numpy as np
import pytest

from latos.analysis.base_analyzer import AnalyzerInputs
from latos.analysis.xps.regions import XpsRegionsAnalyzer
from latos.core.enums import Severity, Technique


def _gauss(x, center, height, width=0.6):
    return height * np.exp(-((x - center) ** 2) / (2 * width**2))


def _region(be_lo, be_hi, *peaks: tuple[float, float]) -> dict[str, np.ndarray]:
    """Synthetic region trace, recorded high→low BE like a real instrument."""
    be = np.linspace(be_hi, be_lo, 800)  # descending, as exported
    intensity = np.full_like(be, 400.0)
    for center, height in peaks:
        intensity += _gauss(be, center, height)
    return {"binding_energy": be, "intensity": intensity}


def _measure_stub(filename: str):
    class _F:
        pass

    class _M:
        pass

    f = _F()
    f.path = f"C:/data/XPS/{filename}"
    m = _M()
    m.files = (f,)
    return m


def _run(arrays, filename="Cu 2p.csv"):
    a = XpsRegionsAnalyzer()
    inputs = AnalyzerInputs(
        measurement=_measure_stub(filename),
        arrays=arrays,
        params=a.merge_params(None),
    )
    return a.analyze(inputs)


class TestMetadata:
    def test_accepts_xps_only(self):
        assert XpsRegionsAnalyzer().accepts_techniques == (Technique.XPS,)


class TestPeaks:
    def test_finds_spin_orbit_doublet(self):
        # Cu 2p3/2 at 932.6 eV and 2p1/2 at 952.3 eV.
        out = _run(_region(925, 965, (932.6, 5000.0), (952.3, 2500.0)))
        assert out.outputs["n_peaks"] == 2
        # Strongest first.
        assert out.outputs["main_peak_be_ev"] == pytest.approx(932.6, abs=0.2)
        assert out.outputs["peak_binding_energies_ev"][1] == pytest.approx(952.3, abs=0.2)

    def test_region_label_from_filename(self):
        out = _run(_region(925, 965, (932.6, 5000.0)), filename="Cu 2p.csv")
        assert out.outputs["region"] == "Cu 2p"

    def test_fwhm_estimated(self):
        out = _run(_region(925, 965, (932.6, 5000.0)))
        # Gaussian width 0.6 eV → FWHM ≈ 2.355·0.6 ≈ 1.4 eV.
        assert out.outputs["peak_fwhms_ev"][0] == pytest.approx(1.4, abs=0.4)

    def test_apex_only_caveat(self):
        out = _run(_region(925, 965, (932.6, 5000.0)))
        assert any(
            i.severity is Severity.INFO and "state assignment" in i.message.lower()
            for i in out.issues
        )


class TestChargeReference:
    def test_c1s_offset_reported(self):
        # Apex at 286.0 eV → +1.2 eV vs the 284.8 reference.
        out = _run(_region(280, 295, (286.0, 3000.0)), filename="C 1s.csv")
        assert out.outputs["charge_offset_vs_c1s_284p8_ev"] == pytest.approx(1.2, abs=0.1)
        assert any(i.field == "charge_reference" for i in out.issues)

    def test_non_c1s_region_has_no_offset(self):
        out = _run(_region(925, 965, (932.6, 5000.0)), filename="Cu 2p.csv")
        assert "charge_offset_vs_c1s_284p8_ev" not in out.outputs


class TestGuards:
    def test_flat_region_errors(self):
        be = np.linspace(300, 280, 100)
        out = _run({"binding_energy": be, "intensity": np.full_like(be, 5.0)})
        assert out.outputs == {}
        assert any(i.severity is Severity.ERROR for i in out.issues)

    def test_missing_arrays_error(self):
        out = _run({})
        assert any(i.severity is Severity.ERROR for i in out.issues)
