# ruff: noqa: N802, N806
# N802: test names include physical-unit suffixes like _meV that are
#       intentionally NOT lowercase (it would mean a milli-electron, not
#       milli-electronvolt).
# N806: scientific synthetic uses uppercase F, R by Tauc convention.
"""Tests for `latos.analysis.uv_drs.tauc.UvDrsTaucAnalyzer`.

These tests check the analyzer in isolation — no DB, no ArrayStore.
They construct synthetic spectra with a known band gap and verify
the extracted gap is close to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latos.analysis.base_analyzer import AnalyzerInputs, AnalyzerOutput
from latos.analysis.uv_drs.tauc import UvDrsTaucAnalyzer
from latos.core.enums import FileRole, Severity, Technique
from latos.core.models import FileRef, Measurement, new_id, utc_now

# Constant from the analyzer module.
_HC_NM_EV = 1240.0


def _measurement(
    *,
    technique: Technique = Technique.UV_DRS,
    n_files: int = 1,
) -> Measurement:
    files = tuple(
        FileRef(
            path=Path(f"/data/uv-{i}.xlsx"),
            sha256=(str(i) * 64)[:64],
            size_bytes=100,
            role=FileRole.RAW,
            scanned_at=utc_now(),
        )
        for i in range(n_files)
    )
    return Measurement(
        id=new_id(),
        sample_id=new_id(),
        technique=technique,
        instrument="UV-DRS",
        measured_at=utc_now(),
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=files,
    )


def _synthetic_uvdrs_spectrum(
    band_gap_ev: float,
    *,
    n_points: int = 401,
    wavelength_min_nm: float = 250.0,
    wavelength_max_nm: float = 800.0,
    gap_type: str = "direct",
    slope: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a (wavelength_nm, reflectance_fraction) pair with a known gap.

    Construction strategy: build the *target* Tauc curve directly so
    that `(F(R) · E)^n = slope · (E - Eg)` is exactly linear above the
    gap (and zero below). Then invert Kubelka-Munk to recover R.

    With n = 2 for direct and n = 1/2 for indirect, this means:
        F(R) = (slope · (E - Eg))^(1/n) / E    when E > Eg
        F(R) = 0                              when E ≤ Eg

    Inverting `F = (1-R)^2 / (2R)`: from `R² - 2(F+1)R + 1 = 0` we get
    `R = (F+1) - sqrt((F+1)² - 1)`, the root in (0, 1].

    The reflectance returned is in fraction units [0, 1]. Tests can
    multiply by 100 to exercise the percent-detection auto-scaling.
    """
    wavelength = np.linspace(wavelength_min_nm, wavelength_max_nm, n_points)
    photon_energy = _HC_NM_EV / wavelength
    above_gap = np.maximum(photon_energy - band_gap_ev, 0.0)

    # n=2 for direct, n=1/2 for indirect — the exponent the analyzer uses.
    n_exp = 2.0 if gap_type == "direct" else 0.5
    # F(R) · E = (slope · (E-Eg))^(1/n)  →  F(R) = (above)^(1/n) / E.
    tauc_y_target = slope * above_gap
    # Use a tiny epsilon to avoid 0^0 = 1 surprises from numpy.
    F = np.where(
        above_gap > 0,
        np.power(np.maximum(tauc_y_target, 1e-30), 1.0 / n_exp) / photon_energy,
        0.0,
    )

    one_plus_F = 1.0 + F
    R = one_plus_F - np.sqrt(np.maximum(one_plus_F**2 - 1.0, 0.0))
    # Floor on R so the analyzer's clip doesn't truncate baseline regions.
    R = np.clip(R, 1e-3, 1.0)
    return wavelength, R


class TestClassAttributes:
    def test_metadata(self) -> None:
        analyzer = UvDrsTaucAnalyzer()
        assert analyzer.name == "uvdrs-tauc"
        assert analyzer.version == "1.0.0"
        assert Technique.UV_DRS in analyzer.accepts_techniques
        # default_params is the contract the UI surfaces.
        assert "band_gap_type" in analyzer.default_params
        assert "fit_window_y_min_frac" in analyzer.default_params


class TestAccepts:
    def test_accepts_uvdrs_measurement(self) -> None:
        assert UvDrsTaucAnalyzer().accepts(_measurement()) is True

    def test_rejects_zero_file_measurement(self) -> None:
        m = _measurement(n_files=0)
        assert UvDrsTaucAnalyzer().accepts(m) is False


class TestSyntheticDirectGap:
    """Recover a known direct band gap from a synthetic spectrum."""

    @pytest.mark.parametrize("expected_gap", [1.7, 2.05, 2.5, 3.1])
    def test_recovers_gap_within_50_meV(self, expected_gap: float) -> None:
        wavelength, refl = _synthetic_uvdrs_spectrum(
            band_gap_ev=expected_gap,
            gap_type="direct",
        )
        analyzer = UvDrsTaucAnalyzer()
        out = analyzer.analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": wavelength, "reflectance": refl},
                params=analyzer.default_params,
            ),
        )
        assert "band_gap_ev" in out.outputs
        recovered = float(out.outputs["band_gap_ev"])
        # 50 meV tolerance is generous — the synthetic curve is ideal,
        # the Tauc fit windowing introduces a small bias. Real data
        # has much larger uncertainties.
        assert abs(recovered - expected_gap) < 0.05, (
            f"Recovered {recovered:.3f} eV vs expected {expected_gap:.3f} eV"
        )

    def test_r_squared_is_high_on_synthetic(self) -> None:
        wavelength, refl = _synthetic_uvdrs_spectrum(band_gap_ev=2.0, gap_type="direct")
        analyzer = UvDrsTaucAnalyzer()
        out = analyzer.analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": wavelength, "reflectance": refl},
                params=analyzer.default_params,
            ),
        )
        assert out.outputs["r_squared"] > 0.99

    def test_returns_derived_arrays(self) -> None:
        wavelength, refl = _synthetic_uvdrs_spectrum(band_gap_ev=2.0, gap_type="direct")
        analyzer = UvDrsTaucAnalyzer()
        out = analyzer.analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": wavelength, "reflectance": refl},
                params=analyzer.default_params,
            ),
        )
        assert set(out.derived_arrays) == {
            "photon_energy_ev",
            "kubelka_munk",
            "tauc_y",
            "fit_line",
        }
        # All derived arrays must be 1-D and co-indexed.
        sizes = {a.shape for a in out.derived_arrays.values()}
        assert len(sizes) == 1


class TestSyntheticIndirectGap:
    @pytest.mark.parametrize("expected_gap", [1.1, 1.5, 2.0])
    def test_recovers_indirect_gap(self, expected_gap: float) -> None:
        wavelength, refl = _synthetic_uvdrs_spectrum(
            band_gap_ev=expected_gap,
            gap_type="indirect",
        )
        analyzer = UvDrsTaucAnalyzer()
        out = analyzer.analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": wavelength, "reflectance": refl},
                params={"band_gap_type": "indirect"},
            ),
        )
        recovered = float(out.outputs["band_gap_ev"])
        # Indirect-gap recovery is slightly less precise than direct due to
        # the higher exponent; 100 meV tolerance.
        assert abs(recovered - expected_gap) < 0.1


class TestPercentDetection:
    def test_auto_scales_percent_reflectance(self) -> None:
        """Reflectance in 0..100% must be auto-scaled and analyzed."""
        wavelength, refl = _synthetic_uvdrs_spectrum(band_gap_ev=2.0, gap_type="direct")
        analyzer = UvDrsTaucAnalyzer()
        out_fraction = analyzer.analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": wavelength, "reflectance": refl},
                params=analyzer.default_params,
            ),
        )
        out_percent = analyzer.analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": wavelength, "reflectance": refl * 100.0},
                params=analyzer.default_params,
            ),
        )
        # Same band gap regardless of scale.
        assert abs(
            out_fraction.outputs["band_gap_ev"] - out_percent.outputs["band_gap_ev"]
        ) < 1e-6
        # Percent path emits an INFO issue.
        kinds = {i.severity for i in out_percent.issues}
        assert Severity.INFO in kinds


class TestParamHandling:
    def test_unrecognized_gap_type_falls_back_to_direct(self) -> None:
        wavelength, refl = _synthetic_uvdrs_spectrum(band_gap_ev=2.0)
        analyzer = UvDrsTaucAnalyzer()
        out = analyzer.analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": wavelength, "reflectance": refl},
                params={"band_gap_type": "wibble"},
            ),
        )
        assert out.outputs["band_gap_type"] == "direct"
        assert any(i.field == "band_gap_type" for i in out.issues)

    def test_fit_window_is_respected(self) -> None:
        """Different y-window fractions produce different fit windows."""
        wavelength, refl = _synthetic_uvdrs_spectrum(band_gap_ev=2.0)
        analyzer = UvDrsTaucAnalyzer()
        narrow = analyzer.analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": wavelength, "reflectance": refl},
                params={"fit_window_y_min_frac": 0.30, "fit_window_y_max_frac": 0.50},
            ),
        )
        wide = analyzer.analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": wavelength, "reflectance": refl},
                params={"fit_window_y_min_frac": 0.10, "fit_window_y_max_frac": 0.80},
            ),
        )
        assert narrow.outputs["n_points_fit"] < wide.outputs["n_points_fit"]


class TestFailureModes:
    """The analyzer must never raise — only return error-issued outputs."""

    def test_missing_arrays(self) -> None:
        out = UvDrsTaucAnalyzer().analyze(
            AnalyzerInputs(measurement=_measurement(), arrays={}, params={}),
        )
        assert out.outputs == {}
        assert any(i.severity is Severity.ERROR for i in out.issues)

    def test_mismatched_shapes(self) -> None:
        out = UvDrsTaucAnalyzer().analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={"wavelength": np.linspace(300, 700, 10), "reflectance": np.zeros(5)},
                params={},
            ),
        )
        assert any(i.severity is Severity.ERROR for i in out.issues)

    def test_too_few_points(self) -> None:
        out = UvDrsTaucAnalyzer().analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={
                    "wavelength": np.linspace(300, 700, 5),
                    "reflectance": np.full(5, 0.5),
                },
                params={},
            ),
        )
        assert any(i.severity is Severity.ERROR for i in out.issues)

    def test_never_raises_on_arbitrary_input(self) -> None:
        """Defensive: garbage-in must produce error-issued output, not crash."""
        out = UvDrsTaucAnalyzer().analyze(
            AnalyzerInputs(
                measurement=_measurement(),
                arrays={
                    "wavelength": np.array([0.0, 0.0, 0.0]),
                    "reflectance": np.array([np.nan, np.nan, np.nan]),
                },
                params={},
            ),
        )
        assert isinstance(out, AnalyzerOutput)
        assert any(i.severity is Severity.ERROR for i in out.issues)
