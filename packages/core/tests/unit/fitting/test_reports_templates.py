"""Tests for fit reports and template round-tripping."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from latos.fitting import (
    BackgroundKind,
    BackgroundSpec,
    FitSpec,
    FixedDelta,
    FixedRatio,
    PeakInit,
    PeakShape,
    SharedWidth,
    csv_table,
    fit_spectrum,
    latex_table,
    load_template,
    markdown_report,
    save_template,
    spec_from_dict,
    spec_to_dict,
    xps_doublet_preset,
)


def _fit():
    x = np.linspace(0.0, 100.0, 500)
    height = 80.0 / (5.0 * np.sqrt(2.0 * np.pi))
    y = 10.0 + height * np.exp(-0.5 * ((x - 50.0) / 5.0) ** 2)
    # A little noise so lmfit can estimate parameter uncertainties (a
    # zero-residual fit has a singular covariance → no stderr).
    rng = np.random.default_rng(0)
    y = y + rng.normal(0.0, 0.05, size=x.size)
    return fit_spectrum(x, y, FitSpec(PeakShape.GAUSSIAN, [PeakInit(50.0)]))


class TestReports:
    def test_markdown_has_gof_and_component_row(self):
        md = markdown_report(_fit(), title="Se 3d fit")
        assert "# Se 3d fit" in md
        assert "R²" in md and "Reduced χ²" in md
        assert "| Peak | Center |" in md
        assert md.count("\n| 1 ") == 1  # one component row

    def test_csv_is_parseable(self):
        csv = csv_table(_fit())
        header, *rows = csv.splitlines()
        assert header == "parameter,value,stderr"
        assert any(r.startswith("p0_center,") for r in rows)
        # value column is numeric.
        float(rows[0].split(",")[1])

    def test_latex_is_wellformed(self):
        tex = latex_table(_fit())
        assert tex.startswith("\\begin{table}")
        assert "\\begin{tabular}" in tex
        assert "\\pm" in tex  # uncertainties rendered


class TestTemplates:
    def test_dict_round_trip_preserves_spec(self):
        spec = xps_doublet_preset(932.6, delta_be=19.8, area_ratio=0.5)
        back = spec_from_dict(spec_to_dict(spec))
        assert back.peak_shape is spec.peak_shape
        assert back.background.kind is BackgroundKind.SHIRLEY
        assert len(back.peaks) == 2
        assert back.peaks[0].center == 932.6
        assert len(back.constraints) == 3

    def test_all_constraint_types_round_trip(self):
        spec = FitSpec(
            PeakShape.VOIGT,
            [PeakInit(1.0), PeakInit(2.0)],
            BackgroundSpec(kind=BackgroundKind.ALS, lam=1e4, p=0.02),
            constraints=[
                FixedDelta(0, 1, 1.0),
                FixedRatio(0, 1, 0.5),
                SharedWidth(0, 1),
            ],
        )
        back = spec_from_dict(spec_to_dict(spec))
        assert [type(c).__name__ for c in back.constraints] == [
            "FixedDelta",
            "FixedRatio",
            "SharedWidth",
        ]
        assert back.background.lam == 1e4
        assert back.background.p == 0.02

    def test_save_and_load_file(self, tmp_path: Path):
        spec = xps_doublet_preset(932.6, delta_be=19.8)
        path = tmp_path / "cu2p.fit.json"
        save_template(spec, path)
        assert path.exists()
        loaded = load_template(path)
        assert loaded.peak_shape is spec.peak_shape
        assert len(loaded.constraints) == 3
