"""Tests for `ShockSummaryCsvParser`.

The shock-summary format is a small, self-describing CSV our split step
writes (per-composition peak forces from a drop test), so we synthesize
one here rather than committing a real (material) fixture file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from latos.core.enums import Technique
from latos.ingestion.parsers.shock_summary_csv import (
    ShockSummaryCsvParser,
    is_shock_summary_header,
)
from latos.ingestion.parsers.shock_tektronix_csv import ShockTektronixCsvParser
from latos.ingestion.parsers.xps_casaxps_csv import CasaXpsCsvParser
from latos.ingestion.registry import default_registry

# A minimal, synthetic shock summary: three replicate drops of one
# composition. Forces 56/74/68 -> mean 66.0, sample SD (ddof=1) ~9.165.
_SUMMARY_CSV = (
    "Latos Shock Summary,1\n"
    "Ionic Liquid Mass g,1.0033\n"
    "Acrylic Particle Mass g,0.6736\n"
    "Particle Mass Fraction wt%,40.169\n"
    "Particle Volume Fraction vol%,43.840\n"
    "Peak Force Calibration N per V,50\n"
    "Replicate,Peak Voltage V,Peak Force N\n"
    "1,1.12,56\n"
    "2,1.48,74\n"
    "3,1.36,68\n"
)


def _write_summary(tmp_path: Path, name: str = "shock.csv", text: str = _SUMMARY_CSV) -> Path:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert ShockSummaryCsvParser.name == "shock-summary-csv"

    def test_technique(self):
        assert ShockSummaryCsvParser.technique is Technique.SHOCK

    def test_extensions(self):
        assert ShockSummaryCsvParser.supported_extensions == (".csv",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = ShockSummaryCsvParser()

    def test_returns_one_for_summary_csv(self, tmp_path: Path):
        assert self.parser.can_parse(_write_summary(tmp_path)) == 1.0

    def test_header_helper_is_case_and_space_insensitive(self):
        assert is_shock_summary_header("Latos Shock Summary,1")
        assert is_shock_summary_header("  latos shock summary ,1")
        assert not is_shock_summary_header("Model,TBS1052C")

    def test_returns_zero_for_plain_csv(self, tmp_path: Path):
        f = tmp_path / "plain.csv"
        f.write_text("a,b\n1,2\n", encoding="utf-8")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        f = tmp_path / "summary.txt"
        f.write_text(_SUMMARY_CSV, encoding="utf-8")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_missing_file(self, tmp_path: Path):
        assert self.parser.can_parse(tmp_path / "missing.csv") == 0.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def test_technique_and_no_arrays(self, tmp_path: Path):
        result = ShockSummaryCsvParser().parse(_write_summary(tmp_path))
        assert result.technique is Technique.SHOCK
        assert result.arrays == {}
        assert not result.issues

    def test_peak_force_and_voltage_are_means(self, tmp_path: Path):
        result = ShockSummaryCsvParser().parse(_write_summary(tmp_path))
        assert result.features["peak_force_n"] == pytest.approx(66.0)
        assert result.features["peak_voltage_v"] == pytest.approx(1.32)

    def test_scatter_is_sample_standard_deviation(self, tmp_path: Path):
        result = ShockSummaryCsvParser().parse(_write_summary(tmp_path))
        # ddof=1 standard deviation of 56 / 74 / 68.
        assert result.features["peak_force_sd_n"] == pytest.approx(9.16515, rel=1e-4)

    def test_composition_promoted_to_features(self, tmp_path: Path):
        result = ShockSummaryCsvParser().parse(_write_summary(tmp_path))
        assert result.features["particle_vol_pct"] == pytest.approx(43.840)
        assert result.features["particle_wt_pct"] == pytest.approx(40.169)

    def test_metadata(self, tmp_path: Path):
        result = ShockSummaryCsvParser().parse(_write_summary(tmp_path))
        assert result.metadata["ionic_liquid_mass_g"] == pytest.approx(1.0033)
        assert result.metadata["acrylic_particle_mass_g"] == pytest.approx(0.6736)
        assert result.metadata["force_calibration_n_per_v"] == pytest.approx(50.0)
        assert result.metadata["n_replicates"] == 3
        assert result.instrument == "drop test (peak summary)"

    def test_parser_identity(self, tmp_path: Path):
        result = ShockSummaryCsvParser().parse(_write_summary(tmp_path))
        assert result.parser_name == ShockSummaryCsvParser.name
        assert result.parser_version == ShockSummaryCsvParser.version


# ─── Parse — edge cases ─────────────────────────────────────────────
class TestParseEdgeCases:
    def test_no_replicate_rows_is_an_issue(self, tmp_path: Path):
        text = (
            "Latos Shock Summary,1\n"
            "Particle Volume Fraction vol%,43.840\n"
            "Replicate,Peak Voltage V,Peak Force N\n"
        )
        result = ShockSummaryCsvParser().parse(_write_summary(tmp_path, text=text))
        assert result.features == {}
        assert any(i.field == "data" for i in result.issues)

    def test_single_replicate_has_no_scatter(self, tmp_path: Path):
        text = (
            "Latos Shock Summary,1\n"
            "Particle Volume Fraction vol%,43.840\n"
            "Replicate,Peak Voltage V,Peak Force N\n"
            "1,1.12,56\n"
        )
        result = ShockSummaryCsvParser().parse(_write_summary(tmp_path, text=text))
        assert result.features["peak_force_n"] == pytest.approx(56.0)
        assert "peak_force_sd_n" not in result.features


# ─── Registry dispatch / regression ─────────────────────────────────
class TestRegistryDispatch:
    def test_registry_dispatches_to_shock_summary(self, tmp_path: Path):
        match = default_registry().find_parser(_write_summary(tmp_path))
        assert match is not None
        assert match.parser.name == "shock-summary-csv"
        assert match.confidence == 1.0

    def test_tektronix_rejects_summary(self, tmp_path: Path):
        assert ShockTektronixCsvParser().can_parse(_write_summary(tmp_path)) == 0.0

    def test_greedy_casaxps_does_not_win_dispatch(self, tmp_path: Path):
        # CasaXPS is a deliberately greedy .csv matcher, registered LAST so
        # it can't beat anything else. It also claims a shock summary, but
        # the shock-summary parser is registered earlier, so dispatch still
        # lands on shock-summary (verified above).
        f = _write_summary(tmp_path)
        assert CasaXpsCsvParser().can_parse(f) > 0.0
        assert default_registry().find_parser(f).parser.name == "shock-summary-csv"
