"""Tests for `ThermoelectricXlsxParser`."""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
from pytest_snapshot.plugin import Snapshot

from latos.core.enums import Technique
from latos.ingestion.parsers.thermoelectric_xlsx import ThermoelectricXlsxParser

from ._helpers import FIXTURES_DIR, SNAPSHOTS_DIR, parsed_to_snapshot

GOLDEN = FIXTURES_DIR / "thermoelectric" / "zt_calc.xlsx"


def _make_xlsx(path: Path, sheets: dict[str, list[tuple]]):
    """Build a minimal `.xlsx` for testing."""
    wb = openpyxl.Workbook()
    first_name = next(iter(sheets))
    wb.active.title = first_name
    for name, rows in sheets.items():
        ws = wb[name] if name == first_name else wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    wb.save(path)


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert ThermoelectricXlsxParser.name == "thermoelectric-xlsx"

    def test_technique(self):
        assert ThermoelectricXlsxParser.technique is Technique.THERMOELECTRIC

    def test_extensions(self):
        assert ThermoelectricXlsxParser.supported_extensions == (".xlsx",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = ThermoelectricXlsxParser()

    def test_returns_one_for_real_fixture(self):
        assert self.parser.can_parse(GOLDEN) == 1.0

    def test_returns_zero_for_random_xlsx(self, tmp_path: Path):
        f = tmp_path / "ledger.xlsx"
        _make_xlsx(f, {"S": [("Name", "Salary"), ("Alice", 50000)]})
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_uvdrs_lookalike(self, tmp_path: Path):
        # UV-DRS has wavelengths — not Temperature/Seebeck.
        f = tmp_path / "uvdrs.xlsx"
        rows = [(200 + i, 50.0) for i in range(10)]
        _make_xlsx(f, {"S": rows})
        assert self.parser.can_parse(f) == 0.0

    def test_returns_one_for_synthetic_te(self, tmp_path: Path):
        f = tmp_path / "syn.xlsx"
        header = (
            "Temperature (K)",
            "Resistivity (Ohm.m)",
            "Thermal conductivity",
            "Seebeck Coefficient",
        )
        rows = [
            header,
            (300, 0.1, 5.0, 7.5),
            (325, 0.11, 5.1, 7.6),
            (350, 0.12, 5.2, 7.7),
        ]
        _make_xlsx(f, {"CS": rows})
        assert self.parser.can_parse(f) == 1.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self.result = ThermoelectricXlsxParser().parse(GOLDEN)

    def test_technique(self):
        assert self.result.technique is Technique.THERMOELECTRIC

    def test_required_arrays_present(self):
        for k in ("temperature_k", "resistivity_ohm_m", "thermal_conductivity", "seebeck_uvk"):
            assert k in self.result.arrays

    def test_arrays_same_length(self):
        lengths = {len(arr) for arr in self.result.arrays.values()}
        assert len(lengths) == 1

    def test_temperature_increasing(self):
        # Real thermoelectric measurements ramp temperature from low to high.
        temp = self.result.arrays["temperature_k"]
        assert temp[0] < temp[-1]

    def test_temperature_in_kelvin_range(self):
        # 300-700 K is the typical TE measurement range.
        temp = self.result.arrays["temperature_k"]
        assert temp.min() >= 200
        assert temp.max() <= 1000

    def test_first_sheet_metadata(self):
        # Fixture sheets: ['CS', 'CSCBI-1', 'CSCBI-3', 'CSCBI 5 ']
        assert self.result.metadata["sheet_name"] == "CS"

    def test_other_sheets_listed(self):
        all_sheets = self.result.metadata["all_sheet_names"]
        assert len(all_sheets) >= 2

    def test_multi_sheet_warning_present(self):
        assert any(i.field == "sheets" for i in self.result.issues)

    def test_no_errors(self):
        assert not self.result.has_errors


# ─── Golden snapshot ────────────────────────────────────────────────
class TestGoldenSnapshot:
    def test_snapshot_match(self, snapshot: Snapshot):
        result = ThermoelectricXlsxParser().parse(GOLDEN)
        snapshot.snapshot_dir = SNAPSHOTS_DIR
        snapshot.assert_match(
            json.dumps(parsed_to_snapshot(result), indent=2, sort_keys=True) + "\n",
            "thermoelectric_zt_calc.json",
        )


# ─── Failure modes ──────────────────────────────────────────────────
class TestFailureModes:
    def setup_method(self):
        self.parser = ThermoelectricXlsxParser()

    def test_missing_required_column_returns_error(self, tmp_path: Path):
        # Has Temperature + Seebeck (so can_parse hits) but no Resistivity.
        f = tmp_path / "missing_col.xlsx"
        rows = [
            ("Temperature (K)", "Seebeck Coefficient (uV/K)"),
            (300, 7.5),
            (325, 7.6),
        ]
        _make_xlsx(f, {"CS": rows})
        result = self.parser.parse(f)
        assert result.has_errors

    def test_no_data_rows_returns_error(self, tmp_path: Path):
        f = tmp_path / "no_data.xlsx"
        header = (
            "Temperature (K)",
            "Resistivity (Ohm.m)",
            "Thermal conductivity",
            "Seebeck Coefficient",
        )
        _make_xlsx(f, {"CS": [header]})
        result = self.parser.parse(f)
        assert result.has_errors

    def test_optional_columns_picked_up_when_present(self, tmp_path: Path):
        f = tmp_path / "with_optional.xlsx"
        header = (
            "Temperature (K)",
            "Resistivity (Ohm.m)",
            "Thermal conductivity",
            "Seebeck Coefficient",
            "Powerfactor",
            "zT",
        )
        rows = [
            header,
            (300, 0.1, 5.0, 7.5, 500.0, 0.03),
            (325, 0.11, 5.1, 7.6, 510.0, 0.035),
        ]
        _make_xlsx(f, {"CS": rows})
        result = self.parser.parse(f)
        assert "power_factor" in result.arrays
        assert "zt" in result.arrays
