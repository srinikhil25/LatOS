"""Tests for `UvDrsXlsxParser`."""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
from pytest_snapshot.plugin import Snapshot

from latos.core.enums import Technique
from latos.ingestion.parsers.uvdrs_xlsx import UvDrsXlsxParser

from ._helpers import FIXTURES_DIR, SNAPSHOTS_DIR, parsed_to_snapshot

GOLDEN = FIXTURES_DIR / "uvdrs" / "uvdrs_cs.xlsx"


def _make_xlsx(path: Path, sheets: dict[str, list[tuple]]):
    """Build a minimal `.xlsx` for testing. `sheets` maps name → list of row tuples."""
    wb = openpyxl.Workbook()
    # openpyxl auto-creates one sheet; rename and reuse it for the first.
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
        assert UvDrsXlsxParser.name == "uvdrs-xlsx"

    def test_technique(self):
        assert UvDrsXlsxParser.technique is Technique.UV_DRS

    def test_extensions(self):
        assert UvDrsXlsxParser.supported_extensions == (".xlsx",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = UvDrsXlsxParser()

    def test_returns_one_for_real_fixture(self):
        assert self.parser.can_parse(GOLDEN) == 1.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        f = tmp_path / "data.xls"
        _make_xlsx(tmp_path / "data.xlsx", {"S": [(200, 0.1)]})
        # Rename to .xls
        (tmp_path / "data.xlsx").rename(f)
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_random_xlsx(self, tmp_path: Path):
        f = tmp_path / "ledger.xlsx"
        _make_xlsx(f, {"Sheet1": [("Name", "Value"), ("Alice", 42), ("Bob", 17)]})
        assert self.parser.can_parse(f) == 0.0

    def test_synthetic_uvdrs_recognized(self, tmp_path: Path):
        f = tmp_path / "syn.xlsx"
        rows = [(200 + i, 50 + 0.1 * i) for i in range(10)]
        _make_xlsx(f, {"CS": rows})
        assert self.parser.can_parse(f) == 1.0

    def test_returns_zero_for_out_of_range_wavelength(self, tmp_path: Path):
        # Column A values way outside UV range → not UV-DRS.
        f = tmp_path / "bad_range.xlsx"
        rows = [(1e6 + i, 0.5) for i in range(5)]
        _make_xlsx(f, {"S": rows})
        assert self.parser.can_parse(f) == 0.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self.result = UvDrsXlsxParser().parse(GOLDEN)

    def test_technique(self):
        assert self.result.technique is Technique.UV_DRS

    def test_arrays_present(self):
        assert "wavelength" in self.result.arrays
        assert "reflectance" in self.result.arrays

    def test_arrays_same_length(self):
        wl = self.result.arrays["wavelength"]
        rf = self.result.arrays["reflectance"]
        assert len(wl) == len(rf)

    def test_wavelength_in_uv_range(self):
        wl = self.result.arrays["wavelength"]
        assert wl.min() >= 100
        assert wl.max() <= 2500

    def test_first_sheet_recorded_in_metadata(self):
        # Fixture sheet names: ['CS', 'CS-1', 'CS-3', 'CS-5'].
        assert self.result.metadata["sheet_name"] == "CS"

    def test_other_sheets_listed_in_metadata(self):
        all_sheets = self.result.metadata["all_sheet_names"]
        assert "CS" in all_sheets
        assert "CS-1" in all_sheets

    def test_multi_sheet_no_longer_warns_about_skipped_sheets(self):
        # As of 1.0.2 the parser exposes `parse_all()` so every sheet
        # is parsed as its own measurement. The old "Workbook has N
        # sheets" warning is therefore gone - skipping no longer
        # happens - and `parse()` returns the first non-empty sheet
        # alone (no warning either).
        assert not any(i.field == "sheets" for i in self.result.issues)

    def test_parse_all_returns_one_result_per_sheet(self):
        # Fixture has 4 sheets, each with valid UV-DRS data → 4 results.
        # The sheet names flow into each result's metadata so the
        # orchestrator can route them to distinct samples.
        results = UvDrsXlsxParser().parse_all(GOLDEN)
        sheet_names = [r.metadata["sheet_name"] for r in results]
        assert len(results) == 4
        assert sheet_names == ["CS", "CS-1", "CS-3", "CS-5"]
        # Each result should carry arrays - empty results are filtered.
        for r in results:
            assert "wavelength" in r.arrays
            assert "reflectance" in r.arrays

    def test_no_errors(self):
        assert not self.result.has_errors


# ─── Golden snapshot ────────────────────────────────────────────────
class TestGoldenSnapshot:
    def test_snapshot_match(self, snapshot: Snapshot):
        result = UvDrsXlsxParser().parse(GOLDEN)
        snapshot.snapshot_dir = SNAPSHOTS_DIR
        snapshot.assert_match(
            json.dumps(parsed_to_snapshot(result), indent=2, sort_keys=True) + "\n",
            "uvdrs_cs.json",
        )


# ─── Failure modes ──────────────────────────────────────────────────
class TestFailureModes:
    def setup_method(self):
        self.parser = UvDrsXlsxParser()

    def test_empty_workbook_returns_error(self, tmp_path: Path):
        f = tmp_path / "empty.xlsx"
        _make_xlsx(f, {"Empty": []})
        result = self.parser.parse(f)
        assert result.has_errors

    def test_no_uvdrs_data_returns_error(self, tmp_path: Path):
        f = tmp_path / "junk.xlsx"
        _make_xlsx(f, {"Junk": [("a", "b"), ("c", "d")]})
        result = self.parser.parse(f)
        assert result.has_errors

    def test_single_sheet_no_multisheet_warning(self, tmp_path: Path):
        f = tmp_path / "single.xlsx"
        rows = [(200 + i, 50.0) for i in range(10)]
        _make_xlsx(f, {"OnlyOne": rows})
        result = self.parser.parse(f)
        assert not any(i.field == "sheets" for i in result.issues)

    def test_skip_leading_blank_rows(self, tmp_path: Path):
        f = tmp_path / "with_blanks.xlsx"
        rows = [(None, None)] * 5 + [(200 + i, 50.0) for i in range(10)]
        _make_xlsx(f, {"S": rows})
        result = self.parser.parse(f)
        assert not result.has_errors
        assert len(result.arrays["wavelength"]) == 10
