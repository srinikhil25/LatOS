"""Tests for `ResistivitySeebeckXlsxParser`."""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from latos.core.enums import Technique
from latos.ingestion.parsers.resistivity_seebeck_xlsx import ResistivitySeebeckXlsxParser

# A few instrument-header rows, then the data block (blank col A + 3 numbers).
_HEADER = [
    ["Date/Time:", "14-02-25", "Sample:", "CS"],
    ["Operator:", "Simon"],
    ["Resistivity"],
    ["Sensor", "Range:", -5, 5, "uOhm*m"],
    ["Absolute", "seebeck", "coefficient"],
]
_DATA = [
    [None, 43.4395, 0.1244, 7.8523],
    [None, 47.3911, 0.13014, 7.9505],
    [None, 65.9154, 0.14691, 8.9498],
]


def _write_rs(path: Path) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in _HEADER:
        ws.append(r)
    for r in _DATA:
        ws.append(r)
    wb.save(path)
    return path


@pytest.fixture()
def rs_file(tmp_path: Path) -> Path:
    return _write_rs(tmp_path / "CS R and S .xlsx")


class TestCanParse:
    def test_matches_rs_workbook(self, rs_file: Path):
        assert ResistivitySeebeckXlsxParser().can_parse(rs_file) == 1.0

    def test_rejects_without_seebeck(self, tmp_path: Path):
        wb = openpyxl.Workbook()
        ws = wb.active
        for r in _DATA:  # data shape but no "seebeck" keyword anywhere
            ws.append(r)
        p = tmp_path / "x.xlsx"
        wb.save(p)
        assert ResistivitySeebeckXlsxParser().can_parse(p) == 0.0

    def test_rejects_wrong_extension(self, tmp_path: Path):
        p = tmp_path / "CS R and S.txt"
        p.write_text("seebeck")
        assert ResistivitySeebeckXlsxParser().can_parse(p) == 0.0


class TestParse:
    def test_extracts_and_converts(self, rs_file: Path):
        d = ResistivitySeebeckXlsxParser().parse(rs_file)
        assert d.technique is Technique.THERMOELECTRIC
        # °C → K conversion applied.
        assert d.arrays["temperature_k"][0] == pytest.approx(43.4395 + 273.15)
        # Resistivity + Seebeck kept in native units.
        assert d.arrays["resistivity_uohm_m"][0] == pytest.approx(0.1244)
        assert d.arrays["seebeck_uv_k"][0] == pytest.approx(7.8523)
        assert d.metadata["n_points"] == 3
        assert d.metadata["measurement_kind"] == "resistivity_seebeck"
        assert d.metadata["units"]["resistivity_uohm_m"] == "uOhm*m"

    def test_ignores_header_rows(self, rs_file: Path):
        # 5 header rows + 3 data rows → only 3 points extracted.
        d = ResistivitySeebeckXlsxParser().parse(rs_file)
        assert len(d.arrays["temperature_k"]) == 3

    def test_sample_name_stripped(self, rs_file: Path):
        assert ResistivitySeebeckXlsxParser().parse(rs_file).metadata["sample_name"] == "CS"

    def test_sample_name_keeps_doping(self, tmp_path: Path):
        f = _write_rs(tmp_path / "CS-CBI-3 R and S .xlsx")
        assert ResistivitySeebeckXlsxParser().parse(f).metadata["sample_name"] == "CS-CBI-3"
