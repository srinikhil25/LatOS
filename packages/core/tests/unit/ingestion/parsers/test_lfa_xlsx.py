"""Tests for `latos.ingestion.parsers.lfa_xlsx.LfaXlsxParser`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pytest

from latos.core.enums import Severity, Technique
from latos.ingestion.parsers.lfa_xlsx import LfaXlsxParser

_HEADER = [
    "#Temperature/K",
    "#Model",
    "#Diffusivity/(mm^2/s)",
    "#Conductivity/(W/(m*K))",
    "#Cp-Calc/(J/(g*K))",
]
_ROWS = [
    (300, "Standard + p.c.(l)", 2.574, 5.1455, 0.3501),
    (325, "Standard + p.c.(l)", 2.458, 5.1206, 0.3648),
    (350, "Standard + p.c.(l)", 2.367, 4.9464, 0.3660),
]


def _write_lfa(path: Path, *, blank_leading: int = 3) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    for _ in range(blank_leading):
        ws.append([])
    ws.append(_HEADER)
    for r in _ROWS:
        ws.append(list(r))
    wb.save(path)
    return path


@pytest.fixture()
def lfa_file(tmp_path: Path) -> Path:
    return _write_lfa(tmp_path / "CS LFA.xlsx")


class TestCanParse:
    def test_matches_lfa_workbook(self, lfa_file: Path):
        assert LfaXlsxParser().can_parse(lfa_file) == 1.0

    def test_rejects_non_lfa_xlsx(self, tmp_path: Path):
        wb = openpyxl.Workbook()
        wb.active.append(["wavelength", "reflectance"])
        wb.active.append([400, 12.3])
        p = tmp_path / "other.xlsx"
        wb.save(p)
        assert LfaXlsxParser().can_parse(p) == 0.0

    def test_rejects_wrong_extension(self, tmp_path: Path):
        p = tmp_path / "CS LFA.txt"
        p.write_text("x")
        assert LfaXlsxParser().can_parse(p) == 0.0


class TestParse:
    def test_extracts_conductivity_and_temperature(self, lfa_file: Path):
        d = LfaXlsxParser().parse(lfa_file)
        assert d.technique is Technique.THERMOELECTRIC
        assert list(d.arrays["temperature_k"]) == [300, 325, 350]
        assert d.arrays["thermal_conductivity"][0] == pytest.approx(5.1455)
        assert d.arrays["diffusivity_mm2_s"][0] == pytest.approx(2.574)
        assert d.metadata["measurement_kind"] == "lfa"
        assert d.metadata["n_points"] == 3

    def test_sample_name_stripped_from_filename(self, lfa_file: Path):
        d = LfaXlsxParser().parse(lfa_file)
        assert d.metadata["sample_name"] == "CS"

    def test_sample_name_keeps_doping_label(self, tmp_path: Path):
        f = _write_lfa(tmp_path / "CS-CBI-1 LFA.xlsx")
        d = LfaXlsxParser().parse(f)
        assert d.metadata["sample_name"] == "CS-CBI-1"

    def test_empty_below_header_errors(self, tmp_path: Path):
        wb = openpyxl.Workbook()
        wb.active.append(_HEADER)  # header only, no data
        p = tmp_path / "CS LFA.xlsx"
        wb.save(p)
        d = LfaXlsxParser().parse(p)
        assert d.arrays == {}
        assert any(i.field == "data" for i in d.issues)


def _write_rows(path: Path, rows) -> Path:
    wb = openpyxl.Workbook()
    wb.active.append(_HEADER)
    for r in rows:
        wb.active.append(list(r))
    wb.save(path)
    return path


def _break_first_sheet(path: Path) -> Path:
    """Truncate the sheet XML but leave the workbook openable.

    openpyxl reads a read-only sheet lazily, so this fails during iteration,
    after `load_workbook` has succeeded.
    """
    import zipfile

    with zipfile.ZipFile(path) as source:
        parts = {info.filename: source.read(info.filename) for info in source.infolist()}
    sheet = "xl/worksheets/sheet1.xml"
    parts[sheet] = parts[sheet][: len(parts[sheet]) // 2]
    with zipfile.ZipFile(path, "w") as target:
        for name, data in parts.items():
            target.writestr(name, data)
    return path


class TestIncompleteRows:
    """July #5, re-examined 2026-09-17: nothing consumes the diffusivity array, so
    a missing diffusivity no longer costs the row its conductivity; both gaps are
    now reported."""

    def test_a_complete_file_raises_no_issue(self, lfa_file: Path):
        assert LfaXlsxParser().parse(lfa_file).issues == ()

    def test_a_missing_diffusivity_is_kept_as_missing_and_reported(self, tmp_path: Path):
        rows = [_ROWS[0], (325, "Standard + p.c.(l)", None, 5.1206, 0.3648), _ROWS[2]]
        d = LfaXlsxParser().parse(_write_rows(tmp_path / "CS LFA.xlsx", rows))
        assert list(d.arrays["temperature_k"]) == [300, 325, 350]
        assert d.arrays["thermal_conductivity"][1] == pytest.approx(5.1206)
        assert np.isnan(d.arrays["diffusivity_mm2_s"][1])
        (issue,) = d.issues
        assert issue.field == "diffusivity_mm2_s"
        assert issue.severity is Severity.WARNING

    def test_rows_without_a_conductivity_are_counted(self, tmp_path: Path):
        rows = [(300, "Standard", 2.574, None, 0.35), (325, "Standard", 2.458, "-", 0.36)]
        d = LfaXlsxParser().parse(_write_rows(tmp_path / "CS LFA.xlsx", rows))
        assert d.arrays == {}
        assert any(
            i.field == "thermal_conductivity" and i.message.startswith("2 row(s)") for i in d.issues
        )


class TestDamagedFiles:
    def test_a_sheet_that_fails_to_read_is_reported_not_raised(self, lfa_file: Path):
        d = LfaXlsxParser().parse(_break_first_sheet(lfa_file))
        assert d.arrays == {}
        assert any(i.field == "file" and i.severity is Severity.ERROR for i in d.issues)
