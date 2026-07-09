"""Tests for `latos.ingestion.parsers.uvdrs_txt.UvDrsTxtParser`."""

from __future__ import annotations

from pathlib import Path

import pytest

from latos.core.enums import Technique
from latos.ingestion.parsers.uvdrs_txt import UvDrsTxtParser

_BODY = '"CS-1 - RawData"\n"Wavelength nm.","R%"\n200.00,9.645\n201.00,9.769\n202.00,10.515\n'


def _write(path: Path, body: str = _BODY) -> Path:
    path.write_text(body)
    return path


@pytest.fixture()
def uv_file(tmp_path: Path) -> Path:
    return _write(tmp_path / "CS-1.txt")


class TestCanParse:
    def test_matches_uvdrs_txt(self, uv_file: Path):
        assert UvDrsTxtParser().can_parse(uv_file) == 1.0

    def test_rejects_other_txt(self, tmp_path: Path):
        p = tmp_path / "scan.txt"
        p.write_text("01bf\n100 200\n101 250\n")  # a STEM line-scan, no header
        assert UvDrsTxtParser().can_parse(p) == 0.0

    def test_rejects_wrong_extension(self, tmp_path: Path):
        p = tmp_path / "CS-1.csv"
        p.write_text(_BODY)
        assert UvDrsTxtParser().can_parse(p) == 0.0


class TestParse:
    def test_extracts_wavelength_reflectance(self, uv_file: Path):
        d = UvDrsTxtParser().parse(uv_file)
        assert d.technique is Technique.UV_DRS
        assert d.arrays["wavelength"].tolist() == [200.0, 201.0, 202.0]
        assert d.arrays["reflectance"].tolist() == pytest.approx([9.645, 9.769, 10.515])
        assert d.metadata["n_points"] == 3

    def test_sample_name_from_title(self, uv_file: Path):
        assert UvDrsTxtParser().parse(uv_file).metadata["sample_name"] == "CS-1"

    def test_skips_out_of_range_rows(self, tmp_path: Path):
        body = '"CS - RawData"\n"Wavelength nm.","R%"\n200.00,9.6\n99999,1.0\n300.00,8.0\n'
        d = UvDrsTxtParser().parse(_write(tmp_path / "CS.txt", body))
        assert d.arrays["wavelength"].tolist() == [200.0, 300.0]

    def test_no_data_errors(self, tmp_path: Path):
        d = UvDrsTxtParser().parse(_write(tmp_path / "x.txt", '"title"\n"Wavelength nm.","R%"\n'))
        assert d.arrays == {}
        assert any(i.field == "data" for i in d.issues)
