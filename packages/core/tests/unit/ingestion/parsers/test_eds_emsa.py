"""Tests for `latos.ingestion.parsers.eds_emsa.EdsEmsaParser`."""

from __future__ import annotations

from pathlib import Path

import pytest

from latos.core.enums import Technique
from latos.ingestion.parsers.eds_emsa import EdsEmsaParser

_HEADER = """#FORMAT      : EMSA/MAS Spectral Data File
#VERSION     : 1.0
#TITLE       : test spectrum
#XUNITS      : Energy (eV)
#YUNITS      : Intensity
#NPOINTS     : 4.
#DATATYPE    : Y
#XPERCHAN    : 10.
#OFFSET      : 0.
#BEAMKV      : 200.000000
#LIVETIME    : 50.0
#SPECTRUM    : DATA BEGINS HERE
"""


def _write_emsa(path: Path, body: str = "0.000,\n12.000,\n34.000,\n7.000,\n#ENDOFDATA\n") -> Path:
    path.write_text(_HEADER + body)
    return path


@pytest.fixture()
def emsa_file(tmp_path: Path) -> Path:
    return _write_emsa(tmp_path / "002.emsa")


class TestCanParse:
    def test_matches_emsa(self, emsa_file: Path):
        assert EdsEmsaParser().can_parse(emsa_file) == 1.0

    def test_rejects_wrong_extension(self, tmp_path: Path):
        p = tmp_path / "x.txt"
        p.write_text("#FORMAT : EMSA/MAS")
        assert EdsEmsaParser().can_parse(p) == 0.0

    def test_rejects_non_emsa_content(self, tmp_path: Path):
        p = tmp_path / "x.emsa"
        p.write_text("just some random text")
        assert EdsEmsaParser().can_parse(p) == 0.0


class TestParse:
    def test_datatype_y_synthesizes_energy_in_kev(self, emsa_file: Path):
        d = EdsEmsaParser().parse(emsa_file)
        assert d.technique is Technique.EDS
        # OFFSET 0, XPERCHAN 10 eV → 0,10,20,30 eV → 0,0.01,0.02,0.03 keV.
        assert d.arrays["energy_kev"].tolist() == pytest.approx([0.0, 0.01, 0.02, 0.03])
        assert d.arrays["intensity"].tolist() == [0.0, 12.0, 34.0, 7.0]
        assert d.metadata["beam_kv"] == pytest.approx(200.0)
        assert d.metadata["live_time_s"] == pytest.approx(50.0)
        assert d.metadata["n_points"] == 4

    def test_datatype_xy_reads_pairs(self, tmp_path: Path):
        header = _HEADER.replace("#DATATYPE    : Y", "#DATATYPE    : XY")
        p = tmp_path / "xy.emsa"
        p.write_text(header + "1000, 5\n2000, 9\n#ENDOFDATA\n")
        d = EdsEmsaParser().parse(p)
        # XUNITS eV → keV: 1000 eV = 1.0 keV, 2000 eV = 2.0 keV.
        assert d.arrays["energy_kev"].tolist() == pytest.approx([1.0, 2.0])
        assert d.arrays["intensity"].tolist() == [5.0, 9.0]

    def test_no_data_errors(self, tmp_path: Path):
        p = tmp_path / "empty.emsa"
        p.write_text(_HEADER + "#ENDOFDATA\n")
        d = EdsEmsaParser().parse(p)
        assert d.arrays == {}
        assert any(i.field == "data" for i in d.issues)
