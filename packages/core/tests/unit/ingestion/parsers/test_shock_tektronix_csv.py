"""Tests for `ShockTektronixCsvParser`.

The Tektronix CSV format is plain text, so we synthesize a minimal
waveform here rather than committing a real (material) fixture file.
"""

from __future__ import annotations

from pathlib import Path

from latos.core.enums import Technique
from latos.ingestion.parsers.shock_tektronix_csv import ShockTektronixCsvParser
from latos.ingestion.parsers.xps_casaxps_csv import CasaXpsCsvParser
from latos.ingestion.registry import default_registry

# A minimal, synthetic Tektronix TBS-series export: header, TIME,CH1
# column line, then five samples whose peak is 0.9 V at t = 0.
_SCOPE_CSV = (
    "Model,TBS1052C\n"
    "Firmware Version,v-test\n"
    "\n"
    "Point Format,Y\n"
    "Horizontal Units,S\n"
    "Sample Interval,0.001\n"
    "Record Length,5\n"
    "Vertical Units,V\n"
    "Vertical Scale,0.5\n"
    "Label,\n"
    "TIME,CH1\n"
    "-0.002,0.0\n"
    "-0.001,0.1\n"
    "0.000,0.9\n"
    "0.001,-0.3\n"
    "0.002,0.05\n"
)


def _write_scope(tmp_path: Path, name: str = "shot.csv") -> Path:
    f = tmp_path / name
    f.write_text(_SCOPE_CSV, encoding="utf-8")
    return f


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert ShockTektronixCsvParser.name == "shock-tektronix-csv"

    def test_technique(self):
        assert ShockTektronixCsvParser.technique is Technique.SHOCK

    def test_extensions(self):
        assert ShockTektronixCsvParser.supported_extensions == (".csv",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = ShockTektronixCsvParser()

    def test_returns_one_for_scope_csv(self, tmp_path: Path):
        assert self.parser.can_parse(_write_scope(tmp_path)) == 1.0

    def test_returns_zero_for_plain_csv(self, tmp_path: Path):
        f = tmp_path / "plain.csv"
        f.write_text("296.0,6568.0\n295.9,6590.0\n295.8,6600.0\n", encoding="utf-8")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text(_SCOPE_CSV, encoding="utf-8")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_missing_file(self, tmp_path: Path):
        assert self.parser.can_parse(tmp_path / "missing.csv") == 0.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self._tmp = Path  # placeholder; real parse in each test via fixture

    def test_technique_and_arrays(self, tmp_path: Path):
        result = ShockTektronixCsvParser().parse(_write_scope(tmp_path))
        assert result.technique is Technique.SHOCK
        assert set(result.arrays) == {"time_s", "voltage_v"}
        assert len(result.arrays["voltage_v"]) == 5
        assert not result.issues

    def test_peak_feature(self, tmp_path: Path):
        result = ShockTektronixCsvParser().parse(_write_scope(tmp_path))
        assert result.features["peak_voltage_v"] == 0.9
        assert result.features["peak_time_ms"] == 0.0

    def test_metadata(self, tmp_path: Path):
        result = ShockTektronixCsvParser().parse(_write_scope(tmp_path))
        assert result.metadata["instrument_model"] == "TBS1052C"
        assert result.metadata["n_points"] == 5
        assert result.instrument == "TBS1052C"

    def test_parser_identity(self, tmp_path: Path):
        result = ShockTektronixCsvParser().parse(_write_scope(tmp_path))
        assert result.parser_name == ShockTektronixCsvParser.name
        assert result.parser_version == ShockTektronixCsvParser.version


# ─── Regression: no longer mistaken for XPS ─────────────────────────
class TestNotMistakenForXps:
    def test_casaxps_rejects_scope(self, tmp_path: Path):
        assert CasaXpsCsvParser().can_parse(_write_scope(tmp_path)) == 0.0

    def test_registry_dispatches_to_shock(self, tmp_path: Path):
        match = default_registry().find_parser(_write_scope(tmp_path))
        assert match is not None
        assert match.parser.name == "shock-tektronix-csv"
        assert match.confidence == 1.0
