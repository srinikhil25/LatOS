"""Tests for `CasaXpsCsvParser`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_snapshot.plugin import Snapshot

from latos.core.enums import Technique
from latos.ingestion.parsers.xps_casaxps_csv import CasaXpsCsvParser

from ._helpers import FIXTURES_DIR, SNAPSHOTS_DIR, parsed_to_snapshot

GOLDEN = FIXTURES_DIR / "xps" / "casaxps_c1s.csv"


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert CasaXpsCsvParser.name == "casaxps-csv"

    def test_technique(self):
        assert CasaXpsCsvParser.technique is Technique.XPS

    def test_extensions(self):
        assert CasaXpsCsvParser.supported_extensions == (".csv",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = CasaXpsCsvParser()

    def test_returns_one_for_valid_file(self):
        assert self.parser.can_parse(GOLDEN) == 1.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("3\n \nC1s\n1\n296.0658,6568.7500\n")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_random_csv(self, tmp_path: Path):
        f = tmp_path / "ledger.csv"
        f.write_text("Date,Amount,Note\n2024-01-01,500,Salary\n2024-01-02,42,Lunch\n")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        assert self.parser.can_parse(f) == 0.0

    def test_synthetic_xps_file_recognized(self, tmp_path: Path):
        f = tmp_path / "test.csv"
        rows = "\n".join(f"{280 + i * 0.1},{1000 + i}" for i in range(5))
        f.write_text(f"3\n \nO1s\n1\n{rows}\n")
        assert self.parser.can_parse(f) == 1.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self.result = CasaXpsCsvParser().parse(GOLDEN)

    def test_technique(self):
        assert self.result.technique is Technique.XPS

    def test_arrays_present(self):
        assert "binding_energy" in self.result.arrays
        assert "intensity" in self.result.arrays

    def test_arrays_same_length(self):
        be = self.result.arrays["binding_energy"]
        inten = self.result.arrays["intensity"]
        assert len(be) == len(inten)

    def test_first_data_point(self):
        # First data line: "296.0658,6568.7500"
        assert self.result.arrays["binding_energy"][0] == pytest.approx(296.0658)
        assert self.result.arrays["intensity"][0] == pytest.approx(6568.75)

    def test_region_label_extracted(self):
        # Header has "C1s" on its own line.
        assert self.result.metadata["region"] == "C1s"

    def test_filename_stem_in_metadata(self):
        assert self.result.metadata["filename_stem"] == "casaxps_c1s"

    def test_n_points_consistent(self):
        assert self.result.metadata["n_points"] == len(self.result.arrays["binding_energy"])

    def test_no_errors(self):
        assert not self.result.has_errors

    def test_be_decreases_typical_xps(self):
        # XPS scans high BE → low BE. The fixture starts at 296 and decreases.
        be = self.result.arrays["binding_energy"]
        assert be[0] > be[-1]


# ─── Golden snapshot ────────────────────────────────────────────────
class TestGoldenSnapshot:
    def test_snapshot_match(self, snapshot: Snapshot):
        result = CasaXpsCsvParser().parse(GOLDEN)
        snapshot.snapshot_dir = SNAPSHOTS_DIR
        snapshot.assert_match(
            json.dumps(parsed_to_snapshot(result), indent=2, sort_keys=True) + "\n",
            "casaxps_c1s.json",
        )


# ─── Failure modes ──────────────────────────────────────────────────
class TestFailureModes:
    def setup_method(self):
        self.parser = CasaXpsCsvParser()

    def test_empty_file_returns_error(self, tmp_path: Path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        result = self.parser.parse(f)
        assert result.has_errors

    def test_header_only_returns_error(self, tmp_path: Path):
        f = tmp_path / "header_only.csv"
        f.write_text("3\n \nC1s\n1\n")
        result = self.parser.parse(f)
        assert result.has_errors

    def test_partial_garbage_keeps_valid_rows(self, tmp_path: Path):
        f = tmp_path / "mixed.csv"
        f.write_text("296.0,6500\n295.9,6600\nGARBAGE,LINE\n295.7,6400\n")
        result = self.parser.parse(f)
        assert not result.has_errors  # got data
        assert result.has_warnings
        assert len(result.arrays["binding_energy"]) == 3

    def test_non_monotonic_warns(self, tmp_path: Path):
        f = tmp_path / "non_mono.csv"
        f.write_text("296.0,6500\n295.0,6600\n297.0,6400\n296.5,6300\n")
        result = self.parser.parse(f)
        assert result.has_warnings
        assert any("monotonic" in i.message.lower() for i in result.issues)

    def test_strictly_increasing_be_no_monotonic_warning(self, tmp_path: Path):
        # Either increasing or decreasing is OK; only mixed direction warns.
        f = tmp_path / "increasing.csv"
        f.write_text("280.0,1000\n281.0,1100\n282.0,1200\n")
        result = self.parser.parse(f)
        assert not any("monotonic" in i.message.lower() for i in result.issues)


# ─── Region label parsing edge cases ────────────────────────────────
class TestRegionLabelExtraction:
    def setup_method(self):
        self.parser = CasaXpsCsvParser()

    def test_no_label_when_header_is_only_numbers(self, tmp_path: Path):
        f = tmp_path / "no_label.csv"
        f.write_text("3\n1\n296.0,6500\n295.9,6600\n295.8,6400\n")
        result = self.parser.parse(f)
        # No alphabetic label in the header → region remains None.
        assert result.metadata["region"] is None

    @pytest.mark.parametrize("label", ["C1s", "Cu2p", "O1s", "Se3d", "Ti2p"])
    def test_common_xps_regions_recognized(self, tmp_path: Path, label: str):
        f = tmp_path / "labeled.csv"
        f.write_text(f"3\n \n{label}\n1\n280.0,1000\n279.9,1100\n279.8,1200\n")
        result = self.parser.parse(f)
        assert result.metadata["region"] == label
