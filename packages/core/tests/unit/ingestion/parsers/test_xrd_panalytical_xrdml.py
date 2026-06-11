"""Tests for `PanalyticalXrdmlParser`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pytest_snapshot.plugin import Snapshot

from latos.core.enums import Technique
from latos.ingestion.parsers.xrd_panalytical_xrdml import PanalyticalXrdmlParser

from ._helpers import FIXTURES_DIR, SNAPSHOTS_DIR, parsed_to_snapshot

GOLDEN = FIXTURES_DIR / "xrd" / "panalytical_cscbi1.xrdml"


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert PanalyticalXrdmlParser.name == "panalytical-xrdml"

    def test_technique(self):
        assert PanalyticalXrdmlParser.technique is Technique.XRD

    def test_extensions(self):
        assert PanalyticalXrdmlParser.supported_extensions == (".xrdml",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = PanalyticalXrdmlParser()

    def test_returns_one_for_valid_xrdml(self):
        assert self.parser.can_parse(GOLDEN) == 1.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("<xrdMeasurements xmlns='http://www.xrdml.com/X/1.7'/>")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_random_xml(self, tmp_path: Path):
        f = tmp_path / "data.xrdml"
        f.write_text("<?xml version='1.0'?><root>not xrd data</root>")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_missing_file(self, tmp_path: Path):
        assert self.parser.can_parse(tmp_path / "missing.xrdml") == 0.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self.parser = PanalyticalXrdmlParser()
        self.result = self.parser.parse(GOLDEN)

    def test_technique_is_xrd(self):
        assert self.result.technique is Technique.XRD

    def test_arrays_present(self):
        assert "two_theta" in self.result.arrays
        assert "intensity" in self.result.arrays

    def test_arrays_same_length(self):
        assert len(self.result.arrays["two_theta"]) == len(self.result.arrays["intensity"])

    def test_two_theta_monotonic_increasing(self):
        tt = self.result.arrays["two_theta"]
        assert np.all(np.diff(tt) > 0)

    def test_two_theta_range_matches_metadata(self):
        m = self.result.metadata
        tt = self.result.arrays["two_theta"]
        assert tt[0] == pytest.approx(m["scan_start_deg"])
        assert tt[-1] == pytest.approx(m["scan_finish_deg"])

    def test_metadata_extracted(self):
        m = self.result.metadata
        assert m["sample_id"] == "Dr.MN-dhivya-cscbi1"
        assert m["wavelength_ka1"] == pytest.approx(1.5405980)
        assert m["anode_material"] == "Cu"
        assert m["tube_tension_kv"] == pytest.approx(40.0)
        assert m["tube_current_ma"] == pytest.approx(15.0)

    def test_instrument_from_tube_name(self):
        # The fixture has tube name "Empyrean Cu LFF HR (...)".
        assert self.result.instrument is not None
        assert "Empyrean" in self.result.instrument

    def test_measured_at_extracted(self):
        # Fixture has <startTimeStamp>2024-06-21T11:00:13+05:30</startTimeStamp>.
        assert self.result.measured_at is not None
        assert self.result.measured_at.tzinfo is not None
        assert self.result.measured_at.year == 2024

    def test_no_errors_on_clean_file(self):
        assert not self.result.has_errors


# ─── Golden-file snapshot ───────────────────────────────────────────
class TestGoldenSnapshot:
    def test_snapshot_match(self, snapshot: Snapshot):
        result = PanalyticalXrdmlParser().parse(GOLDEN)
        snapshot.snapshot_dir = SNAPSHOTS_DIR
        snapshot.assert_match(
            json.dumps(parsed_to_snapshot(result), indent=2, sort_keys=True) + "\n",
            "panalytical_cscbi1.json",
        )


# ─── Failure modes ──────────────────────────────────────────────────
class TestFailureModes:
    def setup_method(self):
        self.parser = PanalyticalXrdmlParser()

    def test_malformed_xml_returns_error(self, tmp_path: Path):
        f = tmp_path / "broken.xrdml"
        f.write_text("<xrdMeasurements xmlns='http://www.xrdml.com/X/1.7'>not closed")
        result = self.parser.parse(f)
        assert result.has_errors
        assert result.arrays == {}

    def test_no_intensities_returns_error(self, tmp_path: Path):
        f = tmp_path / "empty.xrdml"
        f.write_text(
            "<?xml version='1.0'?>"
            "<xrdMeasurements xmlns='http://www.xrdml.com/X/1.7'>"
            "<sample><id>X</id></sample></xrdMeasurements>",
        )
        result = self.parser.parse(f)
        assert result.has_errors
        assert result.arrays == {}

    def test_missing_positions_falls_back_to_indices(self, tmp_path: Path):
        # Intensities present but no <positions axis="2Theta"> — emit a
        # warning and use indices as the x-axis.
        f = tmp_path / "no_positions.xrdml"
        f.write_text(
            "<?xml version='1.0'?>"
            "<xrdMeasurements xmlns='http://www.xrdml.com/X/1.7'>"
            "<intensities>10 20 30 40</intensities>"
            "</xrdMeasurements>",
        )
        result = self.parser.parse(f)
        assert not result.has_errors
        assert result.has_warnings
        assert len(result.arrays["intensity"]) == 4
        # Indices fallback: 0, 1, 2, 3.
        np.testing.assert_array_equal(result.arrays["two_theta"], [0.0, 1.0, 2.0, 3.0])
