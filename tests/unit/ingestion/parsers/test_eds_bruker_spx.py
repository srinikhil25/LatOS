"""Tests for `BrukerSpxParser`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pytest_snapshot.plugin import Snapshot

from latos.core.enums import Technique
from latos.ingestion.parsers.eds_bruker_spx import BrukerSpxParser

from ._helpers import FIXTURES_DIR, SNAPSHOTS_DIR, parsed_to_snapshot

GOLDEN = FIXTURES_DIR / "eds" / "bruker_1.spx"


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert BrukerSpxParser.name == "bruker-eds-spx"

    def test_technique(self):
        assert BrukerSpxParser.technique is Technique.EDS

    def test_extensions(self):
        assert BrukerSpxParser.supported_extensions == (".spx",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = BrukerSpxParser()

    def test_returns_one_for_real_fixture(self):
        assert self.parser.can_parse(GOLDEN) == 1.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        f = tmp_path / "data.xml"
        f.write_text("<TRTSpectrum></TRTSpectrum>")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_random_spx(self, tmp_path: Path):
        f = tmp_path / "fake.spx"
        f.write_text("<NotABrukerFile/>")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_missing(self, tmp_path: Path):
        assert self.parser.can_parse(tmp_path / "missing.spx") == 0.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self.result = BrukerSpxParser().parse(GOLDEN)

    def test_technique(self):
        assert self.result.technique is Technique.EDS

    def test_arrays_present(self):
        assert "energy_kev" in self.result.arrays
        assert "intensity" in self.result.arrays

    def test_arrays_same_length(self):
        e = self.result.arrays["energy_kev"]
        c = self.result.arrays["intensity"]
        assert len(e) == len(c)

    def test_channel_count_matches_metadata(self):
        # Fixture has ChannelCount=4096.
        assert self.result.metadata["channel_count"] == 4096
        assert len(self.result.arrays["energy_kev"]) == 4096

    def test_calibration_extracted(self):
        # Fixture: CalibAbs=-0.48..., CalibLin=0.005...
        assert self.result.metadata["calib_abs"] == pytest.approx(-0.48, abs=0.1)
        assert self.result.metadata["calib_lin"] == pytest.approx(0.005, abs=0.001)

    def test_energy_axis_synthesized(self):
        # First channel: energy = CalibAbs + CalibLin * 0 = CalibAbs.
        e = self.result.arrays["energy_kev"]
        assert e[0] == pytest.approx(self.result.metadata["calib_abs"])

    def test_energy_monotonic_increasing(self):
        e = self.result.arrays["energy_kev"]
        assert np.all(np.diff(e) > 0)

    def test_primary_energy_extracted(self):
        # Fixture: PrimaryEnergy=2.5E1 → 25 kV.
        assert self.result.metadata["primary_energy_kv"] == pytest.approx(25.0)

    def test_measured_at_extracted_with_warning(self):
        # Date+Time present → measured_at set, but with warning about
        # missing tz info.
        assert self.result.measured_at is not None
        assert self.result.measured_at.tzinfo is not None
        warning_fields = [i.field for i in self.result.issues]
        assert "measured_at" in warning_fields


# ─── Golden snapshot ────────────────────────────────────────────────
class TestGoldenSnapshot:
    def test_snapshot_match(self, snapshot: Snapshot):
        result = BrukerSpxParser().parse(GOLDEN)
        snapshot.snapshot_dir = SNAPSHOTS_DIR
        snapshot.assert_match(
            json.dumps(parsed_to_snapshot(result), indent=2, sort_keys=True) + "\n",
            "bruker_eds_1.json",
        )


# ─── Failure modes ──────────────────────────────────────────────────
class TestFailureModes:
    def setup_method(self):
        self.parser = BrukerSpxParser()

    def test_malformed_xml_returns_error(self, tmp_path: Path):
        f = tmp_path / "broken.spx"
        f.write_text("<TRTSpectrum><not closed")
        result = self.parser.parse(f)
        assert result.has_errors

    def test_no_channels_returns_error(self, tmp_path: Path):
        f = tmp_path / "no_chan.spx"
        f.write_text("<TRTSpectrum><CalibAbs>0</CalibAbs><CalibLin>0.005</CalibLin></TRTSpectrum>")
        result = self.parser.parse(f)
        assert result.has_errors

    def test_missing_calibration_warns(self, tmp_path: Path):
        f = tmp_path / "no_calib.spx"
        f.write_text("<TRTSpectrum><Channels>1,2,3,4,5</Channels></TRTSpectrum>")
        result = self.parser.parse(f)
        assert not result.has_errors  # we got data
        warning_fields = [i.field for i in result.issues]
        assert "calibration" in warning_fields
        # Energy axis falls back to indices.
        np.testing.assert_array_equal(result.arrays["energy_kev"], [0, 1, 2, 3, 4])
