"""Tests for `RigakuXrdAscParser`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pytest_snapshot.plugin import Snapshot

from latos.core.enums import Technique
from latos.ingestion.parsers.xrd_rigaku_asc import RigakuXrdAscParser

from ._helpers import FIXTURES_DIR, SNAPSHOTS_DIR, parsed_to_snapshot

GOLDEN = FIXTURES_DIR / "xrd" / "rigaku_cs_pure.asc"


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert RigakuXrdAscParser.name == "rigaku-xrd-asc"

    def test_technique(self):
        assert RigakuXrdAscParser.technique is Technique.XRD

    def test_extensions_lowercase(self):
        assert RigakuXrdAscParser.supported_extensions == (".asc",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = RigakuXrdAscParser()

    def test_returns_one_for_valid_two_column(self):
        assert self.parser.can_parse(GOLDEN) == 1.0

    def test_uppercase_extension_matches(self, tmp_path: Path):
        # Real Rigaku files come as `.ASC` (uppercase) — _extension_matches
        # is case-insensitive.
        f = tmp_path / "data.ASC"
        f.write_text("5.0 100\n5.1 110\n5.2 120\n5.3 130\n5.4 140\n")
        assert self.parser.can_parse(f) == 1.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("5.0 100\n5.1 110\n")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_text_content(self, tmp_path: Path):
        f = tmp_path / "data.asc"
        f.write_text("This is not numeric data at all.\n")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.asc"
        f.write_text("")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_missing_file(self, tmp_path: Path):
        assert self.parser.can_parse(tmp_path / "missing.asc") == 0.0

    def test_returns_intermediate_for_mostly_valid(self, tmp_path: Path):
        # 4 valid rows, 1 garbage. _SNIFF_LINES = 5, so 4/5 valid → 0.7.
        f = tmp_path / "mixed.asc"
        f.write_text("5.0 100\n5.1 110\nGARBAGE LINE\n5.3 130\n5.4 140\n")
        assert self.parser.can_parse(f) == 0.7


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self.result = RigakuXrdAscParser().parse(GOLDEN)

    def test_technique(self):
        assert self.result.technique is Technique.XRD

    def test_arrays_present(self):
        assert "two_theta" in self.result.arrays
        assert "intensity" in self.result.arrays

    def test_arrays_same_length(self):
        assert len(self.result.arrays["two_theta"]) == len(self.result.arrays["intensity"])

    def test_two_theta_monotonic(self):
        tt = self.result.arrays["two_theta"]
        assert np.all(np.diff(tt) > 0)

    def test_first_data_point(self):
        # First line: ` 5.01086644E+0000   0.00E+0000`
        assert self.result.arrays["two_theta"][0] == pytest.approx(5.01086644)
        assert self.result.arrays["intensity"][0] == pytest.approx(0.0)

    def test_metadata_includes_n_points(self):
        assert self.result.metadata["n_points"] == len(self.result.arrays["two_theta"])

    def test_no_instrument_no_timestamp(self):
        # `.ASC` carries neither.
        assert self.result.instrument is None
        assert self.result.measured_at is None

    def test_negative_intensity_warning_consistent_with_threshold(self):
        # The warning's presence must match what's actually in the file —
        # not assume what we'd expect. The fixture turns out to have ~31%
        # negative intensities (background-subtracted), comfortably above
        # the parser's 10% threshold, so the warning should fire.
        intensity = self.result.arrays["intensity"]
        neg_frac = float((intensity < 0).sum()) / len(intensity)
        has_warning = any("negative" in i.message.lower() for i in self.result.issues)
        assert (neg_frac >= 0.10) == has_warning


# ─── Golden-file snapshot ───────────────────────────────────────────
class TestGoldenSnapshot:
    def test_snapshot_match(self, snapshot: Snapshot):
        result = RigakuXrdAscParser().parse(GOLDEN)
        snapshot.snapshot_dir = SNAPSHOTS_DIR
        snapshot.assert_match(
            json.dumps(parsed_to_snapshot(result), indent=2, sort_keys=True) + "\n",
            "rigaku_cs_pure.json",
        )


# ─── Failure modes ──────────────────────────────────────────────────
class TestFailureModes:
    def setup_method(self):
        self.parser = RigakuXrdAscParser()

    def test_empty_file_returns_error(self, tmp_path: Path):
        f = tmp_path / "empty.asc"
        f.write_text("")
        result = self.parser.parse(f)
        assert result.has_errors
        assert result.arrays == {}

    def test_garbage_returns_error(self, tmp_path: Path):
        f = tmp_path / "garbage.asc"
        f.write_text("hello\nworld\n")
        result = self.parser.parse(f)
        assert result.has_errors

    def test_decreasing_two_theta_warns(self, tmp_path: Path):
        f = tmp_path / "backwards.asc"
        f.write_text("10.0 100\n9.0 200\n8.0 300\n4.0 400\n3.0 500\n")
        result = self.parser.parse(f)
        assert result.has_warnings
        assert any("monotonic" in i.message.lower() for i in result.issues)

    def test_no_warning_when_negatives_below_threshold(self, tmp_path: Path):
        # 9 positives + 1 negative = 10% exactly. Threshold is `>= 10%`, so
        # this fires; use 11 positives + 1 negative (~8%) to get below.
        f = tmp_path / "few_neg.asc"
        rows = "\n".join(f"{5.0 + i * 0.1} 100" for i in range(11))
        f.write_text(rows + "\n5.5 -10\n")  # 11 positive + 1 negative = 8.3%
        result = self.parser.parse(f)
        assert not any("negative" in i.message.lower() for i in result.issues)

    def test_warning_when_negatives_above_threshold(self, tmp_path: Path):
        # 50% negative — well above the 10% threshold.
        f = tmp_path / "many_neg.asc"
        f.write_text("5.0 100\n5.1 -10\n5.2 120\n5.3 -30\n5.4 140\n5.5 -50\n")
        result = self.parser.parse(f)
        assert any("negative" in i.message.lower() for i in result.issues)

    def test_partial_garbage_emits_warning_keeps_valid_rows(self, tmp_path: Path):
        f = tmp_path / "partial.asc"
        f.write_text(
            "5.0 100\n"
            "garbage\n"
            "5.2 120\n"
            "5.3 only_one_col\n"  # split() gives 1 part → malformed
            "5.4 140\n",
        )
        result = self.parser.parse(f)
        # 3 valid rows survive.
        assert len(result.arrays["two_theta"]) == 3
        assert result.has_warnings
