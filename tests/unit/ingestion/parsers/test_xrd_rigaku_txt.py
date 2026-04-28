"""Tests for `RigakuXrdTxtParser`, including a golden-file snapshot.

Snapshot strategy
-----------------
The fixture `tests/fixtures/parsers/xrd/rigaku_bs3a.txt` is a real Rigaku
Ultima3 export (1151 data points, full header). We parse it and serialize
the result to a JSON-safe dict, then compare against a saved snapshot.

A snapshot mismatch means *something about the parser output changed*.
That's a deliberate signal — bumping `parser_version` and updating the
snapshot is the same atomic motion. Running tests with
`--snapshot-update` regenerates the snapshot file; review the diff
before committing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest_snapshot.plugin import Snapshot

from latos.core.enums import Severity, Technique
from latos.ingestion.parsed_data import ParsedData
from latos.ingestion.parsers.xrd_rigaku_txt import RigakuXrdTxtParser

FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "parsers" / "xrd"
GOLDEN_FILE = FIXTURE_DIR / "rigaku_bs3a.txt"
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


# ─── Helpers ────────────────────────────────────────────────────────
def _parsed_to_snapshot(parsed: ParsedData) -> dict[str, Any]:
    """Serialize a `ParsedData` into a deterministic JSON-safe dict.

    Arrays are summarized as (length, dtype, sha256-of-bytes, head, tail).
    Storing all 1151 floats in the snapshot would be noisy; the SHA-256
    of the array bytes is exact, and head/tail give a human-readable
    sanity check when reviewing diffs.
    """
    return {
        "technique": parsed.technique.value,
        "instrument": parsed.instrument,
        "measured_at": parsed.measured_at.isoformat() if parsed.measured_at else None,
        "parser_name": parsed.parser_name,
        "parser_version": parsed.parser_version,
        "metadata": parsed.metadata,
        "arrays": {
            name: {
                "length": int(arr.shape[0]),
                "dtype": str(arr.dtype),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "head": [float(x) for x in arr[:5]],
                "tail": [float(x) for x in arr[-5:]],
            }
            for name, arr in sorted(parsed.arrays.items())
        },
        "issues": [
            {
                "field": i.field,
                "severity": i.severity.value,
                "message": i.message,
                "acknowledged": i.acknowledged,
            }
            for i in parsed.issues
        ],
    }


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert RigakuXrdTxtParser.name == "rigaku-xrd-txt"

    def test_version_is_semver(self):
        # Stage 1C.3 ships 1.0.0. Bumping requires regenerating snapshot.
        assert RigakuXrdTxtParser.version == "1.0.0"

    def test_technique(self):
        assert RigakuXrdTxtParser.technique is Technique.XRD

    def test_extensions(self):
        assert RigakuXrdTxtParser.supported_extensions == (".txt",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = RigakuXrdTxtParser()

    def test_returns_one_for_full_rigaku_header(self):
        # Both signature keys present → confidence 1.0.
        assert self.parser.can_parse(GOLDEN_FILE) == 1.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text(";SampleName = test\n;KAlpha1 = 1.54\n")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_random_txt(self, tmp_path: Path):
        f = tmp_path / "readme.txt"
        f.write_text("This is just some plain text, not XRD data.")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_intermediate_for_partial_signature(self, tmp_path: Path):
        # Only one of the two signature keys → confidence 0.7.
        f = tmp_path / "partial.txt"
        f.write_text(";SampleName = test\n10.0 100\n")
        assert self.parser.can_parse(f) == 0.7

    def test_returns_zero_for_missing_file(self, tmp_path: Path):
        # No raise on unreadable — return 0.0 silently.
        assert self.parser.can_parse(tmp_path / "does-not-exist.txt") == 0.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self.parser = RigakuXrdTxtParser()
        self.result = self.parser.parse(GOLDEN_FILE)

    def test_returns_parsed_data(self):
        assert isinstance(self.result, ParsedData)

    def test_technique_is_xrd(self):
        assert self.result.technique is Technique.XRD

    def test_arrays_present(self):
        assert "two_theta" in self.result.arrays
        assert "intensity" in self.result.arrays

    def test_arrays_same_length(self):
        # ParsedData enforces this, but we double-check at parser level
        # that we never produce mismatched arrays in the first place.
        assert len(self.result.arrays["two_theta"]) == len(self.result.arrays["intensity"])

    def test_arrays_are_float64(self):
        assert self.result.arrays["two_theta"].dtype == np.float64
        assert self.result.arrays["intensity"].dtype == np.float64

    def test_two_theta_monotonic(self):
        tt = self.result.arrays["two_theta"]
        assert np.all(np.diff(tt) >= 0)

    def test_metadata_extracted(self):
        m = self.result.metadata
        assert m["sample_name"] == "bs3a.raw"
        assert m["target"] == "Cu"
        assert m["wavelength_ka1"] == pytest.approx(1.54056)

    def test_instrument_extracted(self):
        # The fixture file has `;Gonio = Ultima3 Inplane`.
        assert self.result.instrument == "Ultima3 Inplane"

    def test_measured_at_is_none(self):
        # Rigaku .txt format has no acquisition timestamp.
        assert self.result.measured_at is None

    def test_parser_identity_round_trips(self):
        assert self.result.parser_name == RigakuXrdTxtParser.name
        assert self.result.parser_version == RigakuXrdTxtParser.version

    def test_no_errors_on_clean_file(self):
        # Warnings may be present (e.g. minor field issues) but no errors.
        assert not self.result.has_errors

    def test_n_points_in_metadata_matches_arrays(self):
        assert self.result.metadata["n_points"] == len(self.result.arrays["two_theta"])


# ─── Golden-file snapshot ───────────────────────────────────────────
class TestGoldenSnapshot:
    def test_snapshot_match(self, snapshot: Snapshot):
        """Full-output snapshot test against the bs3a.txt fixture.

        If this fails: either the parser changed (intentional → bump
        version, regenerate with --snapshot-update) or there's a
        regression. Diff the snapshot to find out which.
        """
        parser = RigakuXrdTxtParser()
        result = parser.parse(GOLDEN_FILE)
        snapshot_dict = _parsed_to_snapshot(result)
        snapshot.snapshot_dir = SNAPSHOT_DIR
        snapshot.assert_match(
            json.dumps(snapshot_dict, indent=2, sort_keys=True) + "\n",
            "rigaku_bs3a.json",
        )


# ─── Failure modes — never raise, emit issues ───────────────────────
class TestFailureModes:
    def setup_method(self):
        self.parser = RigakuXrdTxtParser()

    def test_empty_file_returns_error(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = self.parser.parse(f)
        assert result.has_errors
        assert result.arrays == {}

    def test_garbage_file_returns_error(self, tmp_path: Path):
        f = tmp_path / "garbage.txt"
        f.write_text("this is not\nan xrd file\nat all")
        result = self.parser.parse(f)
        assert result.has_errors
        # Parser still returns a valid ParsedData; never raises.

    def test_metadata_only_no_data_returns_error(self, tmp_path: Path):
        f = tmp_path / "meta_only.txt"
        f.write_text(";SampleName = test\n;KAlpha1 = 1.54056\n")
        result = self.parser.parse(f)
        assert result.has_errors
        assert result.arrays == {}
        # But metadata WAS extracted — partial success is preserved.
        assert result.metadata["sample_name"] == "test"

    def test_mixed_garbage_and_valid_rows_emits_warning(self, tmp_path: Path):
        f = tmp_path / "mixed.txt"
        f.write_text(
            ";SampleName = mixed\n"
            ";KAlpha1 = 1.54056\n"
            "10.0 100\n"
            "11.0 not_a_number\n"  # malformed
            "12.0 200\n"
            "13.0\n"  # too few columns
            "14.0 400\n",
        )
        result = self.parser.parse(f)
        assert not result.has_errors  # we got SOME data
        assert result.has_warnings
        # 3 valid pairs survive: (10, 100), (12, 200), (14, 400)
        assert len(result.arrays["two_theta"]) == 3

    def test_decreasing_two_theta_emits_warning(self, tmp_path: Path):
        f = tmp_path / "backwards.txt"
        f.write_text(
            ";SampleName = backwards\n;KAlpha1 = 1.54056\n10.0 100\n9.0 200\n8.0 300\n",
        )
        result = self.parser.parse(f)
        # Data was readable; warning notes the unusual order.
        assert result.has_warnings
        warning_messages = [i.message for i in result.issues]
        assert any("monotonic" in m.lower() for m in warning_messages)

    def test_missing_wavelength_emits_warning(self, tmp_path: Path):
        f = tmp_path / "no_wavelength.txt"
        f.write_text(
            ";SampleName = test\n10.0 100\n11.0 200\n",  # no KAlpha1
        )
        result = self.parser.parse(f)
        # We got data, so no error — but a warning about wavelength.
        warnings_for_kalpha1 = [
            i for i in result.issues if i.field == "KAlpha1" and i.severity is Severity.WARNING
        ]
        assert len(warnings_for_kalpha1) == 1

    def test_non_numeric_kv_emits_warning(self, tmp_path: Path):
        f = tmp_path / "bad_kv.txt"
        f.write_text(
            ";SampleName = test\n;KAlpha1 = 1.54056\n;KV = not_a_number\n10.0 100\n",
        )
        result = self.parser.parse(f)
        kv_issues = [i for i in result.issues if i.field == "KV"]
        assert len(kv_issues) == 1
        assert result.metadata["voltage_kv"] is None


# ─── Numeric values from golden file (regression sanity) ────────────
class TestGoldenNumericValues:
    """Pinned numeric checks against the bs3a.txt fixture.

    These complement the JSON snapshot — the snapshot catches structural
    changes, these catch silent numeric corruption (wrong column read,
    accidental scaling, etc.) with values you can read at a glance.
    """

    def setup_method(self):
        self.result = RigakuXrdTxtParser().parse(GOLDEN_FILE)

    def test_first_data_point(self):
        # First line of the data section: `5.0000 208`
        assert self.result.arrays["two_theta"][0] == pytest.approx(5.0)
        assert self.result.arrays["intensity"][0] == pytest.approx(208.0)

    def test_scan_range_matches_header(self):
        m = self.result.metadata
        tt = self.result.arrays["two_theta"]
        assert m["scan_start_deg"] == pytest.approx(5.0)
        assert m["scan_finish_deg"] == pytest.approx(120.0)
        # First/last data points should fall within the declared range.
        assert tt[0] >= m["scan_start_deg"]
        assert tt[-1] <= m["scan_finish_deg"]

    def test_step_size(self):
        # `;Width = 0.1` → 0.1 deg step.
        assert self.result.metadata["scan_step_deg"] == pytest.approx(0.1)

    def test_intensity_all_finite(self):
        # No NaN/Inf creeping in from format quirks.
        assert np.all(np.isfinite(self.result.arrays["intensity"]))

    def test_slits_collected(self):
        # The fixture has 8 SlitName entries (SlitName0..SlitName7),
        # of which a few are blank in the original file.
        slits = self.result.metadata["slits"]
        assert isinstance(slits, list)
        # At minimum, the non-blank ones should be present.
        assert any("Attenuator" in s for s in slits)
        assert any("DivSlit" in s for s in slits)
