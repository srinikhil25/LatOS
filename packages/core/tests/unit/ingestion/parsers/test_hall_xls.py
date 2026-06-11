"""Tests for `HallXlsParser`."""

from __future__ import annotations

import json
from pathlib import Path

from pytest_snapshot.plugin import Snapshot

from latos.core.enums import Technique
from latos.ingestion.parsers.hall_xls import HallXlsParser

from ._helpers import FIXTURES_DIR, SNAPSHOTS_DIR, parsed_to_snapshot

GOLDEN = FIXTURES_DIR / "hall" / "hall_cs.xls"


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert HallXlsParser.name == "hall-xls"

    def test_technique(self):
        assert HallXlsParser.technique is Technique.HALL

    def test_extensions(self):
        assert HallXlsParser.supported_extensions == (".xls",)


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = HallXlsParser()

    def test_returns_one_for_real_fixture(self):
        assert self.parser.can_parse(GOLDEN) == 1.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        # We can't synthesize .xls cheaply, so just test the extension guard.
        f = tmp_path / "data.csv"
        f.write_text("Hall coefficient\n1.0\n")
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_missing_file(self, tmp_path: Path):
        assert self.parser.can_parse(tmp_path / "missing.xls") == 0.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self.result = HallXlsParser().parse(GOLDEN)

    def test_technique(self):
        assert self.result.technique is Technique.HALL

    def test_arrays_empty(self):
        # Hall is scalar at one temperature — no array data.
        assert self.result.arrays == {}

    def test_metadata_contains_temperature(self):
        m = self.result.metadata
        # Look for a key that contains 'temperature' (normalized form).
        temp_keys = [k for k in m if "temperature" in k and not k.endswith("__label")]
        assert len(temp_keys) >= 1
        assert isinstance(m[temp_keys[0]], int | float)

    def test_metadata_contains_mobility(self):
        m = self.result.metadata
        mob_keys = [k for k in m if "mobility" in k and not k.endswith("__label")]
        assert len(mob_keys) >= 1
        assert isinstance(m[mob_keys[0]], int | float)

    def test_metadata_contains_resistivity(self):
        m = self.result.metadata
        res_keys = [k for k in m if "resistivity" in k and not k.endswith("__label")]
        assert len(res_keys) >= 1

    def test_label_columns_preserve_units(self):
        # __label columns hold the original header (e.g. "Mobility (cm²/(V s))")
        # so display code can show units without re-deriving.
        m = self.result.metadata
        label_keys = [k for k in m if k.endswith("__label")]
        assert len(label_keys) >= 1
        # All label values should be strings.
        for k in label_keys:
            assert isinstance(m[k], str)

    def test_metadata_is_json_safe(self):
        # ParsedData enforces this on construction; a successful parse
        # implicitly verifies it. Re-assert for robustness.
        json.dumps(self.result.metadata)

    def test_no_errors(self):
        assert not self.result.has_errors


# ─── Golden snapshot ────────────────────────────────────────────────
class TestGoldenSnapshot:
    def test_snapshot_match(self, snapshot: Snapshot):
        result = HallXlsParser().parse(GOLDEN)
        snapshot.snapshot_dir = SNAPSHOTS_DIR
        snapshot.assert_match(
            json.dumps(parsed_to_snapshot(result), indent=2, sort_keys=True) + "\n",
            "hall_cs.json",
        )


# ─── Column-name normalization ──────────────────────────────────────
class TestColumnNormalization:
    """Indirect tests via the parser's metadata output."""

    def setup_method(self):
        self.metadata = HallXlsParser().parse(GOLDEN).metadata

    def test_keys_are_lowercase(self):
        # Ignore __label keys (those keep original casing wrapped).
        data_keys = [k for k in self.metadata if not k.endswith("__label")]
        for k in data_keys:
            assert k == k.lower()

    def test_keys_are_snake_case(self):
        data_keys = [k for k in self.metadata if not k.endswith("__label")]
        for k in data_keys:
            assert " " not in k
            assert "(" not in k
            assert "/" not in k
