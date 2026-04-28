"""Tests for `MicroscopyTifParser`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from pytest_snapshot.plugin import Snapshot

from latos.core.enums import Technique
from latos.ingestion.parsers.microscopy_tif import MicroscopyTifParser

from ._helpers import FIXTURES_DIR, SNAPSHOTS_DIR, parsed_to_snapshot

GOLDEN = FIXTURES_DIR / "microscopy" / "tem_cs.tif"


def _write_synthetic_tif(path: Path, *, with_datetime: bool = False) -> None:
    """Write a tiny TIFF for negative/edge tests."""
    img = np.zeros((10, 10), dtype=np.uint8)
    extratags: list = []
    if with_datetime:
        # TIFF DateTime tag (306) — type 2 (ASCII).
        extratags.append((306, "s", 0, "2024:06:15 12:30:45", True))
    tifffile.imwrite(path, img, extratags=extratags)


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert MicroscopyTifParser.name == "microscopy-tif"

    def test_technique_default_is_sem(self):
        # Default; Stage 2 will override based on folder context.
        assert MicroscopyTifParser.technique is Technique.SEM

    def test_extensions_both_tif_and_tiff(self):
        assert ".tif" in MicroscopyTifParser.supported_extensions
        assert ".tiff" in MicroscopyTifParser.supported_extensions


# ─── can_parse ──────────────────────────────────────────────────────
class TestCanParse:
    def setup_method(self):
        self.parser = MicroscopyTifParser()

    def test_returns_one_for_real_fixture(self):
        assert self.parser.can_parse(GOLDEN) == 1.0

    def test_returns_one_for_synthetic(self, tmp_path: Path):
        f = tmp_path / "syn.tif"
        _write_synthetic_tif(f)
        assert self.parser.can_parse(f) == 1.0

    def test_returns_zero_for_wrong_extension(self, tmp_path: Path):
        f = tmp_path / "img.png"
        _write_synthetic_tif(tmp_path / "tmp.tif")
        # Move/rename to wrong extension.
        f.write_bytes((tmp_path / "tmp.tif").read_bytes())
        assert self.parser.can_parse(f) == 0.0

    def test_returns_zero_for_non_tiff_content(self, tmp_path: Path):
        f = tmp_path / "fake.tif"
        f.write_bytes(b"This is not a TIFF file.")
        assert self.parser.can_parse(f) == 0.0


# ─── Parse — happy path ─────────────────────────────────────────────
class TestParseHappyPath:
    def setup_method(self):
        self.result = MicroscopyTifParser().parse(GOLDEN)

    def test_technique_default_sem(self):
        assert self.result.technique is Technique.SEM

    def test_arrays_empty(self):
        # Stage 1C: metadata only, no pixels.
        assert self.result.arrays == {}

    def test_dimensions_in_metadata(self):
        m = self.result.metadata
        # Fixture is 1024x1024 RGB.
        assert m["image_width"] == 1024
        assert m["image_height"] == 1024
        assert m["shape"] == [1024, 1024, 3]

    def test_dtype_in_metadata(self):
        # uint8 RGB.
        assert self.result.metadata["dtype"] == "uint8"

    def test_instrument_built_from_make_and_model(self):
        # Fixture: Make='Olympus Soft Imaging Solutions', Model='Tengra'.
        assert self.result.instrument is not None
        assert "Olympus" in self.result.instrument
        assert "Tengra" in self.result.instrument

    def test_measured_at_extracted_with_warning(self):
        # Fixture has DateTime '2025:04:28 11:21:11'.
        assert self.result.measured_at is not None
        assert self.result.measured_at.tzinfo is not None
        warning_fields = [i.field for i in self.result.issues]
        assert "measured_at" in warning_fields

    def test_technique_default_warning_present(self):
        # The parser always emits an INFO note that the technique was
        # defaulted — Stage 2 will refine.
        info_messages = [i.message for i in self.result.issues]
        assert any("technique" in m.lower() for m in info_messages)

    def test_no_errors(self):
        assert not self.result.has_errors


# ─── Golden snapshot ────────────────────────────────────────────────
class TestGoldenSnapshot:
    def test_snapshot_match(self, snapshot: Snapshot):
        result = MicroscopyTifParser().parse(GOLDEN)
        snapshot.snapshot_dir = SNAPSHOTS_DIR
        snapshot.assert_match(
            json.dumps(parsed_to_snapshot(result), indent=2, sort_keys=True) + "\n",
            "microscopy_tem_cs.json",
        )


# ─── Failure modes ──────────────────────────────────────────────────
class TestFailureModes:
    def setup_method(self):
        self.parser = MicroscopyTifParser()

    def test_invalid_tiff_returns_error(self, tmp_path: Path):
        f = tmp_path / "bad.tif"
        f.write_bytes(b"NOT A TIFF")
        result = self.parser.parse(f)
        assert result.has_errors

    def test_synthetic_no_datetime(self, tmp_path: Path):
        f = tmp_path / "no_dt.tif"
        _write_synthetic_tif(f, with_datetime=False)
        result = self.parser.parse(f)
        assert not result.has_errors
        assert result.measured_at is None

    def test_synthetic_with_datetime(self, tmp_path: Path):
        f = tmp_path / "with_dt.tif"
        _write_synthetic_tif(f, with_datetime=True)
        result = self.parser.parse(f)
        assert result.measured_at is not None
        assert result.measured_at.year == 2024
