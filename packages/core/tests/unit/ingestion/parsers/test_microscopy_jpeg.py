"""Tests for `MicroscopyJpegParser`.

The behaviour worth protecting is the info-bar test. These instruments write a
square image area with the bar in extra rows beneath it, so a frame taller than
it is wide carries a scale and a square one does not. A square frame has no
recoverable pixel size — no length measured on it means anything — and saying so
at ingest is the whole reason this parser reads dimensions rather than pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from latos.core.enums import Severity, Technique
from latos.ingestion.parsers.microscopy_jpeg import MicroscopyJpegParser


def _write_jpeg(path: Path, width: int, height: int) -> Path:
    """A JPEG of the given size. Content is irrelevant; the shape is the signal."""
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (height, width), dtype=np.uint8), mode="L").save(
        path, format="JPEG"
    )
    return path


@pytest.fixture
def parser() -> MicroscopyJpegParser:
    return MicroscopyJpegParser()


class TestClassMetadata:
    def test_name(self, parser):
        assert parser.name == "microscopy-jpeg"

    def test_technique_default_is_sem(self, parser):
        # The orchestrator's folder refinement promotes this to TEM/STEM.
        assert parser.technique is Technique.SEM

    def test_extensions(self, parser):
        assert set(parser.supported_extensions) == {".jpg", ".jpeg"}


class TestCanParse:
    def test_claims_a_real_jpeg(self, parser, tmp_path):
        assert parser.can_parse(_write_jpeg(tmp_path / "f.jpg", 64, 80)) == 1.0

    def test_claims_the_jpeg_spelling_too(self, parser, tmp_path):
        assert parser.can_parse(_write_jpeg(tmp_path / "f.jpeg", 64, 80)) == 1.0

    def test_declines_a_png_renamed_to_jpg(self, parser, tmp_path):
        # Extension alone must never be enough; the SOI marker decides.
        path = tmp_path / "liar.jpg"
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(path, format="PNG")
        assert parser.can_parse(path) == 0.0

    def test_declines_a_tif(self, parser, tmp_path):
        (tmp_path / "x.tif").write_bytes(b"II*\x00rest")
        assert parser.can_parse(tmp_path / "x.tif") == 0.0

    def test_declines_a_missing_file(self, parser, tmp_path):
        assert parser.can_parse(tmp_path / "absent.jpg") == 0.0


class TestInfoBar:
    def test_taller_than_wide_has_a_bar(self, parser, tmp_path):
        result = parser.parse(_write_jpeg(tmp_path / "f.jpg", 512, 600))
        assert result.metadata["info_bar_present"] is True
        assert result.metadata["info_bar_height_px"] == 88
        assert result.metadata["image_area_px"] == 512
        assert not [i for i in result.issues if i.field == "info_bar"]

    def test_square_frame_is_flagged(self, parser, tmp_path):
        result = parser.parse(_write_jpeg(tmp_path / "f.jpg", 512, 512))
        assert result.metadata["info_bar_present"] is False
        assert result.metadata["info_bar_height_px"] == 0
        issue = next(i for i in result.issues if i.field == "info_bar")
        assert issue.severity is Severity.WARNING
        # The message has to say what it costs, not merely what is absent.
        assert "no pixel size can be recovered" in issue.message.lower()

    def test_wider_than_tall_is_also_flagged(self, parser, tmp_path):
        # Not a layout these exports write, but it is certainly not a bar
        # underneath a square area, so it must not be read as calibrated.
        result = parser.parse(_write_jpeg(tmp_path / "f.jpg", 600, 512))
        assert result.metadata["info_bar_present"] is False
        assert [i for i in result.issues if i.field == "info_bar"]


class TestParse:
    def test_records_dimensions(self, parser, tmp_path):
        meta = parser.parse(_write_jpeg(tmp_path / "f.jpg", 320, 400)).metadata
        assert meta["image_width"] == 320
        assert meta["image_height"] == 400

    def test_arrays_are_empty(self, parser, tmp_path):
        # Metadata-only, as for TIFF: 609 frames is 1.3 GB and the question
        # answered here needs the header, not the pixels.
        assert parser.parse(_write_jpeg(tmp_path / "f.jpg", 64, 80)).arrays == {}

    def test_unreadable_file_reports_an_error_and_does_not_raise(self, parser, tmp_path):
        path = tmp_path / "truncated.jpg"
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 8)  # claims JPEG, is not one
        result = parser.parse(path)
        assert [i for i in result.issues if i.severity is Severity.ERROR]
        assert result.arrays == {}

    def test_parser_identity_is_recorded(self, parser, tmp_path):
        result = parser.parse(_write_jpeg(tmp_path / "f.jpg", 64, 80))
        assert result.parser_name == "microscopy-jpeg"
        assert result.parser_version == parser.version

    def test_instrument_falls_back_when_exif_is_absent(self, parser, tmp_path):
        # Microscope exports usually write no EXIF at all; that is not an error.
        assert parser.parse(_write_jpeg(tmp_path / "f.jpg", 64, 80)).instrument == (
            "Microscopy (.jpg)"
        )


class TestRegistered:
    def test_the_default_registry_claims_a_jpeg(self, tmp_path):
        # The parser existing is not the fix; being reachable from ingestion is.
        from latos.ingestion.registry import default_registry

        found = default_registry().find_parser(_write_jpeg(tmp_path / "f.jpg", 64, 80))
        assert found is not None
        assert found.parser.name == "microscopy-jpeg"
        assert found.confidence == 1.0
