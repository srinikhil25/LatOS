"""Tests for `MicroscopyBmpParser`.

Two behaviours carry the weight.

The first is the filename decode. BMP has no tag structure, so `View000 Ti K.bmp`
is the *only* record that the image maps titanium — and the decode has to be
strict, because inventing an element from a filename is the confident wrong
answer this tool exists to avoid.

The second is a refusal: this parser must NOT reuse the JPEG parser's info-bar
test. A frame taller than it is wide carries a JEOL info bar in the `.jpg`
exports; here the extra rows are an 8-px element caption, and reading them as
calibration would manufacture a pixel size that does not exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from latos.core.enums import Severity, Technique
from latos.ingestion.parsers.microscopy_bmp import MicroscopyBmpParser


def _write_bmp(path: Path, width: int = 267, height: int = 275) -> Path:
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (height, width), dtype=np.uint8), mode="L").save(
        path, format="BMP"
    )
    return path


@pytest.fixture
def parser() -> MicroscopyBmpParser:
    return MicroscopyBmpParser()


class TestClassMetadata:
    def test_name(self, parser):
        assert parser.name == "microscopy-bmp"

    def test_extensions(self, parser):
        assert parser.supported_extensions == (".bmp",)

    def test_placeholder_technique_is_sem(self, parser):
        # For an unnamed BMP only; a decoded map returns EDS from `parse`.
        assert parser.technique is Technique.SEM


class TestCanParse:
    def test_a_named_map_scores_higher_than_a_bare_bmp(self, parser, tmp_path):
        named = parser.can_parse(_write_bmp(tmp_path / "View000 Ti K.bmp"))
        plain = parser.can_parse(_write_bmp(tmp_path / "screenshot.bmp"))
        assert named == 1.0
        # Still claimed — but the gap records which answer rests on more evidence.
        assert 0.5 < plain < named

    def test_declines_a_png_renamed_to_bmp(self, parser, tmp_path):
        path = tmp_path / "liar.bmp"
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(path, format="PNG")
        assert parser.can_parse(path) == 0.0

    def test_declines_a_missing_file(self, parser, tmp_path):
        assert parser.can_parse(tmp_path / "absent.bmp") == 0.0


class TestFilenameDecode:
    def test_element_map(self, parser, tmp_path):
        result = parser.parse(_write_bmp(tmp_path / "View000 Ti K.bmp"))
        assert result.technique is Technique.EDS
        assert result.metadata["view"] == "View000"
        assert result.metadata["image_kind"] == "element_map"
        assert result.metadata["element"] == "Ti"
        assert result.metadata["xray_line"] == "K"

    def test_l_line(self, parser, tmp_path):
        meta = parser.parse(_write_bmp(tmp_path / "View000 Mo L.bmp")).metadata
        assert (meta["element"], meta["xray_line"]) == ("Mo", "L")

    def test_bright_field_reference(self, parser, tmp_path):
        result = parser.parse(_write_bmp(tmp_path / "View001 BF.bmp"))
        assert result.technique is Technique.EDS
        assert result.metadata["image_kind"] == "bright_field"
        assert "element" not in result.metadata

    def test_view_number_is_normalised(self, parser, tmp_path):
        meta = parser.parse(_write_bmp(tmp_path / "View7 Al K.bmp")).metadata
        assert meta["view"] == "View007"

    @pytest.mark.parametrize(
        "stem",
        [
            "View000 Xx K",  # not an element
            "View000 Ti Q",  # not an X-ray line
            "View000 Ti K extra",  # too many words
            "holiday photo",  # no view prefix
            "000",  # bare number, the real unnamed frame in this dataset
        ],
    )
    def test_a_name_that_is_not_a_map_decodes_to_nothing(self, parser, tmp_path, stem):
        result = parser.parse(_write_bmp(tmp_path / f"{stem}.bmp"))
        assert "element" not in result.metadata
        assert result.technique is Technique.SEM  # left for folder refinement

    def test_a_map_carries_the_qualitative_caveat(self, parser, tmp_path):
        issue = next(
            i
            for i in parser.parse(_write_bmp(tmp_path / "View000 Ti K.bmp")).issues
            if i.field == "eds_map"
        )
        assert issue.severity is Severity.INFO
        assert "not how much" in issue.message

    def test_bright_field_carries_no_caveat(self, parser, tmp_path):
        # There is nothing to over-read in a reference image.
        result = parser.parse(_write_bmp(tmp_path / "View000 BF.bmp"))
        assert not [i for i in result.issues if i.field == "eds_map"]


class TestNoInfoBarClaim:
    """The trailing strip is recorded, never interpreted as calibration."""

    def test_strip_is_measured(self, parser, tmp_path):
        meta = parser.parse(_write_bmp(tmp_path / "View000 Ti K.bmp", 267, 275)).metadata
        assert meta["trailing_strip_px"] == 8

    def test_square_frame_has_no_strip_and_no_warning(self, parser, tmp_path):
        # The JPEG parser warns here. This one must not: it never claimed a BMP
        # carries a decodable bar, so its absence costs nothing.
        result = parser.parse(_write_bmp(tmp_path / "flat.bmp", 256, 256))
        assert result.metadata["trailing_strip_px"] == 0
        assert not [i for i in result.issues if i.field == "info_bar"]

    def test_a_tall_strip_still_claims_no_pixel_size(self, parser, tmp_path):
        meta = parser.parse(_write_bmp(tmp_path / "000.bmp", 512, 568)).metadata
        assert meta["trailing_strip_px"] == 56
        assert "nm_per_px" not in meta
        assert "info_bar_present" not in meta


class TestParse:
    def test_arrays_are_empty(self, parser, tmp_path):
        assert parser.parse(_write_bmp(tmp_path / "View000 Ti K.bmp")).arrays == {}

    def test_unreadable_file_reports_an_error_and_does_not_raise(self, parser, tmp_path):
        path = tmp_path / "truncated.bmp"
        path.write_bytes(b"BM" + b"\x00" * 8)
        result = parser.parse(path)
        assert [i for i in result.issues if i.severity is Severity.ERROR]
        assert result.arrays == {}


class TestRegistered:
    def test_the_default_registry_claims_an_eds_map(self, tmp_path):
        from latos.ingestion.registry import default_registry

        found = default_registry().find_parser(_write_bmp(tmp_path / "View000 Ti K.bmp"))
        assert found is not None
        assert found.parser.name == "microscopy-bmp"
