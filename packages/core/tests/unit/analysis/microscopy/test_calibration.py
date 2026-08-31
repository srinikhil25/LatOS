"""Tests for `latos.analysis.microscopy.calibration`.

Info bars are synthesised rather than fixtured, so the text a bar carries is
known exactly and every assertion has a ground truth. The synthetic bar
reproduces the structural features the decoder actually depends on: drawn cell
borders that would otherwise read as ink, a label row above the value row, two
image sizes with the same layout at different scales, and a background tone
that is not white.

The emphasis throughout is on **refusing to guess**. A misread field of view is
the worst outcome this module can produce, because it silently rescales every
length derived from the image by a factor that still looks plausible. So the
unreadable, unmatched, and physically impossible cases are tested at least as
hard as the successful ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from latos.analysis.microscopy.calibration import (
    JEOL_2100F,
    InfoBarLayout,
    StripTemplates,
    decode_field_of_view,
    harvest_strips,
    measure_scale_bar,
    nm_per_pixel,
    parse_length,
    split_info_bar,
    value_strip,
)

BACKGROUND = 240
INK = 0
BAR_HEIGHT_AT_2048 = 106


def _glyph(char: str, height: int) -> np.ndarray:
    """A deterministic, distinct bitmap for one character.

    Real glyph shapes are irrelevant to the decoder — it compares bitmaps — but
    the *structure* matters: fixed height, variable width, and a blank column
    between glyphs. That is what this reproduces.
    """
    code = ord(char)
    width = 3 if char == "." else 5
    rng = np.random.default_rng(code)
    block = rng.random((height, width)) < 0.55
    block[0, :] = True  # guarantee ink on the first row so heights are stable
    block[-1, :] = True
    return block


def _render(text: str, height: int) -> np.ndarray:
    """Render `text` as an ink bitmap, one blank column between glyphs."""
    parts: list[np.ndarray] = []
    for char in text:
        if char == " ":
            parts.append(np.zeros((height, 4), dtype=bool))
        else:
            parts.append(_glyph(char, height))
            parts.append(np.zeros((height, 1), dtype=bool))
    return np.concatenate(parts, axis=1)


def make_frame(
    field_text: str = "21.7 nm",
    *,
    width: int = 2048,
    scale_bar_px: int | None = None,
    blank_field: bool = False,
) -> np.ndarray:
    """A square image with a JEOL-style info bar beneath it.

    Returns the full frame, image area plus bar, as the decoder receives it.
    """
    scale = width / 2048
    bar_h = round(BAR_HEIGHT_AT_2048 * scale)
    frame = np.full((width + bar_h, width), BACKGROUND, dtype=np.uint8)
    # Vary the image area so it is never confused with the bar.
    frame[:width, :] = np.random.default_rng(0).integers(60, 200, (width, width))

    bar = frame[width:, :]
    bar[0, :] = INK  # top border
    bar[-1, :] = INK  # bottom border
    for x0, x1 in JEOL_2100F.cells.values():
        for x in (int(x0 * scale), int(x1 * scale)):
            if 0 <= x < width:
                bar[:, x] = INK  # cell divider

    label_h = max(4, round(28 * scale))
    value_h = max(4, round(24 * scale))
    label_top = max(2, round(6 * scale))
    value_top = label_top + label_h + max(2, round(8 * scale))

    def place(cell: str, text: str, top: int, height: int) -> None:
        x0, x1 = JEOL_2100F.cell_bounds(cell, width)
        bitmap = _render(text, height)
        if bitmap.shape[1] > (x1 - x0):
            bitmap = bitmap[:, : x1 - x0]
        start = x0 + max(0, ((x1 - x0) - bitmap.shape[1]) // 2)
        region = bar[top : top + height, start : start + bitmap.shape[1]]
        region[bitmap] = INK

    # A label row that differs from the value row, so picking the wrong band
    # is detectable rather than silently equivalent.
    place("field_of_view", "LABEL", label_top, label_h)
    if not blank_field:
        place("field_of_view", field_text, value_top, value_h)
    place("magnification", "10000 x", value_top, value_h)

    if scale_bar_px:
        x0, x1 = JEOL_2100F.cell_bounds("scale_bar", width)
        start = x0 + ((x1 - x0) - scale_bar_px) // 2
        # Real rules are a few pixels thick, not hairlines.
        rule_top = value_top + value_h // 2
        bar[rule_top : rule_top + 3, start : start + scale_bar_px] = INK
    return frame


def templates_for(texts: list[str], widths: tuple[int, ...] = (2048, 1024)) -> StripTemplates:
    """Build a template set the way a human would: label each distinct strip once."""
    templates = StripTemplates()
    for width in widths:
        for text in texts:
            strip = value_strip(make_frame(text, width=width), "field_of_view")
            assert strip is not None, f"failed to render a strip for {text!r}"
            bar_width = width
            templates.add(bar_width, strip, text)
    return templates


class TestSplitInfoBar:
    def test_square_frame_has_no_bar(self) -> None:
        assert split_info_bar(np.zeros((512, 512), dtype=np.uint8)) is None

    def test_tall_frame_splits_into_square_area_and_bar(self) -> None:
        parts = split_info_bar(make_frame())
        assert parts is not None
        area, bar = parts
        assert area.shape == (2048, 2048)
        assert bar.shape == (BAR_HEIGHT_AT_2048, 2048)

    def test_non_two_dimensional_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            split_info_bar(np.zeros((8, 8, 3)))


class TestLayout:
    def test_bounds_scale_with_bar_width(self) -> None:
        big = JEOL_2100F.cell_bounds("field_of_view", 2048)
        small = JEOL_2100F.cell_bounds("field_of_view", 1024)
        assert small[0] == pytest.approx(big[0] / 2, abs=3)
        assert small[1] == pytest.approx(big[1] / 2, abs=3)

    def test_bounds_are_inset_from_the_nominal_cell(self) -> None:
        x0, x1 = JEOL_2100F.cell_bounds("field_of_view", 2048)
        nominal = JEOL_2100F.cells["field_of_view"]
        assert x0 > nominal[0]
        assert x1 < nominal[1]

    def test_unknown_cell_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown cell"):
            JEOL_2100F.cell_bounds("nonexistent", 2048)

    def test_a_custom_layout_is_usable(self) -> None:
        layout = InfoBarLayout(reference_width=100, cells={"field_of_view": (10, 90)}, inset=1)
        assert layout.cell_bounds("field_of_view", 200) == (22, 178)


class TestValueStrip:
    def test_reads_the_value_row_not_the_label_row(self) -> None:
        strip = value_strip(make_frame("21.7 nm"), "field_of_view")
        expected = _render("21.7 nm", max(4, 24))
        assert strip is not None
        # Same glyph count and comparable width as the value, not the label.
        assert strip.shape[1] == pytest.approx(expected.shape[1], abs=6)
        label = _render("LABEL", 28)
        assert abs(strip.shape[1] - label.shape[1]) > 3

    def test_cell_borders_are_not_mistaken_for_ink(self) -> None:
        """Without the inset every column reads as ink and nothing segments."""
        strip = value_strip(make_frame("21.7 nm"), "field_of_view")
        assert strip is not None
        assert (
            strip.shape[1]
            < JEOL_2100F.cells["field_of_view"][1] - (JEOL_2100F.cells["field_of_view"][0])
        )

    def test_works_at_both_image_sizes(self) -> None:
        for width in (2048, 1024):
            assert value_strip(make_frame("44.7 nm", width=width), "field_of_view") is not None

    def test_blank_cell_returns_none(self) -> None:
        # Only a label, no value row: the last ink band is the label, which is
        # still returned; a truly empty cell is what must give None.
        frame = make_frame(blank_field=True)
        x0, x1 = JEOL_2100F.cell_bounds("field_of_view", 2048)
        frame[2048:, x0:x1] = BACKGROUND
        assert value_strip(frame, "field_of_view") is None

    def test_square_frame_returns_none(self) -> None:
        assert value_strip(np.zeros((256, 256), dtype=np.uint8), "field_of_view") is None


class TestParseLength:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("21.7 nm", 21.7),
            ("14.4 nm", 14.4),
            ("2.3 um", 2300.0),
            ("500 pm", 0.5),
            ("1 mm", 1e6),
        ],
    )
    def test_valid_lengths(self, text: str, expected: float) -> None:
        assert parse_length(text) == pytest.approx(expected)

    def test_impossible_field_of_view_is_rejected(self) -> None:
        """The real failure mode: these exports sometimes write '2 m'.

        Two metres is not a microscope field of view. Accepting it would
        rescale every spacing from that frame by nine orders of magnitude.
        """
        assert parse_length("2 m") is None

    @pytest.mark.parametrize(
        "text",
        ["", "21.7", "nm", "21.7 nm extra", "abc nm", "21,7 nm", "-5 nm", "0 nm"],
    )
    def test_unparseable_or_nonphysical_returns_none(self, text: str) -> None:
        assert parse_length(text) is None


class TestNmPerPixel:
    def test_field_of_view_spans_the_image_width(self) -> None:
        assert nm_per_pixel(21.7, 2048) == pytest.approx(21.7 / 2048)

    @pytest.mark.parametrize(("fov", "width"), [(0, 2048), (-1, 2048), (21.7, 0), (21.7, -5)])
    def test_non_positive_inputs_are_rejected(self, fov: float, width: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            nm_per_pixel(fov, width)


class TestStripTemplates:
    def test_matches_a_strip_it_was_built_from(self) -> None:
        templates = templates_for(["21.7 nm"], widths=(2048,))
        strip = value_strip(make_frame("21.7 nm"), "field_of_view")
        assert templates.match(2048, strip) == "21.7 nm"

    def test_does_not_match_a_different_string(self) -> None:
        templates = templates_for(["21.7 nm"], widths=(2048,))
        other = value_strip(make_frame("44.7 nm"), "field_of_view")
        assert templates.match(2048, other) is None

    def test_templates_are_not_shared_between_bar_widths(self) -> None:
        """Glyphs are never rescaled between sizes, so neither are templates."""
        templates = templates_for(["21.7 nm"], widths=(2048,))
        small = value_strip(make_frame("21.7 nm", width=1024), "field_of_view")
        assert templates.match(1024, small) is None

    def test_round_trips_through_disk(self, tmp_path) -> None:
        templates = templates_for(["21.7 nm", "44.7 nm"])
        path = tmp_path / "templates.npz"
        templates.save(path)
        loaded = StripTemplates.load(path)
        for width in (2048, 1024):
            for text in ("21.7 nm", "44.7 nm"):
                strip = value_strip(make_frame(text, width=width), "field_of_view")
                assert loaded.match(width, strip) == text

    def test_empty_set_matches_nothing(self) -> None:
        strip = value_strip(make_frame(), "field_of_view")
        assert StripTemplates().match(2048, strip) is None


class TestDecodeFieldOfView:
    def test_end_to_end_gives_the_expected_scale(self) -> None:
        templates = templates_for(["21.7 nm"], widths=(2048,))
        result = decode_field_of_view(make_frame("21.7 nm"), templates)
        assert result.ok
        assert result.field_of_view_nm == pytest.approx(21.7)
        assert result.nm_per_px == pytest.approx(21.7 / 2048)
        assert result.image_width == 2048

    def test_scale_differs_between_image_sizes_for_the_same_text(self) -> None:
        """The same printed field over half the pixels is twice the pixel size."""
        templates = templates_for(["21.7 nm"])
        big = decode_field_of_view(make_frame("21.7 nm", width=2048), templates)
        small = decode_field_of_view(make_frame("21.7 nm", width=1024), templates)
        assert big.ok and small.ok
        assert small.nm_per_px == pytest.approx(2 * big.nm_per_px)

    def test_unrecognised_text_is_refused_rather_than_guessed(self) -> None:
        templates = templates_for(["21.7 nm"], widths=(2048,))
        result = decode_field_of_view(make_frame("44.7 nm"), templates)
        assert not result.ok
        assert result.nm_per_px is None

    def test_impossible_reading_is_refused_but_reported(self) -> None:
        """A '2 m' frame yields no scale, yet keeps the label for diagnosis."""
        templates = templates_for(["2 m"], widths=(2048,))
        result = decode_field_of_view(make_frame("2 m"), templates)
        assert not result.ok
        assert result.nm_per_px is None
        assert result.label == "2 m"

    def test_frame_without_a_bar_is_refused(self) -> None:
        templates = templates_for(["21.7 nm"], widths=(2048,))
        result = decode_field_of_view(np.zeros((512, 512), dtype=np.uint8), templates)
        assert not result.ok

    def test_never_raises_on_a_degenerate_frame(self) -> None:
        templates = templates_for(["21.7 nm"], widths=(2048,))
        for frame in (
            np.zeros((64, 32), dtype=np.uint8),
            np.full((70, 64), 255, dtype=np.uint8),
        ):
            assert not decode_field_of_view(frame, templates).ok


class TestHarvestStrips:
    def test_identical_strips_collapse_to_one_entry(self) -> None:
        strip = value_strip(make_frame("21.7 nm"), "field_of_view")
        groups = harvest_strips([(2048, strip)] * 7)
        assert len(groups) == 1
        assert groups[0][2] == 7

    def test_distinct_strips_stay_separate_and_sort_by_frequency(self) -> None:
        a = value_strip(make_frame("21.7 nm"), "field_of_view")
        b = value_strip(make_frame("44.7 nm"), "field_of_view")
        groups = harvest_strips([(2048, a)] * 2 + [(2048, b)] * 5)
        assert len(groups) == 2
        assert [g[2] for g in groups] == [5, 2]

    def test_the_same_text_at_two_sizes_is_not_merged(self) -> None:
        big = value_strip(make_frame("21.7 nm", width=2048), "field_of_view")
        small = value_strip(make_frame("21.7 nm", width=1024), "field_of_view")
        groups = harvest_strips([(2048, big), (1024, small)])
        assert len(groups) == 2

    def test_no_strips_gives_no_groups(self) -> None:
        assert harvest_strips([]) == []

    def test_harvest_output_can_seed_a_template_set(self) -> None:
        """The intended workflow: harvest, label once, decode everything."""
        frames = [make_frame("21.7 nm")] * 3 + [make_frame("44.7 nm")] * 2
        strips = [(2048, value_strip(f, "field_of_view")) for f in frames]
        groups = harvest_strips(strips)
        templates = StripTemplates()
        for width, bitmap, _count in groups:
            # A human reads each representative once; here we know the order.
            templates.add(width, bitmap, "21.7 nm" if _count == 3 else "44.7 nm")
        results = [decode_field_of_view(f, templates) for f in frames]
        assert all(r.ok for r in results)
        assert sum(r.field_of_view_nm == pytest.approx(21.7) for r in results) == 3


class TestMeasureScaleBar:
    def test_recovers_the_drawn_rule_length(self) -> None:
        assert measure_scale_bar(make_frame(scale_bar_px=445)) == pytest.approx(445, abs=2)

    def test_absent_rule_gives_none(self) -> None:
        assert measure_scale_bar(make_frame(scale_bar_px=None)) is None

    def test_square_frame_gives_none(self) -> None:
        assert measure_scale_bar(np.zeros((256, 256), dtype=np.uint8)) is None

    def test_cross_checks_the_calibration_convention(self) -> None:
        """The check that field of view spans the image width.

        A 500 nm rule at 2.3 um across 2048 px should be 445 px. Measuring it
        and multiplying back by nm_per_px must reproduce the printed legend; a
        mismatch would mean the convention is wrong and every derived length
        with it.
        """
        templates = templates_for(["2.3 um"], widths=(2048,))
        frame = make_frame("2.3 um", scale_bar_px=445)
        result = decode_field_of_view(frame, templates)
        assert result.ok
        measured_px = measure_scale_bar(frame)
        assert measured_px is not None
        assert measured_px * result.nm_per_px == pytest.approx(500, rel=0.02)
