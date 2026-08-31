"""Tests for publication figure generation."""

from __future__ import annotations

import pytest

from latos.visualization.figures import (
    ScatterPoint,
    correlation_heatmap,
    figure_to_bytes,
    scatter,
)
from latos.visualization.styles import JournalStyle, palette, rc_params


class TestStyles:
    def test_every_style_has_a_figsize(self):
        for style in JournalStyle:
            rc = rc_params(style)
            assert "figure.figsize" in rc
            assert len(rc["figure.figsize"]) == 2

    def test_palette_is_colourblind_set(self):
        assert len(palette()) >= 6
        assert all(c.startswith("#") for c in palette())

    def test_origin_preset_inverts_the_shared_defaults(self):
        """The ORIGIN look is defined by four deliberate departures from `_BASE`.

        Every other preset leaves the top and right spines off, points ticks
        outward and drops the legend box. Origin does the opposite on all four,
        and that combination is what makes a figure read as an Origin figure -
        so it is pinned rather than left to drift back to the shared default.
        """
        rc = rc_params(JournalStyle.ORIGIN)
        assert rc["axes.spines.top"] is True
        assert rc["axes.spines.right"] is True
        assert rc["xtick.direction"] == "in"
        assert rc["ytick.direction"] == "in"
        assert rc["legend.frameon"] is True

    def test_origin_preset_draws_ticks_on_all_four_sides_with_minors(self):
        rc = rc_params(JournalStyle.ORIGIN)
        assert rc["xtick.top"] is True
        assert rc["ytick.right"] is True
        assert rc["xtick.minor.visible"] is True
        assert rc["ytick.minor.visible"] is True
        assert rc["xtick.minor.size"] < rc["xtick.major.size"]

    def test_origin_frame_is_heavier_than_a_journal_preset(self):
        assert (
            rc_params(JournalStyle.ORIGIN)["axes.linewidth"]
            > (rc_params(JournalStyle.NATURE)["axes.linewidth"])
        )

    def test_presets_do_not_share_mutable_state(self):
        """`rc_params` must hand back a fresh dict, or one caller's edit leaks."""
        first = rc_params(JournalStyle.ORIGIN)
        first["axes.linewidth"] = 99
        assert rc_params(JournalStyle.ORIGIN)["axes.linewidth"] != 99


class TestHeatmap:
    def test_renders_svg_with_labels(self):
        props = ["band_gap_ev", "peak_zt", "crystallite_size_nm"]
        matrix = [[1.0, 0.5, None], [0.5, 1.0, 0.9], [None, 0.9, 1.0]]
        svg = figure_to_bytes(correlation_heatmap(props, matrix), "svg")
        assert svg[:5] == b"<?xml" or b"<svg" in svg[:400]
        assert b"band gap ev" in svg  # underscores prettified into the labels

    def test_renders_pdf_and_png(self):
        props = ["a", "b"]
        matrix = [[1.0, 0.3], [0.3, 1.0]]
        pdf = figure_to_bytes(correlation_heatmap(props, matrix), "pdf")
        png = figure_to_bytes(correlation_heatmap(props, matrix), "png")
        assert pdf[:5] == b"%PDF-"
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


class TestScatter:
    def test_renders_and_labels_axes(self):
        pts = [
            ScatterPoint("A", 1.0, 2.0),
            ScatterPoint("B", 2.0, 4.1),
            ScatterPoint("C", 3.0, 5.9),
        ]
        svg = figure_to_bytes(scatter("x_prop", "y_prop", pts, style=JournalStyle.THESIS), "svg")
        assert b"x prop" in svg and b"y prop" in svg
        assert b"r = " in svg  # the fit annotation

    def test_unreliable_points_excluded_from_fit(self):
        # Three reliable colinear points + one wild unreliable outlier.
        pts = [
            ScatterPoint("A", 1.0, 1.0),
            ScatterPoint("B", 2.0, 2.0),
            ScatterPoint("C", 3.0, 3.0),
            ScatterPoint("BAD", 4.0, -50.0, reliable=False),
        ]
        svg = figure_to_bytes(scatter("x", "y", pts), "svg").decode("utf-8", "replace")
        # The reliable points are perfectly correlated -> r = 1.00 despite the outlier.
        assert "r = 1.00" in svg

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            figure_to_bytes(correlation_heatmap(["a"], [[1.0]]), "gif")
