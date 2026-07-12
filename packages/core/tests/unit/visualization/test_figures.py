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
