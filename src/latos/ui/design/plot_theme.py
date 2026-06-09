"""Theme-aware styling for pyqtgraph plots.

Replaces the per-page `setBackground("w")` + hardcoded pen dictionaries
that scattered across `overview.py`, `sample_review.py`, and
`analysis.py`. Plots now match the active light/dark theme and the
brand accent, and every page draws its traces with the same palette.

Usage
-----
    from latos.ui.design import plot_theme

    plot_theme.style_plot_widget(self._plot)         # background + axes
    self._plot.plot(x, y, pen=plot_theme.accent_pen())
    self._plot.plot(xf, yf, pen=plot_theme.fit_pen())

Re-call `style_plot_widget` on theme change to recolor an existing plot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from latos.ui.design import tokens

if TYPE_CHECKING:
    from pyqtgraph import PlotWidget

__all__ = [
    "SERIES_PALETTE",
    "accent_pen",
    "fit_pen",
    "marker_pen",
    "series_pen",
    "style_plot_widget",
]

# Categorical palette for multiple traces on one plot. Accent first so
# single-series plots get the brand color; the rest are distinguishable
# and colorblind-friendly-ish without being garish.
SERIES_PALETTE: tuple[str, ...] = (
    tokens.ACCENT,  # blue (brand)
    "#CA5010",  # orange
    "#107C10",  # green
    "#8764B8",  # purple
    "#C239B3",  # magenta
    "#00808A",  # teal
)

# Alpha for the background grid — faint enough to read as a guide, not
# a cage. Matches the airy aesthetic.
_GRID_ALPHA = 0.12


def style_plot_widget(plot_widget: PlotWidget) -> None:
    """Apply the active theme to a pyqtgraph `PlotWidget`.

    Sets the background to the themed surface color and the axes/labels
    to the themed secondary-text color, and shows a faint grid.
    Idempotent — safe to call again after a theme switch to recolor.
    """
    plot_widget.setBackground(tokens.surface())

    axis_pen = pg.mkPen(color=tokens.border())
    text_pen = pg.mkPen(color=tokens.text_secondary())
    for axis_name in ("left", "bottom"):
        axis = plot_widget.getAxis(axis_name)
        axis.setPen(axis_pen)
        axis.setTextPen(text_pen)

    plot_widget.showGrid(x=True, y=True, alpha=_GRID_ALPHA)


def accent_pen(width: int = 2) -> Any:
    """Pen for the primary data trace (brand accent)."""
    return pg.mkPen(color=tokens.accent(), width=width)


def series_pen(index: int, width: int = 2) -> Any:
    """Pen for the `index`-th trace, cycling the categorical palette."""
    color = QColor(SERIES_PALETTE[index % len(SERIES_PALETTE)])
    return pg.mkPen(color=color, width=width)


def fit_pen() -> Any:
    """Dashed pen for an overlaid fit/model line."""
    return pg.mkPen(
        color=QColor("#CA5010"),
        width=2,
        style=Qt.PenStyle.DashLine,
    )


def marker_pen() -> Any:
    """Dotted pen for a vertical marker (band gap, peak center, …)."""
    return pg.mkPen(
        color=QColor("#107C10"),
        width=1,
        style=Qt.PenStyle.DotLine,
    )
