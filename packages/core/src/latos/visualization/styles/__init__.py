"""Journal figure styles — publication-ready matplotlib rcParams presets.

Each preset encodes a target venue's typography, single-column figure size,
line weights, and export DPI, so a figure rendered with `JournalStyle.NATURE`
drops into a Nature manuscript at the right size and font without manual
tweaking. Colours come from a shared colourblind-safe palette (Wong 2011).

Applied via `matplotlib.rc_context(rc_params(style))` around figure creation
(see `latos.visualization.figures`), which keeps global rcParams untouched
between renders.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["PALETTE", "JournalStyle", "palette", "rc_params"]


class JournalStyle(StrEnum):
    """A target publication style."""

    NATURE = "nature"
    ACS = "acs"
    RSC = "rsc"
    THESIS = "thesis"
    PRESENTATION = "presentation"


# Wong (2011) colourblind-safe qualitative palette.
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#000000",  # black
]

# Shared defaults; each style overrides what it needs.
_BASE: dict[str, object] = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
}

_STYLES: dict[JournalStyle, dict[str, object]] = {
    # Nature: sans-serif, ~7 pt, 89 mm single column.
    JournalStyle.NATURE: {
        **_BASE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "figure.figsize": (3.50, 2.62),
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
    },
    # ACS: Arial/Helvetica, 8 pt, 3.25 in single column.
    JournalStyle.ACS: {
        **_BASE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.figsize": (3.25, 2.44),
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    },
    # RSC: Arial, 8 pt, 8.3 cm single column.
    JournalStyle.RSC: {
        **_BASE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.figsize": (3.27, 2.45),
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    },
    # Thesis: serif, 11 pt, wide.
    JournalStyle.THESIS: {
        **_BASE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.figsize": (6.0, 4.2),
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    },
    # Presentation: large, bold, high contrast.
    JournalStyle.PRESENTATION: {
        **_BASE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "figure.figsize": (8.0, 6.0),
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.5,
        "lines.markersize": 9,
    },
}


def rc_params(style: JournalStyle) -> dict[str, object]:
    """A fresh matplotlib rcParams dict for `style`."""
    return dict(_STYLES[style])


def palette() -> list[str]:
    """A copy of the colourblind-safe qualitative palette."""
    return list(PALETTE)
