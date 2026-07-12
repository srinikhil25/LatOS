"""Publication figure generators for the Stage-6 cross-correlation views.

Two figures, both rendered with a `JournalStyle` and exportable to vector
formats (SVG / PDF) or PNG:

* `correlation_heatmap` — the property × property Pearson matrix.
* `scatter` — one property pair across samples, with a fit line, the
  Pearson r, and *unreliable* points drawn as open markers so a reader can
  see at a glance which values failed a physics / data-quality check.

Uses matplotlib's object-oriented `Figure` API (no pyplot global state) so
it is safe to call from the server, and `rc_context` so a render never
leaks style into the next one.
"""

from __future__ import annotations

import io

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from latos.visualization.styles import JournalStyle, palette, rc_params

__all__ = ["ScatterPoint", "correlation_heatmap", "figure_to_bytes", "scatter"]

_VALID_FORMATS = frozenset({"svg", "pdf", "png"})
# Above this |r|, a heatmap cell is dark enough to need white annotation text.
_TEXT_CONTRAST_R = 0.6
# Minimum reliable points before a scatter fit line is meaningful.
_MIN_FIT_POINTS = 2


class ScatterPoint:
    """One sample's (x, y) with its label and reliability."""

    __slots__ = ("label", "reliable", "x", "y")

    def __init__(self, label: str, x: float, y: float, *, reliable: bool = True) -> None:
        self.label = label
        self.x = x
        self.y = y
        self.reliable = reliable


def _pretty(name: str) -> str:
    """`peak_thermal_conductivity` -> `peak thermal conductivity`."""
    return name.replace("_", " ")


def figure_to_bytes(fig: Figure, fmt: str = "svg") -> bytes:
    """Serialize a figure to `fmt` (svg | pdf | png) bytes."""
    if fmt not in _VALID_FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}; use one of {sorted(_VALID_FORMATS)}.")
    buffer = io.BytesIO()
    # bbox_inches explicit: savefig runs outside the style's rc_context, so
    # the "tight" rcParam is gone by now and long tick labels would clip.
    fig.savefig(buffer, format=fmt, bbox_inches="tight")
    return buffer.getvalue()


def correlation_heatmap(
    properties: list[str],
    matrix: list[list[float | None]],
    *,
    style: JournalStyle = JournalStyle.NATURE,
) -> Figure:
    """A Pearson-r heatmap over `properties` (None cells left blank)."""
    grid = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in matrix], dtype=float
    )
    labels = [_pretty(p) for p in properties]
    with matplotlib.rc_context(rc_params(style)):
        fig = Figure()
        ax = fig.subplots()
        image = ax.imshow(grid, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    shade = "white" if abs(grid[i, j]) > _TEXT_CONTRAST_R else "black"
                    ax.text(
                        j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color=shade, fontsize=6
                    )
        ax.set_title("Cross-property correlation")
        fig.colorbar(image, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
    return fig


def scatter(
    prop_x: str,
    prop_y: str,
    points: list[ScatterPoint],
    *,
    style: JournalStyle = JournalStyle.NATURE,
) -> Figure:
    """A labelled scatter of one property pair, with a fit line and Pearson r.

    Unreliable points (open markers) are drawn but *excluded* from the fit
    line and r, so a flagged value cannot silently drive the trend.
    """
    color = palette()[0]
    with matplotlib.rc_context(rc_params(style)):
        fig = Figure()
        ax = fig.subplots()
        rel_x, rel_y = [], []
        for pt in points:
            if pt.reliable:
                ax.scatter(pt.x, pt.y, color=color, zorder=3)
                rel_x.append(pt.x)
                rel_y.append(pt.y)
            else:
                ax.scatter(pt.x, pt.y, facecolors="none", edgecolors=color, zorder=3)
            ax.annotate(
                pt.label, (pt.x, pt.y), textcoords="offset points", xytext=(4, 3), fontsize=6
            )

        if len(rel_x) >= _MIN_FIT_POINTS and np.ptp(rel_x) > 0:
            xr = np.asarray(rel_x)
            yr = np.asarray(rel_y)
            slope, intercept = np.polyfit(xr, yr, 1)
            line = np.array([xr.min(), xr.max()])
            ax.plot(line, slope * line + intercept, color=color, lw=1.0, alpha=0.6, zorder=2)
            if np.ptp(yr) > 0:
                r = float(np.corrcoef(xr, yr)[0, 1])
                ax.annotate(
                    f"r = {r:.2f}  (n = {len(rel_x)})",
                    xy=(0.04, 0.92),
                    xycoords="axes fraction",
                    fontsize=7,
                )
        ax.set_xlabel(_pretty(prop_x))
        ax.set_ylabel(_pretty(prop_y))
    return fig
