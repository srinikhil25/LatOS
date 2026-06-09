"""Design tokens — the single source of truth for Latos's visual system.

Every page and component pulls spacing, radii, type sizes, and colors
from here instead of hardcoding values. Change a token, and the whole
app moves with it.

Design language
---------------
Minimal / airy (Linear / Notion family): generous whitespace, soft
cards, a muted neutral palette, and a single accent color. The levers
that create the "calm" feeling are mostly *spacing* and *restraint*,
not decoration — so the spacing scale below is the most important part
of this file.

Theme awareness
---------------
Color tokens are *functions*, not constants, because the resolved value
depends on the active light/dark theme (`qfluentwidgets.isDarkTheme()`).
Call them at paint/build time, and re-call them when the theme changes.
Sizing tokens (spacing, radii, type) are theme-independent constants.
"""

from __future__ import annotations

from PySide6.QtGui import QColor
from qfluentwidgets import isDarkTheme

from latos.core.enums import Severity, Technique

__all__ = [
    "ACCENT",
    "CONTENT_MAX_WIDTH",
    "PAGE_MARGIN",
    "PAGE_MARGIN_TIGHT",
    "RADIUS_LG",
    "RADIUS_MD",
    "RADIUS_SM",
    "SECTION_GAP",
    "SPACE_2XL",
    "SPACE_2XS",
    "SPACE_3XL",
    "SPACE_4XL",
    "SPACE_LG",
    "SPACE_MD",
    "SPACE_SM",
    "SPACE_XL",
    "SPACE_XS",
    "TECHNIQUE_TINT_ALPHA",
    "Type",
    "accent",
    "accent_hex",
    "border",
    "hex_of",
    "muted_surface",
    "page_bg",
    "severity_color",
    "severity_hex",
    "surface",
    "technique_color",
    "technique_hex",
    "technique_tint",
    "text_primary",
    "text_secondary",
]


# ─── Spacing scale (px) ──────────────────────────────────────────────
# A 4px-based scale. Use these everywhere instead of magic numbers; the
# airy feel comes from reaching for the *larger* steps (XL/2XL/3XL) for
# page margins and section gaps.
SPACE_2XS = 2
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24
SPACE_2XL = 32
SPACE_3XL = 40
SPACE_4XL = 56

# ─── Corner radii (px) ───────────────────────────────────────────────
RADIUS_SM = 6
RADIUS_MD = 10
RADIUS_LG = 14

# ─── Layout ──────────────────────────────────────────────────────────
# Generous page padding + a capped content width keep long lines
# readable and give the airy, centered-column feel of Linear/Notion.
PAGE_MARGIN = SPACE_3XL  # 40 — default page padding
PAGE_MARGIN_TIGHT = SPACE_XL  # 24 — for denser master/detail panes
CONTENT_MAX_WIDTH = 1120  # cap so content doesn't sprawl on wide monitors
SECTION_GAP = SPACE_2XL  # 32 — vertical gap between page sections


class Type:
    """Type scale (point sizes).

    QFluentWidgets ships semantic label classes (`TitleLabel`,
    `BodyLabel`, …) that we use for most text; these point sizes are for
    the cases where we set a font size directly (custom cards, plot
    captions). Kept deliberately small and few — restraint is the point.
    """

    DISPLAY = 28
    TITLE = 20
    SUBTITLE = 15
    BODY = 13
    CAPTION = 11


# ─── Color ───────────────────────────────────────────────────────────
# Brand accent — a slightly desaturated blue that reads well in both
# themes (matches `themes.LATOS_ACCENT`).
ACCENT = "#3B7DD8"

# Neutral palettes. Light is near-white with soft grays (airy); dark is
# a soft charcoal, not pure black, to avoid harsh contrast.
_LIGHT: dict[str, str] = {
    "page_bg": "#F7F8FA",
    "surface": "#FFFFFF",
    "muted_surface": "#F1F3F5",
    "border": "#E5E7EB",
    "text_primary": "#1F2328",
    "text_secondary": "#6B7280",
}
_DARK: dict[str, str] = {
    "page_bg": "#1C1D1F",
    "surface": "#26282B",
    "muted_surface": "#2F3134",
    "border": "#3A3C40",
    "text_primary": "#E8EAED",
    "text_secondary": "#9AA0A6",
}

# Severity colors are theme-independent (same hue in light/dark) so a
# warning is unmistakably a warning regardless of mode. Consolidated
# here from the per-page copies that used to live in analysis.py and
# sample_review.py.
_SEVERITY: dict[Severity, str] = {
    Severity.ERROR: "#D13438",
    Severity.WARNING: "#CA5010",
    Severity.INFO: "#0F6CBD",
}


def _palette() -> dict[str, str]:
    """Return the active palette for the current theme."""
    return _DARK if isDarkTheme() else _LIGHT


def page_bg() -> QColor:
    """Background color for a page surface."""
    return QColor(_palette()["page_bg"])


def surface() -> QColor:
    """Card / elevated-surface background color."""
    return QColor(_palette()["surface"])


def muted_surface() -> QColor:
    """Subtle inset background (e.g. param panels, list rows)."""
    return QColor(_palette()["muted_surface"])


def border() -> QColor:
    """Hairline border / divider color."""
    return QColor(_palette()["border"])


def text_primary() -> QColor:
    """Primary text color."""
    return QColor(_palette()["text_primary"])


def text_secondary() -> QColor:
    """Secondary / muted text color (captions, hints)."""
    return QColor(_palette()["text_secondary"])


def accent() -> QColor:
    """Brand accent color as a QColor."""
    return QColor(ACCENT)


def accent_hex() -> str:
    """Brand accent color as a hex string (for stylesheets)."""
    return ACCENT


def severity_color(severity: Severity) -> QColor:
    """Color for a validation/analysis issue of the given severity."""
    return QColor(_SEVERITY[severity])


def severity_hex(severity: Severity) -> str:
    """Hex string for a severity (for stylesheets / rich text)."""
    return _SEVERITY[severity]


def hex_of(color: QColor) -> str:
    """`#RRGGBB` string for a QColor — convenience for stylesheet building."""
    return color.name()


# ─── Technique colors ────────────────────────────────────────────────
# One hue per characterization technique, used by chips, plot traces,
# and thumbnails so a technique is recognizable at a glance anywhere in
# the app. Mid-saturation values chosen to read on both themes; loosely
# mnemonic (thermoelectric = heat orange, UV-DRS = amber optical,
# TEM/SEM = magenta/rose imaging family).
_TECHNIQUE: dict[Technique, str] = {
    Technique.XRD: "#4A8CE8",
    Technique.XPS: "#8764B8",
    Technique.UV_DRS: "#C98A1B",
    Technique.HALL: "#00808A",
    Technique.THERMOELECTRIC: "#D2542C",
    Technique.EDS: "#3D8B40",
    Technique.TEM: "#C239B3",
    Technique.SEM: "#D24A6E",
    Technique.STEM: "#5E63D1",
    Technique.RAMAN: "#2AA0C4",
    Technique.UNKNOWN: "#8A8F98",
}

# Alpha (0-255) for chip/tint backgrounds derived from technique colors.
TECHNIQUE_TINT_ALPHA = 46


def technique_color(technique: Technique) -> QColor:
    """The identity color for a technique."""
    return QColor(_TECHNIQUE[technique])


def technique_hex(technique: Technique) -> str:
    """Hex string of the technique identity color (for stylesheets)."""
    return _TECHNIQUE[technique]


def technique_tint(technique: Technique) -> QColor:
    """Low-alpha tint of the technique color — chip/row backgrounds."""
    color = QColor(_TECHNIQUE[technique])
    color.setAlpha(TECHNIQUE_TINT_ALPHA)
    return color
