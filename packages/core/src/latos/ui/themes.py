"""Theme management for Latos.

Wraps QFluentWidgets' theme primitives (`Theme`, `setTheme`,
`setThemeColor`) behind two functions: `apply_theme()` to set a
specific theme, and `apply_system_theme()` to follow the OS.

Why a thin wrapper instead of calling QFluentWidgets directly
-------------------------------------------------------------
The accent color, theme mode, and a couple of font tweaks are the
three things we'd want to centralize when we add user-configurable
appearance later (Stage 8). Wrapping them now means one file changes,
not the entire UI surface.
"""

from __future__ import annotations

from qfluentwidgets import Theme, setTheme, setThemeColor

__all__ = ["LATOS_ACCENT", "apply_theme", "apply_system_theme"]


# Latos brand color — a slightly desaturated blue that reads well in
# both dark and light themes. Picked to match scientific/lab software
# conventions without being the cliché "Microsoft blue".
LATOS_ACCENT = "#3B7DD8"


def apply_theme(theme: Theme = Theme.AUTO) -> None:
    """Apply a Latos theme to the current application.

    Args:
        theme: `Theme.LIGHT`, `Theme.DARK`, or `Theme.AUTO` (follows OS).
            Defaults to AUTO so users get whatever their system prefers.
    """
    setTheme(theme)
    setThemeColor(LATOS_ACCENT)


def apply_system_theme() -> None:
    """Follow the OS dark/light setting. Convenience alias for `apply_theme(AUTO)`."""
    apply_theme(Theme.AUTO)
