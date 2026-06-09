"""Latos UI design system — tokens and themed plot styling.

Import the modules, not their members, so call sites read as
`tokens.surface()` / `plot_theme.accent_pen()` — self-documenting about
where a value comes from.

    from latos.ui.design import tokens, plot_theme
"""

from __future__ import annotations

from latos.ui.design import plot_theme, tokens

__all__ = ["plot_theme", "tokens"]
