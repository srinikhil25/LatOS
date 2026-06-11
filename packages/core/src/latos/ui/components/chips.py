"""`TechniqueChip` — a small colored badge identifying a technique.

The chip is the technique's visual signature across the app: the same
hue appears in the Samples tree, measurement detail headers, and
(later) plot traces, so users learn "magenta = TEM" once and read it
everywhere.

Rendered as a QLabel with a rounded, tinted background and the
technique color as text. Pure stylesheet — no custom paint event — so
it stays cheap to construct in long tree/list rows.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from latos.core.enums import Technique
from latos.ui.design import tokens

__all__ = ["TechniqueChip", "technique_label"]

# Display names that differ from a plain `.value.upper()`.
_LABELS: dict[Technique, str] = {
    Technique.UV_DRS: "UV-DRS",
    Technique.THERMOELECTRIC: "TE",
    Technique.UNKNOWN: "?",
}


def technique_label(technique: Technique) -> str:
    """Short display label for a technique (e.g. `UV-DRS`, `TEM`)."""
    label = _LABELS.get(technique)
    if label is not None:
        return label
    return technique.value.upper()


class TechniqueChip(QLabel):
    """A compact rounded badge in the technique's identity color."""

    def __init__(
        self,
        technique: Technique,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(technique_label(technique), parent)
        self._technique = technique
        self.setObjectName(f"TechniqueChip_{technique.value}")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_style()

    @property
    def technique(self) -> Technique:
        """The technique this chip represents."""
        return self._technique

    def _apply_style(self) -> None:
        """Tinted pill background + technique-colored text."""
        color = tokens.technique_hex(self._technique)
        tint = tokens.technique_tint(self._technique)
        # rgba() keeps the tint translucent over any surface (cards,
        # tree rows, headers) without needing per-surface variants.
        background = f"rgba({tint.red()}, {tint.green()}, {tint.blue()}, {tint.alpha()})"
        self.setStyleSheet(
            f"""
            QLabel#{self.objectName()} {{
                color: {color};
                background-color: {background};
                border-radius: {tokens.RADIUS_SM}px;
                padding: 1px {tokens.SPACE_SM}px;
                font-size: {tokens.Type.CAPTION}pt;
                font-weight: 600;
            }}
            """,
        )
