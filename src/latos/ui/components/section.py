"""`SectionHeader` — a consistent title block for page sections.

A bold title, an optional muted caption beneath it, and an optional
right-aligned action slot (e.g. a "Run on all" button). Using this
everywhere keeps section typography and spacing identical across pages.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, StrongBodyLabel

from latos.ui.design import tokens

__all__ = ["SectionHeader"]


class SectionHeader(QWidget):
    """Title (+ optional caption) with an optional right-aligned action."""

    def __init__(
        self,
        title: str,
        *,
        caption: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(tokens.SPACE_MD)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(tokens.SPACE_2XS)

        self._title = StrongBodyLabel(title, self)
        text_col.addWidget(self._title)

        # CaptionLabel is already theme-muted by QFluentWidgets — don't
        # override its color, or it stops adapting on theme switch.
        self._caption: CaptionLabel | None = None
        if caption is not None:
            self._caption = CaptionLabel(caption, self)
            text_col.addWidget(self._caption)

        row.addLayout(text_col, 1)
        row.addStretch(1)

        # Action slot lives on the right, vertically centered.
        self._action_holder = QHBoxLayout()
        self._action_holder.setContentsMargins(0, 0, 0, 0)
        self._action_holder.setSpacing(tokens.SPACE_SM)
        row.addLayout(self._action_holder, 0)

    def set_title(self, title: str) -> None:
        """Update the section title text."""
        self._title.setText(title)

    def set_action(self, widget: QWidget) -> None:
        """Place a widget (usually a button) in the right-aligned slot."""
        self._action_holder.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)
