"""Card components — surfaces that group related content.

- `Card`: a plain padded surface with a vertical layout. The generic
  building block for grouped content.
- `StatTile`: a compact card showing a big number over a caption (the
  Overview summary row).
- `ActivityCard`: a large clickable card — icon, title, one-line
  description, chevron — used on the Project Hub to launch a section.

All three build on QFluentWidgets' `CardWidget`, so they inherit its
themed surface, subtle border, and hover treatment; we only standardize
padding and content.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    IconWidget,
    StrongBodyLabel,
    TitleLabel,
)

from latos.ui.design import tokens

__all__ = ["ActivityCard", "Card", "StatTile"]


class Card(CardWidget):  # type: ignore[misc]
    """A padded surface with a vertical content layout.

    Add children with `add` / `add_layout`, or use `.content_layout`.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        padding: int = tokens.SPACE_XL,
        spacing: int = tokens.SPACE_MD,
    ) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(spacing)

    @property
    def content_layout(self) -> QVBoxLayout:
        """The card's vertical content layout."""
        return self._layout

    def add(self, widget: QWidget) -> None:
        """Append a widget to the card."""
        self._layout.addWidget(widget)


class StatTile(CardWidget):  # type: ignore[misc]
    """A compact summary tile: a big value over a small caption."""

    def __init__(
        self,
        label: str,
        value: str | int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(f"StatTile_{label}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            tokens.SPACE_XL,
            tokens.SPACE_LG,
            tokens.SPACE_XL,
            tokens.SPACE_LG,
        )
        layout.setSpacing(tokens.SPACE_2XS)

        self._value = TitleLabel(str(value), self)
        self._caption = CaptionLabel(label, self)

        layout.addWidget(self._value)
        layout.addWidget(self._caption)

    def set_value(self, value: str | int) -> None:
        """Update the big number."""
        self._value.setText(str(value))


class ActivityCard(CardWidget):  # type: ignore[misc]
    """A large clickable launcher card for the Project Hub.

    Emits `activated(key)` when clicked, where `key` is the caller-
    supplied route identifier (e.g. "analysis"). The hub maps the key
    to a navigation action.
    """

    activated = Signal(str)

    def __init__(
        self,
        *,
        key: str,
        icon: Any,
        title: str,
        description: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._key = key
        self.setObjectName(f"ActivityCard_{key}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        row = QHBoxLayout(self)
        row.setContentsMargins(
            tokens.SPACE_XL,
            tokens.SPACE_LG,
            tokens.SPACE_XL,
            tokens.SPACE_LG,
        )
        row.setSpacing(tokens.SPACE_LG)

        icon_widget = IconWidget(icon, self)
        icon_widget.setFixedSize(28, 28)
        row.addWidget(icon_widget, 0, Qt.AlignmentFlag.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(tokens.SPACE_2XS)
        title_label = StrongBodyLabel(title, self)
        description_label = BodyLabel(description, self)
        description_label.setWordWrap(True)
        text_col.addWidget(title_label)
        text_col.addWidget(description_label)
        row.addLayout(text_col, 1)

        chevron = IconWidget(FluentIcon.CHEVRON_RIGHT, self)
        chevron.setFixedSize(16, 16)
        row.addWidget(chevron, 0, Qt.AlignmentFlag.AlignVCenter)

        # CardWidget fires a zero-arg `clicked` on mouse release; re-emit
        # it as `activated(key)` so the hub knows which card was hit.
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self) -> None:
        self.activated.emit(self._key)
