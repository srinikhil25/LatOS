"""Layout containers that enforce the airy page rhythm.

`PageContainer` is the outer shell every page should sit in: generous
margins, a capped content width so lines don't sprawl on wide monitors,
and a centered column. Pages add their sections to `.body` and get
consistent spacing for free.
"""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLayout, QVBoxLayout, QWidget

from latos.ui.design import tokens

__all__ = ["PageContainer"]


class PageContainer(QWidget):
    """A centered, max-width, generously-padded page body.

    Layout: `[ margin | stretch | column(≤CONTENT_MAX_WIDTH) | stretch | margin ]`.
    On narrow windows the column fills the available width (minus
    margins); on wide windows it centers and stops growing, which is
    what gives the calm Linear/Notion column feel.

    Add content via `add_widget` / `add_layout` / `add_stretch`, or grab
    `.body` directly. Sections are separated by `SECTION_GAP` automatically.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        margin: int = tokens.PAGE_MARGIN,
        max_width: int = tokens.CONTENT_MAX_WIDTH,
    ) -> None:
        super().__init__(parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(margin, margin, margin, margin)
        outer.setSpacing(0)

        outer.addStretch(1)

        self._column = QWidget(self)
        self._column.setMaximumWidth(max_width)
        self._body = QVBoxLayout(self._column)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(tokens.SECTION_GAP)

        outer.addWidget(self._column, stretch=10)
        outer.addStretch(1)

    @property
    def body(self) -> QVBoxLayout:
        """The vertical layout holding page sections."""
        return self._body

    def add_widget(self, widget: QWidget) -> None:
        """Append a section widget to the page column."""
        self._body.addWidget(widget)

    def add_layout(self, layout: QLayout) -> None:
        """Append a sub-layout to the page column."""
        self._body.addLayout(layout)

    def add_stretch(self, stretch: int = 1) -> None:
        """Push subsequent content up by absorbing trailing space."""
        self._body.addStretch(stretch)
