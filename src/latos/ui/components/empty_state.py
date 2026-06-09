"""`EmptyState` — a friendly placeholder for "nothing here yet" views.

A centered, muted icon over a short title and one explanatory line,
with an optional call-to-action button. Used wherever a page has no
data to show (no project open, no samples detected, no analyzers
applicable) so dead-ends feel intentional and guide the next step
instead of looking broken.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, IconWidget, PrimaryPushButton, SubtitleLabel

from latos.ui.design import tokens

__all__ = ["EmptyState"]

# Size of the decorative icon — large enough to anchor the centered
# block without dominating it.
_ICON_SIZE = 48


class EmptyState(QWidget):
    """Centered icon + title + message, with an optional action button."""

    actionClicked = Signal()  # noqa: N815  (Qt signals use mixedCase)

    def __init__(
        self,
        *,
        icon: Any,
        title: str,
        message: str,
        action_text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(tokens.SPACE_MD)
        layout.setContentsMargins(
            tokens.SPACE_XL,
            tokens.SPACE_3XL,
            tokens.SPACE_XL,
            tokens.SPACE_3XL,
        )

        icon_widget = IconWidget(icon, self)
        icon_widget.setFixedSize(_ICON_SIZE, _ICON_SIZE)
        layout.addWidget(icon_widget, 0, Qt.AlignmentFlag.AlignHCenter)

        title_label = SubtitleLabel(title, self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        message_label = BodyLabel(message, self)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setWordWrap(True)
        layout.addWidget(message_label)

        if action_text is not None:
            button = PrimaryPushButton(action_text, self)
            button.setMinimumHeight(36)
            button.clicked.connect(self.actionClicked)
            # Wrap so the button keeps its natural width and stays centered.
            layout.addSpacing(tokens.SPACE_SM)
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
