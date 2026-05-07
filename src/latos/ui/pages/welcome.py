"""The Welcome page — first thing the user sees on launch.

Stage 1E.1 ships a minimal placeholder. Stage 1E.2 wires up the actual
"Open Folder" / "Open Recent" buttons.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import StrongBodyLabel, SubtitleLabel, TitleLabel

__all__ = ["WelcomePage"]


class WelcomePage(QWidget):
    """The landing page shown when no project is open.

    Just a centered heading + tagline for Stage 1E.1. Buttons land in 1E.2.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Object name is what `FluentWindow.addSubInterface` uses to wire
        # navigation; must be unique across pages and stable for tests.
        self.setObjectName("WelcomePage")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        title = TitleLabel("Latos")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = SubtitleLabel("Multi-modal materials characterization")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hint = StrongBodyLabel("Open a folder to ingest your data.")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(hint)
        layout.addStretch(1)
