"""Latos desktop entry point.

Invoked by the `latos-app` console script (defined in `pyproject.toml`'s
`[project.gui-scripts]`). Boots the QApplication, applies the Latos
theme, constructs the main window, and runs the event loop.

Headless / programmatic launch
------------------------------
Tests use `pytest-qt`'s `qtbot` fixture which manages its own
`QApplication` — they instantiate `LatosMainWindow` directly without
calling `main()`. So `main()` only handles the "real launch via the
console script" path; everything else uses the lower-level pieces.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from latos.ui.main_window import LatosMainWindow
from latos.ui.themes import apply_system_theme

__all__ = ["main"]


def main() -> int:
    """Boot the Latos desktop app. Returns the QApplication exit code.

    Designed to be wired to a `gui-script` entry point — the
    `sys.exit(main())` idiom in `__main__` lets `python -m latos.ui.app`
    also work for development.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    apply_system_theme()

    window = LatosMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
