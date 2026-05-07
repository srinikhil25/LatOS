"""Tests for `latos.ui.pages.welcome.WelcomePage`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from latos.ui.pages.welcome import WelcomePage

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.ui


class TestWelcomePage:
    def test_constructs_standalone(self, qtbot: QtBot):
        page = WelcomePage()
        qtbot.addWidget(page)
        assert page.objectName() == "WelcomePage"

    def test_renders_brand_text(self, qtbot: QtBot):
        page = WelcomePage()
        qtbot.addWidget(page)
        # Walk the children for the title text — keeps the test resilient
        # to layout reorganization.
        from PySide6.QtWidgets import QLabel

        labels = [child.text() for child in page.findChildren(QLabel)]
        joined = " | ".join(labels)
        assert "Latos" in joined
        assert "characterization" in joined.lower()
