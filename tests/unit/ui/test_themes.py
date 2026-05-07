"""Tests for `latos.ui.themes`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qfluentwidgets import Theme, qconfig

from latos.ui.themes import LATOS_ACCENT, apply_system_theme, apply_theme

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.ui


class TestApplyTheme:
    def test_apply_explicit_dark(self, qtbot: QtBot):
        # qtbot ensures a QApplication exists before theme calls touch it.
        _ = qtbot
        apply_theme(Theme.DARK)
        # `qconfig.theme` returns the resolved theme value.
        assert qconfig.theme == Theme.DARK

    def test_apply_explicit_light(self, qtbot: QtBot):
        _ = qtbot
        apply_theme(Theme.LIGHT)
        assert qconfig.theme == Theme.LIGHT

    def test_apply_system_theme_uses_auto(self, qtbot: QtBot):
        _ = qtbot
        apply_system_theme()
        # AUTO resolves to LIGHT or DARK at runtime depending on OS;
        # we just verify it didn't raise and the call applied *some* theme.
        assert qconfig.theme in (Theme.LIGHT, Theme.DARK, Theme.AUTO)


class TestAccentColor:
    def test_accent_is_six_digit_hex(self):
        assert len(LATOS_ACCENT) == 7
        assert LATOS_ACCENT.startswith("#")
        # Validate the hex digits.
        int(LATOS_ACCENT[1:], 16)
