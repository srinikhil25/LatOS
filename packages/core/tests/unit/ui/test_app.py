"""Tests for `latos.ui.app.main`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from latos.ui import app as app_module

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.ui


class TestMain:
    def test_main_constructs_window_and_returns_exec_code(self, qtbot: QtBot):
        """`main()` should set up the window and return the QApp exit code.

        We patch `QApplication.exec` to return immediately so the test
        doesn't block. Real launch behavior is exercised by the desktop
        when the user runs `latos-app`.
        """
        _ = qtbot
        with patch("latos.ui.app.QApplication.exec", return_value=0) as exec_mock:
            code = app_module.main()
        assert code == 0
        exec_mock.assert_called_once()
