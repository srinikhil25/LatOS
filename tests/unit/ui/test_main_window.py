"""Tests for `latos.ui.main_window.LatosMainWindow`."""

from __future__ import annotations

import pytest

from latos.ui.main_window import LatosMainWindow
from latos.ui.pages.welcome import WelcomePage

pytestmark = pytest.mark.ui


class TestWindowConstruction:
    def test_window_constructs(self, latos_window: LatosMainWindow):
        # If the constructor raises (Qt theme miswire, missing widget,
        # etc.), the test fails here before any assertion runs.
        assert latos_window is not None

    def test_window_title(self, latos_window: LatosMainWindow):
        assert latos_window.windowTitle() == "Latos"

    def test_default_window_size(self, latos_window: LatosMainWindow):
        # Matches `_DEFAULT_WINDOW_SIZE`. Fixed values catch accidental
        # regressions from someone "tidying" the constants.
        size = latos_window.size()
        assert size.width() == 1280
        assert size.height() == 800

    def test_minimum_window_size(self, latos_window: LatosMainWindow):
        min_size = latos_window.minimumSize()
        assert min_size.width() == 960
        assert min_size.height() == 600


class TestWelcomePageRegistered:
    def test_welcome_page_present_in_widget_tree(self, latos_window: LatosMainWindow):
        # `addSubInterface` parents the page to the FluentWindow's
        # stacked widget. A simple findChild verifies the registration
        # succeeded.
        welcome = latos_window.findChild(WelcomePage, "WelcomePage")
        assert welcome is not None
