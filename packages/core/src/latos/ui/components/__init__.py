"""Reusable, token-driven UI components for Latos.

These replace the ad-hoc layouts and hardcoded styling that used to
live inside each page. Build pages from these so spacing, surfaces, and
empty states stay identical across the app.
"""

from __future__ import annotations

from latos.ui.components.cards import ActivityCard, Card, StatTile
from latos.ui.components.chips import TechniqueChip, technique_label
from latos.ui.components.containers import PageContainer
from latos.ui.components.empty_state import EmptyState
from latos.ui.components.section import SectionHeader

__all__ = [
    "ActivityCard",
    "Card",
    "EmptyState",
    "PageContainer",
    "SectionHeader",
    "StatTile",
    "TechniqueChip",
    "technique_label",
]
