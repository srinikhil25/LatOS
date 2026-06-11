"""Smoke tests — verify the package is importable and basic invariants hold.

These run on every CI build to catch packaging regressions early.
"""

from __future__ import annotations

import latos


def test_package_imports() -> None:
    """Latos package can be imported without side effects."""
    assert latos is not None


def test_version_exists() -> None:
    """Latos exports a __version__ string."""
    assert hasattr(latos, "__version__")
    assert isinstance(latos.__version__, str)
    assert len(latos.__version__) > 0


def test_version_format() -> None:
    """Version follows semver-ish format (X.Y.Z[-suffix])."""
    parts = latos.__version__.split("-")[0].split(".")
    assert len(parts) == 3, f"Expected X.Y.Z, got {latos.__version__}"
    for part in parts:
        assert part.isdigit(), f"Non-numeric version component: {part}"
