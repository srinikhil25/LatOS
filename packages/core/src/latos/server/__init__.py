"""Latos sidecar server — localhost-only FastAPI app over the core.

The desktop shell spawns `python -m latos.server` and talks to it on
127.0.0.1. See `app.py` for the endpoint surface.
"""

from __future__ import annotations

from latos.server.app import create_app

__all__ = ["create_app"]
