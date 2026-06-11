"""Desktop UI layer for Latos.

PySide6 + QFluentWidgets, structured around a single `FluentWindow` with a
sidebar (navigation) and a stacked content area (pages). Every layer below
this one (core, persistence, ingestion) must remain importable headlessly —
nothing under `latos.ui.*` may be imported from there.
"""
