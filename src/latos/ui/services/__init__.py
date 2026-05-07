"""UI-layer services — small persistent stores backing the desktop UI.

Each service owns one piece of cross-page state (recent projects, window
geometry, future "current sample"). They live next to the UI because the
UI is the only consumer; nothing under `latos.core` / `latos.persistence` /
`latos.ingestion` may import from here.

Services are intentionally Qt-free dataclasses + plain functions so they
can be unit-tested headlessly without `qtbot`.
"""
