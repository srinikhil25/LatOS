"""`RecentProjectsService` — persistent MRU list of project folders.

The "Open Recent" rail on the project picker page reads from this service.
State lives at `$LATOS_HOME/recent.json` (defaulting to `~/.latos/`), so it
survives across restarts and is shared between debug + packaged runs of
the app on the same machine.

Design choices
--------------
- **Pure Python.** No Qt. Testable headlessly without `qtbot`.
- **MRU semantics.** `add(path)` deduplicates and moves the entry to the
  top. List is capped at `max_entries` (default 20) to keep the picker
  rail readable.
- **Filter-on-read.** Entries whose path no longer exists on disk are
  silently dropped from `list()` and from any subsequent persisted write.
  We don't proactively re-scan a separate "vacuum" — eventual consistency
  via the next `add`/`remove` is sufficient.
- **Atomic writes.** Same `.tmp` + `os.replace()` pattern used elsewhere
  (ParsedData arrays, Alembic). A Ctrl+C mid-write leaves the previous
  valid `recent.json` intact.
- **Tolerant load.** Corrupt JSON, missing file, schema drift → treat as
  an empty list rather than crashing the app on startup. Recents are
  convenience state, not a source of truth.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "RecentProject",
    "RecentProjectsService",
    "default_state_path",
]

_LOGGER = logging.getLogger(__name__)

# 20 keeps the picker rail short enough to scan at a glance. Increase only
# if user research shows people regularly hunt past entry 20.
DEFAULT_MAX_ENTRIES = 20

# Schema version for the on-disk file. Bump if the JSON shape changes in a
# breaking way; the loader returns an empty list on mismatch rather than
# trying to migrate (recents are throwaway state).
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RecentProject:
    """One entry in the recent-projects list.

    `path` is stored as an absolute `Path`. `name` defaults to the folder
    name but can be overridden later (e.g. read from a project metadata
    file). `last_opened_at` is when the user last opened this project via
    the app — used to order the MRU list.
    """

    path: Path
    name: str
    last_opened_at: datetime


def default_state_path() -> Path:
    """Return the default location for the recent-projects state file.

    Honors `$LATOS_HOME` so tests and CI can redirect to a tmp dir without
    touching the real `~/.latos`. Falls back to `~/.latos/` otherwise.
    """
    override = os.environ.get("LATOS_HOME")
    base = Path(override) if override else Path.home() / ".latos"
    return base / "recent.json"


class RecentProjectsService:
    """Persistent MRU list of project folders.

    Each public method reads + filters the on-disk state, mutates in
    memory, and writes back. The instance holds no in-memory cache, so two
    services pointed at the same file (e.g. dev runs of the app) stay in
    sync without an explicit refresh — at the cost of a JSON read per
    call. The list is capped at 20 entries so the cost is negligible.
    """

    def __init__(
        self,
        state_path: Path | None = None,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """Build a service rooted at `state_path` (or the default location).

        Args:
            state_path: Where the JSON file lives. `None` resolves to
                `default_state_path()`. Tests typically pass a `tmp_path`.
            max_entries: How many entries to keep before trimming.
        """
        self._state_path = state_path or default_state_path()
        self._max_entries = max_entries

    @property
    def state_path(self) -> Path:
        """Filesystem location of the persisted state file."""
        return self._state_path

    def entries(self) -> list[RecentProject]:
        """Return entries in MRU order, dropping any whose path is gone.

        Named `entries` rather than `list` so the method doesn't shadow
        `builtins.list` inside the class scope (which breaks `list[...]`
        return-type annotations under mypy strict).
        """
        return self._load_filtered()

    def add(self, path: Path, *, name: str | None = None) -> RecentProject:
        """Insert (or promote) `path` to the top of the list.

        Returns the entry as written so the caller can read its
        `last_opened_at` if desired. The path is resolved to an absolute
        `Path`; relative paths from `QFileDialog` are normalized here.
        """
        resolved = path.resolve()
        entry = RecentProject(
            path=resolved,
            name=name or resolved.name or str(resolved),
            last_opened_at=datetime.now(UTC),
        )

        existing = self._load_filtered()
        # Drop any prior entry pointing at the same path so we don't end
        # up with duplicates after the prepend.
        kept = [e for e in existing if e.path != resolved]
        new_list = [entry, *kept][: self._max_entries]
        self._save(new_list)
        return entry

    def remove(self, path: Path) -> None:
        """Remove the entry for `path` if present. No-op otherwise."""
        resolved = path.resolve()
        kept = [e for e in self._load_filtered() if e.path != resolved]
        self._save(kept)

    def clear(self) -> None:
        """Empty the list and persist."""
        self._save([])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_filtered(self) -> list[RecentProject]:
        """Read the JSON, drop missing-on-disk paths, return MRU-ordered."""
        raw = self._load_raw()
        return [e for e in raw if e.path.exists() and e.path.is_dir()]

    def _load_raw(self) -> list[RecentProject]:
        """Parse the state file. Returns `[]` on any error."""
        if not self._state_path.exists():
            return []
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _LOGGER.warning(
                "recent.json unreadable at %s; treating as empty",
                self._state_path,
            )
            return []

        if not isinstance(data, dict) or data.get("version") != _SCHEMA_VERSION:
            _LOGGER.warning(
                "recent.json schema mismatch at %s; treating as empty",
                self._state_path,
            )
            return []

        entries_raw = data.get("entries")
        if not isinstance(entries_raw, list):
            return []

        out: list[RecentProject] = []
        for raw in entries_raw:
            entry = self._parse_entry(raw)
            if entry is not None:
                out.append(entry)
        return out

    @staticmethod
    def _parse_entry(raw: Any) -> RecentProject | None:
        """Convert one JSON dict into a `RecentProject`. None on bad shape."""
        if not isinstance(raw, dict):
            return None
        try:
            return RecentProject(
                path=Path(raw["path"]),
                name=str(raw["name"]),
                last_opened_at=datetime.fromisoformat(raw["last_opened_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _save(self, entries: list[RecentProject]) -> None:
        """Atomically persist `entries` to the state file."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": _SCHEMA_VERSION,
            "entries": [
                {
                    "path": str(e.path),
                    "name": e.name,
                    "last_opened_at": e.last_opened_at.isoformat(),
                }
                for e in entries
            ],
        }

        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._state_path)
