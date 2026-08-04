"""Samples the researcher has marked as not to be trusted.

Latos already distrusts a measurement when one of its own physics checks
rejects it (Wiedemann-Franz, Hall cross-configuration, the single-parabolic
-band ceiling). Those checks are automatic, and they only see the numbers.

A researcher sees more. They were in the room: the powder clumped, the
sonicator was skipped, the sensor was knocked between runs. None of that is
visible in the exported data, and until now there was no way to tell the
optimizer about it.

This is the human half of the quality signal, and it exists because the
automatic half cannot be complete. Harris et al. (npj Comput. Mater. 11, 23,
2025) make the same argument: their Dual-GP constrains the search from an
automated quality score *and* keeps a human-in-the-loop override, because a
score computed from the data cannot know why the data looks the way it does.

A distrusted sample is **not deleted**. It is fitted with a larger assumed
noise, exactly like a physics-flagged one, so the model still sees it but
stops chasing it. Deleting data silently is a different kind of dishonesty.

Stored as its own sidecar rather than inside the synthesis parameters: those
are the optimizer's *input space*, and a boolean flag among them would show up
as a candidate search axis.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["load_distrusted", "save_distrusted", "set_distrusted"]


def _path(root: Path) -> Path:
    return root / ".latos" / "distrusted_samples.json"


def load_distrusted(root: Path) -> set[str]:
    """Sample ids the researcher has marked untrusted; empty set if none."""
    path = _path(root)
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw if isinstance(item, str)}


def save_distrusted(root: Path, sample_ids: set[str]) -> None:
    """Persist the set atomically (tmp + os.replace), sorted for stable diffs."""
    path = _path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(sample_ids), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def set_distrusted(root: Path, sample_id: str, distrusted: bool) -> set[str]:
    """Mark one sample trusted or not; returns the updated set."""
    current = load_distrusted(root)
    if distrusted:
        current.add(sample_id)
    else:
        current.discard(sample_id)
    save_distrusted(root, current)
    return current
