"""Per-sample synthesis parameters — the BO input space (X).

Latos's parsers capture characterization *outputs*; Bayesian
optimization also needs the synthesis *inputs* (doping %, annealing
temperature, time, ...). Those are the researcher's own knowledge,
entered in the Optimize screen and persisted here as a small JSON
sidecar under `<root>/.latos/`, keyed by sample id.

A JSON sidecar (not a schema column) keeps this a lightweight overlay —
no migration, easy to read, and the synthesis recipe lives right next
to the project it belongs to. Writes are atomic (tmp + os.replace).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["load_params", "save_params", "set_sample_params"]

# Mapping: sample_id -> {parameter_name: value}.
SynthesisParams = dict[str, dict[str, float]]


def _path(root: Path) -> Path:
    return root / ".latos" / "synthesis_params.json"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_params(root: Path) -> SynthesisParams:
    """Load all per-sample parameters; `{}` if none / unreadable."""
    path = _path(root)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: SynthesisParams = {}
    if isinstance(raw, dict):
        for sample_id, params in raw.items():
            if isinstance(params, dict):
                out[sample_id] = {k: float(v) for k, v in params.items() if _is_number(v)}
    return out


def save_params(root: Path, params: SynthesisParams) -> None:
    """Atomically persist the full parameters map."""
    path = _path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def set_sample_params(
    root: Path,
    sample_id: str,
    params: dict[str, float],
) -> SynthesisParams:
    """Replace one sample's parameters and persist. Returns the full map."""
    all_params = load_params(root)
    cleaned = {k: float(v) for k, v in params.items() if _is_number(v)}
    if cleaned:
        all_params[sample_id] = cleaned
    else:
        all_params.pop(sample_id, None)
    save_params(root, all_params)
    return all_params
