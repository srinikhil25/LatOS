"""Fit templates — serialize a `FitSpec` so a fit recipe can be reused.

A researcher who dials in a good XPS or XRD fit wants to reuse that exact
recipe (shape, background, peak count, constraints) on the next sample. A
template is just a JSON-round-trippable dict of a `FitSpec`. Peak *positions*
are intentionally kept — a template seeds the next fit and the user nudges
from there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from latos.fitting.constraints import (
    Constraint,
    FixedDelta,
    FixedRatio,
    SharedWidth,
)
from latos.fitting.engine import BackgroundKind, BackgroundSpec, FitSpec, PeakInit
from latos.fitting.peak_shapes import PeakShape

__all__ = ["load_template", "save_template", "spec_from_dict", "spec_to_dict"]

_TEMPLATE_VERSION = 1


def _constraint_to_dict(c: Constraint) -> dict[str, Any]:
    if isinstance(c, FixedDelta):
        return {"type": "fixed_delta", "ref": c.ref, "target": c.target, "delta": c.delta}
    if isinstance(c, FixedRatio):
        return {"type": "fixed_ratio", "ref": c.ref, "target": c.target, "ratio": c.ratio}
    return {"type": "shared_width", "ref": c.ref, "target": c.target}


def _constraint_from_dict(d: dict[str, Any]) -> Constraint:
    kind = d["type"]
    if kind == "fixed_delta":
        return FixedDelta(ref=d["ref"], target=d["target"], delta=d["delta"])
    if kind == "fixed_ratio":
        return FixedRatio(ref=d["ref"], target=d["target"], ratio=d["ratio"])
    if kind == "shared_width":
        return SharedWidth(ref=d["ref"], target=d["target"])
    raise ValueError(f"Unknown constraint type {kind!r}")


def spec_to_dict(spec: FitSpec) -> dict[str, Any]:
    """A JSON-serializable dict for a `FitSpec`."""
    return {
        "template_version": _TEMPLATE_VERSION,
        "peak_shape": spec.peak_shape.value,
        "peaks": [
            {"center": p.center, "amplitude": p.amplitude, "sigma": p.sigma} for p in spec.peaks
        ],
        "background": {
            "kind": spec.background.kind.value,
            "degree": spec.background.degree,
            "lam": spec.background.lam,
            "p": spec.background.p,
        },
        "constraints": [_constraint_to_dict(c) for c in spec.constraints],
    }


def spec_from_dict(d: dict[str, Any]) -> FitSpec:
    """Rebuild a `FitSpec` from `spec_to_dict`'s output."""
    bg = d["background"]
    return FitSpec(
        peak_shape=PeakShape(d["peak_shape"]),
        peaks=[
            PeakInit(center=p["center"], amplitude=p.get("amplitude"), sigma=p.get("sigma"))
            for p in d["peaks"]
        ],
        background=BackgroundSpec(
            kind=BackgroundKind(bg["kind"]),
            degree=bg.get("degree", 2),
            lam=bg.get("lam", 1e5),
            p=bg.get("p", 0.01),
        ),
        constraints=[_constraint_from_dict(c) for c in d.get("constraints", [])],
    )


def save_template(spec: FitSpec, path: Path) -> None:
    """Write `spec` to `path` as a JSON template."""
    path.write_text(json.dumps(spec_to_dict(spec), indent=2), encoding="utf-8")


def load_template(path: Path) -> FitSpec:
    """Read a JSON template written by `save_template`."""
    return spec_from_dict(json.loads(path.read_text(encoding="utf-8")))
