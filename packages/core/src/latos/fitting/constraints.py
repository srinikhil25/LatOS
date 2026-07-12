"""Inter-peak constraints for the fit engine.

Physical peaks are rarely independent. A spin-orbit doublet in XPS has a
*fixed* energy splitting and a *fixed* area ratio set by the degeneracy of
the levels, and its two components share a line width. Encoding those as
hard constraints turns an under-determined multi-peak fit into a stable
one — fewer free parameters, physically guaranteed results.

Each constraint ties one peak's parameter to another's via an algebraic
expression (lmfit's `expr` mechanism), so the tied parameter stops varying
and is computed from its reference. Peaks are addressed by their index in
the `FitSpec.peaks` list (``0`` is the first peak).

Constraints:

* **FixedDelta** — ``target.center = ref.center + delta`` (e.g. the 19.8 eV
  Cu 2p₃/₂→2p₁/₂ spin-orbit splitting).
* **FixedRatio** — ``target.amplitude = ratio · ref.amplitude`` (e.g. the
  2:1 area ratio of a p-doublet, so ``ratio = 0.5``).
* **SharedWidth** — ``target.sigma = ref.sigma`` (both components broadened
  alike).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Constraint", "FixedDelta", "FixedRatio", "SharedWidth", "apply_constraints"]


@dataclass(frozen=True)
class FixedDelta:
    """Pin `target`'s center a fixed `delta` away from `ref`'s center."""

    ref: int
    target: int
    delta: float


@dataclass(frozen=True)
class FixedRatio:
    """Pin `target`'s amplitude (area) to `ratio` × `ref`'s amplitude."""

    ref: int
    target: int
    ratio: float


@dataclass(frozen=True)
class SharedWidth:
    """Force `target`'s sigma to equal `ref`'s sigma (shared line width)."""

    ref: int
    target: int


Constraint = FixedDelta | FixedRatio | SharedWidth


def apply_constraints(params: object, constraints: list[Constraint]) -> None:
    """Tie parameters together in-place on an lmfit `Parameters` object.

    `params` must already contain the ``p{i}_center`` / ``p{i}_amplitude`` /
    ``p{i}_sigma`` entries for every referenced peak index.
    """
    for c in constraints:
        if isinstance(c, FixedDelta):
            params[f"p{c.target}_center"].set(expr=f"p{c.ref}_center + {c.delta!r}")  # type: ignore[index]
        elif isinstance(c, FixedRatio):
            params[f"p{c.target}_amplitude"].set(expr=f"{c.ratio!r} * p{c.ref}_amplitude")  # type: ignore[index]
        else:  # SharedWidth
            params[f"p{c.target}_sigma"].set(expr=f"p{c.ref}_sigma")  # type: ignore[index]
