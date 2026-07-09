"""Bayesian-optimization layer — the closed-loop "what to make next".

A GP surrogate over (synthesis parameter -> measured property) plus an
Expected-Improvement acquisition function and a convergence verdict.
See `engine.py` for the method.

`optimize()` returns a frozen `BoConfig` alongside the recommendation, and
`prereg.freeze()` writes that (plus the predicted value + predictive interval)
to an auditable, timestamped record — so a recommendation can be committed
*before* the sample is made. `length_scale_robustness()` checks the pick isn't
a kernel artifact.
"""

from __future__ import annotations

from latos.optimization.engine import (
    BoConfig,
    OptimizationError,
    OptimizationResult,
    Recommendation,
    RobustnessEntry,
    RobustnessReport,
    length_scale_robustness,
    optimize,
)
from latos.optimization.prereg import build_record, freeze, write_record

__all__ = [
    "BoConfig",
    "OptimizationError",
    "OptimizationResult",
    "Recommendation",
    "RobustnessEntry",
    "RobustnessReport",
    "build_record",
    "freeze",
    "length_scale_robustness",
    "optimize",
    "write_record",
]
