"""Bayesian-optimization layer — the closed-loop "what to make next".

A GP surrogate over (synthesis parameter -> measured property) plus an
Expected-Improvement acquisition function and a convergence verdict.
See `engine.py` for the method.
"""

from __future__ import annotations

from latos.optimization.engine import (
    OptimizationError,
    OptimizationResult,
    Recommendation,
    optimize,
)

__all__ = [
    "OptimizationError",
    "OptimizationResult",
    "Recommendation",
    "optimize",
]
