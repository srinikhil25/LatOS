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

from latos.optimization import benchmarks, spb
from latos.optimization.campaign import (
    CampaignDrift,
    DriftStep,
    recommendation_drift,
)
from latos.optimization.engine import (
    BoConfig,
    BoConfigND,
    OptimizationError,
    OptimizationResult,
    OptimizationResultND,
    Recommendation,
    RecommendationND,
    ReliabilityReport,
    RobustnessEntry,
    RobustnessReport,
    length_scale_robustness,
    optimize,
    optimize_nd,
)
from latos.optimization.prereg import build_record, freeze, write_record
from latos.optimization.validate import (
    OutcomeVerdict,
    PreregEntry,
    list_preregistrations,
    outcome_path_for,
    validate_outcome,
    write_outcome,
)

__all__ = [
    "BoConfig",
    "benchmarks",
    "optimize_nd",
    "RecommendationND",
    "OptimizationResultND",
    "BoConfigND",
    "CampaignDrift",
    "DriftStep",
    "OptimizationError",
    "OptimizationResult",
    "OutcomeVerdict",
    "PreregEntry",
    "Recommendation",
    "ReliabilityReport",
    "RobustnessEntry",
    "RobustnessReport",
    "build_record",
    "freeze",
    "length_scale_robustness",
    "list_preregistrations",
    "optimize",
    "outcome_path_for",
    "recommendation_drift",
    "spb",
    "validate_outcome",
    "write_outcome",
    "write_record",
]
