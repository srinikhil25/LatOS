"""Freeze a recommendation into an auditable pre-registration record.

The scientific point: a Bayesian-optimization recommendation is only
*prospective* if the model configuration and the predicted value (with a
predictive confidence interval) are committed **before** the recommended
sample is made. This module turns an `OptimizationResult` (plus, optionally,
a `RobustnessReport`) into a timestamped JSON + Markdown record you can commit
to version control — the tool-enforced version of "screenshot the config and
write down the prediction first".

The later validation reads directly off this record:
* **calibration** — did the measured value land inside `predictive_interval_95`?
* **improvement** — did it beat `prior_best`?
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from latos.optimization.engine import OptimizationResult, RobustnessReport

__all__ = ["build_record", "freeze", "write_record"]


def build_record(
    result: OptimizationResult,
    *,
    prior_best: float,
    robustness: RobustnessReport | None = None,
) -> dict[str, Any]:
    """Assemble the pre-registration record as a JSON-serializable dict."""
    cfg = result.config
    rec = result.recommendation
    lo95 = rec.predicted_mean - rec.ci95_predictive
    hi95 = rec.predicted_mean + rec.ci95_predictive
    record: dict[str, Any] = {
        "kind": "latos.bo.prereg",
        "created_at": cfg.created_at.isoformat(),
        "objective": {
            "property": cfg.objective,
            "direction": cfg.direction,
            "aggregation": cfg.objective_aggregation,
            "input_variable": cfg.input_name,
            "search_bounds": list(cfg.bounds),
        },
        "frozen_config": {
            "kernel": cfg.kernel,
            "length_scale": cfg.length_scale,
            "length_scale_fitted": cfg.length_scale_fitted,
            "length_scale_bounds": list(cfg.length_scale_bounds),
            "xi": cfg.xi,
            "rel_noise": cfg.rel_noise,
            "noise_std": cfg.noise_std,
            "grid_size": cfg.grid_size,
            "seed": cfg.seed,
            "n_observations": cfg.n_observations,
        },
        "prediction_at_recommendation": {
            "x": rec.x,
            "predicted_mean": rec.predicted_mean,
            "ci95_model": rec.ci95,
            "predictive_sd": rec.predictive_sd,
            "ci95_predictive": rec.ci95_predictive,
            "predictive_interval_95": [lo95, hi95],
        },
        "prior_best": prior_best,
        "converged": result.converged,
        "validation_criteria": {
            "calibration": "measured value falls within predictive_interval_95",
            "improvement": "measured value exceeds prior_best",
        },
    }
    if robustness is not None:
        record["robustness"] = {
            "recommended_x_spread": robustness.recommended_x_spread,
            "search_span": robustness.search_span,
            "tolerance": robustness.tolerance,
            "stable": robustness.stable,
            "entries": [asdict(e) for e in robustness.entries],
        }
    if result.reliability is not None:
        # The reliability the tool assigned itself at freeze time — an
        # auditor can judge how much the frozen claim was worth without
        # rerunning anything.
        record["reliability"] = asdict(result.reliability)
    return record


def _to_markdown(record: dict[str, Any]) -> str:
    """Render the record as a human-readable pre-registration note."""
    obj = record["objective"]
    cfg = record["frozen_config"]
    pred = record["prediction_at_recommendation"]
    lo, hi = pred["predictive_interval_95"]
    lines = [
        "# Bayesian-optimization pre-registration",
        "",
        f"_Committed {record['created_at']} — before the recommended sample is made._",
        "",
        "## Objective (frozen)",
        f"- {obj.get('direction', 'maximize').capitalize()} **{obj['property']}** "
        f"(aggregation: {obj['aggregation']})",
        f"- Input variable: `{obj['input_variable']}`",
        f"- Search bounds: {obj['search_bounds']}",
        "",
        "## Frozen model configuration",
        f"- Kernel: {cfg['kernel']}",
        f"- Length-scale: {cfg['length_scale']:.4g} "
        f"({'fitted' if cfg['length_scale_fitted'] else 'fixed'}, "
        f"bounds {cfg['length_scale_bounds']})",
        f"- Exploration xi: {cfg['xi']}",
        f"- Noise: rel {cfg['rel_noise']}, std {cfg['noise_std']:.4g}",
        f"- Grid size: {cfg['grid_size']} · seed: {cfg['seed']} · "
        f"observations: {cfg['n_observations']}",
        "",
        "## Prediction at the recommended experiment (committed in advance)",
        f"- Recommended `{obj['input_variable']}` = **{pred['x']:.4g}**",
        f"- Predicted {obj['property']} = **{pred['predicted_mean']:.4g}**",
        f"- 95% predictive interval: **[{lo:.4g}, {hi:.4g}]**",
        f"- Prior best: {record['prior_best']:.4g}",
        "",
        "## Validation criteria (decided in advance)",
        "- **Calibration:** the measured value falls within the 95% predictive interval.",
        "- **Improvement:** the measured value exceeds the prior best.",
    ]
    if "robustness" in record:
        rob = record["robustness"]
        verdict = (
            "the pick does not depend on the kernel length-scale."
            if rob["stable"]
            else "the pick depends on the length-scale; data may be too sparse "
            "to recommend confidently."
        )
        lines += [
            "",
            "## Kernel robustness",
            f"- Recommended point spread across length-scales: "
            f"{rob['recommended_x_spread']:.4g} "
            f"(tolerance {rob['tolerance']:.4g} of span {rob['search_span']:.4g})",
            f"- **{'STABLE' if rob['stable'] else 'UNSTABLE'}** — {verdict}",
        ]
    if "reliability" in record:
        rel = record["reliability"]
        lines += [
            "",
            "## Reliability (self-assessed at freeze time)",
            f"- Level: **{rel['level'].upper()}** "
            f"({rel['n_observations']} observations; leave-one-out "
            f"{rel['loo_inside']}/{rel['loo_total']} inside the 95% band)",
            f"- {rel['note']}",
        ]
    return "\n".join(lines) + "\n"


def write_record(record: dict[str, Any], path: Path) -> Path:
    """Write the record to `path` (JSON) and a sibling `.md`; return the JSON path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(_to_markdown(record), encoding="utf-8")
    return path


def freeze(
    result: OptimizationResult,
    path: Path,
    *,
    prior_best: float,
    robustness: RobustnessReport | None = None,
) -> Path:
    """Build the record and write it in one call. Returns the JSON path."""
    return write_record(build_record(result, prior_best=prior_best, robustness=robustness), path)
