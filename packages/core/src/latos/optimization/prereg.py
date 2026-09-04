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

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from latos import __version__

if TYPE_CHECKING:
    from latos.optimization.engine import OptimizationResult, RobustnessReport

__all__ = ["build_record", "freeze", "observations_digest", "prereg_dir", "write_record"]

# Where frozen records live, declared once because it was previously declared
# four times across three modules and one copy disagreed: `campaign_cycle`
# froze into `<workbook>/preregistrations/` while the validation module and the
# server both read `<root>/.latos/prereg/`. Nothing failed loudly. The bench
# command printed "Pre-registered: ..." and wrote a real record; the screen
# that scores it simply listed nothing, and the server's path-confinement
# check would have refused the record even if handed it directly. So the
# closed loop was open at its last joint, silently. A freeze no reader can
# find is not a commitment, which is the one thing a pre-registration is for.
_PREREG_PARENT = ".latos"
_PREREG_SUBDIR = "prereg"


def prereg_dir(root: Path) -> Path:
    """The directory holding frozen pre-registrations for the project at `root`.

    Every writer and every reader goes through this. See the comment above for
    what happened when they did not.
    """
    return Path(root) / _PREREG_PARENT / _PREREG_SUBDIR


# How the observations are canonicalised before hashing, recorded inside every
# record so an auditor can recompute the digest without reading this source.
_CANONICAL_FORM = (
    "rows of x, y[, sigma] as Python float repr, tab-separated, sorted ascending, "
    "newline-joined, UTF-8, SHA-256"
)


def observations_digest(
    x: Sequence[float],
    y: Sequence[float],
    *,
    sigma: Sequence[float] | None = None,
) -> str:
    """SHA-256 of the observations a model was fit to.

    The record already pins the configuration. What it could not pin was the
    *data*: `n_observations` is a count, so two records fit to entirely
    different measurements were distinguishable only by their timestamps. A
    frozen prediction whose training set cannot be identified is not evidence
    of anything, which matters because pre-registration is the claim this whole
    module exists to support.

    `sigma` is included when per-observation standard deviations were supplied.
    Two runs with identical (x, y) and different weights are different fits and
    produce different recommendations — `BoConfig.point_noise_used` says that
    happened, and this says with what.

    Rows are **sorted**, so the same dataset listed in a different order gives
    the same digest. That is deliberate: a reordering is the same set of
    measurements, and a check that cried mismatch over row order would be a
    check nobody trusts. `repr` of a float round-trips exactly in Python, so
    the canonical text loses no precision.
    """
    if len(x) != len(y):
        raise ValueError(f"x has {len(x)} points and y has {len(y)}")
    if sigma is not None and len(sigma) != len(x):
        raise ValueError(f"sigma has {len(sigma)} points and x has {len(x)}")

    rows: list[tuple[float, ...]]
    if sigma is None:
        rows = [(float(a), float(b)) for a, b in zip(x, y, strict=True)]
    else:
        rows = [(float(a), float(b), float(c)) for a, b, c in zip(x, y, sigma, strict=True)]
    text = "\n".join("\t".join(repr(v) for v in row) for row in sorted(rows))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_record(
    result: OptimizationResult,
    *,
    prior_best: float,
    robustness: RobustnessReport | None = None,
) -> dict[str, Any]:
    """Assemble the pre-registration record as a JSON-serializable dict."""
    cfg = result.config
    rec = result.recommendation
    # Exact, physically-bounded interval from the engine (asymmetric for a
    # log-space fit) — not a symmetric ± half-width.
    lo95, hi95 = rec.predictive_interval_95
    record: dict[str, Any] = {
        "kind": "latos.bo.prereg",
        "created_at": cfg.created_at.isoformat(),
        # Which build produced this. The engine's defaults have already moved
        # once (`xi` changed from absolute to fractional on 2026-08-10), so a
        # record that does not say which version wrote it cannot be replayed.
        "latos_version": __version__,
        "objective": {
            "property": cfg.objective,
            "direction": cfg.direction,
            "y_transform": cfg.y_transform,
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
            # Whether each observation carried its own variance. Two runs can
            # otherwise carry identical frozen configs and have weighed the same
            # points differently, which would make this record unable to explain
            # the recommendation it is freezing.
            "point_noise_used": cfg.point_noise_used,
            # The weights themselves, so the training-data digest below can
            # actually be recomputed. Claiming a hash is falsifiable while
            # withholding an input to it is not a claim anyone can check.
            "point_noise_scale": (
                list(cfg.point_noise_scale) if cfg.point_noise_scale is not None else None
            ),
            "grid_size": cfg.grid_size,
            "seed": cfg.seed,
            "n_observations": cfg.n_observations,
        },
        # What the model actually saw. See `observations_digest`.
        "training_data": {
            "sha256": observations_digest(
                result.observed_x, result.observed_y, sigma=cfg.point_noise_scale
            ),
            # The observations themselves, at full precision. A digest whose
            # inputs are absent is only checkable by someone who can re-run the
            # fit; carrying them makes the record auditable on its own, and the
            # values are exact floats because a hash over rounded ones will not
            # reproduce.
            "x": list(result.observed_x),
            "y": list(result.observed_y),
            "n_observations": len(result.observed_x),
            "point_noise_used": cfg.point_noise_used,
            "digest_covers_point_noise": cfg.point_noise_scale is not None,
            "canonical_form": _CANONICAL_FORM,
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
        # The stopping claim, committed with everything else. Without this the
        # tool could assert "94% sure you are already done", the researcher
        # could stop on it, and nobody could later check what was claimed.
        # `probability` is conditional on the frozen model, which is exactly
        # why `reliability` below is recorded beside it.
        "stopping_claim": {
            "epsilon": result.epsilon,
            "delta": result.delta,
            "probability_within_epsilon": result.prob_within_epsilon,
            "met": result.epsilon_delta_met,
            "best_measured": result.best_y,
            "n_unreliable_observations": result.n_unreliable,
        },
        "validation_criteria": {
            "calibration": "measured value falls within predictive_interval_95",
            "improvement": "measured value exceeds prior_best",
            "stopping_claim": (
                "the best value found over the campaign improves on the frozen "
                "best_measured by no more than epsilon"
            ),
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
    # Records written before 2026-09-05 carry no training-data block.
    data = record.get("training_data", {})
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
        f"- Produced by: latos {record.get('latos_version', 'unknown')}",
        "",
        "## Training data (frozen)",
        f"- SHA-256: `{data.get('sha256', 'unknown')}`",
        f"- {data.get('n_observations', cfg['n_observations'])} observations"
        + (
            ", digest covers the per-point weights"
            if data.get("digest_covers_point_noise")
            else ", equal weighting"
        ),
        "- The observations and any weights are stored beside the digest in the "
        "JSON, at full precision, so it can be recomputed from this record alone.",
        "- **Falsifiable by:** re-hashing them. A different digest means a "
        "different training set, whatever the timestamps say.",
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
    claim = record.get("stopping_claim")
    if claim:
        pct = claim["probability_within_epsilon"] * 100.0
        bar = (1.0 - claim["delta"]) * 100.0
        lines += [
            "",
            "## Stopping claim (committed in advance)",
            f"- Best measured at freeze time: **{claim['best_measured']:.4g}**",
            f"- Under the frozen model, that is within **{claim['epsilon']:.4g}** of the "
            f"optimum with probability **{pct:.0f}%** "
            f"({'meets' if claim['met'] else 'below'} the {bar:.0f}% bar).",
            "- **Falsifiable by:** finding a value that beats the frozen best by more "
            f"than {claim['epsilon']:.4g}.",
        ]
        if claim.get("n_unreliable_observations"):
            lines.append(
                f"- {claim['n_unreliable_observations']} observation(s) were "
                "down-weighted because a physics check rejected them."
            )
        lines.append(
            "- This probability is conditional on the frozen model; read it with "
            "the reliability level below."
        )
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
