"""Close the loop: judge a synthesized outcome against the frozen prediction.

Latos's recommendation is only *prospective* because the prediction and
model configuration were frozen to a pre-registration record before the
recommended sample was made (see `prereg.py`). This module is the return
path: once that sample has been synthesized and measured, it scores the
measured value against the frozen record on the two criteria the record
itself declared in advance:

* **Calibration** — did the measured value fall inside the 95% predictive
  interval? If yes, the model's stated uncertainty was honest here; if no,
  the model was over-confident (or the recommendation missed).
* **Improvement** — did the measured value beat the prior best, in the
  direction being optimized (higher for *maximize*, lower for *minimize*)?

The verdict is written to a sibling ``*.outcome.json`` next to the frozen
record, so the closed loop is itself auditable: prediction and outcome
live side by side and neither can be edited to fit the other after the
fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from latos.core.models import utc_now

__all__ = [
    "OutcomeVerdict",
    "PreregEntry",
    "list_preregistrations",
    "outcome_path_for",
    "validate_outcome",
    "write_outcome",
]

# Below this magnitude a relative error is meaningless (division blows up).
_REL_ERROR_MIN_DENOM = 1e-12


@dataclass(frozen=True, slots=True)
class OutcomeVerdict:
    """How a measured outcome scored against a frozen prediction."""

    measured: float
    predicted_mean: float
    predictive_interval_95: tuple[float, float]
    prior_best: float
    direction: str  # "maximize" | "minimize"
    within_interval: bool  # calibration criterion
    improved: bool  # improvement criterion (direction-aware)
    signed_error: float  # measured - predicted_mean
    absolute_error: float  # |measured - predicted_mean|
    relative_error: float | None  # abs_error / |measured|; None if measured ≈ 0
    summary: str  # plain-language, materials-scientist-facing
    validated_at: str  # ISO timestamp


def _summarize(
    *,
    property_name: str,
    measured: float,
    lo: float,
    hi: float,
    prior_best: float,
    direction: str,
    within: bool,
    improved: bool,
) -> str:
    """One or two plain sentences: calibration then improvement."""
    calib = (
        f"Measured {property_name} {measured:.4g} falls within the predicted 95% "
        f"interval [{lo:.4g}, {hi:.4g}] — the model's uncertainty was honest here."
        if within
        else f"Measured {property_name} {measured:.4g} falls OUTSIDE the predicted 95% "
        f"interval [{lo:.4g}, {hi:.4g}] — the model was over-confident (or the "
        f"recommendation missed)."
    )
    better = "higher" if direction == "maximize" else "lower"
    if improved:
        impr = (
            f"It improved on the prior best ({prior_best:.4g} → {measured:.4g}, "
            f"{better} is better)."
        )
    else:
        impr = (
            f"It did not beat the prior best ({prior_best:.4g}; {better} is better) — "
            f"the optimum likely still stands."
        )
    return f"{calib} {impr}"


def validate_outcome(record: dict[str, Any], measured: float) -> OutcomeVerdict:
    """Score `measured` against a frozen pre-registration `record`.

    `record` is a parsed prereg JSON (see `prereg.build_record`). The
    measured value must be in the same units as the frozen prediction
    (the property the freeze optimized).
    """
    pred = record["prediction_at_recommendation"]
    predicted_mean = float(pred["predicted_mean"])
    lo, hi = (float(v) for v in pred["predictive_interval_95"])
    prior_best = float(record["prior_best"])
    obj = record.get("objective", {})
    direction = str(obj.get("direction", "maximize"))
    property_name = str(obj.get("property", "value"))

    within = lo <= measured <= hi
    improved = measured > prior_best if direction == "maximize" else measured < prior_best
    signed = measured - predicted_mean
    abs_err = abs(signed)
    rel_err = abs_err / abs(measured) if abs(measured) > _REL_ERROR_MIN_DENOM else None

    return OutcomeVerdict(
        measured=measured,
        predicted_mean=predicted_mean,
        predictive_interval_95=(lo, hi),
        prior_best=prior_best,
        direction=direction,
        within_interval=within,
        improved=improved,
        signed_error=signed,
        absolute_error=abs_err,
        relative_error=rel_err,
        summary=_summarize(
            property_name=property_name,
            measured=measured,
            lo=lo,
            hi=hi,
            prior_best=prior_best,
            direction=direction,
            within=within,
            improved=improved,
        ),
        validated_at=utc_now().isoformat(),
    )


def outcome_path_for(prereg_path: Path) -> Path:
    """The sibling outcome file for a frozen record (``*.outcome.json``)."""
    return prereg_path.with_suffix(".outcome.json")


def write_outcome(prereg_path: Path, verdict: OutcomeVerdict) -> Path:
    """Persist a verdict next to its frozen record; return the outcome path."""
    out = outcome_path_for(prereg_path)
    payload = {"kind": "latos.bo.outcome", "prereg": prereg_path.name, **asdict(verdict)}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


@dataclass(frozen=True, slots=True)
class PreregEntry:
    """A frozen pre-registration, with its recorded outcome if any.

    Everything the UI needs to list a prediction and, once the sample is
    made, either show its validation verdict or offer to enter one.
    """

    path: str
    created_at: str
    input_variable: str
    property_name: str
    direction: str
    recommended_x: float
    predicted_mean: float
    predictive_interval_95: tuple[float, float]
    prior_best: float
    reliability_level: str
    outcome: dict[str, Any] | None  # the recorded verdict, or None


def _prereg_dir(root: Path) -> Path:
    return root / ".latos" / "prereg"


def list_preregistrations(root: Path) -> list[PreregEntry]:
    """Every frozen pre-registration under `root`, newest first.

    Skips the ``*.outcome.json`` siblings themselves and attaches each
    recorded outcome (if present) to its parent entry. Malformed records
    are skipped, not fatal.
    """
    directory = _prereg_dir(root)
    if not directory.is_dir():
        return []
    entries: list[PreregEntry] = []
    for path in directory.glob("prereg_*.json"):
        if ".outcome." in path.name:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            pred = record["prediction_at_recommendation"]
            obj = record.get("objective", {})
            lo, hi = (float(v) for v in pred["predictive_interval_95"])
            outcome_file = outcome_path_for(path)
            outcome = (
                json.loads(outcome_file.read_text(encoding="utf-8"))
                if outcome_file.is_file()
                else None
            )
            entries.append(
                PreregEntry(
                    path=str(path),
                    created_at=str(record.get("created_at", "")),
                    input_variable=str(obj.get("input_variable", "")),
                    property_name=str(obj.get("property", "value")),
                    direction=str(obj.get("direction", "maximize")),
                    recommended_x=float(pred["x"]),
                    predicted_mean=float(pred["predicted_mean"]),
                    predictive_interval_95=(lo, hi),
                    prior_best=float(record.get("prior_best", float("nan"))),
                    reliability_level=str(record.get("reliability", {}).get("level", "unknown")),
                    outcome=outcome,
                )
            )
        except (KeyError, ValueError, OSError):
            continue
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries
