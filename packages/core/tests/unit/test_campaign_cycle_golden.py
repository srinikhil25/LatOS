"""Golden tests for the campaign cycle: what `run_cycle` produces today, pinned.

These are Step 0 of the experiment-definition refactor. The refactor moves the
ionic-liquid cycle onto a generic experiment definition, and its promise is
that nothing a researcher sees changes. These tests are how that
promise gets checked rather than asserted. Seven synthetic campaigns are run
through the cycle as it stands, and everything it produces is compared with a
JSON file written before the refactor began.

What each file pins:

- every field of every `SampleFit` and every `DesignPoint`;
- the recommendation, `max_ei`, `noise_threshold`, `converged`, the stopping
  verdict, the reliability report, every `BoConfig` field except `created_at`,
  and the digest of the training data;
- the messages, and the report text with the record's path and any timestamp
  masked;
- the frozen pre-registration record, except `created_at`.

How it is compared. Floats agree to `rel=1e-9, abs=1e-12`, because CI runs on
three operating systems and the GP's L-BFGS fit and BLAS summation can differ in
the last bits. Everything else must match exactly, type included. That covers
the digest, whose inputs come from the slope fit, and that fit uses only
`np.mean` and `np.sum`.

The inputs are generated in-test by `_campaign`, the workbook writer the
cycle's own tests use. No workbook and no real data are committed.

Each fixture is named for the branch it exists to reach, and
`TestFixturesReachTheirBranch` checks that it still does. Without that check, a
regenerated file that had quietly stopped exercising anything would pass
forever.

Regenerating is deliberate. It is done only in a commit that names the
behaviour it changes:

    LATOS_REGENERATE_GOLDEN=1 python -m pytest tests/unit/test_campaign_cycle_golden.py

Without that variable a mismatch fails, and nothing is rewritten.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latos.campaign_cycle import run_cycle
from latos.optimization import OptimizationResult, observations_digest, prereg_dir
from tests.unit.test_campaign_cycle import DELTAS, _campaign

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden" / "campaign_cycle"
REGENERATE = os.environ.get("LATOS_REGENERATE_GOLDEN") == "1"

# Tolerance for floats. Anything that is not a float is compared exactly.
REL = 1e-9
ABS = 1e-12

# A time in ISO 8601, or the compact stamp in a record's filename.
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?|\d{8}T\d{6}Z"
)


def _inputs(samples, *, noise, deltas=DELTAS, offset=0.35, polarity="V+ on cold electrode"):
    """One fixture's arguments to `_campaign`, every one of them spelled out.

    They are passed explicitly rather than left to the helper's defaults, so an
    edit to those defaults cannot move the pins without the inputs check below
    saying so.
    """
    return {
        "samples": samples,
        "noise": noise,
        "deltas": deltas,
        "offset": offset,
        "polarity": polarity,
    }


# The campaign of the existing full-cycle test: four compositions, peak at x = 0.5.
_FOUR = {"IL-001": (0.0, 1.1), "IL-002": (1.0, 1.6), "IL-003": (0.5, 2.35), "IL-004": (0.25, 1.8)}

FIXTURES: dict[str, dict[str, Any]] = {
    # IL-005 repeats x = 0.5. Exercises the sigma labels `replicates` and
    # `pooled`, the pooled note, the collapse note, and the design-point table.
    "same_sign_replicated": _inputs({**_FOUR, "IL-005": (0.5, 2.25)}, noise=0.04),
    # One sample per condition. Exercises the label `fit` (a lower bound) and
    # the median fit SE as the noise level.
    "no_replicates": _inputs(_FOUR, noise=0.04),
    # S changes sign across the range. Exercises the zero-crossing message.
    "opposite_sign_endpoints": _inputs(
        {"IL-001": (0.0, -1.8), "IL-002": (1.0, 2.1), "IL-003": (0.5, 0.4)}, noise=0.02
    ),
    # Two replicates at x = 0.5 that disagree in sign. Exercises the mean |S|
    # and the replicate-sign note. This pins the behaviour from before the
    # change to a signed mean (1C-b).
    "mixed_sign_replicates": _inputs(
        {"IL-001": (0.0, 1.1), "IL-002": (1.0, 1.6), "IL-003": (0.5, 0.3), "IL-004": (0.5, -0.4)},
        noise=0.02,
    ),
    # Two temperature steps leave each slope fit no degrees of freedom, so no
    # sigma is known. Exercises the spread fallback.
    "two_point_series": _inputs(_FOUR, noise=0.04, deltas=(2.0, 5.0)),
    # Every series lies exactly on its line. Exercises the degenerate-sigma
    # guard and its message.
    "noise_free": _inputs(_FOUR, noise=0.0),
    # Three samples on only two compositions. Exercises the early return, with
    # no record written.
    "too_few_conditions": _inputs(
        {"IL-001": (0.0, 1.1), "IL-002": (1.0, 1.6), "IL-003": (1.0, 1.55)}, noise=0.04
    ),
}


def _scalar(value: Any) -> Any:
    """Let `json` store a numpy scalar as the Python number it holds."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} has no JSON form")


def _dump(value: Any) -> str:
    """The golden-file format: indented, keys sorted so a diff shows only real change."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_scalar) + "\n"


def _as_stored(value: Any) -> Any:
    """`value` exactly as a golden file holds it once read back."""
    return json.loads(_dump(value))


def _fields(obj: Any, *, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """Every dataclass field of `obj`, so a field added later is pinned too."""
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.name not in skip}


def _pick(obj: Any, *names: str) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in names}


def _mask(text: str, record: Path | None) -> str:
    """Remove what differs from run to run: the record's path, and the time."""
    if record is not None:
        text = text.replace(str(record), "<RECORD>")
    return _TIMESTAMP.sub("<TIMESTAMP>", text)


def _result(result: OptimizationResult) -> dict[str, Any]:
    cfg = result.config
    return {
        "recommendation": _pick(
            result.recommendation,
            "x",
            "predicted_mean",
            "ci95",
            "predictive_sd",
            "ci95_predictive",
            "predictive_interval_95",
        ),
        "max_ei": result.max_ei,
        "noise_threshold": result.noise_threshold,
        "converged": result.converged,
        "stopping": _pick(
            result.stopping,
            "action",
            "probability",
            "epsilon",
            "delta",
            "signal_exhausted",
            "data_sufficient",
        ),
        "reliability": _pick(
            result.reliability,
            "level",
            "n_observations",
            "loo_inside",
            "loo_total",
            "loo_coverage",
            "fill_distance",
            "fill_limit",
        ),
        "config": _fields(cfg, skip=("created_at",)),
        # Computed exactly as the record computes it.
        "training_digest": observations_digest(
            result.observed_x, result.observed_y, sigma=cfg.point_noise_scale
        ),
    }


def _run(tmp_path: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """Run one cycle and reduce it to the pinned outputs, as a golden file holds them."""
    workbook = _campaign(tmp_path, **inputs)
    outcome = run_cycle(workbook)

    record = None if outcome.prereg_path is None else Path(outcome.prereg_path)
    # "No record written" is read from the disk, not taken from the outcome:
    # the only record on disk must be the one the outcome names.
    written = sorted(prereg_dir(workbook.parent).glob("*.json"))
    assert written == ([record] if record is not None else []), written
    frozen = None
    if record is not None:
        frozen = json.loads(record.read_text(encoding="utf-8"))
        del frozen["created_at"]

    return _as_stored(
        {
            "fits": [_fields(fit) for fit in outcome.fits],
            "points": [_fields(point) for point in outcome.points],
            "result": None if outcome.result is None else _result(outcome.result),
            "messages": list(outcome.messages),
            # One entry per line, so a regenerated report diffs line by line.
            "report": _mask(outcome.report(), record).split("\n"),
            "record": frozen,
        }
    )


def _same(pinned: Any, now: Any) -> bool:
    """Floats within tolerance; everything else, type included, exactly."""
    if type(pinned) is not type(now):
        return False
    if isinstance(pinned, float):
        both_nan = math.isnan(pinned) and math.isnan(now)
        return both_nan or now == pytest.approx(pinned, rel=REL, abs=ABS)
    return pinned == now


def _differences(pinned: Any, now: Any, where: str = "outputs") -> list[str]:
    """Every place `now` departs from `pinned`, one readable line each."""
    if isinstance(pinned, dict) and isinstance(now, dict):
        found = [f"{where}.{key}: no longer produced" for key in sorted(pinned.keys() - now.keys())]
        found += [
            f"{where}.{key}: produced, not pinned" for key in sorted(now.keys() - pinned.keys())
        ]
        for key in sorted(pinned.keys() & now.keys()):
            found += _differences(pinned[key], now[key], f"{where}.{key}")
        return found
    if isinstance(pinned, list) and isinstance(now, list):
        found = []
        for i, (p, n) in enumerate(zip(pinned, now, strict=False)):
            found += _differences(p, n, f"{where}[{i}]")
        if len(pinned) != len(now):
            found.append(f"{where}: {len(pinned)} entries pinned, {len(now)} now")
        return found
    return [] if _same(pinned, now) else [f"{where}: pinned {pinned!r}, now {now!r}"]


def _golden(name: str) -> dict[str, Any]:
    path = GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(f"{path} is missing. Generate it with LATOS_REGENERATE_GOLDEN=1.")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", list(FIXTURES))
def test_cycle_matches_golden(name, tmp_path):
    inputs = _as_stored(FIXTURES[name])
    outputs = _run(tmp_path, FIXTURES[name])

    if REGENERATE:
        path = GOLDEN_DIR / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _dump({"fixture": name, "inputs": inputs, "outputs": outputs}),
            encoding="utf-8",
            newline="\n",
        )
        pytest.skip(f"regenerated {path.name}; review its diff before committing it")

    golden = _golden(name)
    assert golden["inputs"] == inputs, (
        f"{name}: the fixture's inputs no longer match its golden file. Changing a "
        "fixture means regenerating it, with LATOS_REGENERATE_GOLDEN=1."
    )
    differences = _differences(golden["outputs"], outputs)
    assert not differences, (
        f"{name}: the cycle departs from its golden file in {len(differences)} place(s):\n  "
        + "\n  ".join(differences)
        + "\nIf the change is intended, regenerate with LATOS_REGENERATE_GOLDEN=1, "
        "in a commit that names the behaviour it changes."
    )


def test_golden_directory_holds_one_file_per_fixture():
    # A renamed fixture must not leave its old file behind, pinned to nothing.
    assert sorted(path.stem for path in GOLDEN_DIR.glob("*.json")) == sorted(FIXTURES)


class TestFixturesReachTheirBranch:
    """Each fixture still exercises what it is named for.

    These read the golden files rather than re-running the cycle. The test above
    holds the cycle to the files; these hold the files to their purpose.
    """

    def test_same_sign_replicated(self):
        out = _golden("same_sign_replicated")["outputs"]
        assert {p["sigma_source"] for p in out["points"]} == {"replicates", "pooled"}
        assert any(m.startswith("Replicate scatter measured") for m in out["messages"])
        assert any(" collapsed onto " in m for m in out["messages"])
        assert "Design points given to the surrogate:" in out["report"]

    def test_no_replicates(self):
        out = _golden("no_replicates")["outputs"]
        assert {p["sigma_source"] for p in out["points"]} == {"fit"}
        median_fit_se = statistics.median(p["sigma"] for p in out["points"])
        noise_std = out["result"]["config"]["noise_std"]
        assert noise_std == pytest.approx(median_fit_se, rel=REL, abs=ABS)

    def test_opposite_sign_endpoints(self):
        out = _golden("opposite_sign_endpoints")["outputs"]
        assert any("crosses zero" in m for m in out["messages"])

    def test_mixed_sign_replicates(self):
        out = _golden("mixed_sign_replicates")["outputs"]
        (point,) = [p for p in out["points"] if p["n_replicates"] > 1]
        signed = [f["seebeck_mv_k"] for f in out["fits"] if f["sample_id"] in point["sample_ids"]]
        assert min(signed) < 0 < max(signed)
        # Today the magnitudes are averaged. The change to a signed mean, then
        # its magnitude (1C-b), changes this assertion with it.
        mean_magnitude = statistics.fmean(abs(s) for s in signed)
        assert point["value"] == pytest.approx(mean_magnitude, rel=REL, abs=ABS)
        assert any("disagree in the SIGN" in note for note in point["notes"])

    def test_two_point_series(self):
        out = _golden("two_point_series")["outputs"]
        assert {p["sigma_source"] for p in out["points"]} == {"undetermined"}
        spread = statistics.pstdev(p["value"] for p in out["points"])
        noise_std = out["result"]["config"]["noise_std"]
        assert noise_std == pytest.approx(spread, rel=REL, abs=ABS)

    def test_noise_free(self):
        out = _golden("noise_free")["outputs"]
        assert any("vanishingly small" in m for m in out["messages"])
        assert out["result"]["config"]["point_noise_used"] is False

    def test_too_few_conditions(self):
        out = _golden("too_few_conditions")["outputs"]
        assert out["result"] is None
        assert out["record"] is None
        assert any("at least 3 DISTINCT compositions" in m for m in out["messages"])
