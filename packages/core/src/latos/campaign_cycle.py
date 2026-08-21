"""One command per experimental cycle: read the workbook, decide, pre-register.

Closing the loop is four steps — parse the bench record, fit each sample, ask
the optimizer, freeze the prediction before anything is made — and every one of
them already exists. What did not exist was a way to run them in the right order
without writing a script first.

That matters more than it sounds. A hand-written script during a busy lab week
gets skipped, and a loop that is closed inconsistently is not closed at all: the
pre-registration is the whole evidentiary basis for the closed-loop claim, and
it only counts if it was written *before* the sample. Making that the path of
least resistance is the point of this module.

The cycle
---------
1. Read the recording workbook. One entry per sample, each carrying its
   (ΔT, ΔV) series and the composition actually weighed.
2. Fit `ΔV = S·ΔT + b` per sample. The slope is the Seebeck coefficient and its
   standard error is how well that particular sample is known.
3. Fit the surrogate over composition, with those standard errors as per-point
   variance, so precise samples pull the surface and vague ones do not.
4. Print the stopping verdict, and freeze the recommendation with its predicted
   value and interval.

Objective
---------
The magnitude of the Seebeck coefficient. Sign carries the carrier type, and a
campaign that maximised the signed value would treat a strongly n-type mixture
as the worst result rather than a differently useful one. When the samples so
far disagree in sign the report says so, because that changes the target: the
magnitude then has an interior *minimum*, the endpoints win, and the interesting
composition is the crossing rather than a peak.

What this does not do
---------------------
It does not touch the project database. The workbook is the record for this
campaign, and adding a persistence round-trip would buy nothing the file does
not already give while making the command harder to run from a bench laptop.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from latos.analysis.thermovoltage.slope import fit_seebeck_slope
from latos.core.enums import Severity
from latos.ingestion.parsed_data import ParsedData
from latos.ingestion.parsers.ite_workbook import IteWorkbookParser
from latos.optimization.engine import OptimizationResult, optimize
from latos.optimization.prereg import freeze

__all__ = ["CycleOutcome", "SampleFit", "main", "run_cycle"]

# Composition is a mass fraction, so the search range is the whole simplex.
_BOUNDS = (0.0, 1.0)
_INPUT_NAME = "mass_fraction_x"
_TARGET_NAME = "abs_seebeck_mv_k"

# Two points fit a line exactly and leave nothing to judge it by, so a sample
# with fewer contributes its composition to the campaign and no value.
_MIN_POINTS_PER_SAMPLE = 2

# The surrogate needs somewhere to start. Below this the honest answer is the
# seed design the plan already specifies, not a recommendation dressed up as one.
_MIN_SAMPLES_FOR_A_MODEL = 3

# Below this share of the signal a claimed uncertainty is not precision, it is
# the absence of an estimate. No real thermovoltage measurement is known to a
# part in a million.
_DEGENERATE_SIGMA_FRACTION = 1e-6


@dataclass(frozen=True, slots=True)
class SampleFit:
    """One sample reduced to the two numbers the optimizer consumes."""

    sample_id: str
    composition: float  # mass fraction of ionic liquid A
    seebeck_mv_k: float  # the fitted slope, signed
    stderr_mv_k: float | None  # its standard error, None when undetermined
    offset_mv: float  # the electrode-polarisation intercept
    n_points: int
    notes: tuple[str, ...]  # anything the fit or the workbook flagged


@dataclass(frozen=True, slots=True)
class CycleOutcome:
    """What the cycle concluded, and where it wrote the evidence."""

    fits: tuple[SampleFit, ...]
    result: OptimizationResult | None
    prereg_path: Path | None
    messages: tuple[str, ...]

    def report(self) -> str:
        """The whole cycle as text, for a terminal or a lab notebook."""
        lines: list[str] = ["Samples read from the workbook:"]
        if not self.fits:
            lines.append("  (none usable)")
        for fit in self.fits:
            err = "  +/- undetermined" if fit.stderr_mv_k is None else f" +/- {fit.stderr_mv_k:.3f}"
            lines.append(
                f"  {fit.sample_id:<12} x = {fit.composition:.3f}   "
                f"S = {fit.seebeck_mv_k:+.3f}{err} mV/K   "
                f"offset {fit.offset_mv:+.3f} mV   ({fit.n_points} points)"
            )
            lines.extend(f"      - {note}" for note in fit.notes)

        lines.extend(("", *self.messages))

        if self.result is not None:
            stopping = self.result.stopping
            rec = self.result.recommendation
            low, high = rec.predictive_interval_95
            lines.extend(
                (
                    "",
                    f"NEXT: mix at x = {rec.x:.4f}",
                    f"  predicted |S| = {rec.predicted_mean:.3f} mV/K "
                    f"(95% interval {low:.3f} to {high:.3f})",
                    "",
                    f"  {stopping.action.upper() if stopping else 'CONTINUE'}: "
                    f"{stopping.reason if stopping else ''}",
                )
            )
        if self.prereg_path is not None:
            lines.extend(
                (
                    "",
                    f"Pre-registered: {self.prereg_path}",
                    "  Written before the sample exists. Do not edit it afterwards.",
                )
            )
        return "\n".join(lines)


def run_cycle(
    workbook: Path, *, out_dir: Path | None = None, freeze_prereg: bool = True
) -> CycleOutcome:
    """Read the workbook, fit every sample, and recommend the next composition.

    Args:
        workbook: The filled recording workbook.
        out_dir: Where the pre-registration is written. Defaults to a
            `preregistrations/` directory beside the workbook, so the evidence
            lands next to the record it came from rather than wherever the
            command happened to be run.
        freeze_prereg: Write the pre-registration (default). False is for
            previewing a recommendation without committing to it. Note what
            that costs: a prediction that was not frozen before the sample
            cannot afterwards be claimed to have been.

    Returns:
        A `CycleOutcome`. `result` is None when there is too little data to fit
        a surrogate, which is a legitimate answer and not a failure.
    """
    parsed = IteWorkbookParser().parse_all(workbook)
    fatal = [
        issue
        for entry in parsed
        for issue in entry.issues
        if issue.severity is Severity.ERROR and issue.field == "file"
    ]
    if fatal:
        return CycleOutcome(
            (), None, None, tuple(f"Could not read the workbook: {i.message}" for i in fatal)
        )

    fits = tuple(fit for fit in (_fit_one(entry) for entry in parsed) if fit is not None)
    messages: list[str] = []

    skipped = len(parsed) - len(fits)
    if skipped:
        messages.append(
            f"{skipped} sample(s) contributed no value: fewer than "
            f"{_MIN_POINTS_PER_SAMPLE} usable (delta-T, delta-V) points, or no composition."
        )

    signs = {math.copysign(1.0, fit.seebeck_mv_k) for fit in fits if fit.seebeck_mv_k != 0.0}
    if len(signs) > 1:
        messages.append(
            "The samples so far disagree in sign, so the coefficient crosses zero "
            "somewhere in this range. The magnitude therefore has an interior MINIMUM "
            "and the best |S| sits at an endpoint. Optimising |S| will spend the budget "
            "learning that; the composition worth finding is the crossing, which gives a "
            "matched p-type and n-type pair from one liquid pair."
        )

    if len(fits) < _MIN_SAMPLES_FOR_A_MODEL:
        messages.append(
            f"{len(fits)} sample(s) fitted. A surrogate needs at least "
            f"{_MIN_SAMPLES_FOR_A_MODEL}; measure both pure liquids and the midpoint "
            "first, which the campaign needs anyway as mixing-law anchors and drift "
            "controls."
        )
        return CycleOutcome(fits, None, None, tuple(messages))

    result, noise_note = _optimize(fits)
    if noise_note:
        messages.append(noise_note)
    if not freeze_prereg:
        messages.append(
            "Nothing was pre-registered. This recommendation cannot later be presented "
            "as a prediction made before the sample."
        )
        return CycleOutcome(fits, result, None, tuple(messages))

    destination = out_dir if out_dir is not None else workbook.parent / "preregistrations"
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = freeze(result, destination / f"prereg_{stamp}.json", prior_best=result.best_y)

    return CycleOutcome(fits, result, path, tuple(messages))


def _fit_one(entry: ParsedData) -> SampleFit | None:
    """Reduce one parsed sample to a composition and a fitted coefficient."""
    arrays, metadata = entry.arrays, entry.metadata
    sample_id = str(metadata.get("sample_id", "(unlabelled)"))

    composition = metadata.get("mass_fraction_x")
    delta_t = arrays.get("delta_t_k")
    delta_v = arrays.get("delta_v_mv")
    if composition is None or delta_t is None or delta_v is None:
        return None
    if np.asarray(delta_t).size < _MIN_POINTS_PER_SAMPLE:
        return None

    fit = fit_seebeck_slope(np.asarray(delta_t, dtype=float), np.asarray(delta_v, dtype=float))
    if not math.isfinite(fit.slope):
        return None

    notes = tuple(issue.message for issue in entry.issues if issue.severity is not Severity.INFO)
    return SampleFit(
        sample_id=sample_id,
        composition=float(composition),
        seebeck_mv_k=fit.slope,
        stderr_mv_k=fit.slope_stderr if math.isfinite(fit.slope_stderr) else None,
        offset_mv=fit.intercept,
        n_points=fit.n,
        notes=notes,
    )


def _optimize(fits: tuple[SampleFit, ...]) -> tuple[OptimizationResult, str | None]:
    """Fit the surrogate over composition, weighting each sample by its own error.

    A sample whose standard error could not be determined — a two-point fit has
    no degrees of freedom left — is given the largest error in the campaign
    rather than the smallest. Treating "unknown" as "excellent" is how one
    under-measured point comes to dominate a surface.

    Returns the result and, when one applies, a note for the report.
    """
    x = np.asarray([fit.composition for fit in fits], dtype=float)
    y = np.asarray([abs(fit.seebeck_mv_k) for fit in fits], dtype=float)

    known = [fit.stderr_mv_k for fit in fits if fit.stderr_mv_k is not None]
    fallback = max(known) if known else float(np.std(y)) or 1.0
    sigma = np.asarray(
        [fit.stderr_mv_k if fit.stderr_mv_k is not None else fallback for fit in fits],
        dtype=float,
    )

    # A campaign whose fitted errors are all vanishing has not achieved perfect
    # precision; it has failed to estimate its own noise, usually because every
    # series happened to lie exactly on its line. Passing that through would set
    # the convergence floor and the epsilon tolerance to roughly zero, and the
    # engine would then report being within 1e-16 of the optimum, which is not a
    # statement about anything. Decline it and let the engine's own relative
    # assumption stand. The per-point ratios are no use here either: when every
    # sample claims the same vanishing error, there is no ordering to preserve.
    scale = float(np.mean(np.abs(y)))
    degenerate = scale > 0 and float(np.median(sigma)) < _DEGENERATE_SIGMA_FRACTION * scale
    note = (
        "Every fitted standard error was vanishingly small, so this campaign carries "
        "no measurement-noise estimate and the engine's default assumption applies. "
        "Real replicate scatter would not look like this; check that the recorded "
        "voltages are measurements rather than computed values."
        if degenerate
        else None
    )

    result = optimize(
        x,
        y,
        bounds=_BOUNDS,
        input_name=_INPUT_NAME,
        target_name=_TARGET_NAME,
        direction="maximize",
        point_noise=None if degenerate or not np.any(sigma > 0) else sigma,
        objective_aggregation="abs_slope",
    )
    return result, note


def main(argv: list[str] | None = None) -> int:
    """`python -m latos next` — run one cycle and print what to make."""
    parser = argparse.ArgumentParser(
        prog="latos next",
        description="Read the recording workbook and recommend the next composition.",
    )
    parser.add_argument("workbook", type=Path, help="the filled recording workbook (.xlsx)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the pre-registration (default: preregistrations/ beside the workbook)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the recommendation without freezing a pre-registration",
    )
    args = parser.parse_args(argv)

    if not args.workbook.is_file():
        print(f"No such workbook: {args.workbook}")
        return 2

    outcome = run_cycle(args.workbook, out_dir=args.out, freeze_prereg=not args.dry_run)
    print(outcome.report())
    return 0 if outcome.fits else 1
