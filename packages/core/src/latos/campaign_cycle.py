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

__all__ = [
    "CycleOutcome",
    "DesignPoint",
    "SampleFit",
    "aggregate_replicates",
    "main",
    "run_cycle",
]

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

# Two samples weighed to 30.00 % and 30.02 % are one condition attempted twice,
# not two conditions. Compositions within this absolute distance are treated as
# the same design point. Deliberately larger than a balance's resolution and far
# smaller than any spacing a campaign would design on purpose.
_COMPOSITION_TOLERANCE = 0.005

# A within-point standard deviation needs degrees of freedom to mean anything.
# One replicated point with n = 2 contributes a single degree of freedom, and a
# variance on one degree of freedom is close to worthless: its sampling
# distribution is wide enough that the estimate is routinely out by a factor of
# several. Pooling across every replicated point is the standard remedy and
# costs nothing, so the pooled figure is used whenever the total reaches this
# many degrees of freedom. Below it the estimate is still used, with the caveat
# stated in the report, because a wide honest estimate beats the flat 8 %
# assumption it replaces.
_MIN_POOLED_DF = 3

# A condition attempted once has nothing to compare itself with, so it
# contributes no degrees of freedom to the pooled estimate.
_MIN_REPLICATES_FOR_SPREAD = 2


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
class DesignPoint:
    """One composition, and everything measured at it.

    A design point is what the surrogate actually observes. When a composition
    was made only once it is the sample; when it was made several times it is
    their mean, and the scatter between them is the honest measure of how well
    that condition is known.

    That distinction matters more than it looks. The standard error of a slope
    fit describes how well a line was drawn through one specimen's points. It
    says nothing about whether making the specimen again would give the same
    answer, and between-specimen scatter is the larger term: it carries the
    weighing, the mixing, the mounting and the contacts as well as the voltmeter.
    Feeding a fit error to the optimizer as though it were the measurement
    uncertainty therefore understates the noise, and an over-confident surrogate
    is precisely the failure the reliability layer exists to catch.
    """

    composition: float  # the mean composition of the replicates
    value: float  # mean |S| across replicates, in mV/K
    sigma: float | None  # standard error of that mean, None when undetermined
    n_replicates: int
    sample_ids: tuple[str, ...]
    sigma_source: str  # "replicates" | "pooled" | "fit" | "undetermined"
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CycleOutcome:
    """What the cycle concluded, and where it wrote the evidence."""

    fits: tuple[SampleFit, ...]
    result: OptimizationResult | None
    prereg_path: Path | None
    messages: tuple[str, ...]
    points: tuple[DesignPoint, ...] = ()

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

        # Only worth a second table when aggregation actually did something.
        if any(p.n_replicates > 1 for p in self.points):
            lines.extend(("", "Design points given to the surrogate:"))
            for p in self.points:
                unc = "undetermined" if p.sigma is None else f"{p.sigma:.4f}"
                lines.append(
                    f"  x = {p.composition:.3f}   |S| = {p.value:.3f} +/- {unc} mV/K   "
                    f"n = {p.n_replicates} ({p.sigma_source})   "
                    f"[{', '.join(p.sample_ids)}]"
                )

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

    # Replicates of one condition are one observation, not several. The surrogate
    # is fitted over design points, so every count and every gate below refers to
    # those rather than to samples.
    points, replicate_notes = aggregate_replicates(fits)
    messages.extend(replicate_notes)
    messages.extend(note for point in points for note in point.notes)

    if len(points) < _MIN_SAMPLES_FOR_A_MODEL:
        extra = (
            ""
            if len(points) == len(fits)
            else f" ({len(fits)} sample(s) collapsed onto {len(points)} distinct composition(s))"
        )
        messages.append(
            f"{len(points)} design point(s) available{extra}. A surrogate needs at least "
            f"{_MIN_SAMPLES_FOR_A_MODEL} DISTINCT compositions; measure both pure liquids "
            "and the midpoint first, which the campaign needs anyway as mixing-law anchors "
            "and drift controls."
        )
        return CycleOutcome(fits, None, None, tuple(messages), points)

    result, noise_note = _optimize(points)
    if noise_note:
        messages.append(noise_note)
    if not freeze_prereg:
        messages.append(
            "Nothing was pre-registered. This recommendation cannot later be presented "
            "as a prediction made before the sample."
        )
        return CycleOutcome(fits, result, None, tuple(messages), points)

    destination = out_dir if out_dir is not None else workbook.parent / "preregistrations"
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = freeze(result, destination / f"prereg_{stamp}.json", prior_best=result.best_y)

    return CycleOutcome(fits, result, path, tuple(messages), points)


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


def _cluster(fits: tuple[SampleFit, ...]) -> list[list[SampleFit]]:
    """Group samples whose compositions are the same condition attempted twice.

    Single-linkage on a sorted list: a sample joins the open group while it is
    within `_COMPOSITION_TOLERANCE` of the previous one. Chaining is possible in
    principle — a run of samples each just inside the tolerance of its neighbour
    would merge into one group spanning more than the tolerance — but a campaign
    that dense in composition is not one this module is for, and the alternative
    (fixed bins) would split a genuine pair that straddles a bin edge, which is
    the worse failure of the two.
    """
    if not fits:
        return []
    ordered = sorted(fits, key=lambda f: f.composition)
    groups: list[list[SampleFit]] = [[ordered[0]]]
    for fit in ordered[1:]:
        if abs(fit.composition - groups[-1][-1].composition) <= _COMPOSITION_TOLERANCE:
            groups[-1].append(fit)
        else:
            groups.append([fit])
    return groups


def aggregate_replicates(fits: tuple[SampleFit, ...]) -> tuple[tuple[DesignPoint, ...], list[str]]:
    """Collapse samples onto design points, measuring the noise where possible.

    Independent specimens made at the same composition are replicates of one
    condition. Their scatter is what the optimizer should be told, because it
    includes everything that varies between one attempt and the next: weighing,
    mixing, mounting, contacts and the voltmeter. A slope-fit standard error
    includes only the last of those.

    The within-point standard deviations are **pooled** across every replicated
    condition rather than used one at a time. Two replicates give one degree of
    freedom, and a variance on one degree of freedom is so noisy as to be
    misleading; pooling k conditions gives the sum of their degrees of freedom
    for no extra experiments. Each replicated point then receives the pooled
    standard deviation divided by the root of its own replicate count, and each
    unreplicated point receives the pooled standard deviation itself, which is
    the correct uncertainty for a single observation drawn from that population
    and is usually larger, and more honest, than its own fit error.

    Returns the design points and any notes for the report.
    """
    notes: list[str] = []
    groups = _cluster(fits)

    # Within-point spread, one entry per replicated condition.
    ssq = 0.0
    dof = 0
    for group in groups:
        if len(group) < _MIN_REPLICATES_FOR_SPREAD:
            continue
        values = np.asarray([abs(f.seebeck_mv_k) for f in group], dtype=float)
        ssq += float(np.sum((values - values.mean()) ** 2))
        dof += len(group) - 1

    pooled_sd: float | None = None
    if dof > 0:
        pooled_sd = math.sqrt(ssq / dof)
        replicated = sum(1 for g in groups if len(g) > 1)
        detail = (
            f"Replicate scatter measured at {replicated} condition(s): pooled standard "
            f"deviation {pooled_sd:.4f} mV/K on {dof} degree(s) of freedom. This is the "
            f"measurement uncertainty the surrogate was given, in place of the engine's "
            f"assumed default."
        )
        if dof < _MIN_POOLED_DF:
            detail += (
                f" With only {dof} degree(s) of freedom this estimate is itself uncertain "
                f"by roughly a factor of two; treat it as an order of magnitude, and "
                f"replicate a second condition to sharpen it."
            )
        if pooled_sd <= 0.0:
            pooled_sd = None
            detail = (
                "Replicates at the same condition gave identical values to full precision. "
                "That is not agreement, it is a sign the recorded numbers are computed "
                "rather than measured. The replicate estimate was discarded."
            )
        notes.append(detail)

    points: list[DesignPoint] = []
    for group in groups:
        values = np.asarray([abs(f.seebeck_mv_k) for f in group], dtype=float)
        composition = float(np.mean([f.composition for f in group]))
        ids = tuple(f.sample_id for f in group)
        local: list[str] = []

        signs = {math.copysign(1.0, f.seebeck_mv_k) for f in group if f.seebeck_mv_k != 0.0}
        if len(signs) > 1:
            local.append(
                "Replicates at this condition disagree in the SIGN of the coefficient. "
                "That is a difference in carrier type, not measurement scatter, and "
                "averaging their magnitudes hides it. Check the wiring polarity and the "
                "specimen orientation before trusting this point."
            )

        if len(group) > 1 and pooled_sd is not None:
            sigma: float | None = pooled_sd / math.sqrt(len(group))
            source = "replicates"
        elif pooled_sd is not None:
            # One attempt at this condition. A single draw from a population whose
            # spread we have measured elsewhere carries that whole spread.
            sigma = pooled_sd
            source = "pooled"
        else:
            sigma = group[0].stderr_mv_k if len(group) == 1 else None
            source = "fit" if sigma is not None else "undetermined"

        points.append(
            DesignPoint(
                composition=composition,
                value=float(values.mean()),
                sigma=sigma,
                n_replicates=len(group),
                sample_ids=ids,
                sigma_source=source,
                notes=tuple(local),
            )
        )

    collapsed = len(fits) - len(points)
    if collapsed > 0:
        notes.append(
            f"{len(fits)} sample(s) collapsed onto {len(points)} design point(s); "
            f"{collapsed} were replicates of a condition already present."
        )
    return tuple(points), notes


def _optimize(points: tuple[DesignPoint, ...]) -> tuple[OptimizationResult, str | None]:
    """Fit the surrogate over composition, weighting each point by its own error.

    A point whose uncertainty could not be determined — a two-point slope fit has
    no degrees of freedom left, and a lone sample at an unreplicated condition
    has nothing to compare itself with — is given the largest error in the
    campaign rather than the smallest. Treating "unknown" as "excellent" is how
    one under-measured point comes to dominate a surface.

    Returns the result and, when one applies, a note for the report.
    """
    x = np.asarray([p.composition for p in points], dtype=float)
    y = np.asarray([p.value for p in points], dtype=float)

    known = [p.sigma for p in points if p.sigma is not None]
    fallback = max(known) if known else float(np.std(y)) or 1.0
    sigma = np.asarray(
        [p.sigma if p.sigma is not None else fallback for p in points],
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
        "Every uncertainty was vanishingly small, so this campaign carries "
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
