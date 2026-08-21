"""Rehearse a campaign before spending a sample on it.

Bayesian optimization is normally judged after the fact, when the experiments
are already gone. For a lab that can afford eight or twelve samples, the useful
questions come earlier: roughly how many will this take, and is the plan sound
before the first mixture is weighed?

Both can be answered by simulation, because the *algorithm* can be exercised
against objectives whose optimum is known even though the real one is not. The
campaign is replayed against a family of plausible response shapes, with
realistic scatter, and scored on the true value at the points the campaign
actually chose to synthesise. That is what the lab ends up holding.

What this is not
----------------
It proves the algorithm behaves sensibly on curves that are plausible. It says
nothing about whether nature follows any of them. Every report carries that
sentence in `RehearsalReport.caveat`, because a number reported without it would
manufacture exactly the false confidence this project exists to remove.

Auditioning a prior
-------------------
The most valuable use is settling an argument that would otherwise be settled by
taste. A physics-informed prior mean feels like free information, and three
separate ones have now been measured not to be: a self-fitted single-parabolic-
band prior lost 288 to 390 on 888 literature curves, its 1/T variant lost 287 to
436, and a linear mixing law dropped the success rate on a sharp interior peak
from 95 % to 15 %.

Passing `prior_mean` runs the whole rehearsal twice and returns HELPS, NEUTRAL or
HARMS. That converts "should we trust this prior?" from a discussion into a
measurement taken before any sample is spent on the answer.

A design trap, handled explicitly
---------------------------------
When the optimum sits at an endpoint, the seed design finds it immediately and
every strategy looks equally good. In the first version of this experiment three
of five shapes were solved at the second experiment before any model ran, which
is the same trap that made 54.5 % of an early Starrydata benchmark meaningless.
Those shapes are still reported, because a campaign really might be that easy,
but the headline numbers come only from shapes whose optimum lies strictly
inside the range. `ShapeOutcome.discriminating` marks which is which.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from latos.optimization.engine import optimize

__all__ = [
    "HARMS",
    "HELPS",
    "NEUTRAL",
    "RehearsalReport",
    "Shape",
    "ShapeOutcome",
    "default_shapes",
    "rehearse",
]

HELPS = "helps"
NEUTRAL = "neutral"
HARMS = "harms"

CAVEAT = (
    "This rehearsal shows how the optimizer behaves on plausible response shapes "
    "with known optima. It is not evidence that the real system follows any of "
    "them, and the budget it reports is a guide rather than a guarantee."
)

# Below this many discriminating outcomes the comparison is noise. Two shapes at
# forty seeds gives a binomial standard error near 5.6 %, so a difference has to
# clear roughly two of those before it means anything.
_PRIOR_MARGIN = 0.10

# A prior has to save a whole experiment before it counts as helping. A lab
# cannot run four fifths of a sample, and these medians are only as stable as
# the seed count behind them.
_PRIOR_MIN_EXPERIMENTS_SAVED = 1

# One point fixes nothing and two define a line exactly, so a rehearsal needs
# at least two experiments before there is anything to rehearse.
_MIN_BUDGET = 2

_DEFAULT_SEEDS = 40
_DEFAULT_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class Shape:
    """A candidate response curve, with the optimum known in advance."""

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    note: str


@dataclass(frozen=True, slots=True)
class ShapeOutcome:
    """How the campaign fared against one shape, over many random seeds."""

    name: str
    note: str
    discriminating: bool  # the optimum is interior, so the seed design cannot find it free
    median_experiments: int | None  # None when over half the runs never got there
    solved_fraction: float  # share of runs reaching the tolerance within budget


@dataclass(frozen=True, slots=True)
class RehearsalReport:
    """What a rehearsal concluded, and what it is entitled to conclude."""

    outcomes: tuple[ShapeOutcome, ...]
    median_experiments: int | None  # across discriminating shapes only
    solved_fraction: float  # likewise
    budget: int
    tolerance: float
    noise: float
    n_seeds: int
    caveat: str = CAVEAT
    prior_verdict: str | None = None  # HELPS, NEUTRAL, HARMS, or None if not auditioned
    prior_detail: str = ""
    prior_outcomes: tuple[ShapeOutcome, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """A few lines a person can paste into a lab notebook."""
        lines = [
            f"Rehearsal over {len(self.outcomes)} shapes, {self.n_seeds} seeds, "
            f"budget {self.budget}, noise {self.noise:.0%}, tolerance {self.tolerance:.0%}.",
        ]
        if self.median_experiments is None:
            lines.append(
                "On the shapes that discriminate, over half the runs did not reach the "
                "tolerance within budget. Plan for more experiments than this budget."
            )
        else:
            lines.append(
                f"Median experiments to within {self.tolerance:.0%} of the optimum: "
                f"{self.median_experiments}. Reached within budget in "
                f"{self.solved_fraction:.0%} of runs."
            )
        for outcome in self.outcomes:
            got = "never" if outcome.median_experiments is None else str(outcome.median_experiments)
            mark = "" if outcome.discriminating else "   (optimum at an endpoint)"
            lines.append(f"  {outcome.name:<28} {got:>6}  {outcome.solved_fraction:>4.0%}{mark}")
        if self.prior_verdict is not None:
            lines.append(f"Prior: {self.prior_verdict.upper()}. {self.prior_detail}")
        lines.append(self.caveat)
        return "\n".join(lines)


def default_shapes(bounds: tuple[float, float], height: float = 1.0) -> tuple[Shape, ...]:
    """Five response curves spanning the ways a one-variable campaign can go.

    Nobody knows which the real system follows, and a budget that only holds for
    the easy case is not a budget. The sign flip is included because it is
    documented behaviour in ionic thermoelectrics, where a mixing ratio can carry
    the Seebeck coefficient through zero; when it happens, the magnitude has an
    interior *minimum* and the endpoints win, so a campaign optimizing |S| spends
    everything learning that.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    span = hi - lo

    def at(fraction: float) -> float:
        return lo + fraction * span

    def linear(x: np.ndarray) -> np.ndarray:
        return height * (0.2 + 0.8 * (x - lo) / span)

    def interior(x: np.ndarray) -> np.ndarray:
        u = (x - lo) / span
        return height * (0.2 + 0.8 * u + 1.4 * u * (1.0 - u))

    def sign_flip(x: np.ndarray) -> np.ndarray:
        return height * (-0.6 + 1.8 * (x - lo) / span)

    def saturating(x: np.ndarray) -> np.ndarray:
        return height * 1.2 * (1.0 - np.exp(-6.0 * (x - lo) / span))

    def sharp(x: np.ndarray) -> np.ndarray:
        width = 0.09 * span
        return height * (0.3 + 1.2 * np.exp(-((x - at(0.72)) ** 2) / (2 * width**2)))

    return (
        Shape("ideal / linear", linear, "optimum at an endpoint"),
        Shape("interior maximum", interior, "the case worth optimizing"),
        Shape("sign flip", sign_flip, "magnitude has an interior minimum"),
        Shape("saturating", saturating, "flat across most of the range"),
        Shape("sharp peak", sharp, "hardest: narrow, off centre"),
    )


def rehearse(
    *,
    bounds: tuple[float, float],
    noise: float,
    budget: int,
    seed_design: Sequence[float] | None = None,
    prior_mean: Callable[[np.ndarray], np.ndarray] | None = None,
    shapes: Sequence[Shape] | None = None,
    n_seeds: int = _DEFAULT_SEEDS,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> RehearsalReport:
    """Replay the planned campaign against known objectives, before running it.

    Args:
        bounds: The search range, in the knob's own units.
        noise: Measurement scatter as a fraction of the objective's full range.
            The replicate spread from a real protocol is the number to use.
        budget: Total experiments available, seed design included.
        seed_design: Points measured before the optimizer is consulted. Defaults
            to both endpoints and the midpoint, which is what a mixture campaign
            measures anyway for its own reasons.
        prior_mean: A physics-informed prior to audition. When given, the whole
            rehearsal runs twice and the report carries a verdict.
        shapes: Override the default response family.
        n_seeds: Random repetitions per shape. The medians are only as stable as
            this number.
        tolerance: How close to the true optimum counts as solved, as a fraction.

    Returns:
        A `RehearsalReport`. Read `caveat` before quoting anything from it.
    """
    if budget < _MIN_BUDGET:
        raise ValueError(f"budget must allow at least two experiments; got {budget}")
    if not 0.0 <= noise < 1.0:
        raise ValueError(f"noise is a fraction of the signal range; got {noise}")
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance is a fraction; got {tolerance}")

    lo, hi = float(bounds[0]), float(bounds[1])
    design = tuple(seed_design) if seed_design is not None else (lo, (lo + hi) / 2.0, hi)
    if len(design) > budget:
        raise ValueError(
            f"the seed design uses {len(design)} of a {budget}-experiment budget, "
            "leaving nothing for the optimizer."
        )

    family = tuple(shapes) if shapes is not None else default_shapes((lo, hi))
    plain = tuple(
        _rehearse_shape(
            shape,
            bounds=(lo, hi),
            noise=noise,
            budget=budget,
            design=design,
            prior_mean=None,
            n_seeds=n_seeds,
            tolerance=tolerance,
        )
        for shape in family
    )
    median, solved = _headline(plain)

    if prior_mean is None:
        return RehearsalReport(
            outcomes=plain,
            median_experiments=median,
            solved_fraction=solved,
            budget=budget,
            tolerance=tolerance,
            noise=noise,
            n_seeds=n_seeds,
        )

    primed = tuple(
        _rehearse_shape(
            shape,
            bounds=(lo, hi),
            noise=noise,
            budget=budget,
            design=design,
            prior_mean=prior_mean,
            n_seeds=n_seeds,
            tolerance=tolerance,
        )
        for shape in family
    )
    verdict, detail = _judge_prior((median, solved), _headline(primed))

    return RehearsalReport(
        outcomes=plain,
        median_experiments=median,
        solved_fraction=solved,
        budget=budget,
        tolerance=tolerance,
        noise=noise,
        n_seeds=n_seeds,
        prior_verdict=verdict,
        prior_detail=detail,
        prior_outcomes=primed,
    )


def _rehearse_shape(
    shape: Shape,
    *,
    bounds: tuple[float, float],
    noise: float,
    budget: int,
    design: tuple[float, ...],
    prior_mean: Callable[[np.ndarray], np.ndarray] | None,
    n_seeds: int,
    tolerance: float,
) -> ShapeOutcome:
    grid = np.linspace(bounds[0], bounds[1], 1001)
    truth = np.abs(shape.fn(grid))
    best_true = float(np.max(truth))
    sigma = noise * best_true

    counts = [
        _one_run(
            shape,
            bounds=bounds,
            sigma=sigma,
            budget=budget,
            design=design,
            prior_mean=prior_mean,
            seed=seed,
            target=(1.0 - tolerance) * best_true,
        )
        for seed in range(n_seeds)
    ]
    reached = [c for c in counts if c is not None]
    median = int(np.median(reached)) if len(reached) > n_seeds / 2 else None

    return ShapeOutcome(
        name=shape.name,
        note=shape.note,
        discriminating=_has_interior_optimum(truth),
        median_experiments=median,
        solved_fraction=len(reached) / n_seeds,
    )


def _one_run(
    shape: Shape,
    *,
    bounds: tuple[float, float],
    sigma: float,
    budget: int,
    design: tuple[float, ...],
    prior_mean: Callable[[np.ndarray], np.ndarray] | None,
    seed: int,
    target: float,
) -> int | None:
    """Experiments until the best sample *made* is good enough, or None.

    Scored on the true value at the compositions actually synthesised, not on
    the noisy reading and not on the model's belief, because a real campaign
    ends holding a physical sample rather than a posterior.
    """
    rng = np.random.default_rng(seed)
    xs: list[float] = []
    ys: list[float] = []
    running = -math.inf

    def measure(point: float) -> None:
        nonlocal running
        true = abs(float(shape.fn(np.atleast_1d(point))[0]))
        xs.append(point)
        ys.append(true + float(rng.normal(0.0, sigma)))
        running = max(running, true)

    for point in design:
        measure(point)
        if running >= target:
            return len(xs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while len(xs) < budget:
            result = optimize(
                np.asarray(xs, dtype=float),
                np.asarray(ys, dtype=float),
                bounds=bounds,
                input_name="knob",
                target_name="objective",
                direction="maximize",
                measured_noise=max(sigma, 1e-12),
                prior_mean=prior_mean,
                with_reliability=False,
                seed=int(rng.integers(0, 10_000)),
            )
            measure(float(result.recommendation.x))
            if running >= target:
                return len(xs)
    return None


def _has_interior_optimum(truth: np.ndarray) -> bool:
    """True when the best value is not at either end of the range.

    A shape whose optimum sits at an endpoint is solved by the seed design
    before any model runs, so it cannot separate one strategy from another.
    """
    best = int(np.argmax(truth))
    margin = max(1, truth.size // 50)
    return margin < best < truth.size - margin - 1


def _headline(outcomes: tuple[ShapeOutcome, ...]) -> tuple[int | None, float]:
    """Median and solve rate over the shapes that actually discriminate."""
    useful = [o for o in outcomes if o.discriminating]
    if not useful:
        return None, 0.0
    medians = [o.median_experiments for o in useful if o.median_experiments is not None]
    median = int(np.median(medians)) if len(medians) > len(useful) / 2 else None
    return median, float(np.mean([o.solved_fraction for o in useful]))


def _judge_prior(
    without: tuple[int | None, float], with_prior: tuple[int | None, float]
) -> tuple[str, str]:
    """Say plainly whether the prior earned its place.

    Two things can improve and they are not interchangeable. Reaching the
    optimum at all is checked first, because a prior that costs you the answer
    is not redeemed by being quick about it. Only when the solve rates are
    comparable does the number of experiments decide, and it has to decide by a
    whole experiment: a lab cannot run four fifths of a sample, and the medians
    are only as stable as the seed count behind them.
    """
    median_plain, solved_plain = without
    median_primed, solved_primed = with_prior
    rates = (
        f"Reaching the tolerance in {solved_primed:.0%} of runs with the prior against "
        f"{solved_plain:.0%} without it"
    )

    if solved_primed - solved_plain < -_PRIOR_MARGIN:
        return HARMS, (
            f"{rates}. A prior that describes the wrong shape suppresses the search: the "
            "surrogate explains the data as prior plus small residual, expected "
            "improvement stays low where the prior says nothing interesting lives, and "
            "the optimizer never looks there. Do not use it."
        )
    if solved_primed - solved_plain > _PRIOR_MARGIN:
        return HELPS, f"{rates}. Worth using, and worth recording that it was tested."

    if median_plain is not None and median_primed is not None:
        saved = median_plain - median_primed
        if saved >= _PRIOR_MIN_EXPERIMENTS_SAVED:
            return HELPS, (
                f"{rates}, but the prior reaches it in {median_primed} experiments against "
                f"{median_plain}. On this budget that is {saved} sample(s) saved."
            )
        if -saved >= _PRIOR_MIN_EXPERIMENTS_SAVED:
            return HARMS, (
                f"{rates}, but the prior takes {median_primed} experiments against "
                f"{median_plain}. It is steering the search away from the answer."
            )

    return NEUTRAL, (
        f"{rates}, with no material difference in experiments either. The prior is not "
        "earning its place; prefer the simpler model."
    )
