"""Closed-loop validation against functions whose optimum is already known.

Everything else in this package is judged on real experiments, where nobody
knows the right answer — which is exactly why none of it can be *validated*
there. A benchmark with a published optimum is the only place the loop can be
graded rather than merely inspected: run it, and either it finds the answer or
it does not.

Two standard problems, both minimisation, both used throughout the Bayesian
optimization literature:

    Branin      2-D, three equal global minima at f = 0.397887
    Hartmann-3  3-D, one global minimum at f = -3.86278

`run_campaign` simulates the whole loop rather than a single round: propose,
"synthesise" by evaluating the true function, append, refit, repeat. That is
the thing worth testing, because a recommendation that is individually sensible
can still fail to converge over a campaign.

`strategy="random"` runs the identical loop drawing uniformly instead of by
Expected Improvement. Reporting Bayesian optimization without that baseline is
how a method gets credited for work the sampling budget did on its own — on a
small budget random search is a stronger competitor than it sounds, so if the
engine cannot beat it, that is the finding.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from latos.optimization.engine import optimize_nd

__all__ = [
    "BENCHMARKS",
    "Benchmark",
    "CampaignResult",
    "branin",
    "hartmann3",
    "run_campaign",
]


def branin(x: np.ndarray) -> np.ndarray:
    """Branin-Hoo, on the usual domain x1 in [-5, 10], x2 in [0, 15].

    Three global minima of equal value, which makes it a fair test of whether
    a method finds *an* optimum rather than memorising one location.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    x1, x2 = x[:, 0], x[:, 1]
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
_H3_P = 1e-4 * np.array(
    [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
)


def hartmann3(x: np.ndarray) -> np.ndarray:
    """Hartmann-3 on [0, 1]^3. One global minimum, several local ones."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    inner = np.einsum("ij,kij->ki", _H3_A, (x[:, None, :] - _H3_P[None, :, :]) ** 2)
    return -np.einsum("j,kj->k", _H3_ALPHA, np.exp(-inner))


@dataclass(frozen=True, slots=True)
class Benchmark:
    """A test function with its published optimum."""

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    bounds: tuple[tuple[float, float], ...]
    optimum: float  # the known minimum value
    n_dims: int


BENCHMARKS: dict[str, Benchmark] = {
    "branin": Benchmark("branin", branin, ((-5.0, 10.0), (0.0, 15.0)), 0.397887, 2),
    "hartmann3": Benchmark("hartmann3", hartmann3, ((0.0, 1.0),) * 3, -3.86278, 3),
}


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """What a simulated closed loop achieved.

    `regret` is simple regret: how far the best point found sits above the
    known optimum. It is the number that matters — not whether the model's
    curve looked plausible.
    """

    benchmark: str
    strategy: str
    n_initial: int
    n_rounds: int
    seed: int
    best_y: float
    regret: float
    best_x: tuple[float, ...]
    history: tuple[float, ...]  # best-so-far after each round
    reliability_levels: tuple[str, ...]  # grade after each round, if assessed


def _initial_design(bounds: np.ndarray, n: int, seed: int) -> np.ndarray:
    """A scrambled Sobol start, so neither strategy is handed a lucky corner."""
    d = bounds.shape[0]
    m = max(1, int(np.ceil(np.log2(max(n, 2)))))
    unit = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m)[:n]
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def run_campaign(
    benchmark: str | Benchmark,
    *,
    n_initial: int = 8,
    n_rounds: int = 12,
    seed: int = 0,
    strategy: str = "bo",
    rel_noise: float = 0.01,
    with_reliability: bool = False,
    n_candidates: int = 1024,
    length_scale_bounds: tuple[float, float] | None = None,
) -> CampaignResult:
    """Simulate a closed loop and report how close it got.

    Args:
        benchmark: a key of `BENCHMARKS` or a `Benchmark`.
        n_initial: points in the space-filling start, shared by both strategies.
        n_rounds: proposals after the initial design.
        seed: controls the initial design, the GP restarts and the random
            baseline's draws, so a (seed, strategy) pair is reproducible.
        strategy: "bo" to propose by Expected Improvement, "random" to draw
            uniformly — the baseline any claim of benefit has to clear.
        rel_noise: relative measurement noise handed to the engine. The
            benchmark itself is evaluated exactly; this is what the *model* is
            told to assume, and a materials tool should never be tested as if
            measurements were perfect.
        with_reliability: run the leave-one-out grade each round. Off by
            default because it costs n extra GP fits per round and the regret
            does not depend on it; switch it on to watch the grade evolve.
        n_candidates: Sobol points the acquisition is maximised over each round.
        length_scale_bounds: override the ARD length-scale range. `None` uses
            the engine default, which is the 1-D value and measurably too
            coarse for structured multi-axis targets.

    Returns:
        A `CampaignResult`.
    """
    bench = BENCHMARKS[benchmark] if isinstance(benchmark, str) else benchmark
    if strategy not in ("bo", "random"):
        raise ValueError(f"strategy must be 'bo' or 'random'; got {strategy!r}")

    box = np.asarray(bench.bounds, dtype=float)
    x = _initial_design(box, n_initial, seed)
    y = np.asarray(bench.fn(x), dtype=float)

    rng = np.random.default_rng(seed)
    history: list[float] = [float(y.min())]
    levels: list[str] = []

    for _ in range(n_rounds):
        if strategy == "random":
            nxt = box[:, 0] + rng.random(box.shape[0]) * (box[:, 1] - box[:, 0])
        else:
            res = optimize_nd(
                x,
                y,
                bounds=bench.bounds,
                input_names=tuple(f"x{i}" for i in range(bench.n_dims)),
                target_name=bench.name,
                direction="minimize",
                rel_noise=rel_noise,
                seed=seed,
                n_candidates=n_candidates,
                with_reliability=with_reliability,
                **(
                    {}
                    if length_scale_bounds is None
                    else {"length_scale_bounds": length_scale_bounds}
                ),
            )
            nxt = np.asarray(res.recommendation.x, dtype=float)
            if res.reliability is not None:
                levels.append(res.reliability.level)

        x = np.vstack([x, nxt])
        y = np.append(y, float(bench.fn(nxt.reshape(1, -1))[0]))
        history.append(float(y.min()))

    best = int(np.argmin(y))
    return CampaignResult(
        benchmark=bench.name,
        strategy=strategy,
        n_initial=n_initial,
        n_rounds=n_rounds,
        seed=seed,
        best_y=float(y[best]),
        regret=float(y[best] - bench.optimum),
        best_x=tuple(float(v) for v in x[best]),
        history=tuple(history),
        reliability_levels=tuple(levels),
    )


def compare(
    benchmark: str,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    **kwargs: object,
) -> dict[str, tuple[float, ...]]:
    """Median-friendly regret for each strategy across several seeds.

    Several seeds because one campaign proves nothing: Bayesian optimization
    beating random once is within the noise of the initial design.
    """
    out: dict[str, tuple[float, ...]] = {}
    for strategy in ("bo", "random"):
        out[strategy] = tuple(
            run_campaign(benchmark, seed=s, strategy=strategy, **kwargs).regret  # type: ignore[arg-type]
            for s in seeds
        )
    return out
