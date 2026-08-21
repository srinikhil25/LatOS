"""Run many simulated-synthesis campaigns in parallel.

A single campaign on the corrected `synthesis_sim` harness costs about 96
seconds single-core, and a study cell is 30 of them. The AX sweep — four
configurations across three process windows — is 360 campaigns, or roughly ten
hours serially. That is the measurement AX2 owes AX3 before the anisotropy sweep
can be scoped, and the answer is that it does not fit without parallelism.

Campaigns are embarrassingly parallel: each one is an independent problem
instance. The only subtlety is that NumPy's BLAS already threads internally, so
running sixteen worker processes on top of it oversubscribes the machine and can
end up *slower* than serial. The GP fits here are small (n < 100), where threaded
BLAS buys almost nothing, so each worker is pinned to a single thread and the
parallelism is taken at the campaign level instead.

This lives in the package rather than in a scratch directory on purpose. The
Starrydata sample-efficiency benchmark was written as a throwaway script, the
scratch directory was wiped, and the harness behind a headline result had to be
reconstructed from memory. Research drivers are results infrastructure.

**Callers must guard their entry point.** Windows spawns worker processes by
re-importing the calling module, so a driver script without

    if __name__ == "__main__":
        ...

re-runs its own body in every worker, which then tries to spawn workers of its
own. `run_cells` raises rather than letting that happen, because the failure
mode otherwise is a wall of tracebacks with no obvious cause.
"""

from __future__ import annotations

import multiprocessing
import os
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from latos.optimization.synthesis_sim import (
    make_model_function,
    n90_from_trials,
    run_grid_campaign,
)

__all__ = ["CellResult", "CellSpec", "run_cells"]

# Pin BLAS before NumPy is imported in a worker. Set in the parent too, since
# the parent imports NumPy transitively and children inherit the environment.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _pin_threads() -> None:
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")


_pin_threads()


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One (configuration, process window) cell of a sweep.

    `kwargs` is passed straight through to `run_grid_campaign`, so a cell is
    fully described by this object — which is what lets a result be traced back
    to the exact settings that produced it.
    """

    label: str
    n_dims: int
    pw: float
    n_functions: int = 30
    budget: int = 100
    aspect: float = 1.0
    # Shifts the seed range, so a cell can be re-run on a *different* set of
    # model functions. Independent replication is the only way to tell a real
    # marginal effect from one of the several comparisons that happened to land
    # under 0.05 — re-running the same seeds just reproduces the same numbers.
    seed_offset: int = 0
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CellResult:
    """What a cell measured, with enough detail to re-derive the statistic."""

    spec: CellSpec
    n90: float | None
    median: float | None
    success_rate: float
    trials: tuple[int, ...]  # successes only, sorted — for reading N90% off
    # Per-seed, in seed order, with None for campaigns that never succeeded.
    # Sorting discards which model function produced which result, and cells that
    # share a seed range are running the *same* problems — so keeping the order
    # is what makes a paired comparison possible. Comparing two independent N90%
    # point estimates throws away that pairing and buries a real effect under
    # between-function variance, which is much larger than the effect itself.
    trials_by_seed: tuple[int | None, ...]
    seconds: float


def _one_campaign(args: tuple[CellSpec, int]) -> int | None:
    """Run one campaign; return trials-to-success, or None if it never got there.

    Top-level so it survives pickling under the spawn start method Windows uses.
    """
    _pin_threads()
    warnings.filterwarnings("ignore")
    spec, seed = args
    model = make_model_function(spec.n_dims, spec.pw, seed=seed, aspect=spec.aspect)
    outcome = run_grid_campaign(model, budget=spec.budget, seed=seed, **spec.kwargs)  # type: ignore[arg-type]
    return outcome.trials_to_success


def run_cells(
    specs: Sequence[CellSpec],
    *,
    max_workers: int | None = None,
    on_result: object = None,
) -> list[CellResult]:
    """Measure every cell, parallel across campaigns.

    Args:
        specs: the cells to run.
        max_workers: defaults to two fewer than the core count, leaving room for
            the machine to stay usable.
        on_result: optional callable invoked with each `CellResult` as it lands,
            so a long sweep reports progress instead of going quiet for an hour.

    Returns:
        One `CellResult` per spec, in the order given.
    """
    if multiprocessing.current_process().name != "MainProcess":
        raise RuntimeError(
            "run_cells was called from a worker process. On Windows this means "
            "the calling script has no `if __name__ == '__main__':` guard, so "
            "every spawned worker re-ran it."
        )

    workers = max_workers or max(1, (os.cpu_count() or 4) - 2)
    results: list[CellResult] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for spec in specs:
            started = time.time()
            trials = list(
                pool.map(
                    _one_campaign, [(spec, spec.seed_offset + s) for s in range(spec.n_functions)]
                )
            )
            elapsed = time.time() - started

            value = n90_from_trials(trials)
            successes = sorted(t for t in trials if t is not None)
            median = float(successes[len(successes) // 2]) if successes else None

            result = CellResult(
                spec=spec,
                n90=value,
                median=median,
                success_rate=len(successes) / len(trials),
                trials=tuple(successes),
                trials_by_seed=tuple(trials),
                seconds=elapsed,
            )
            results.append(result)
            if callable(on_result):
                on_result(result)

    return results
