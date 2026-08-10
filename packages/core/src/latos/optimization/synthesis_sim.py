"""Simulated thin-film synthesis campaigns with anisotropic process windows.

`benchmarks.py` grades the optimizer on Branin and Hartmann-3, which are the
right tools for "does the loop work" and the wrong ones for "how many syntheses
will this cost". Those functions were not designed to look like a deposition
chamber, and their optima are not process windows.

This module rebuilds the model that Xu, Nakayama, Kimura, Shimizu, Ando,
Kobayashi, Yasuo, Sekijima & Hitosugi used in *Tuning Bayesian optimization for
materials synthesis: simulating two- and three-dimensional cases*, Sci. Technol.
Adv. Mater.: Methods **3**, 2210251 (2023) — sums of Gaussian peaks on a 51-point
grid per axis, standing in for temperature, oxygen partial pressure and sputtering
power — and adds the one thing they explicitly left out:

    "In this study, as the simplest case, we assumed that the shape of the global
    and local optimum peaks in materials synthesis is isotropic. […] We note
    that, in actual materials synthesis, the shape of the global and local
    optimum peaks could be anisotropic. Future investigations into these cases
    will be reported in due course."

Anisotropy is the normal case, not a corner case: a process window is rarely
equally tight in every knob. Temperature might tolerate +/-100 C while oxygen
partial pressure is tight to +/-0.2e-4 Pa. Xu et al. removed this by
construction, setting the Gaussian width equal across axes in grid units.

Two things make the comparison honest rather than decorative:

* **The isotropic arm reproduces their published N90%.** If it does not, the
  harness is wrong and every anisotropic number downstream is worthless. See
  `PUBLISHED_N90` and the tests that check against it.
* **Anisotropy holds peak volume fixed.** The per-axis widths are scaled by
  `sqrt(aspect)` and `1/sqrt(aspect)` about the nominal window, so a stretched
  peak is not also a *bigger* peak. Without that, "anisotropy costs experiments"
  would just be measuring that small targets are harder to find.

The search is over a discrete grid, as theirs is: a proposal is snapped to the
nearest unvisited grid point before it is evaluated. Optimising continuously and
reporting against a grid-based N90% would be comparing different quantities.

Reproduction status — READ THIS BEFORE QUOTING ANY NUMBER FROM HERE
-------------------------------------------------------------------
Measured against `PUBLISHED_N90`, 30 model functions per cell, 2-D:

    process window    this harness    Xu et al.
    wide  (Pw=7)              18        16.8 +- 0.8
    medium(Pw=5)              30        21.4 +- 1.5
    narrow(Pw=3)     >100 (67% ok)      49.8 +- 2.0

The qualitative behaviour reproduces: N90% climbs steeply as the process window
narrows, which is the property any statement about window *shape* depends on.
The wide-window value agrees. **The narrow-window value does not** — this
harness is systematically harder, increasingly so as the window tightens, and
the cause is not known.

Three explanations were tested and eliminated, so none of them should be
re-proposed without new evidence:

  * *Length-scale.* Pinning l at their reported best of 1 grid makes results
    much worse, not better. Their l = 1 is an **initial** value for a marginal-
    likelihood fit, not a fixed one, so free fitting is the faithful setting.
  * *Snapping.* The concern was that rerouting a repeated proposal to its
    nearest unvisited neighbour gives a free local hill-climb. Measured: 0% of
    proposals were duplicates. It never fires.
  * *Success criterion.* Their "90% or more (approximate region of Pw) of the
    global maximum" admits three readings (see `_SUCCESS_BASES`). All three
    give the same N90% to within a trial or two.

Untested and still plausible: peak placement geometry — the paper fixes the
minimum separation but says nothing about insetting peaks from the boundary,
and a broad background hump sitting near the global peak would signpost it in a
way this generator does not guarantee; and the variance hyperparameter, which
they fit over 0.1-10 while Latos normalises y and uses a fixed ConstantKernel
range.

What this means for use: comparisons **within** this harness — isotropic versus
ARD at a given aspect ratio, or one aspect ratio against another — are
internally controlled and sound. Comparisons of an absolute N90% against the
published table are not, and must be reported qualitatively.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from latos.optimization.engine import optimize_nd

__all__ = [
    "LS_BOUNDS_ANISOTROPY",
    "LS_BOUNDS_PAPER",
    "PUBLISHED_N90",
    "CampaignOutcome",
    "ModelFunction",
    "Peak",
    "fitted_aspect_ratio",
    "make_model_function",
    "n90",
    "n90_from_trials",
    "run_grid_campaign",
]

# Their grid: 51 points per axis, so one axis spans 50 grid steps.
GRIDS = 51

# Nominal process windows, as the standard deviation of the global peak in grid
# units. Xu et al.'s three settings: Pw = (60 C, 0.6e-4 Pa) -> 3 grids,
# (100 C, 1.0e-4 Pa) -> 5, (140 C, 1.4e-4 Pa) -> 7. Naming them by width keeps
# the call sites readable — "narrow" is the hard one.
PW_NARROW, PW_MEDIUM, PW_WIDE = 3.0, 5.0, 7.0

# Fixed widths for the distractors, also from the paper: the local optimum is
# broader than the global one (120 C), and the background is broad enough to
# tilt the whole space (400 C).
_SIGMA_LOCAL = 6.0
_SIGMA_BACKGROUND = 20.0

# Peak heights. The global optimum is only 1.7x the local one, which is what
# makes the problem a search rather than a gradient walk.
_H_2D = (1.2, 0.7, 0.3)
_H_3D = (1.2, 0.7, 0.6, 0.3)
_TWO_D = 2  # the paper's two cases are 2-D and 3-D; named so comparisons read

# Independent RNG streams for "where are the peaks" and "where do we start".
#
# Both used to be `default_rng(seed)`, and callers naturally pass the same seed
# to both so a run is reproducible. That made the first initial point a
# deterministic function of the peak position: the peak is drawn from
# uniform(inset, 50-inset) and the first point from uniform(0, 50), off the same
# variate, so they differed by at most `inset` grids. The first "random" point
# landed on the optimum every time — and more precisely for narrower windows,
# because the inset shrinks with the peak. That inverted the process-window
# trend the entire study depends on, while looking like a very good optimizer.
#
# Spawning from a SeedSequence keeps (seed -> problem) reproducible while making
# the two streams statistically independent.
_STREAM_MODEL = 0x5EED_9EA5
_STREAM_CAMPAIGN = 0xCA_3A_16_47


def _stream(stream_id: int, seed: int) -> np.random.Generator:
    """A generator that is reproducible in `seed` but independent per stream."""
    return np.random.default_rng(np.random.SeedSequence([stream_id, seed]))


# A run counts as a success once it has found this fraction of the target value.
_SUCCESS_FRACTION = 0.9
# The quantile N90% reports. Numerically equal to _SUCCESS_FRACTION and
# conceptually unrelated — one is "how good counts as found", the other is "how
# many campaigns must have found it". Kept separate so changing either is safe.
_N90_QUANTILE = 0.9

# Xu et al. define success as: "When 90% or more (approximate region of Pw) of
# the global maximum was found". That sentence admits three readings, and they
# are not close to equivalent once the process window is narrow:
#
#   "surface"  90% of the maximum of the whole surface — peak plus the local
#              and background tails underneath it. The strictest reading, and
#              the literal one.
#   "peak"     90% of the global peak's own height, ignoring what sits beneath
#              it. A larger target.
#   "window"   within Pw of the peak centre. This is what their parenthetical
#              actually says, and it is the loosest: a Gaussian is above 90% of
#              its peak only within 0.459 sigma, so "the 90% region" and "the
#              region of Pw" cannot both be true. Their own equivalence is
#              internally inconsistent, which is why all three are implemented
#              rather than one being guessed at.
_SUCCESS_BASES = ("surface", "peak", "window")

# Their published N90% values, for the validation gate. Reproducing these is the
# precondition for trusting anything this module says about anisotropy.
PUBLISHED_N90: dict[tuple[int, float], tuple[float, float]] = {
    (2, PW_WIDE): (16.8, 0.8),
    (2, PW_MEDIUM): (21.4, 1.5),
    (2, PW_NARROW): (49.8, 2.0),
    (3, PW_MEDIUM): (74.8, 2.5),
    (3, PW_NARROW): (361.8, 30.5),
}

# Their measured optimum was a length-scale of 1 grid, fitted within 0.5-5 grids.
# The engine works in units where a search span is `_SPAN_UNITS`, so a grid step
# is span/(GRIDS-1) of that. Passing these explicitly is what makes the
# reproduction a reproduction rather than a different experiment.
_SPAN = 4.0
_GRID_IN_ENGINE_UNITS = _SPAN / (GRIDS - 1)
LS_BOUNDS_PAPER = (0.5 * _GRID_IN_ENGINE_UNITS, 5.0 * _GRID_IN_ENGINE_UNITS)

# Anisotropy studies must NOT use the range above. It was chosen for isotropic
# peaks no wider than Pw = 7 grids, and it tops out at 5 — so on a stretched
# window the long axis is clipped at the ceiling and the fitted length-scale
# ratio reports the *bounds* rather than the peak. Measured directly: at aspect
# 5 the fit returns (0.171, 0.400) where 0.400 is exactly the ceiling, giving a
# ratio of 2.3 against a true 5.0.
#
# That is not a small bias, it is the measurement destroying the thing it
# measures — and it would have produced a confident "ARD does not recover
# anisotropy" result. The widened range spans every true width the sweep
# generates: at Pw = 5 and aspect 10 the axes are 1.6 and 15.8 grids, so 0.25
# to 25 grids leaves headroom at both ends without pinning either.
LS_BOUNDS_ANISOTROPY = (0.25 * _GRID_IN_ENGINE_UNITS, 25.0 * _GRID_IN_ENGINE_UNITS)


@dataclass(frozen=True, slots=True)
class Peak:
    """One Gaussian component, in grid coordinates.

    `sigma` is per-axis, which is the whole point: an isotropic peak has equal
    entries, an anisotropic one does not.
    """

    height: float
    centre: tuple[float, ...]
    sigma: tuple[float, ...]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Contribution of this peak at grid coordinates `x`, shape (..., d)."""
        c = np.asarray(self.centre, dtype=float)
        s = np.asarray(self.sigma, dtype=float)
        return self.height * np.exp(-np.sum(((x - c) / s) ** 2, axis=-1) / 2.0)


@dataclass(frozen=True, slots=True)
class ModelFunction:
    """A synthetic property surface: peaks plus measurement noise.

    Evaluated in grid coordinates so the search space is exactly the lattice the
    paper searches. `optimum` is the noise-free maximum over that lattice, which
    is what success is judged against — judging against the continuous supremum
    would make the target unreachable by a grid search.
    """

    peaks: tuple[Peak, ...]
    n_dims: int
    noise_fraction: float
    seed: int

    def clean(self, x: np.ndarray) -> np.ndarray:
        """Noise-free value at grid coordinates `x`, shape (..., d)."""
        x = np.asarray(x, dtype=float)
        return sum(p(x) for p in self.peaks)  # type: ignore[return-value]

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Value with measurement noise, as a real experiment would return it."""
        clean = self.clean(x)
        scale = self.noise_fraction * max(p.height for p in self.peaks)
        return clean + scale * rng.standard_normal(np.shape(clean))

    @property
    def lattice(self) -> np.ndarray:
        """Every grid point, shape (GRIDS**d, d)."""
        axes = np.meshgrid(*([np.arange(GRIDS, dtype=float)] * self.n_dims), indexing="ij")
        return np.column_stack([a.ravel() for a in axes])

    @property
    def optimum(self) -> float:
        """Noise-free maximum over the lattice — what success is judged against."""
        return float(self.clean(self.lattice).max())


def _anisotropic_sigma(nominal: float, aspect: float, n_dims: int) -> tuple[float, ...]:
    """Per-axis widths with the given aspect ratio at fixed peak volume.

    The widths are spread geometrically about `nominal` so their product is
    unchanged — a stretched window is not also a larger one. Without this,
    sweeping the aspect ratio would confound peak shape with peak size, and the
    result would say "smaller targets are harder", which nobody needed a
    simulation to learn.
    """
    if aspect <= 0:
        raise ValueError(f"aspect must be positive; got {aspect}")
    if n_dims == 1 or aspect == 1.0:
        return (nominal,) * n_dims
    # Exponents symmetric about zero: e.g. d=2 -> (-1/2, +1/2), d=3 -> (-1, 0, 1)
    # scaled so the extremes are exactly `aspect` apart in ratio.
    steps = np.linspace(-0.5, 0.5, n_dims)
    widths = nominal * aspect**steps
    # Renormalise so the geometric mean is exactly `nominal`.
    widths *= nominal / float(np.exp(np.mean(np.log(widths))))
    return tuple(float(w) for w in widths)


def make_model_function(
    n_dims: int,
    pw: float,
    *,
    seed: int,
    aspect: float = 1.0,
    noise_fraction: float = 0.001,
    inset_peaks: bool = True,
) -> ModelFunction:
    """Build one model surface with a randomly placed global optimum.

    Args:
        n_dims: 2 or 3, matching the paper's two cases.
        pw: process window — the global peak's nominal standard deviation in
            grid units. `PW_NARROW` / `PW_MEDIUM` / `PW_WIDE`.
        seed: controls peak placement, so a (seed, pw, aspect) triple is one
            reproducible problem instance.
        aspect: ratio between the widest and narrowest axis of the *global*
            peak, at fixed volume. 1.0 reproduces Xu et al. exactly.
        noise_fraction: measurement noise as a fraction of the tallest peak.
            Their default is 0.1%.
        inset_peaks: keep every peak at least its own width from the box edge,
            so the optimum is interior. The paper does not state whether it did
            this, and it matters — an optimum against a wall is a different and
            generally harder search, because the model has data on one side of
            it only. Exposed so the assumption can be ablated rather than
            assumed.

    Returns:
        A `ModelFunction`.
    """
    if n_dims not in (_TWO_D, _TWO_D + 1):
        raise ValueError(f"n_dims must be 2 or 3; got {n_dims}")
    rng = _stream(_STREAM_MODEL, seed)
    heights = _H_2D if n_dims == _TWO_D else _H_3D

    sigmas = [_anisotropic_sigma(pw, aspect, n_dims)]
    sigmas += [(_SIGMA_LOCAL,) * n_dims] * (len(heights) - 2)
    sigmas.append((_SIGMA_BACKGROUND,) * n_dims)

    # Peaks are separated by at least twice the local peak's width, as the paper
    # specifies, so the global maximum is not absorbed into a neighbour. Each
    # peak is also inset by roughly its own width per axis, which keeps the
    # optimum interior — a maximum pinned to a wall is a different search
    # problem, easier along that axis, and would quietly change what N90% means.
    #
    # The inset is per-axis and capped at a quarter of the range, because a
    # strongly anisotropic peak can be wider than the box is generous: at aspect
    # 10 the long axis is 15.8 grids, and insetting by that on both sides would
    # leave nowhere to put it.
    limit = (GRIDS - 1) / 4.0
    min_gap = 2.0 * _SIGMA_LOCAL

    centres: list[tuple[float, ...]] = []
    for sigma in sigmas:
        inset = (
            np.minimum(np.asarray(sigma, dtype=float), limit) if inset_peaks else np.zeros(n_dims)
        )
        lo, hi = inset, (GRIDS - 1) - inset
        for _attempt in range(2000):
            c = tuple(float(v) for v in rng.uniform(lo, hi))
            if all(math.dist(c, other) >= min_gap for other in centres):
                centres.append(c)
                break
        else:  # pragma: no cover — only reachable if the box is over-packed
            raise RuntimeError(
                f"could not place {len(heights)} peaks {min_gap:g} grids apart "
                f"in {n_dims}-D with sigma={sigma}"
            )

    return ModelFunction(
        peaks=tuple(
            Peak(height=h, centre=c, sigma=s)
            for h, c, s in zip(heights, centres, sigmas, strict=True)
        ),
        n_dims=n_dims,
        noise_fraction=noise_fraction,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class CampaignOutcome:
    """What one simulated campaign achieved.

    `trials_to_success` is None when the budget ran out first — which is
    information, not a failure to record, and `n90` needs it to be honest about
    censored runs.
    """

    found: bool
    trials_to_success: int | None
    n_trials: int
    best_clean: float
    optimum: float
    # ARD length-scales from the last fit, one per axis, in engine-normalized
    # units. This is what H3 tests: on an anisotropic window, does their ratio
    # recover the true aspect ratio? If it does, the tool can say *which knob is
    # critical* — a statement about the experiment, not just a faster search.
    # None when the campaign never fitted a model (the random baseline).
    length_scales: tuple[float, ...] | None = None
    true_aspect: float = 1.0


def _snap(point: np.ndarray, visited: set[tuple[int, ...]]) -> tuple[int, ...]:
    """Nearest unvisited lattice point to a continuous proposal.

    The optimizer proposes anywhere in the box; the paper's search is over the
    lattice. Re-proposing an already-measured point would waste a trial and
    silently inflate the trial count, so the nearest *unvisited* point is taken
    instead, spiralling outward one ring at a time.
    """
    base = np.clip(np.rint(point), 0, GRIDS - 1).astype(int)
    key = tuple(int(v) for v in base)
    if key not in visited:
        return key
    d = len(base)
    for radius in range(1, GRIDS):
        offsets = (
            np.array(np.meshgrid(*([np.arange(-radius, radius + 1)] * d), indexing="ij"))
            .reshape(d, -1)
            .T
        )
        ring = offsets[np.max(np.abs(offsets), axis=1) == radius]
        candidates = base + ring
        inside = candidates[np.all((candidates >= 0) & (candidates < GRIDS), axis=1)]
        for cand in inside:
            k = tuple(int(v) for v in cand)
            if k not in visited:
                return k
    raise RuntimeError("lattice exhausted")  # pragma: no cover


def run_grid_campaign(
    model: ModelFunction,
    *,
    n_initial: int = 5,
    budget: int = 100,
    seed: int = 0,
    strategy: str = "bo",
    xi: float = 0.01,
    length_scale_bounds: tuple[float, float] | None = None,
    kernel_ard: bool = True,
    polish: bool = True,
    success_basis: str = "surface",
) -> CampaignOutcome:
    """Simulate one closed loop over the lattice and report when it succeeded.

    Args:
        model: the surface to search.
        n_initial: randomly chosen starting points (5 in the paper).
        budget: maximum trials, including the initial points.
        seed: controls the initial points, the noise draws and the GP restarts.
        strategy: "bo" or "random" — the baseline any claim has to clear.
        xi: EI exploration sweetener; 0.01 is their measured optimum.
        length_scale_bounds: defaults to `LS_BOUNDS_PAPER`, which is their
            0.5-5 grid range expressed in engine units.
        kernel_ard: True fits one length-scale per axis; False forces a single
            shared scale, which is the isotropic-kernel arm of the anisotropy
            study — and also what Xu et al. used.
        polish: refine the proposal continuously before snapping it to the
            lattice. Their acquisition was maximised on the lattice directly, so
            False is the paper-faithful setting; True is what Latos ships.
        success_basis: which reading of "90% of the global maximum" to score
            against — see `_SUCCESS_BASES`. Defaults to the strictest.

    Returns:
        A `CampaignOutcome`.
    """
    if strategy not in ("bo", "random"):
        raise ValueError(f"strategy must be 'bo' or 'random'; got {strategy!r}")

    if success_basis not in _SUCCESS_BASES:
        raise ValueError(f"success_basis must be one of {_SUCCESS_BASES}; got {success_basis!r}")

    rng = _stream(_STREAM_CAMPAIGN, seed)
    global_peak = model.peaks[0]
    target = _SUCCESS_FRACTION * (global_peak.height if success_basis == "peak" else model.optimum)
    centre = np.asarray(global_peak.centre, dtype=float)
    window = float(np.mean(global_peak.sigma))
    bounds = [(0.0, float(GRIDS - 1))] * model.n_dims
    names = tuple(f"x{i}" for i in range(model.n_dims))
    ls_bounds = LS_BOUNDS_PAPER if length_scale_bounds is None else length_scale_bounds

    visited: set[tuple[int, ...]] = set()
    xs: list[tuple[int, ...]] = []
    ys: list[float] = []
    trials_to_success: int | None = None
    length_scales: tuple[float, ...] | None = None

    def record(point: tuple[int, ...]) -> None:
        nonlocal trials_to_success
        visited.add(point)
        xs.append(point)
        arr = np.asarray(point, dtype=float)
        ys.append(float(model(arr, rng)))
        reached = (
            float(np.linalg.norm(arr - centre)) <= window
            if success_basis == "window"
            else float(model.clean(arr)) >= target
        )
        if trials_to_success is None and reached:
            trials_to_success = len(xs)

    while len(xs) < n_initial:
        record(_snap(rng.uniform(0, GRIDS - 1, size=model.n_dims), visited))

    while len(xs) < budget and trials_to_success is None:
        if strategy == "random":
            record(_snap(rng.uniform(0, GRIDS - 1, size=model.n_dims), visited))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_nd(
                np.asarray(xs, dtype=float),
                np.asarray(ys, dtype=float),
                bounds=bounds,
                input_names=names,
                target_name="property",
                xi=xi,
                seed=seed,
                length_scale_bounds=ls_bounds,
                isotropic=not kernel_ard,
                polish=polish,
                with_reliability=False,
            )
        length_scales = result.config.length_scales
        record(_snap(np.asarray(result.recommendation.x, dtype=float), visited))

    best_clean = float(model.clean(np.asarray(xs, dtype=float)).max())
    return CampaignOutcome(
        found=trials_to_success is not None,
        trials_to_success=trials_to_success,
        n_trials=len(xs),
        best_clean=best_clean,
        optimum=model.optimum,
        length_scales=length_scales,
        true_aspect=float(max(global_peak.sigma) / min(global_peak.sigma)),
    )


def fitted_aspect_ratio(
    n_dims: int,
    pw: float,
    aspect: float,
    n_points: int,
    seed: int,
    *,
    length_scale_bounds: tuple[float, float] | None = None,
) -> float:
    """Fit an ARD GP to a space-filling design and report its length-scale ratio.

    This is H3 asked cleanly. Reading the ratio off the end of a *campaign*
    conflates two questions, because a campaign stops as soon as it succeeds —
    often after eight or twenty points — so a null result there could mean either
    "an anisotropic kernel cannot identify an elongated window" or merely "it
    cannot identify one from eight points". Those have very different
    implications: the first says the approach is wrong, the second says the
    sample size is.

    Here the design is fixed and space-filling, so `n_points` is a controlled
    variable and the answer becomes a curve: at what sample size, if any, does
    the fitted ratio track the true one? A materials campaign lives at the small
    end of that curve whether or not the large end works.

    Returns the ratio of largest to smallest fitted length-scale, which is 1.0
    for a model that sees no anisotropy at all.
    """
    model = make_model_function(n_dims, pw, seed=seed, aspect=aspect)
    rng = _stream(_STREAM_CAMPAIGN, seed)
    m = max(1, int(np.ceil(np.log2(max(n_points, 2)))))
    unit = qmc.Sobol(d=n_dims, scramble=True, seed=int(rng.integers(2**31))).random_base2(m)
    design = unit[:n_points] * (GRIDS - 1)
    y = np.asarray([model(row, rng) for row in design], dtype=float)

    bounds = [(0.0, float(GRIDS - 1))] * n_dims
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize_nd(
            design,
            y,
            bounds=bounds,
            input_names=tuple(f"x{i}" for i in range(n_dims)),
            target_name="property",
            length_scale_bounds=length_scale_bounds or LS_BOUNDS_ANISOTROPY,
            with_reliability=False,
            n_candidates=64,  # the recommendation is irrelevant here; only the fit matters
            polish=False,
        )
    scales = result.config.length_scales
    return float(max(scales) / min(scales))


def n90_from_trials(trials: Sequence[int | None]) -> float | None:
    """Xu et al.'s N90% from per-campaign trials-to-success.

    `None` entries are campaigns that never succeeded. They are counted in the
    denominator but cannot supply a value, so when too few succeeded the answer
    is `None` — "more than the budget" — rather than a number obtained by
    pretending censored runs finished exactly at the budget, which would flatter
    the method precisely where it is doing worst.

    Takes bare trial counts rather than outcomes so the rule has one definition
    that survives a process boundary; `n90` is the convenience wrapper.
    """
    if not trials:
        raise ValueError("no outcomes")
    successes = sorted(t for t in trials if t is not None)
    needed = math.ceil(_N90_QUANTILE * len(trials))
    if len(successes) < needed:
        return None
    return float(successes[needed - 1])


def n90(outcomes: Sequence[CampaignOutcome]) -> float | None:
    """Trials needed to succeed in 90% of these campaigns."""
    return n90_from_trials([o.trials_to_success for o in outcomes])
