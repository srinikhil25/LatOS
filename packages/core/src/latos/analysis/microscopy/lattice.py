# N806 (uppercase locals) is OK here: scientific code uses single uppercase
# letters (P, R, N) by convention; renaming to lowercase obscures rather than
# clarifies. RUF002/003 covers the Greek in docstrings (θ, σ, Å).
"""Lattice spacing from high-resolution TEM images, by windowed FFT.

A layered material imaged edge-on shows parallel lattice fringes. Their
periodicity appears in the power spectrum as a pair of symmetric streaks normal
to the fringes, and the spacing follows from the streak radius:

    d = N · (nm per pixel) / r

where N is the window size in pixels and r the radius of the peak in pixels.

Why not simply take the strongest peak
--------------------------------------
Because it is often the wrong one. For a layered (00l) series the reflections
sit at radii in the ratio 1:2:3…, and the second order is frequently *stronger*
than the first — dynamical diffraction and defocus both do this. Taking the
global maximum therefore reports d/2 rather than d, which is a factor-of-two
error that looks entirely plausible in a results table.

The opposite fix is equally wrong. Scoring candidates by how many harmonics
they explain mechanically favours the smallest radius, because at small r the
multiples 2r, 3r, 4r all land inside the search band and something is almost
always there to match them. That drives the answer toward the largest d in
range regardless of the image.

So the fundamental is sought inside a physically motivated window supplied by
the caller, and harmonics are recorded as corroboration only — they never pick
the answer. The window is an assumption and is meant to be checked against an
independent technique (see `latos.analysis.xrd`), not taken on trust.

What makes a detection trustworthy
----------------------------------
Three numbers are reported alongside every spacing rather than being folded
into a single pass/fail:

``contrast``
    Streak power divided by the **azimuthal median at the same radius**. This
    is the discriminator that matters. A radial background alone cannot tell a
    ring (polycrystalline, or an artefact) from a spot (an oriented lattice),
    because both raise the radial average identically. Amorphous carbon and
    shot noise give ≈1; a genuine fringe streak gives tens to thousands.

``n_orders``
    How many members of the harmonic series were found. Two or more is strong
    evidence the fundamental is real rather than an accidental maximum.

``n_repeats``
    Lattice repeats inside the window. The fractional resolution of an FFT peak
    is roughly 1/n_repeats, so **this is the measurement's own uncertainty**,
    and it is why large windows beat small ones here. A window holding eight
    repeats cannot pin d to better than about 12% no matter how clean it looks.
    Counter-intuitively this means a lower-magnification frame can measure a
    given spacing more precisely than a higher-magnification one, because it
    contains more repeats.

Windowing
---------
A Hann window is applied before the transform. Without it the discontinuity at
the tile edge spreads a cross of spectral leakage through the origin, which is
exactly where the low-order reflections of a large-d material sit.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from statistics import median

import numpy as np

__all__ = [
    "DEFAULT_D_WINDOW_NM",
    "DEFAULT_MIN_REPEATS",
    "DEFAULT_SEARCH_WINDOW_NM",
    "FrameSpacing",
    "LatticePeak",
    "SpacingEstimate",
    "aggregate_frames",
    "analyse_tile",
    "iter_tiles",
    "scan_frame",
]

# Bracket for the fundamental. The default spans the basal spacings of MXene and
# MAX phases; callers measuring something else must pass their own.
DEFAULT_D_WINDOW_NM = (0.70, 2.80)

# Peaks are *found* over a much wider range than the fundamental is *chosen*
# from. Keeping the two separate is what makes harmonic corroboration possible
# at all: the second order of a 1.2 nm spacing sits at 0.6 nm, below the
# fundamental window, so a detector that only looked inside that window would
# never see it and every spacing would report a single unsupported order. The
# wide range also lets a strong reflection lying outside the caller's window be
# seen, which is what `min_relative_power` needs in order to reject its leakage.
DEFAULT_SEARCH_WINDOW_NM = (0.15, 3.50)

# Absolute floor on the peak radius. Below this sit the Hann window's own
# transform and any residual illumination gradient, never a lattice.
_DC_GUARD_PX = 4

# Lattice repeats the window must contain before a spacing is believed. Because
# the peak radius in pixels IS the number of repeats, this doubles as the real
# guard against the DC shoulder: a smooth illumination gradient produces a
# local maximum a few bins out that is otherwise indistinguishable from a very
# weak reflection, and it arrives with enormous contrast because the azimuthal
# median it is measured against is near zero there. Six repeats corresponds to
# a 17% resolution limit, which is about the loosest a reported spacing can be
# and still mean anything.
DEFAULT_MIN_REPEATS = 6.0

# Half-width of the angular wedge averaged along the streak direction.
_WEDGE_DEG = 12.0

# Highest harmonic order looked for when corroborating a fundamental.
_MAX_ORDER = 4

# Fractional tolerance when matching an observed peak to a predicted harmonic.
_HARMONIC_TOL = 0.06
_HARMONIC_TOL_FLOOR_PX = 1.5

# Smallest window worth transforming: below this the search band collapses.
_MIN_TILE_PX = 16

# Spectrum pixels the wedge must hold before its profile means anything.
_MIN_WEDGE_PIXELS = 20

# Angular resolution of the direction search, and the smoothing span over it.
_ANGLE_BINS = 180
_ANGLE_SMOOTH = 7

_NDIM_2D = 2


@dataclass(frozen=True, slots=True)
class LatticePeak:
    """One spacing measured from one window.

    Attributes:
        d_nm: The fundamental spacing.
        r_px: Radius of the fundamental in the power spectrum, sub-pixel refined.
        angle_deg: Direction of the streak, 0-180. Fringes run perpendicular
            to this, so it doubles as a flake-orientation readout.
        contrast: Peak power over the azimuthal median at the same radius.
        n_orders: Harmonic orders observed, including the fundamental.
        n_repeats: Lattice repeats inside the window; ~1/n_repeats is the
            fractional resolution of the measurement.
        d_err_nm: ``d_nm / n_repeats`` — the FFT resolution limit expressed as
            an absolute uncertainty. This is a floor, not a full error budget:
            it excludes microscope calibration error, which on a routine
            instrument is a few percent and is systematic across all frames.
        all_d_nm: Every peak found along the streak, largest d first. Kept so a
            caller can audit what the fundamental was chosen from.
    """

    d_nm: float
    r_px: float
    angle_deg: float
    contrast: float
    n_orders: int
    n_repeats: float
    d_err_nm: float
    all_d_nm: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FrameSpacing:
    """One frame's spacing, pooled over the windows that yielded a detection."""

    key: str
    d_nm: float
    n_tiles: int


@dataclass(frozen=True, slots=True)
class SpacingEstimate:
    """A group's spacing, pooled over frames.

    ``spread`` and the quartiles describe variation **across frames**, which is
    the quantity that reflects real sample heterogeneity. Pooling raw windows
    instead would let one densely tiled frame dominate a group, since windows
    from the same frame largely re-measure the same flake.
    """

    d_nm: float
    spread_nm: float
    q1_nm: float
    q3_nm: float
    n_frames: int
    n_tiles: int


def _power_spectrum(tile: np.ndarray) -> np.ndarray:
    t = np.asarray(tile, dtype=np.float64)
    t = t - t.mean()
    window = np.hanning(t.shape[0])[:, None] * np.hanning(t.shape[1])[None, :]
    return np.fft.fftshift(np.abs(np.fft.fft2(t * window)) ** 2)


def _azimuthal_median(power: np.ndarray, radius_int: np.ndarray) -> np.ndarray:
    """Median power at each integer radius — the isotropic reference."""
    counts = np.bincount(radius_int.ravel())
    order = np.argsort(radius_int.ravel(), kind="stable")
    values = power.ravel()[order]
    out = np.zeros(len(counts))
    start = 0
    for k, count in enumerate(counts):
        if count:
            out[k] = np.median(values[start : start + count])
        start += count
    return out


def _local_maxima(profile: np.ndarray, lo: int, hi: int, floor: float) -> list[tuple[float, float]]:
    """Interior maxima of a 1-D profile, refined to sub-pixel by parabola.

    The scan starts strictly ABOVE `lo`, never on it. When `lo` is the DC guard
    radius, a maximum sitting exactly on it is the shoulder of the Hann
    window's own transform, or of a residual illumination gradient - precisely
    what the guard exists to exclude. Admitting it yields a large, confident and
    entirely spurious spacing, because a radius that close to the origin maps to
    a d near the top of any plausible window.
    """
    peaks: list[tuple[float, float]] = []
    for i in range(max(lo + 1, 1), min(hi, len(profile) - 1)):
        a, b, c = profile[i - 1], profile[i], profile[i + 1]
        if b >= a and b > c and b >= floor:
            denom = a - 2.0 * b + c
            shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
            peaks.append((i + float(np.clip(shift, -1.0, 1.0)), float(b)))
    return peaks


def _checked_window(window: tuple[float, float], name: str) -> tuple[float, float]:
    lo, hi = window
    if not (0 < lo < hi):
        raise ValueError(f"{name} must be an increasing positive pair, got {window!r}")
    return float(lo), float(hi)


def _polar_maps(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radius, integer radius, and angle of every spectrum pixel.

    Angles run 0-180 rather than 0-360: a real image has a centrosymmetric
    power spectrum, so the two lobes of a streak carry identical information
    and folding them together doubles the statistics behind each direction.
    """
    yy, xx = np.indices((n, n))
    centre = n // 2
    dy, dx = yy - centre, xx - centre
    radius = np.hypot(dy, dx)
    angle = np.degrees(np.arctan2(dy, dx)) % 180.0
    return radius, radius.astype(int), angle


def _streak_direction(excess: np.ndarray, angle: np.ndarray, band: np.ndarray) -> float:
    """Direction, in degrees, of the strongest anisotropy in the search band.

    Smoothing wraps around, so a streak straddling the 0/180 boundary is not
    split between the first and last bins and thereby lost.
    """
    bins = np.clip(angle[band].astype(int), 0, _ANGLE_BINS - 1)
    weight = np.bincount(bins, excess[band], minlength=_ANGLE_BINS)
    wrapped = np.concatenate([weight, weight, weight])
    kernel = np.ones(_ANGLE_SMOOTH) / _ANGLE_SMOOTH
    smoothed = np.convolve(wrapped, kernel, mode="same")[_ANGLE_BINS : 2 * _ANGLE_BINS]
    return float(np.argmax(smoothed))


def _wedge_profiles(
    excess: np.ndarray,
    power: np.ndarray,
    *,
    radius_int: np.ndarray,
    angle: np.ndarray,
    theta: float,
    band: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Radial profiles along the streak: contrast, and raw power.

    Both are needed. Contrast finds peaks against the isotropic background, but
    it is normalised per radius, so a leakage sidelobe sitting on an empty
    background scores as highly as the parent peak it leaked from. Raw power
    keeps the difference in magnitude that separates the two.
    """
    separation = np.minimum(np.abs(angle - theta), 180.0 - np.abs(angle - theta))
    wedge = (separation <= _WEDGE_DEG) & band
    if wedge.sum() < _MIN_WEDGE_PIXELS:
        return None
    bins = radius_int[wedge]
    counts = np.maximum(np.bincount(bins, minlength=n_bins), 1)
    contrast = np.bincount(bins, excess[wedge], minlength=n_bins) / counts
    raw = np.bincount(bins, power[wedge], minlength=n_bins) / counts
    return contrast, raw


def analyse_tile(  # noqa: PLR0911
    tile: np.ndarray,
    nm_per_px: float,
    *,
    d_window_nm: tuple[float, float] = DEFAULT_D_WINDOW_NM,
    search_window_nm: tuple[float, float] = DEFAULT_SEARCH_WINDOW_NM,
    min_contrast: float = 8.0,
    min_relative_power: float = 1e-3,
    min_repeats: float = DEFAULT_MIN_REPEATS,
) -> LatticePeak | None:
    """Measure the lattice spacing in a single square window.

    Args:
        tile: Square 2-D array of image intensities.
        nm_per_px: Calibration, from the instrument's field of view divided by
            the image width in pixels.
        d_window_nm: ``(min, max)`` bracket the fundamental must fall inside.
            Choose it to bracket the expected spacing. A window spanning more
            than a factor of two can hold both a fundamental and its second
            order, and if the second order is the stronger it will be the one
            reported; `all_d_nm` is there so that case can be audited.
        search_window_nm: Wider ``(min, max)`` over which peaks are detected.
            Harmonics are counted here, and a reflection outside `d_window_nm`
            is seen here so its leakage can be rejected.
        min_contrast: Floor on streak power over the azimuthal median at the
            same radius. Peaks below this are not considered at all.
        min_relative_power: Floor on the fundamental's raw power as a fraction
            of the strongest peak anywhere in `search_window_nm`. This rejects
            spectral leakage: when the real periodicity lies outside
            `d_window_nm`, its sidelobes still fall inside, and on a clean
            image they can clear `min_contrast` easily because the azimuthal
            median they are measured against is near zero. Sidelobes sit far
            below the parent peak, so a relative-power floor removes them
            while leaving a genuinely weak fundamental intact.
        min_repeats: Lattice repeats the window must contain. This is the
            guard against the DC shoulder of a smooth illumination gradient,
            which otherwise reports as a large, very high-contrast spacing.
            It is also the honest lower bound on the measurement: fewer
            repeats than this and the FFT cannot resolve the peak well enough
            for the answer to mean anything.

    Returns:
        A `LatticePeak`, or None when the window holds no periodic signal in
        the requested range. None is the expected outcome for a large fraction
        of windows in a real dataset: most of any TEM frame is support film,
        vacuum, or a flake in the wrong orientation.

    Raises:
        ValueError: If the tile is not square 2-D, or `nm_per_px` is not
            positive, or either window's bounds are not ordered and positive.
    """
    tile = np.asarray(tile)
    if tile.ndim != _NDIM_2D or tile.shape[0] != tile.shape[1]:
        raise ValueError(f"tile must be square 2-D, got shape {tile.shape}")
    if not np.isfinite(nm_per_px) or nm_per_px <= 0:
        raise ValueError(f"nm_per_px must be positive and finite, got {nm_per_px!r}")
    d_lo, d_hi = _checked_window(d_window_nm, "d_window_nm")
    s_lo, s_hi = _checked_window(search_window_nm, "search_window_nm")
    # The search range must contain the window, or the fundamental could be
    # selected from peaks that were never looked for.
    s_lo, s_hi = min(s_lo, d_lo), max(s_hi, d_hi)

    n = tile.shape[0]
    if n < _MIN_TILE_PX:
        return None

    power = _power_spectrum(tile)
    radius, radius_int, angle = _polar_maps(n)
    isotropic = _azimuthal_median(power, radius_int)
    excess = power / np.maximum(isotropic[np.clip(radius_int, 0, len(isotropic) - 1)], 1e-12)

    scale = n * nm_per_px
    r_min = max(float(_DC_GUARD_PX), float(min_repeats), scale / s_hi)
    r_max = min(n / 2.0 - 2.0, scale / s_lo)
    if r_max <= r_min + 2.0:
        return None
    band = (radius >= r_min) & (radius <= r_max)
    if not band.any():
        return None

    theta = _streak_direction(excess, angle, band)
    profiles = _wedge_profiles(
        excess,
        power,
        radius_int=radius_int,
        angle=angle,
        theta=theta,
        band=band,
        n_bins=len(isotropic),
    )
    if profiles is None:
        return None
    profile, raw = profiles

    peaks = _local_maxima(profile, int(r_min), int(r_max), min_contrast)
    if not peaks:
        return None
    in_window = [(r, amp) for r, amp in peaks if d_lo <= scale / r <= d_hi]
    if not in_window:
        return None
    r_fund, amplitude = max(in_window, key=lambda p: p[1])

    strongest = max(raw[round(r)] for r, _ in peaks)
    if strongest > 0 and raw[round(r_fund)] / strongest < min_relative_power:
        return None

    orders = 1
    for m in range(2, _MAX_ORDER + 1):
        target = r_fund * m
        if target > r_max:
            break
        tol = max(_HARMONIC_TOL_FLOOR_PX, _HARMONIC_TOL * target)
        if any(abs(r - target) <= tol for r, _ in peaks):
            orders += 1

    d = scale / r_fund
    repeats = scale / d
    return LatticePeak(
        d_nm=d,
        r_px=r_fund,
        angle_deg=theta,
        contrast=float(amplitude),
        n_orders=orders,
        n_repeats=repeats,
        d_err_nm=d / max(repeats, 1.0),
        all_d_nm=tuple(sorted((scale / r for r, _ in peaks), reverse=True)),
    )


def iter_tiles(
    image: np.ndarray,
    size: int,
    stride: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(row, col, tile)`` square windows over an image.

    The window is clipped to the image when the image is smaller than `size`,
    so a small frame still yields one full-frame window rather than nothing.

    Raises:
        ValueError: If `size` or `stride` is not positive.
    """
    if size <= 0 or stride <= 0:
        raise ValueError(f"size and stride must be positive, got {size!r}, {stride!r}")
    image = np.asarray(image)
    if image.ndim != _NDIM_2D:
        raise ValueError(f"image must be 2-D, got shape {image.shape}")
    height, width = image.shape
    n = min(size, height, width)
    for row in range(0, max(1, height - n + 1), stride):
        for col in range(0, max(1, width - n + 1), stride):
            yield row, col, image[row : row + n, col : col + n]


def scan_frame(
    image: np.ndarray,
    nm_per_px: float,
    *,
    tile_size: int = 1024,
    stride: int | None = None,
    d_window_nm: tuple[float, float] = DEFAULT_D_WINDOW_NM,
    search_window_nm: tuple[float, float] = DEFAULT_SEARCH_WINDOW_NM,
    min_contrast: float = 8.0,
    min_relative_power: float = 1e-3,
    min_repeats: float = DEFAULT_MIN_REPEATS,
) -> tuple[LatticePeak, ...]:
    """Run `analyse_tile` over every window of one frame.

    Scanning rather than measuring the centre is not an optimisation — it is
    required. Crystalline regions occupy a small and unpredictable part of a
    typical frame, so a single central window misses most of the signal in the
    dataset and, worse, misses it non-randomly.
    """
    if stride is None:
        stride = max(1, tile_size // 2)
    found = []
    for _row, _col, tile in iter_tiles(image, tile_size, stride):
        peak = analyse_tile(
            tile,
            nm_per_px,
            d_window_nm=d_window_nm,
            search_window_nm=search_window_nm,
            min_contrast=min_contrast,
            min_relative_power=min_relative_power,
            min_repeats=min_repeats,
        )
        if peak is not None:
            found.append(peak)
    return tuple(found)


def aggregate_frames(
    detections: dict[str, list[float]],
) -> SpacingEstimate | None:
    """Pool per-window spacings into one estimate, frame by frame.

    Args:
        detections: ``{frame_key: [d_nm, ...]}`` — the spacings accepted from
            each frame. Frames with no detections may be present with an empty
            list, or omitted; both are treated the same.

    Returns:
        A `SpacingEstimate`, or None if no frame contributed a detection.

    The two-stage median (within frame, then across frames) is what keeps a
    single heavily-tiled frame from dominating. The reported spread is across
    frames, so it describes sample variation rather than the repeatability of
    re-measuring one flake.
    """
    per_frame = [median(values) for values in detections.values() if values]
    if not per_frame:
        return None
    array = np.asarray(per_frame, dtype=float)
    centre = float(np.median(array))
    return SpacingEstimate(
        d_nm=centre,
        spread_nm=float(np.median(np.abs(array - centre))),
        q1_nm=float(np.percentile(array, 25)),
        q3_nm=float(np.percentile(array, 75)),
        n_frames=len(per_frame),
        n_tiles=sum(len(v) for v in detections.values()),
    )
