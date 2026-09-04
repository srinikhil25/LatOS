"""Automatic peak detection — seed a fit without hand-placing every peak.

Given a raw ``(x, y)`` spectrum, return ranked candidate peak positions the
user can accept, prune, or extend before fitting. The approach is the
noise-aware prominence detection the XRD analyzer already proved on real
scans: smooth for *detection only* (never for fitting — smoothing biases
amplitudes), estimate the noise floor robustly from the median absolute
deviation, and keep peaks whose prominence clears both a fixed fraction of
the signal range and a multiple of that noise floor.

``scipy.signal.find_peaks`` (prominence + separation) is used rather than
the multi-scale ``find_peaks_cwt`` — it is far more robust on real, unevenly
broadened spectra and needs no wavelet-width guessing.

References:
- Rousseeuw P.J., Croux C. (1993). *J. Am. Stat. Assoc.* 88, 1273 (MAD→σ).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter

__all__ = [
    "PeakCandidate",
    "detect_peaks",
    "detect_peaks_detailed",
    "measure_widths",
]

# MAD→σ consistency constant under Gaussian noise.
_MAD_TO_SIGMA = 1.4826
# Prominence floor as a fraction of the signal peak-to-peak range.
_DEFAULT_PROMINENCE_FRAC = 0.02
# Noise-aware prominence: this many robust σ above the noise floor.
_NOISE_SIGMA_MULT = 3.0
# Savitzky-Golay smoothing window (samples) and polyorder for detection.
_SMOOTH_WINDOW = 7
_SMOOTH_POLYORDER = 2

# How far either side of a peak prominence may look, as a multiple of the
# minimum peak separation. Bounding this is what makes a width usable: with an
# unbounded window a sharp peak riding on a broad amorphous hump takes its
# prominence from the far side of the hump and measures tens of degrees wide,
# which is a property of the background, not of the peak.
_PROMINENCE_WLEN_MULT = 4


@dataclass(frozen=True, slots=True)
class PeakCandidate:
    """One detected peak: where it is, how wide, and how far it stands out.

    `width` and `prominence` are measured on the smoothed trace with a bounded
    prominence window, so both describe the peak rather than the ground beneath
    it. `width` is a FWHM in the units of `x`.
    """

    center: float
    width: float
    prominence: float


def measure_widths(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    centers: Sequence[float],
    *,
    min_distance_frac: float = 0.01,
) -> list[float]:
    """FWHM at each given center, in the units of `x`; 0.0 where unmeasurable.

    The same bounded-prominence measurement `detect_peaks_detailed` uses, but
    at positions the caller already has — a fit seeded by hand, or by an earlier
    detection whose widths were not kept. Bounding matters as much here: without
    it a sharp reflection on a broad hump measures as wide as the hump.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < _SMOOTH_WINDOW or x.size != n or not len(centers):
        return [0.0] * len(centers)

    window = min(_SMOOTH_WINDOW, n if n % 2 else n - 1)
    smoothed = savgol_filter(y, window, min(_SMOOTH_POLYORDER, window - 1))
    indices = np.clip(np.searchsorted(x, np.asarray(centers, dtype=float)), 0, n - 1).astype(int)

    distance = max(1, int(min_distance_frac * n))
    wlen = max(5, _PROMINENCE_WLEN_MULT * distance) | 1
    with warnings.catch_warnings():
        # A position that is not a local maximum has zero prominence, which
        # scipy warns about. That is a legitimate answer here — the caller may
        # have placed a peak by hand on a shoulder — and 0.0 is what we return.
        warnings.simplefilter("ignore")
        try:
            bounded = peak_prominences(smoothed, indices, wlen=wlen)
            widths, *_ = peak_widths(smoothed, indices, rel_height=0.5, prominence_data=bounded)
        except (ValueError, IndexError):
            return [0.0] * len(centers)

    step = abs(float(np.median(np.diff(x)))) if n > 1 else 1.0
    return [float(w) * step if np.isfinite(w) else 0.0 for w in widths]


def detect_peaks(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    max_peaks: int = 30,
    min_prominence_frac: float = _DEFAULT_PROMINENCE_FRAC,
    min_distance_frac: float = 0.01,
) -> list[float]:
    """Candidate peak centers (x-values), strongest first.

    A thin view over `detect_peaks_detailed` for callers that only need
    positions. Anything that goes on to FIT these peaks should prefer the
    detailed form: the width measured here is what lets the fitter group
    peaks that overlap and separate ones that do not.
    """
    return [
        c.center
        for c in detect_peaks_detailed(
            x,
            y,
            max_peaks=max_peaks,
            min_prominence_frac=min_prominence_frac,
            min_distance_frac=min_distance_frac,
        )
    ]


def detect_peaks_detailed(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    max_peaks: int = 30,
    min_prominence_frac: float = _DEFAULT_PROMINENCE_FRAC,
    min_distance_frac: float = 0.01,
) -> list[PeakCandidate]:
    """Candidate peaks with their widths and prominences, strongest first.

    Args:
        x: the spectrum's abscissa (need not be evenly spaced).
        y: the spectrum's intensity, same length as x.
        max_peaks: cap on how many candidates to return.
        min_prominence_frac: prominence floor as a fraction of y's range.
        min_distance_frac: minimum peak separation as a fraction of the
            number of samples (dedupes the same peak found twice).

    Returns an empty list if the trace is too short or flat.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < _SMOOTH_WINDOW or x.size != n:
        return []

    y_range = float(np.ptp(y))
    if y_range <= 0:
        return []

    # Smooth for detection only.
    window = min(_SMOOTH_WINDOW, n if n % 2 else n - 1)
    smoothed = savgol_filter(y, window, min(_SMOOTH_POLYORDER, window - 1))

    # Robust noise floor from the MAD of the detrended residual.
    residual = y - smoothed
    mad = float(np.median(np.abs(residual - np.median(residual))))
    noise_sigma = _MAD_TO_SIGMA * mad
    prominence = max(min_prominence_frac * y_range, _NOISE_SIGMA_MULT * noise_sigma)
    distance = max(1, int(min_distance_frac * n))

    indices, props = find_peaks(smoothed, prominence=prominence, distance=distance)
    if indices.size == 0:
        return []

    # Rank by prominence, strongest first, then cap.
    order = np.argsort(props["prominences"])[::-1][:max_peaks]
    kept = indices[order]

    # Widths, measured with a BOUNDED prominence window. scipy's default looks
    # outward until the signal rises again, which on a spectrum with broad
    # structure means the far side of that structure — so a 0.15 deg reflection
    # comes back tens of degrees wide. `wlen` confines it to the peak's own
    # neighbourhood.
    wlen = max(5, _PROMINENCE_WLEN_MULT * distance) | 1
    bounded = peak_prominences(smoothed, kept, wlen=wlen)
    widths_samples, *_ = peak_widths(smoothed, kept, rel_height=0.5, prominence_data=bounded)
    step = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    return [
        PeakCandidate(
            center=float(x[i]),
            width=float(w) * abs(step),
            prominence=float(p),
        )
        for i, w, p in zip(kept, widths_samples, bounded[0], strict=True)
    ]
