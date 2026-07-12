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

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, savgol_filter

__all__ = ["detect_peaks"]

# MAD→σ consistency constant under Gaussian noise.
_MAD_TO_SIGMA = 1.4826
# Prominence floor as a fraction of the signal peak-to-peak range.
_DEFAULT_PROMINENCE_FRAC = 0.02
# Noise-aware prominence: this many robust σ above the noise floor.
_NOISE_SIGMA_MULT = 3.0
# Savitzky-Golay smoothing window (samples) and polyorder for detection.
_SMOOTH_WINDOW = 7
_SMOOTH_POLYORDER = 2


def detect_peaks(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    max_peaks: int = 30,
    min_prominence_frac: float = _DEFAULT_PROMINENCE_FRAC,
    min_distance_frac: float = 0.01,
) -> list[float]:
    """Candidate peak centers (x-values), strongest first.

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
    return [float(x[i]) for i in indices[order]]
