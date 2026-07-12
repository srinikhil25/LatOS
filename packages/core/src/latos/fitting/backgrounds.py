"""Background / baseline models for the universal fit engine.

A spectrum is *peaks on a background*. Getting the background right is half
the battle in XPS (Shirley/Tougaard) and Raman (ALS), and lmfit ships none
of these — so they live here as pure ``(x, y) -> baseline`` functions that
the engine subtracts before fitting peaks. Each returns a baseline array the
same length as ``y``.

Backgrounds provided:

* **constant** — the minimum level; the crudest offset removal.
* **linear** — a straight line between the two endpoints.
* **polynomial** — a least-squares polynomial of a given degree.
* **Shirley** — the iterative XPS background whose height at each point is
  proportional to the integrated photoemission intensity above it
  [Shirley 1972; Proctor & Sherwood 1982]. The physics: inelastically
  scattered electrons pile up on the high-binding-energy side of a peak.
* **ALS** (Asymmetric Least Squares) — a smooth baseline that hugs the
  valleys, ignoring positive peaks; the standard for fluorescent Raman
  backgrounds [Eilers & Boelens 2005].

References:
- Shirley D.A. (1972). *Phys. Rev. B* 5, 4709.
- Proctor A., Sherwood P.M.A. (1982). *Anal. Chem.* 54, 13.
- Eilers P.H.C., Boelens H.F.M. (2005). *Baseline correction with
  asymmetric least squares smoothing.* Leiden Univ. Medical Centre report.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

__all__ = [
    "als_baseline",
    "constant_baseline",
    "linear_baseline",
    "polynomial_baseline",
    "shirley_baseline",
]

# A baseline needs at least two points to have two anchors.
_MIN_BASELINE_POINTS = 2


def constant_baseline(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """A flat baseline at the minimum of `y`."""
    y = np.asarray(y, dtype=float)
    return np.full_like(y, float(np.min(y)))


def linear_baseline(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """A straight line through the first and last (x, y) points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x[-1] == x[0]:
        return np.full_like(y, float(y[0]))
    slope = (y[-1] - y[0]) / (x[-1] - x[0])
    return np.asarray(y[0] + slope * (x - x[0]), dtype=float)


def polynomial_baseline(
    x: NDArray[np.float64], y: NDArray[np.float64], degree: int = 2
) -> NDArray[np.float64]:
    """A least-squares polynomial baseline of the given degree.

    Fits the whole trace, so it is only a sensible background when peaks
    are sparse; for dense spectra prefer Shirley or ALS.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs = np.polyfit(x, y, degree)
    return np.asarray(np.polyval(coeffs, x), dtype=float)


def shirley_baseline(
    y: NDArray[np.float64], *, max_iter: int = 100, tol: float = 1e-6
) -> NDArray[np.float64]:
    """Iterative Shirley background for an XPS region.

    The baseline is anchored to the endpoint intensities ``y[0]`` and
    ``y[-1]`` and rises across the peak in proportion to the integrated
    peak area still to the high-index side — the classic Shirley
    construction. Pass ``y`` ordered so the endpoints sit on genuine
    background (the usual case for an isolated region). Converges when the
    baseline stops changing to within `tol` (relative to the endpoint
    step); falls back to the last iterate at `max_iter`.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < _MIN_BASELINE_POINTS:
        return np.zeros_like(y)
    y_lo, y_hi = float(y[0]), float(y[-1])
    step = abs(y_hi - y_lo) or 1.0
    background = np.full(n, min(y_lo, y_hi))
    for _ in range(max_iter):
        # Cumulative area of (y - background) from each point to the end.
        # Clamp the integrand to ≥ 0: it keeps the area monotonic (hence the
        # background monotonic between anchors) and stops the baseline
        # overshooting the data where a prior iterate dipped above it.
        area_above = np.cumsum(np.clip(y - background, 0.0, None)[::-1])[::-1]
        total = area_above[0]
        if total <= 0:
            break
        updated = y_hi + (y_lo - y_hi) * area_above / total
        if np.max(np.abs(updated - background)) < tol * step:
            background = updated
            break
        background = updated
    return background


def als_baseline(
    y: NDArray[np.float64],
    *,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10,
) -> NDArray[np.float64]:
    """Asymmetric Least Squares baseline (Eilers & Boelens 2005).

    Balances smoothness (``lam`` — larger is stiffer) against fidelity,
    weighting points below the current baseline by ``p`` and points above
    (i.e. peaks) by ``1 - p`` so the baseline is pulled toward the valleys.
    ``n_iter`` reweighting passes. Good for broad fluorescent Raman humps
    where no endpoint anchoring is available.
    """
    y = np.asarray(y, dtype=float)
    length = y.size
    # Second-difference operator D; lam·DᵀD is the smoothness penalty.
    diff = sparse.eye(length, format="csc")
    diff = diff[1:] - diff[:-1]
    diff = diff[1:] - diff[:-1]
    penalty = lam * (diff.T @ diff)
    weights = np.ones(length)
    baseline = y.copy()
    for _ in range(n_iter):
        weight_mat = sparse.diags(weights, 0, shape=(length, length), format="csc")
        baseline = spsolve(weight_mat + penalty, weights * y)
        weights = p * (y > baseline) + (1.0 - p) * (y < baseline)
    return np.asarray(baseline, dtype=float)
