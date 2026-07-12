"""Per-technique fit presets — sensible `FitSpec`s so callers don't wire lmfit.

Each preset encodes the community-standard line shape, background, and
constraints for a technique, given only the peak positions the user (or an
auto-detector) supplies. They return a `FitSpec` the engine runs directly.

* **XRD** — pseudo-Voigt peaks on a polynomial background (the Rietveld/
  profile-fitting workhorse); peaks independent.
* **XPS core-level doublet** — a spin-orbit pair on a Shirley background
  with the splitting, area ratio, and width tied by physics (see
  `constraints`). Common splittings are tabulated in `XPS_DOUBLETS`.
* **Raman** — Lorentzian peaks on an asymmetric-least-squares background
  (removes the fluorescent hump); peaks independent.

References:
- Thompson P. et al. (1987). *J. Appl. Cryst.* 20, 79 (pseudo-Voigt).
- Moulder J.F. et al. (1992). *Handbook of X-ray Photoelectron
  Spectroscopy* (spin-orbit splittings & ratios).
"""

from __future__ import annotations

from latos.fitting.constraints import FixedDelta, FixedRatio, SharedWidth
from latos.fitting.engine import BackgroundKind, BackgroundSpec, FitSpec, PeakInit
from latos.fitting.peak_shapes import PeakShape

__all__ = ["XPS_DOUBLETS", "raman_preset", "xps_doublet_preset", "xrd_preset"]

# Common XPS spin-orbit doublets: name -> (ΔBE in eV, area ratio
# lower/higher-j). p: 2:1 → 0.5; d: 3:2 → 0.667; f: 4:3 → 0.75.
XPS_DOUBLETS: dict[str, tuple[float, float]] = {
    "Cu 2p": (19.8, 0.5),
    "Se 3d": (0.86, 0.667),
    "Bi 4f": (5.31, 0.75),
    "I 3d": (11.5, 0.667),
    "generic p": (0.0, 0.5),
    "generic d": (0.0, 0.667),
    "generic f": (0.0, 0.75),
}

# Polynomial degree for the XRD background — cubic follows gentle
# amorphous humps without chasing Bragg peaks.
_XRD_BACKGROUND_DEGREE = 3


def xrd_preset(peak_centers: list[float]) -> FitSpec:
    """Pseudo-Voigt peaks on a cubic-polynomial background (XRD profile fit)."""
    return FitSpec(
        peak_shape=PeakShape.PSEUDO_VOIGT,
        peaks=[PeakInit(center=c) for c in peak_centers],
        background=BackgroundSpec(kind=BackgroundKind.POLYNOMIAL, degree=_XRD_BACKGROUND_DEGREE),
    )


def xps_doublet_preset(
    center_high_j: float,
    *,
    delta_be: float,
    area_ratio: float = 0.5,
    shape: PeakShape = PeakShape.PSEUDO_VOIGT,
) -> FitSpec:
    """A spin-orbit doublet on a Shirley background, tied by physics.

    `center_high_j` is the binding energy of the higher-degeneracy
    component (e.g. 2p₃/₂); the lower-j component sits at
    ``center_high_j + delta_be`` with ``area_ratio`` of its area and a
    shared width. Look splittings up in `XPS_DOUBLETS`.
    """
    return FitSpec(
        peak_shape=shape,
        peaks=[PeakInit(center=center_high_j), PeakInit(center=center_high_j + delta_be)],
        background=BackgroundSpec(kind=BackgroundKind.SHIRLEY),
        constraints=[
            FixedDelta(ref=0, target=1, delta=delta_be),
            FixedRatio(ref=0, target=1, ratio=area_ratio),
            SharedWidth(ref=0, target=1),
        ],
    )


def raman_preset(peak_centers: list[float]) -> FitSpec:
    """Lorentzian peaks on an ALS background (Raman, fluorescence removed)."""
    return FitSpec(
        peak_shape=PeakShape.LORENTZIAN,
        peaks=[PeakInit(center=c) for c in peak_centers],
        background=BackgroundSpec(kind=BackgroundKind.ALS),
    )
