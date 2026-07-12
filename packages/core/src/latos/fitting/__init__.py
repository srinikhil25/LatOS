"""Universal peak-fit engine — Latos's "replace Origin" fitting core.

`fit_spectrum(x, y, FitSpec(...))` fits peaks (a shared `PeakShape`) over a
chosen background and returns a `FitResult` with parameters ± uncertainty,
the component decomposition, the background, the residual, and goodness of
fit. Per-technique presets and an interactive editor build on this by
assembling a `FitSpec`; the background library (`backgrounds`) is reusable
on its own.
"""

from __future__ import annotations

from latos.fitting.engine import (
    BackgroundKind,
    BackgroundSpec,
    FitError,
    FitResult,
    FitSpec,
    FittedComponent,
    PeakInit,
    compute_baseline,
    fit_spectrum,
)
from latos.fitting.peak_shapes import PeakShape, peak_model

__all__ = [
    "BackgroundKind",
    "BackgroundSpec",
    "FitError",
    "FitResult",
    "FitSpec",
    "FittedComponent",
    "PeakInit",
    "PeakShape",
    "compute_baseline",
    "fit_spectrum",
    "peak_model",
]
