"""Peak line-shape vocabulary for the universal fit engine.

A thin, typed façade over lmfit's built-in models so the rest of Latos
(and, later, an interactive fit editor) refers to peak shapes by a stable
enum rather than importing lmfit model classes directly. Every shape below
exposes the same core parameters — ``amplitude`` (area), ``center``,
``sigma`` — plus lmfit's derived ``fwhm`` and ``height``, so result
extraction is uniform across shapes.

Shapes and when to reach for them:

* **Gaussian** — instrument-broadening-dominated peaks (idealised).
* **Lorentzian** — lifetime-broadening-dominated peaks.
* **Voigt** — a true Gaussian⊗Lorentzian convolution (physically correct
  for most spectroscopy; ``gamma`` is the Lorentzian half-width).
* **Pseudo-Voigt** — a cheap Gaussian/Lorentzian linear mix (``fraction``);
  the workhorse for XRD.
* **Doniach–Šunjić** — asymmetric core-level line for *metallic* XPS
  (``gamma`` = asymmetry) [Doniach & Šunjić 1970].
* **Skewed Voigt** — a generic asymmetric Voigt (``skew``) for tails that
  aren't captured by a symmetric shape.

References:
- Newville M. et al. (2014). *LMFIT: Non-Linear Least-Square Minimization
  and Curve-Fitting for Python.* Zenodo. doi:10.5281/zenodo.11813.
- Doniach S., Šunjić M. (1970). *J. Phys. C* 3, 285.
"""

from __future__ import annotations

from enum import StrEnum

from lmfit.model import Model
from lmfit.models import (
    DoniachModel,
    GaussianModel,
    LorentzianModel,
    PseudoVoigtModel,
    SkewedVoigtModel,
    VoigtModel,
)

__all__ = ["PeakShape", "peak_model"]


class PeakShape(StrEnum):
    """A peak line shape, resolvable to an lmfit model via `peak_model`."""

    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"
    PSEUDO_VOIGT = "pseudo_voigt"
    DONIACH = "doniach"  # Doniach–Šunjić — asymmetric metallic XPS core levels
    SKEWED_VOIGT = "skewed_voigt"  # generic asymmetric Voigt


_MODEL_FACTORIES: dict[PeakShape, type[Model]] = {
    PeakShape.GAUSSIAN: GaussianModel,
    PeakShape.LORENTZIAN: LorentzianModel,
    PeakShape.VOIGT: VoigtModel,
    PeakShape.PSEUDO_VOIGT: PseudoVoigtModel,
    PeakShape.DONIACH: DoniachModel,
    PeakShape.SKEWED_VOIGT: SkewedVoigtModel,
}


def peak_model(shape: PeakShape, prefix: str) -> Model:
    """An lmfit model for one peak of `shape`, namespaced by `prefix`.

    `prefix` (e.g. ``"p0_"``) keeps each peak's parameters distinct in a
    composite (multi-peak) model. All returned models share the
    ``amplitude`` / ``center`` / ``sigma`` parameter names and provide
    derived ``fwhm`` / ``height`` parameters.
    """
    return _MODEL_FACTORIES[shape](prefix=prefix)
