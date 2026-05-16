# ruff: noqa: N806, RUF002
# N806: scientific code uses single uppercase letters (E, Y, F, R) by convention.
# RUF002: Greek letters in docstrings are intentional physics notation.
"""UV-DRS Tauc-plot band gap analyzer.

The Tauc method extracts a semiconductor's optical band gap from a
diffuse-reflectance measurement. Three steps:

1. **Kubelka-Munk transform.** Diffuse-reflectance R (fraction in
   [0, 1]) is converted to the Kubelka-Munk function

       F(R) = (1 - R)^2 / (2 R)

   which, for a thick, weakly-absorbing scatterer, is proportional to
   the absorption coefficient α. This is what makes diffuse
   reflectance interpretable as absorption.

2. **Tauc transform.** Plot `(F(R) · E)^n` vs photon energy E, where

       E[eV] = 1240 / λ[nm]    (hc / λ; the 1240 nm·eV constant)

   and `n = 2` for a direct allowed transition or `n = 1/2` for an
   indirect allowed transition. The linear region above the absorption
   edge, extrapolated back to the x-axis, gives the band gap Eg.

3. **Linear fit + extrapolation.** Pick a fit window on the rising
   edge (default: data points whose y-value lies between 20% and 60%
   of the maximum). Least-squares fit `y = m·E + b`. The band gap is
   the x-intercept `Eg = -b / m`.

This analyzer is deliberately *literate* — every step is named so a
student reading the code can map it back to the materials-science
textbook. Real Stage 4+ work may layer on automatic-edge-detection
heuristics; the manual edge-window override is the escape hatch.

Inputs expected on the measurement:
- `arrays["wavelength"]` in nm
- `arrays["reflectance"]` either as a fraction in [0, 1] or as a
  percentage in [0, 100]. The analyzer detects which by looking at
  the data range and converts to fraction internally.

Output payload:
- `band_gap_ev` (float): the extracted band gap
- `slope`, `intercept` (floats): the linear-fit parameters in
  (eV·F(R))^n vs eV coordinates
- `r_squared` (float): goodness of fit on the chosen window
- `fit_window_ev` (list[float]): [E_min, E_max] of points actually fit
- `n_points_fit` (int): number of points in the fit window

Derived arrays (Parquet sidecar):
- `photon_energy_ev`
- `kubelka_munk`
- `tauc_y`  (= (F(R) · E)^n  -- what gets plotted on the Tauc axis)
- `fit_line` (linear extrapolation evaluated at every E, for plotting)
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from latos.analysis.base_analyzer import (
    AnalyzerInputs,
    AnalyzerOutput,
    BaseAnalyzer,
)
from latos.core.enums import Severity, Technique
from latos.core.models import Measurement, ValidationIssue, utc_now

__all__ = ["UvDrsTaucAnalyzer"]

# Planck's constant times the speed of light in convenient units:
# E[eV] = 1240 / wavelength[nm]. This is `h·c / e`, with the 0.066%
# rounding error baked in — the physics community uses 1240 nm·eV
# universally so we match that convention.
_HC_NM_EV = 1240.0

# Reflectance auto-detection threshold. If the max value in the
# reflectance array exceeds 1.5, we assume the data are in percent
# (the human convention) and divide by 100. The 1.5 (rather than 1.0)
# gives a little headroom for noisy data that exceeds 100% reflectance
# in the baseline before correction.
_PERCENT_DETECTION_THRESHOLD = 1.5

# Minimum number of (wavelength, reflectance) points required to attempt
# a fit. Below this, the analyzer emits an error issue and returns
# empty outputs.
_MIN_POINTS_FOR_FIT = 20

# Floor applied to reflectance before dividing in the Kubelka-Munk
# formula. Without it, R = 0 (e.g. a perfect-absorber baseline pixel)
# would blow up F(R) to infinity. 1e-4 is a couple orders of magnitude
# below typical instrument noise and well below the smallest legitimate
# R value we expect to see.
_R_FLOOR = 1e-4

# Minimum points to attempt a least-squares fit. Three is the hard
# mathematical floor (line through two points is trivially R²=1, no
# diagnostic value); below three the analyzer returns an error issue
# rather than a misleading fit.
_MIN_FIT_POINTS = 3

# R² threshold below which the analyzer flags the fit as low-quality.
# 0.95 is empirically a clean line through the rising-edge data —
# anything worse usually means the user picked the wrong window.
_R_SQUARED_WARN_THRESHOLD = 0.95


class UvDrsTaucAnalyzer(BaseAnalyzer):
    """Band-gap extraction from UV-DRS reflectance via the Tauc method."""

    name: ClassVar[str] = "uvdrs-tauc"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.UV_DRS,)
    default_params: ClassVar[dict[str, Any]] = {
        # "direct" → exponent 2; "indirect" → exponent 1/2.
        "band_gap_type": "direct",
        # Window on the rising edge expressed as fractions of the max
        # Tauc-y value. Picks the region most likely to be linear; the
        # user can override either bound to inspect a different segment.
        "fit_window_y_min_frac": 0.20,
        "fit_window_y_max_frac": 0.60,
    }

    # ─── accepts ─────────────────────────────────────────────────────
    def accepts(self, measurement: Measurement) -> bool:
        """True if this measurement looks like a UV-DRS scan we can analyze.

        We can't see the arrays from a Measurement alone (those live
        in ArrayStore), so the accepts test is metadata-based: the
        technique is UV-DRS (enforced by `accepts_techniques`) and
        the parser version is one we know writes a wavelength /
        reflectance array pair. For Stage 1's only UV-DRS parser
        (`uvdrs-xlsx`), that's always true — but stamping the check
        here makes us forward-compatible with future UV-DRS parsers
        that might write differently-shaped arrays.
        """
        # Conservative: only accept if there's at least one file. A
        # zero-file measurement is a corrupt state we'd rather skip.
        return len(measurement.files) > 0

    # ─── analyze ─────────────────────────────────────────────────────
    def analyze(  # noqa: PLR0911, PLR0915
        self, inputs: AnalyzerInputs
    ) -> AnalyzerOutput:
        """Run the Tauc-plot extraction. Never raises.

        Long by design: the Tauc procedure has six labelled stages
        (normalize → KM → photon energy → Tauc transform → fit-window
        pick → linear fit). Splitting into helper methods would scatter
        the variable bindings each step depends on, hurting readability
        more than the length costs.
        """
        issues: list[ValidationIssue] = []

        wavelength_nm = inputs.arrays.get("wavelength")
        reflectance = inputs.arrays.get("reflectance")
        if wavelength_nm is None or reflectance is None:
            return _error_output(
                "Missing wavelength/reflectance arrays — cannot run Tauc analysis.",
            )
        if wavelength_nm.shape != reflectance.shape:
            return _error_output(
                f"Array shape mismatch: wavelength={wavelength_nm.shape}, "
                f"reflectance={reflectance.shape}",
            )
        if wavelength_nm.size < _MIN_POINTS_FOR_FIT:
            return _error_output(
                f"Too few data points for a stable Tauc fit "
                f"(have {wavelength_nm.size}, need ≥ {_MIN_POINTS_FOR_FIT}).",
            )

        params = inputs.params
        gap_type = str(params.get("band_gap_type", "direct")).lower()
        if gap_type not in {"direct", "indirect"}:
            issues.append(
                ValidationIssue(
                    field="band_gap_type",
                    severity=Severity.WARNING,
                    message=(f"Unrecognized band_gap_type={gap_type!r}; falling back to 'direct'."),
                    detected_at=utc_now(),
                ),
            )
            gap_type = "direct"

        # n=2 for direct allowed; n=1/2 for indirect allowed. The
        # power applied to (F(R)·E) — NOT to F(R) alone — is what makes
        # the rising edge linear under the relevant assumption.
        exponent = 2.0 if gap_type == "direct" else 0.5

        # 1. Normalize reflectance to fraction in [0, 1].
        refl = np.asarray(reflectance, dtype=np.float64)
        if float(np.nanmax(refl)) > _PERCENT_DETECTION_THRESHOLD:
            refl = refl / 100.0
            issues.append(
                ValidationIssue(
                    field="reflectance",
                    severity=Severity.INFO,
                    message="Reflectance auto-scaled from percent (0..100) to fraction (0..1).",
                    detected_at=utc_now(),
                ),
            )

        # Clip to the [floor, 1] range so the Kubelka-Munk denominator
        # is well-defined and we don't get negative absorption from
        # noisy slightly-over-100% baselines.
        refl = np.clip(refl, _R_FLOOR, 1.0)

        # 2. Photon energy and Kubelka-Munk function.
        wl = np.asarray(wavelength_nm, dtype=np.float64)
        # Guard wavelength against zero/negative values (sanity check;
        # the UV-DRS parser already enforces positive wavelengths but
        # defence in depth is cheap).
        wl = np.where(wl > 0, wl, np.nan)
        photon_energy_ev = _HC_NM_EV / wl
        kubelka_munk = (1.0 - refl) ** 2 / (2.0 * refl)

        # 3. Tauc transform.
        tauc_y = np.power(kubelka_munk * photon_energy_ev, exponent)

        # Sort by photon energy ascending so the "rising edge" runs
        # left-to-right. Wavelength is usually monotone decreasing in
        # photon energy, so we typically reverse the array — but doing
        # an explicit sort handles both conventions.
        order = np.argsort(photon_energy_ev)
        E = photon_energy_ev[order]
        Y = tauc_y[order]

        # Drop NaN / inf entries that came from R floor / zero wavelength.
        finite = np.isfinite(E) & np.isfinite(Y)
        E = E[finite]
        Y = Y[finite]
        if E.size < _MIN_POINTS_FOR_FIT:
            return _error_output(
                f"After cleaning, only {E.size} finite Tauc points remain; "
                f"need ≥ {_MIN_POINTS_FOR_FIT}.",
            )

        # 4. Pick the fit window. Default uses y-percentile cuts so
        # the linear region of the rising edge is selected
        # automatically; user overrides flow through `params`.
        y_min_frac = float(params.get("fit_window_y_min_frac", 0.20))
        y_max_frac = float(params.get("fit_window_y_max_frac", 0.60))
        y_max = float(np.max(Y))
        if y_max <= 0:
            return _error_output(
                "All Tauc values are non-positive — no rising edge to fit.",
            )
        mask = (y_min_frac * y_max <= Y) & (y_max_frac * y_max >= Y)
        if int(mask.sum()) < _MIN_POINTS_FOR_FIT:
            issues.append(
                ValidationIssue(
                    field="fit_window",
                    severity=Severity.WARNING,
                    message=(
                        f"Default fit window has only {int(mask.sum())} points; "
                        "consider widening fit_window_y_*_frac."
                    ),
                    detected_at=utc_now(),
                ),
            )

        if int(mask.sum()) < _MIN_FIT_POINTS:
            return _error_output(
                f"Fit window has fewer than {_MIN_FIT_POINTS} points — "
                "cannot perform linear regression.",
            )

        E_fit = E[mask]
        Y_fit = Y[mask]

        # 5. Linear least-squares fit. polyfit is overkill here but its
        # numerics are battle-tested and we get the residuals for free.
        slope, intercept = np.polyfit(E_fit, Y_fit, deg=1)
        slope = float(slope)
        intercept = float(intercept)

        # Goodness of fit on the fit window only.
        y_pred = slope * E_fit + intercept
        ss_res = float(np.sum((Y_fit - y_pred) ** 2))
        ss_tot = float(np.sum((Y_fit - np.mean(Y_fit)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if slope <= 0:
            return _error_output(
                "Fit slope is non-positive — the chosen window is not a rising edge.",
            )

        # 6. Band gap = x-intercept of the fit line.
        band_gap_ev = -intercept / slope

        if band_gap_ev <= 0 or not np.isfinite(band_gap_ev):
            issues.append(
                ValidationIssue(
                    field="band_gap_ev",
                    severity=Severity.ERROR,
                    message=(
                        f"Extracted band gap is unphysical: {band_gap_ev:.3f} eV. "
                        "Check the fit window."
                    ),
                    detected_at=utc_now(),
                ),
            )

        if r_squared < _R_SQUARED_WARN_THRESHOLD:
            issues.append(
                ValidationIssue(
                    field="r_squared",
                    severity=Severity.WARNING,
                    message=(
                        f"Linear fit quality is low (R² = {r_squared:.3f}); "
                        "the chosen window may not be the linear region."
                    ),
                    detected_at=utc_now(),
                ),
            )

        # Build a fit line evaluated at every photon-energy point —
        # purely for plotting. The user sees the Tauc curve and the
        # extrapolated line crossing the x-axis at band_gap_ev.
        fit_line = slope * E + intercept

        outputs: dict[str, Any] = {
            "band_gap_ev": float(band_gap_ev),
            "band_gap_type": gap_type,
            "slope": slope,
            "intercept": intercept,
            "r_squared": float(r_squared),
            "fit_window_ev": [float(E_fit.min()), float(E_fit.max())],
            "n_points_fit": int(E_fit.size),
            "n_points_total": int(E.size),
        }

        derived_arrays = {
            "photon_energy_ev": E,
            "kubelka_munk": kubelka_munk[order][finite],
            "tauc_y": Y,
            "fit_line": fit_line,
        }

        return AnalyzerOutput(
            outputs=outputs,
            derived_arrays=derived_arrays,
            issues=tuple(issues),
        )


# ─── Module helpers ─────────────────────────────────────────────────
def _error_output(message: str) -> AnalyzerOutput:
    """Build an AnalyzerOutput carrying a single error-level issue.

    The standard "analysis could not be performed" return shape.
    Empty `outputs`, empty `derived_arrays`, one ERROR issue describing
    what went wrong. The UI shows the issue to the user and the
    cache key still records the attempt so repeated runs don't
    re-trigger the same failure.
    """
    issue = ValidationIssue(
        field="analyze",
        severity=Severity.ERROR,
        message=message,
        detected_at=utc_now(),
    )
    return AnalyzerOutput(outputs={}, derived_arrays={}, issues=(issue,))
