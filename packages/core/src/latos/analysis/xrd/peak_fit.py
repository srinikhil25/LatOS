# N806 (uppercase locals) is OK here: scientific code uses single
# uppercase letters (A, x, FWHM) by convention; renaming to lowercase
# obscures rather than clarifies. We rely on the project-wide RUF002/003
# ignore for Greek letters in docstrings (2θ, σ, η are intentional
# physics notation, not typos of o/y).
"""XRD peak-fit analyzer.

Takes a 1-D powder-diffraction scan (intensity vs 2θ) and returns the
positions, heights, FWHMs, and integrated areas of every peak above a
prominence threshold. The implementation follows the consensus
practice of modern XRD analysis tooling [Daniels 2020, Toby 2013,
Newville 2014]:

1. **Baseline subtraction** via SNIP [Ryan 1988] — the de-facto
   standard in XRF/XRD for separating peaks from slowly-varying
   instrumental + amorphous background. Iteratively replaces each
   point with the minimum of itself and the average of its
   neighbours at growing window width, geometrically "pressing
   down" until only smooth background remains. Single hyperparameter
   (max window width); robust to peak count.

2. **Peak detection** via `scipy.signal.find_peaks` [Virtanen 2020]
   on a Savitzky-Golay-smoothed [Savitzky 1964] copy of the corrected
   curve. Threshold combines a fraction-of-max floor with a
   noise-σ-based floor (`max(3·σ_MAD, 0.01·max)`); minimum
   peak-separation defends against fitting one peak as two.

3. **Peak fitting** via lmfit's `PseudoVoigtModel` [Newville 2014].
   Detected peaks are clustered by overlapping fit windows so that
   neighbours (e.g. Kα₁/Kα₂ doublets, overlapping reflections from
   solid solutions) are fit *simultaneously* as a composite model
   rather than independently. Pseudo-Voigt — a linear combination of
   Gaussian and Lorentzian sharing FWHM — is the field default
   [Thompson 1987]; the η ∈ [0, 1] mixing parameter is fit alongside
   amplitude, center, and σ. The fit is performed on the
   baseline-corrected curve (not the smoothed one — smoothing biases
   amplitudes); the baseline is added back into the reported
   `fit_line` for overlay on the observed data.

Profile choice
--------------
Pseudo-Voigt with shared FWHM (one σ, one η per peak) is what the
field reaches for first for lab-grade powder XRD: it captures the
Gaussian (instrumental, strain) and Lorentzian (size, defect)
contributions in one analytical form. Thompson-Cox-Hastings
pseudo-Voigt [Thompson 1987] with separate Gaussian/Lorentzian widths
is the next step up (used in GSAS-II, FullProf); we defer to a
future version-1.1 if the extra parameter buys measurable accuracy
on our datasets.

Output payload
--------------
- `n_peaks` (int)
- `peak_centers_2theta` (list[float])
- `peak_heights` (list[float])
- `peak_fwhms_2theta` (list[float])
- `peak_areas` (list[float])     — integrated peak area
- `peak_fractions` (list[float]) — η per peak (Gaussian↔Lorentzian)
- `r_squared` (float)            — overall fit quality on the
                                   baseline-corrected curve
- `reduced_chi_square` (float)   — lmfit's χ²/dof

Derived arrays (Parquet sidecar)
--------------------------------
- `two_theta`
- `intensity_observed`
- `baseline`            — SNIP estimate
- `fit_line`            — sum of pseudo-Voigts + baseline (UI overlay)
- `residual`            — intensity_observed − fit_line

Limitations / future work
-------------------------
- **No Kα₂ stripping.** Rachinger correction [Rachinger 1948] for
  Cu Kα is the standard preprocessing step for non-Rietveld
  workflows. For datasets where Kα₂ is resolved, the analyzer will
  fit each component as a separate peak. A future "Strip Kα₂" toggle
  on import is on the roadmap.
- **No instrumental broadening subtraction.** The Caglioti formula
  [Caglioti 1958] requires a LaB6 / Si SRM 660c calibration scan to
  determine instrument-specific U, V, W parameters. Without that,
  the reported FWHMs are total (instrumental + sample) widths —
  fine for relative comparison, not for Scherrer crystallite-size
  estimates. That's a Stage 4 cross-modal task.

References:
- Caglioti G, Paoletti A, Ricci FP (1958). Nucl. Instr. Methods 3, 223.
- Daniels P, Connolley T (2020). J. Open Source Software 5(54), 2381.
- Newville M et al. (2014). lmfit v0.8, Zenodo.
- Rachinger WA (1948). J. Sci. Instrum. 25, 254.
- Ryan CG et al. (1988). Nucl. Instr. Methods B34, 396.
- Savitzky A, Golay MJE (1964). Anal. Chem. 36, 1627.
- Thompson P, Cox DE, Hastings JB (1987). J. Appl. Cryst. 20, 79.
- Toby BH, Von Dreele RB (2013). J. Appl. Cryst. 46, 544.
- Virtanen P et al. (2020). Nature Methods 17, 261.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from lmfit.model import CompositeModel, ModelResult
from lmfit.models import PseudoVoigtModel
from pybaselines.smooth import snip
from scipy.signal import find_peaks, peak_widths, savgol_filter

from latos.analysis.base_analyzer import (
    AnalyzerInputs,
    AnalyzerOutput,
    BaseAnalyzer,
)
from latos.core.enums import Severity, Technique
from latos.core.models import Measurement, ValidationIssue, utc_now

__all__ = ["XrdPeakFitAnalyzer"]


# FWHM = 2√(2 ln 2) · σ for a Gaussian. Named here so the conversion
# inside per-peak result extraction reads as physics, not magic.
_GAUSSIAN_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Minimum total points before peak fitting is meaningful. A ~50-point
# scan is essentially noise; below this the analyzer refuses to run.
_MIN_POINTS_FOR_ANALYSIS = 50

# Minimum points within a fit window for lmfit to converge usefully.
# A pseudo-Voigt has 4 parameters per peak; 8 points per peak gives a
# meaningful residual degree of freedom on the simplest cluster.
_MIN_FIT_POINTS_PER_PEAK = 8

# Floor on the prominence threshold expressed as a fraction of max
# corrected intensity. The brief from the literature review:
# `max(3·σ_MAD, 0.01·max(corrected))` — 1% catches every "real" peak
# while suppressing baseline-residual ripples.
_PROMINENCE_FRAC_FLOOR = 0.01

# Multiplier on the noise σ (MAD-based) for the noise-aware
# component of the prominence threshold. The Daniels (xrdfit) and
# scikit-ued convention is 3σ — keeps false positives below the
# Gaussian 3σ rate (~0.3%).
_NOISE_PROMINENCE_MULTIPLIER = 3.0

# 1.4826 converts median-absolute-deviation to a robust σ estimate
# under Gaussian noise (consistency constant). See Rousseeuw &
# Croux 1993.
_MAD_TO_SIGMA = 1.4826

# R² below which we attach a warning issue. Empirical: clean data fits
# > 0.95; noisy real-world spectra sit around 0.85; below 0.8 something
# is wrong (wrong model, missed peaks, miscalibrated baseline).
_R_SQUARED_WARN_THRESHOLD = 0.8

# Minimum number of points crossing the half-max line before we trust
# the empirical FWHM measurement. Below 2, fall back to the 2σ closed
# form (pathological edge cases like extremely narrow synthetic peaks).
_MIN_HALF_MAX_CROSSINGS = 2


class XrdPeakFitAnalyzer(BaseAnalyzer):
    """Detect and pseudo-Voigt-fit peaks in an XRD intensity-vs-2θ scan.

    See module docstring for method details and citations.
    """

    name: ClassVar[str] = "xrd-peak-fit"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.XRD,)
    default_params: ClassVar[dict[str, Any]] = {
        # SNIP baseline. max_half_window expressed in 2θ degrees;
        # converted to samples via the scan step. Bigger window =
        # more aggressive flattening (clips wider amorphous humps).
        # 5° is wide enough to capture typical glass / amorphous
        # backgrounds while small enough to preserve true broad
        # crystalline features.
        "baseline_max_window_2theta": 5.0,
        # Savitzky-Golay smoothing for peak detection only (not for
        # fitting — smoothing biases peak amplitudes). Window must be
        # odd; polyorder < window.
        "smoothing_window": 7,
        "smoothing_polyorder": 2,
        # Peak detection. Prominence is the larger of:
        # - `min_prominence_fraction × max(corrected)` (default 1%)
        # - `3 × σ_noise` estimated from MAD of the detrended curve
        "min_prominence_fraction": _PROMINENCE_FRAC_FLOOR,
        # Minimum 2θ separation between peaks. 0.1° resolves the
        # high-angle Cu Kα₁/Kα₂ doublet (~0.05° split at low angle,
        # ~0.2° at high angle) as two peaks rather than one merged.
        "min_peak_distance_2theta": 0.1,
        # Minimum peak width in samples. 2 rejects single-point spikes
        # without losing legitimately narrow Bragg peaks.
        "min_peak_width_samples": 2,
        # Per-peak / per-cluster fit window: ±k·FWHM around the peak
        # (or around the cluster extents). 4·FWHM contains essentially
        # all of a pseudo-Voigt; going much wider lets neighbours bleed
        # in.
        "fit_window_fwhm_multiplier": 4.0,
        # Safety cap. A scan picking up >80 peaks is almost always a
        # noise-driven storm; the cap surfaces a warning rather than
        # churning through hundreds of futile fits.
        "max_peaks": 80,
    }

    # ─── accepts ─────────────────────────────────────────────────────
    def accepts(self, measurement: Measurement) -> bool:
        """True if this measurement is a fittable XRD scan.

        Technique gating is upstream (`accepts_techniques`); here we
        just require at least one file. Per-array shape validation
        happens inside `analyze()`.
        """
        return len(measurement.files) > 0

    # ─── analyze ─────────────────────────────────────────────────────
    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:  # noqa: PLR0911, PLR0912, PLR0915
        """Run the full baseline → detect → fit pipeline. Never raises.

        Long by design: every numerical stage is named with an early-
        return error path. Splitting into helpers scatters the
        variable bindings each stage depends on (x, y, baseline,
        corrected, smoothed, prominence_threshold, distance_samples,
        cluster_list, ...).
        """
        issues: list[ValidationIssue] = []

        two_theta = inputs.arrays.get("two_theta")
        intensity = inputs.arrays.get("intensity")
        if two_theta is None or intensity is None:
            return _error_output(
                "Missing two_theta/intensity arrays — cannot run XRD peak fit.",
            )
        if two_theta.shape != intensity.shape:
            return _error_output(
                f"Array shape mismatch: two_theta={two_theta.shape}, intensity={intensity.shape}",
            )
        if two_theta.size < _MIN_POINTS_FOR_ANALYSIS:
            return _error_output(
                f"Too few data points for peak fitting "
                f"(have {two_theta.size}, need ≥ {_MIN_POINTS_FOR_ANALYSIS}).",
            )

        # Defensive: sort by 2θ ascending and drop non-finite samples.
        # Most XRD files come in monotonic order, but parsing oddities
        # do happen.
        x = np.asarray(two_theta, dtype=np.float64)
        y = np.asarray(intensity, dtype=np.float64)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < _MIN_POINTS_FOR_ANALYSIS:
            return _error_output(
                f"After cleaning, only {x.size} finite points remain; "
                f"need ≥ {_MIN_POINTS_FOR_ANALYSIS}.",
            )

        # Median step (robust to small irregularities); used to convert
        # every "in degrees" parameter into a "in samples" count.
        step = float(np.median(np.diff(x)))
        if step <= 0 or not np.isfinite(step):
            return _error_output(
                f"2θ axis is not strictly increasing (median step = {step}); cannot run analysis.",
            )

        params = inputs.params
        # 1. Baseline (SNIP). pybaselines.smooth.snip returns a
        # (baseline_array, params_dict) tuple; we only need the array.
        baseline_window_2t = float(params.get("baseline_max_window_2theta", 5.0))
        max_half_window = max(3, round(baseline_window_2t / step))
        baseline_array, _ = snip(y, max_half_window=max_half_window)
        baseline = np.asarray(baseline_array, dtype=np.float64)
        corrected = y - baseline

        if float(np.max(corrected)) <= 0:
            return _error_output(
                "Baseline-corrected intensity is non-positive everywhere — no peaks to fit.",
            )

        # 2. Smoothing for detection only. Window must be odd and
        # smaller than data size. Polyorder must be strictly less than
        # the window length.
        smooth_window = _coerce_odd(int(params.get("smoothing_window", 7)))
        smooth_polyorder = int(params.get("smoothing_polyorder", 2))
        smooth_window = min(smooth_window, _coerce_odd(corrected.size - 1))
        smooth_polyorder = min(smooth_polyorder, smooth_window - 1)
        smoothed = savgol_filter(corrected, smooth_window, smooth_polyorder)

        # 3. Noise σ via MAD of first differences. For independent
        # Gaussian noise of σ per point, `diff(y)` has σ_diff = σ·√2;
        # the consistency factor MAD-to-σ is 1.4826 [Rousseeuw & Croux
        # 1993]. Peaks contribute large differences only at their
        # ascending / descending edges (a small fraction of samples),
        # so the median absolute difference is robust to peak count
        # and amplitude — this is the standard estimator in modern
        # XRD baselines (xrdfit, scikit-ued).
        #
        # Earlier attempt: MAD(corrected - savgol_smoothed). That
        # failed on noise-only data because a low-order Savitzky-Golay
        # filter tracks pure noise, leaving a near-zero residual and a
        # near-zero σ estimate — which made the 3σ prominence
        # threshold collapse and waved 17 noise-spike "peaks" through.
        diffs = np.diff(corrected)
        sigma_noise = float(
            _MAD_TO_SIGMA * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2.0)
        )

        # 4. Prominence threshold: max of the noise-aware and the
        # max-fraction floor (literature consensus).
        max_corrected = float(np.max(corrected))
        prom_frac = float(
            params.get("min_prominence_fraction", _PROMINENCE_FRAC_FLOOR),
        )
        prominence_threshold = max(
            _NOISE_PROMINENCE_MULTIPLIER * sigma_noise,
            prom_frac * max_corrected,
        )

        # 5. Peak detection on the smoothed curve.
        min_peak_dist_2t = float(params.get("min_peak_distance_2theta", 0.1))
        distance_samples = max(1, round(min_peak_dist_2t / step))
        min_width_samples = int(params.get("min_peak_width_samples", 2))
        peak_indices, _ = find_peaks(
            smoothed,
            prominence=prominence_threshold,
            distance=distance_samples,
            width=min_width_samples,
        )

        if peak_indices.size == 0:
            issues.append(
                ValidationIssue(
                    field="peaks",
                    severity=Severity.WARNING,
                    message=(
                        "No peaks detected above the prominence threshold "
                        f"({prominence_threshold:.3g}); consider lowering "
                        "`min_prominence_fraction`."
                    ),
                    detected_at=utc_now(),
                ),
            )
            return _empty_peak_output(x=x, y=y, baseline=baseline, issues=issues)

        # Honour the safety cap. Pick the most-prominent peaks.
        max_peaks = int(params.get("max_peaks", 80))
        if peak_indices.size > max_peaks:
            issues.append(
                ValidationIssue(
                    field="peaks",
                    severity=Severity.WARNING,
                    message=(
                        f"Detected {peak_indices.size} peaks; capping at "
                        f"{max_peaks}. Raise `min_prominence_fraction` to "
                        "filter noise."
                    ),
                    detected_at=utc_now(),
                ),
            )
            keep = np.argsort(smoothed[peak_indices])[::-1][:max_peaks]
            peak_indices = np.sort(peak_indices[keep])

        # 6. Per-peak FWHM estimates from peak_widths-at-half-maximum
        # (in sample units). Used both as initial σ guesses and for
        # cluster window sizing.
        widths_samples, _, _, _ = peak_widths(smoothed, peak_indices, rel_height=0.5)

        # 7. Cluster peaks whose fit windows overlap. Isolated peaks
        # become 1-peak clusters; close neighbours (doublets, shoulder
        # peaks) become N-peak clusters fit simultaneously.
        fit_window_mult = float(params.get("fit_window_fwhm_multiplier", 4.0))
        clusters = _cluster_peaks(
            peak_indices=peak_indices,
            widths_samples=widths_samples,
            fit_window_mult=fit_window_mult,
        )

        # 8. Per-cluster fits on the *unsmoothed* corrected curve.
        # Each fit produces zero or more (center, height, fwhm, area,
        # fraction) tuples.
        centers: list[float] = []
        heights: list[float] = []
        fwhms: list[float] = []
        areas: list[float] = []
        fractions: list[float] = []
        fit_curve = np.zeros_like(x)

        for cluster in clusters:
            cluster_widths = [
                float(widths_samples[np.where(peak_indices == idx)[0][0]]) for idx in cluster
            ]
            fit_result = _fit_cluster(
                x=x,
                y_corrected=corrected,
                cluster=cluster,
                cluster_widths_samples=cluster_widths,
                step=step,
                fit_window_mult=fit_window_mult,
                issues=issues,
            )
            if fit_result is None:
                continue
            # Evaluate the cluster's composite over the full x-axis so
            # `fit_line` shows the fit everywhere, not just inside the
            # window we trained on.
            evaluated = fit_result.eval(x=x)
            fit_curve += np.asarray(evaluated, dtype=np.float64)
            # Extract per-peak parameters. lmfit auto-derives `height`
            # and `fwhm` from amplitude/sigma/fraction.
            for j in range(len(cluster)):
                prefix = f"p{j}_"
                ctr = float(fit_result.params[f"{prefix}center"].value)
                hgt = float(fit_result.params[f"{prefix}height"].value)
                sigma_j = float(fit_result.params[f"{prefix}sigma"].value)
                area = float(fit_result.params[f"{prefix}amplitude"].value)
                frac = float(fit_result.params[f"{prefix}fraction"].value)
                # Compute the *actual* FWHM of the pseudo-Voigt curve.
                # lmfit's derived `fwhm` parameter is defined as
                # `2·sigma` regardless of fraction — that's an internal
                # convention, NOT the physical full-width-at-half-max.
                # For the materials-science user-facing report we want
                # the genuine FWHM (2.355·σ for pure Gaussian; 2·σ for
                # pure Lorentzian; an interpolation in between).
                fwhm = _empirical_fwhm(sigma=sigma_j, fraction=frac)
                # Drop unphysical peaks. A center outside the data
                # range, negative height, or unconstrained fraction
                # means the optimizer wandered.
                if not (x[0] <= ctr <= x[-1]) or hgt <= 0 or fwhm <= 0:
                    issues.append(
                        ValidationIssue(
                            field="peak_fit",
                            severity=Severity.WARNING,
                            message=(
                                f"Peak at 2θ≈{ctr:.3f}° has unphysical fit "
                                f"(height={hgt:.3g}, fwhm={fwhm:.3g}); dropped."
                            ),
                            detected_at=utc_now(),
                        ),
                    )
                    continue
                centers.append(ctr)
                heights.append(hgt)
                fwhms.append(fwhm)
                areas.append(area)
                fractions.append(frac)

        # Sort peaks by ascending 2θ for stable reporting.
        if centers:
            order_peaks = np.argsort(centers)
            centers = [centers[i] for i in order_peaks]
            heights = [heights[i] for i in order_peaks]
            fwhms = [fwhms[i] for i in order_peaks]
            areas = [areas[i] for i in order_peaks]
            fractions = [fractions[i] for i in order_peaks]

        # 9. Goodness of fit on the baseline-corrected curve. We
        # compute reduced χ² ourselves rather than concatenating lmfit
        # results across clusters (which would mix windows of
        # different size).
        ss_res = float(np.sum((corrected - fit_curve) ** 2))
        ss_tot = float(np.sum((corrected - np.mean(corrected)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # dof = N_points - 4·N_peaks. lmfit uses the same formula
        # internally for composite models.
        dof = max(1, x.size - 4 * len(centers))
        # σ_noise² ≈ sigma_noise² gives a noise-normalized χ²; without
        # explicit per-point weights, this is the right scale.
        reduced_chi_square = ss_res / dof / max(sigma_noise * sigma_noise, 1e-30)

        if r_squared < _R_SQUARED_WARN_THRESHOLD and centers:
            issues.append(
                ValidationIssue(
                    field="r_squared",
                    severity=Severity.WARNING,
                    message=(
                        f"Overall fit quality is low (R² = {r_squared:.3f}). "
                        "Consider adjusting prominence threshold or fit window."
                    ),
                    detected_at=utc_now(),
                ),
            )

        outputs: dict[str, Any] = {
            "n_peaks": len(centers),
            "peak_centers_2theta": centers,
            "peak_heights": heights,
            "peak_fwhms_2theta": fwhms,
            "peak_areas": areas,
            "peak_fractions": fractions,
            "r_squared": float(r_squared),
            "reduced_chi_square": float(reduced_chi_square),
            "noise_sigma_estimate": float(sigma_noise),
        }
        # The visible fit overlay adds the baseline back onto the
        # corrected-curve fit, so it sits on top of the *raw* observed
        # curve in the UI.
        fit_line = fit_curve + baseline
        derived_arrays = {
            "two_theta": x,
            "intensity_observed": y,
            "baseline": baseline,
            "fit_line": fit_line,
            "residual": y - fit_line,
        }
        return AnalyzerOutput(
            outputs=outputs,
            derived_arrays=derived_arrays,
            issues=tuple(issues),
        )


# ─── Module helpers ─────────────────────────────────────────────────
def _empirical_fwhm(*, sigma: float, fraction: float) -> float:
    """Compute the true FWHM of an lmfit pseudo-Voigt with given σ, η.

    lmfit's `PseudoVoigtModel` is the linear combination

        pV(x) = (1−η)·G(x; σ_g=σ) + η·L(x; γ_L=σ)

    with a SHARED `sigma` parameter feeding both components. The
    Gaussian FWHM is `2σ·√(2·ln 2)`; the Lorentzian FWHM is `2σ`.
    These are NOT equal, so the linear combination's FWHM depends on
    η in a non-trivial way that lmfit's derived `fwhm` parameter (=
    2σ regardless of η) does not capture.

    We compute the FWHM empirically: evaluate the normalized profile
    on a fine grid (±5σ_G covers >99.99% of the Gaussian and >97%
    of the Lorentzian) and find the half-max crossings. Stable for
    any fraction in [0, 1] and any sigma > 0. The factor 5 is the
    accepted "tails covered" multiplier for pseudo-Voigt-style
    profiles in the lmfit cookbook examples.
    """
    if sigma <= 0:
        return 0.0
    # Evaluate at center first to fix the half-max target.
    sigma_g = sigma  # lmfit's Gaussian σ_g = sigma directly
    # Peak amplitude per unit `amplitude` parameter — analytic.
    height_per_amp = (1.0 - fraction) / (sigma_g * np.sqrt(2.0 * np.pi)) + (
        fraction / (np.pi * sigma)
    )
    # Discretize ±5σ around the center on a fine grid. 5001 points
    # gives sub-percent FWHM resolution for any reasonable sigma.
    x = np.linspace(-5.0 * sigma_g, 5.0 * sigma_g, 5001)
    gauss = (
        (1.0 - fraction)
        / (sigma_g * np.sqrt(2.0 * np.pi))
        * np.exp(-(x * x) / (2.0 * sigma_g * sigma_g))
    )
    lorentz = fraction * (1.0 / np.pi) * sigma / (x * x + sigma * sigma)
    profile = gauss + lorentz
    half_max = height_per_amp / 2.0
    # Bracket the half-max crossings on each side of the center.
    above = profile >= half_max
    indices_above = np.where(above)[0]
    if indices_above.size < _MIN_HALF_MAX_CROSSINGS:
        # Pathological: extremely narrow / numerical edge case.
        return float(2.0 * sigma)
    return float(x[indices_above[-1]] - x[indices_above[0]])


def _coerce_odd(n: int) -> int:
    """Return the largest odd integer ≤ n, with a minimum of 3.

    `scipy.signal.savgol_filter` requires an odd window length; this
    coerces any user input to a valid value rather than raising.
    """
    n = max(3, int(n))
    return n if n % 2 == 1 else n - 1


def _cluster_peaks(
    *,
    peak_indices: np.ndarray,
    widths_samples: np.ndarray,
    fit_window_mult: float,
) -> list[list[int]]:
    """Greedy cluster peaks whose fit windows overlap.

    A peak's fit window in sample units is `[idx - k·FWHM/2 - ...]`
    — but we don't need the exact bounds, just the overlap test
    `right_of_previous ≥ left_of_current`. Result is a list of
    clusters, each cluster a list of peak indices in 2θ-ascending
    order.

    The brief from the literature review:
        "Detect peaks first; cluster peaks whose windows overlap;
         fit each cluster as a multi-peak composite … Isolated
         peaks → single fits."
    """
    if peak_indices.size == 0:
        return []
    # Sort by 2θ index (peak_indices comes back sorted from find_peaks
    # already, but be defensive).
    order = np.argsort(peak_indices)
    sorted_indices = peak_indices[order]
    sorted_widths = widths_samples[order]
    clusters: list[list[int]] = []
    current: list[int] = []
    current_right = -np.inf
    for idx, width in zip(sorted_indices, sorted_widths, strict=True):
        left = float(idx) - fit_window_mult * float(width)
        right = float(idx) + fit_window_mult * float(width)
        if current and left <= current_right:
            current.append(int(idx))
            current_right = max(current_right, right)
        else:
            if current:
                clusters.append(current)
            current = [int(idx)]
            current_right = right
    if current:
        clusters.append(current)
    return clusters


def _fit_cluster(
    *,
    x: np.ndarray,
    y_corrected: np.ndarray,
    cluster: list[int],
    cluster_widths_samples: list[float],
    step: float,
    fit_window_mult: float,
    issues: list[ValidationIssue],
) -> ModelResult | None:
    """Fit an N-peak pseudo-Voigt composite over a cluster's window.

    Initial guesses follow the literature consensus (xrdfit / scikit-
    ued / lmfit examples):
        - center₀ = 2θ at the detected peak index
        - amplitude₀ = local height × (FWHM × √(2π))   (Gaussian area)
        - sigma₀ = FWHM_init / (2√(2 ln 2))
        - fraction₀ = 0.5 (midpoint Gaussian↔Lorentzian)

    Constraints (also literature consensus):
        - amplitude > 0
        - sigma > one scan-step (rejects pathological narrow peaks)
        - 0 ≤ fraction ≤ 1
        - center within ±2·σ_init of the detected position

    On any convergence failure (lmfit raises or returns success=False),
    we record a Severity.WARNING issue and return None. The caller
    drops the cluster from the result set rather than aborting the
    whole analysis.
    """
    # Cluster window: union of all per-peak windows. Clip to data.
    lo_samples = max(
        0,
        int(min(cluster) - fit_window_mult * max(cluster_widths_samples)),
    )
    hi_samples = min(
        x.size,
        int(max(cluster) + fit_window_mult * max(cluster_widths_samples)) + 1,
    )
    if hi_samples - lo_samples < _MIN_FIT_POINTS_PER_PEAK * len(cluster):
        issues.append(
            ValidationIssue(
                field="peak_fit",
                severity=Severity.WARNING,
                message=(
                    f"Cluster of {len(cluster)} peaks at "
                    f"2θ≈{x[cluster[0]]:.3f}°: fit window too narrow "
                    f"({hi_samples - lo_samples} points); skipped."
                ),
                detected_at=utc_now(),
            ),
        )
        return None
    x_win = x[lo_samples:hi_samples]
    y_win = y_corrected[lo_samples:hi_samples]

    # Build the composite model. `make_params()` is important: it
    # creates the full parameter set including each PseudoVoigt's
    # auto-derived `height` and `fwhm` expressions, which we read
    # after fitting. Building Parameters() manually with `.add()`
    # would skip the derived ones and trigger KeyError on lookup.
    composite: CompositeModel | PseudoVoigtModel | None = None
    for j in range(len(cluster)):
        prefix = f"p{j}_"
        model = PseudoVoigtModel(prefix=prefix)
        composite = model if composite is None else composite + model
    assert composite is not None  # mypy hint — cluster is non-empty
    params = composite.make_params()
    for j, (peak_idx, fwhm_samples) in enumerate(
        zip(cluster, cluster_widths_samples, strict=True),
    ):
        prefix = f"p{j}_"
        center_init = float(x[peak_idx])
        height_init = float(y_corrected[peak_idx])
        fwhm_init = max(step, float(fwhm_samples) * step)
        sigma_init = max(step, fwhm_init / _GAUSSIAN_FWHM_PER_SIGMA)
        # PseudoVoigtModel's `amplitude` is the integrated area, NOT
        # the peak height. Approximate area by Gaussian: A ≈ h·σ·√(2π).
        amplitude_init = max(1e-12, height_init * sigma_init * np.sqrt(2.0 * np.pi))
        # ±2·σ window on the center is generous enough for a
        # genuinely-misidentified peak to slide into place, narrow
        # enough that the optimizer can't run away to a neighbour.
        params[f"{prefix}center"].set(
            value=center_init,
            min=center_init - 2.0 * sigma_init,
            max=center_init + 2.0 * sigma_init,
        )
        params[f"{prefix}amplitude"].set(value=amplitude_init, min=0.0)
        # Upper-bound σ at 3× the initial estimate. Without this,
        # asymmetric multi-peak clusters (a strong peak next to a
        # weaker one) can converge with the weaker peak ballooning to
        # FWHM > 1° to absorb the strong peak's tail, swallowing the
        # gap between them. 3·σ_init still allows for a peak twice
        # as wide as the initial detection if that's what the data
        # genuinely shows.
        params[f"{prefix}sigma"].set(value=sigma_init, min=step, max=3.0 * sigma_init)
        params[f"{prefix}fraction"].set(value=0.5, min=0.0, max=1.0)

    try:
        result = composite.fit(y_win, params, x=x_win)
    except (ValueError, TypeError, RuntimeError) as exc:
        issues.append(
            ValidationIssue(
                field="peak_fit",
                severity=Severity.WARNING,
                message=(
                    f"Cluster of {len(cluster)} peaks at "
                    f"2θ≈{x[cluster[0]]:.3f}° failed to converge: {exc}"
                ),
                detected_at=utc_now(),
            ),
        )
        return None
    if not result.success:
        issues.append(
            ValidationIssue(
                field="peak_fit",
                severity=Severity.WARNING,
                message=(
                    f"Cluster of {len(cluster)} peaks at "
                    f"2θ≈{x[cluster[0]]:.3f}°: fit did not converge "
                    f"(message: {result.message})."
                ),
                detected_at=utc_now(),
            ),
        )
        return None
    return result


def _error_output(message: str) -> AnalyzerOutput:
    """Build a no-result AnalyzerOutput carrying a single error issue.

    Returned from `analyze()` when the input is degenerate enough that
    no further processing is meaningful (missing arrays, shape
    mismatch, non-monotonic axis, etc.).
    """
    issue = ValidationIssue(
        field="analyze",
        severity=Severity.ERROR,
        message=message,
        detected_at=utc_now(),
    )
    return AnalyzerOutput(outputs={}, derived_arrays={}, issues=(issue,))


def _empty_peak_output(
    *,
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    issues: list[ValidationIssue],
) -> AnalyzerOutput:
    """Return shape when baseline ran cleanly but no peaks crossed threshold.

    Still ships the derived arrays so the user can inspect the
    baseline curve and decide whether to lower the prominence
    threshold; peak lists are empty and `fit_line` is just the
    baseline.
    """
    return AnalyzerOutput(
        outputs={
            "n_peaks": 0,
            "peak_centers_2theta": [],
            "peak_heights": [],
            "peak_fwhms_2theta": [],
            "peak_areas": [],
            "peak_fractions": [],
            "r_squared": 0.0,
            "reduced_chi_square": 0.0,
            "noise_sigma_estimate": 0.0,
        },
        derived_arrays={
            "two_theta": x,
            "intensity_observed": y,
            "baseline": baseline,
            "fit_line": baseline.copy(),
            "residual": y - baseline,
        },
        issues=tuple(issues),
    )
