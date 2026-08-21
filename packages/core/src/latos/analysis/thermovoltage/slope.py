"""Ionic Seebeck coefficient as a fitted slope, not a single reading.

In a thermodiffusive ionic thermoelectric cell the measured open-circuit
voltage is not the thermovoltage alone. Ions adsorb and desorb at the
electrodes, and that interface contributes a term which does not scale with the
temperature difference:

    ΔV = S · ΔT + ΔV_electrode

A measurement at one ΔT gives one number and no way to separate the two. The
usual response — divide the voltage by the temperature difference and call it S
— silently folds the electrode offset into the reported coefficient, and its
size is invisible.

Measuring at three or more ΔT values fixes this. `S` is the slope, the offset is
the intercept, and the residuals say whether the relationship was linear at all.
That turns an unverifiable reading into a fitted quantity with an uncertainty
attached, which is the same move the transport kernel makes when it derives zT
from arrays rather than trusting a headline value.

Two further things fall out of the fit, and both matter downstream:

* **The standard error of the slope.** The optimizer currently receives
  reliability as one boolean per point plus a single campaign-wide noise
  figure. A per-point `σ_S` computed here is what lets it become a real
  variance instead. This analyzer is the first producer of that number.

  One caveat travels with it. At three ΔT values the residual variance has a
  single degree of freedom, and a chi-squared on one degree of freedom is
  heavily right-skewed — its median is 0.455 of its mean. The reported
  standard error therefore lands below the truth more often than not, by
  roughly a factor of two at the median. A three-point campaign hands the
  surrogate an over-confident noise estimate on exactly the sparse data where
  over-confidence costs most. Four ΔT values is the cheap fix, and the
  behaviour is pinned by a test so it cannot be forgotten.
* **The intercept as a fraction of signal.** Reporting `ΔV_electrode` in
  millivolts is honest but hard to act on. Reporting it as a share of the
  voltage at the largest ΔT answers the question a reviewer actually asks:
  how much of what you measured was never thermoelectric?

Fitting is done in closed form rather than through `numpy.polyfit(cov=True)`,
which needs more than three points before it will return a covariance. Three is
exactly the interesting case here, and the textbook expressions are short enough
to read.

Inputs expected on the measurement, in either of two forms:

* `arrays["delta_t_k"]` and `arrays["delta_v_mv"]` — already reduced to pairs
* `arrays["t_hot_c"]`, `arrays["t_cold_c"]` and `arrays["voltage_mv"]` — the
  raw form a continuous temperature ramp produces, reduced here

Output payload:

* `seebeck_mv_k`, `seebeck_stderr_mv_k` — the coefficient and its uncertainty
* `offset_mv`, `offset_stderr_mv` — the electrode-polarisation term
* `offset_fraction` — |offset| as a share of |S·ΔT| at the largest ΔT
* `r_squared`, `residual_max_mv`, `n_points`, `delta_t_span_k`
* `offset_significant` — True when the intercept exceeds twice its own error

Derived arrays: `fit_delta_t_k`, `fit_delta_v_mv`, `residual_mv`.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, ClassVar

import numpy as np

from latos.analysis.base_analyzer import (
    AnalyzerInputs,
    AnalyzerOutput,
    BaseAnalyzer,
)
from latos.core.enums import Severity, Technique
from latos.core.models import Measurement, ValidationIssue, utc_now

__all__ = ["ThermovoltageSlopeAnalyzer", "fit_seebeck_slope"]

# Two points define a line exactly, leaving no residual and no way to judge the
# fit. We still report the slope — it is the best available estimate and
# refusing to compute it helps nobody — but with no uncertainty and a warning.
_MIN_POINTS_FOR_FIT = 2

# Below three points the offset and the slope cannot be told apart with any
# confidence: the fit passes through both points exactly whatever the true
# intercept is.
_MIN_POINTS_FOR_OFFSET = 3

# The underlying relationship is linear by construction, so a low R² means
# something went wrong (drift, leakage, a cell not at steady state) rather than
# that the material is interesting. Higher than a typical goodness-of-fit
# threshold for that reason: at a realistic 5% scatter over a 2 K to 10 K
# sweep the signal range dwarfs the noise and R² sits well above this, so
# anything below it is a signal rather than ordinary imprecision.
_R_SQUARED_WARN = 0.99

# Curvature needs a separate test, because R² does not detect it on a handful
# of points. A deliberately quadratic five-point series still scores 0.963:
# comfortably "good fit" by any usual standard, and completely wrong.
#
# Four points is the minimum a parabola can be tested with — three fit it
# exactly, leaving nothing to judge the fit by.
_MIN_POINTS_FOR_CURVATURE = 4

# F above which the quadratic term counts as real. Around the 2% level for the
# small degrees of freedom this analyzer works with, chosen so ordinary scatter
# on a nine-point series raises a false alarm roughly once in fifty runs. A
# curvature warning people learn to dismiss is worse than none.
_CURVATURE_F = 10.0

# An intercept larger than this share of the signal at the widest ΔT means a
# single-point measurement would have been materially wrong. Ten percent is the
# point at which the error exceeds typical replicate scatter in this technique.
_OFFSET_FRACTION_WARN = 0.10

# ΔT values must actually differ. Expressed relative to the mean |ΔT| so it
# holds whether the campaign works at 2 K or 40 K.
_MIN_RELATIVE_SPAN = 0.05

# An intercept counts as real when it clears two standard errors, the usual
# two-sigma convention.
_OFFSET_SIGNIFICANCE_SIGMA = 2.0


class SlopeFit:
    """Least-squares fit of `ΔV = S·ΔT + b`, with the errors that go with it.

    A small container rather than a bare tuple: six correlated numbers travel
    together here, and every consumer needs a different subset.
    """

    __slots__ = (
        "intercept",
        "intercept_stderr",
        "n",
        "r_squared",
        "residuals",
        "slope",
        "slope_stderr",
    )

    def __init__(
        self,
        *,
        slope: float,
        intercept: float,
        slope_stderr: float,
        intercept_stderr: float,
        r_squared: float,
        residuals: np.ndarray,
        n: int,
    ) -> None:
        self.slope = slope
        self.intercept = intercept
        self.slope_stderr = slope_stderr
        self.intercept_stderr = intercept_stderr
        self.r_squared = r_squared
        self.residuals = residuals
        self.n = n


def fit_seebeck_slope(delta_t_k: np.ndarray, delta_v_mv: np.ndarray) -> SlopeFit:
    """Fit `ΔV = S·ΔT + b` and return the parameters with their standard errors.

    Closed-form ordinary least squares. With `n` points and two parameters the
    residual variance carries `n - 2` degrees of freedom, so a two-point fit has
    none and its standard errors are reported as NaN rather than as zero — an
    exact fit through two points is not a precise one.

    Args:
        delta_t_k: Temperature differences, in kelvin.
        delta_v_mv: Measured voltages, in millivolts, same length.

    Returns:
        A `SlopeFit`. `slope` is the Seebeck coefficient in mV/K.
    """
    x = np.asarray(delta_t_k, dtype=np.float64)
    y = np.asarray(delta_v_mv, dtype=np.float64)
    n = int(x.size)

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    s_xx = float(np.sum(dx * dx))

    slope = float(np.sum(dx * (y - y_mean)) / s_xx) if s_xx > 0 else math.nan
    intercept = y_mean - slope * x_mean if math.isfinite(slope) else math.nan

    residuals = y - (slope * x + intercept) if math.isfinite(slope) else np.full_like(y, math.nan)

    dof = n - 2
    if dof > 0 and s_xx > 0:
        # Residual variance, then the standard textbook propagation onto each
        # parameter. The intercept carries the extra x̄²/S_xx term because it is
        # an extrapolation back to ΔT = 0, and extrapolating from data centred
        # far from zero is less certain than the slope itself.
        resid_var = float(np.sum(residuals**2) / dof)
        slope_stderr = math.sqrt(resid_var / s_xx)
        intercept_stderr = math.sqrt(resid_var * (1.0 / n + x_mean**2 / s_xx))
    else:
        slope_stderr = math.nan
        intercept_stderr = math.nan

    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum(residuals**2)) if math.isfinite(slope) else math.nan
    if ss_tot > 0 and math.isfinite(ss_res):
        r_squared = 1.0 - ss_res / ss_tot
    elif math.isfinite(ss_res):
        # Every y identical. A flat line explains it perfectly, but there is no
        # variance to explain, so R² is meaningless rather than 1.
        r_squared = math.nan
    else:
        r_squared = math.nan

    return SlopeFit(
        slope=slope,
        intercept=intercept,
        slope_stderr=slope_stderr,
        intercept_stderr=intercept_stderr,
        r_squared=r_squared,
        residuals=residuals,
        n=n,
    )


class ThermovoltageSlopeAnalyzer(BaseAnalyzer):
    """Seebeck coefficient and electrode offset from a ΔV-versus-ΔT series."""

    name: ClassVar[str] = "thermovoltage-slope"
    version: ClassVar[str] = "1.0.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (Technique.THERMOELECTRIC,)
    default_params: ClassVar[dict[str, Any]] = {}

    def accepts(self, measurement: Measurement) -> bool:
        """Accept any thermoelectric measurement with a source file.

        The array-level decision belongs in `analyze`, where a missing column
        can be reported as an issue the reviewer sees, rather than here, where
        it would silently drop the measurement from the run.
        """
        return len(measurement.files) > 0

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        """Reduce to (ΔT, ΔV) pairs, fit, and report what the fit exposes."""
        pairs = _extract_pairs(inputs.arrays)
        if isinstance(pairs, str):
            return _error(pairs)
        delta_t, delta_v = pairs

        finite = np.isfinite(delta_t) & np.isfinite(delta_v)
        dropped = int(delta_t.size - int(np.count_nonzero(finite)))
        delta_t, delta_v = delta_t[finite], delta_v[finite]

        if delta_t.size < _MIN_POINTS_FOR_FIT:
            return _error(
                f"Need at least {_MIN_POINTS_FOR_FIT} finite (ΔT, ΔV) points to fit a "
                f"slope; got {delta_t.size}.",
            )

        span = float(np.max(delta_t) - np.min(delta_t))
        mean_abs = float(np.mean(np.abs(delta_t)))
        if mean_abs > 0 and span / mean_abs < _MIN_RELATIVE_SPAN:
            return _error(
                f"ΔT spans only {span:.3g} K across {delta_t.size} points, which is "
                f"under {_MIN_RELATIVE_SPAN:.0%} of the mean |ΔT| ({mean_abs:.3g} K). "
                "The slope is not determined; measure at genuinely different ΔT.",
            )

        fit = fit_seebeck_slope(delta_t, delta_v)
        if not math.isfinite(fit.slope):
            return _error("Slope is undefined — every ΔT value is identical.")

        issues = list(_judge(fit, delta_t, delta_v, dropped))
        outputs = _payload(fit, delta_t, span)
        derived = {
            "fit_delta_t_k": delta_t,
            "fit_delta_v_mv": fit.slope * delta_t + fit.intercept,
            "residual_mv": fit.residuals,
        }
        return AnalyzerOutput(outputs=outputs, derived_arrays=derived, issues=tuple(issues))


def _extract_pairs(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | str:
    """Get (ΔT, ΔV) from either the reduced or the raw-ramp array form.

    Returns the pair, or an error message describing what was missing. A ramp
    is stored as hot and cold temperatures against a voltage trace, and a
    discrete series is usually already reduced; accepting both means the
    analyzer does not care how the run was performed.
    """
    if "delta_t_k" in arrays and "delta_v_mv" in arrays:
        delta_t = np.asarray(arrays["delta_t_k"], dtype=np.float64).ravel()
        delta_v = np.asarray(arrays["delta_v_mv"], dtype=np.float64).ravel()
    elif {"t_hot_c", "t_cold_c", "voltage_mv"} <= set(arrays):
        hot = np.asarray(arrays["t_hot_c"], dtype=np.float64).ravel()
        cold = np.asarray(arrays["t_cold_c"], dtype=np.float64).ravel()
        if hot.size != cold.size:
            return (
                f"t_hot_c and t_cold_c differ in length ({hot.size} vs {cold.size}); "
                "they must be sampled together."
            )
        # A difference in Celsius is a difference in kelvin, so no offset applies.
        delta_t = hot - cold
        delta_v = np.asarray(arrays["voltage_mv"], dtype=np.float64).ravel()
    else:
        return (
            "Missing arrays — expected either (delta_t_k, delta_v_mv) or "
            "(t_hot_c, t_cold_c, voltage_mv)."
        )

    if delta_t.size != delta_v.size:
        return f"ΔT and ΔV differ in length ({delta_t.size} vs {delta_v.size})."
    if delta_t.size == 0:
        return "No data points supplied."
    return delta_t, delta_v


def _payload(fit: SlopeFit, delta_t: np.ndarray, span: float) -> dict[str, Any]:
    """The JSON-safe output record."""
    widest = float(delta_t[int(np.argmax(np.abs(delta_t)))])
    signal_at_widest = abs(fit.slope * widest)
    offset_fraction = abs(fit.intercept) / signal_at_widest if signal_at_widest > 0 else math.nan
    significant = (
        math.isfinite(fit.intercept_stderr)
        and fit.intercept_stderr > 0
        and abs(fit.intercept) > _OFFSET_SIGNIFICANCE_SIGMA * fit.intercept_stderr
    )
    return {
        "seebeck_mv_k": round(fit.slope, 6),
        "seebeck_stderr_mv_k": _clean(fit.slope_stderr),
        "offset_mv": round(fit.intercept, 6),
        "offset_stderr_mv": _clean(fit.intercept_stderr),
        "offset_fraction": _clean(offset_fraction),
        "offset_significant": bool(significant),
        "r_squared": _clean(fit.r_squared),
        "residual_max_mv": _clean(float(np.max(np.abs(fit.residuals)))),
        "n_points": fit.n,
        "delta_t_span_k": round(span, 6),
        "delta_t_min_k": round(float(np.min(delta_t)), 6),
        "delta_t_max_k": round(float(np.max(delta_t)), 6),
    }


def _judge(
    fit: SlopeFit, delta_t: np.ndarray, delta_v: np.ndarray, dropped: int
) -> list[ValidationIssue]:
    """Everything the fit says that a reviewer needs told, in plain terms."""
    issues: list[ValidationIssue] = []

    if dropped:
        issues.append(
            _warn(
                "delta_v_mv",
                f"{dropped} point(s) dropped for non-finite ΔT or ΔV.",
            )
        )

    if fit.n < _MIN_POINTS_FOR_OFFSET:
        issues.append(
            _warn(
                "n_points",
                f"Only {fit.n} points: a line through them is exact, so the electrode "
                "offset cannot be separated from the Seebeck coefficient and no "
                "uncertainty is available. Measure at three or more ΔT values.",
            )
        )

    if math.isfinite(fit.r_squared) and fit.r_squared < _R_SQUARED_WARN:
        issues.append(
            _warn(
                "r_squared",
                f"ΔV is not linear in ΔT (R² = {fit.r_squared:.3f}, below "
                f"{_R_SQUARED_WARN}). The relationship is linear by construction, so "
                "this points at drift, leakage, or a cell read before steady state.",
            )
        )

    if _bows(delta_t, delta_v, fit.residuals):
        issues.append(
            _warn(
                "residual_mv",
                "Residuals bow rather than scatter: they sit on one side of the fit at "
                "both ends of the ΔT range and on the other in the middle. That is "
                "curvature, which a straight-line model cannot represent, so the "
                "reported coefficient is an average over a changing slope.",
            )
        )

    widest = float(delta_t[int(np.argmax(np.abs(delta_t)))])
    signal = abs(fit.slope * widest)
    if signal > 0:
        fraction = abs(fit.intercept) / signal
        if fraction > _OFFSET_FRACTION_WARN:
            issues.append(
                _warn(
                    "offset_mv",
                    f"Electrode offset is {fit.intercept:+.4g} mV, which is "
                    f"{fraction:.0%} of the {signal:.4g} mV signal at ΔT = "
                    f"{widest:.3g} K. A single-point measurement would have "
                    "attributed that share to the Seebeck coefficient.",
                )
            )

    if (
        math.isfinite(fit.intercept_stderr)
        and fit.intercept_stderr > 0
        and abs(fit.intercept) > _OFFSET_SIGNIFICANCE_SIGMA * fit.intercept_stderr
    ):
        issues.append(
            _warn(
                "offset_mv",
                f"Intercept {fit.intercept:+.4g} ± {fit.intercept_stderr:.4g} mV differs "
                "from zero by more than two standard errors, so the offset is a real "
                "feature of the cell rather than fit noise.",
            )
        )

    return issues


def _bows(delta_t: np.ndarray, delta_v: np.ndarray, linear_residuals: np.ndarray) -> bool:
    """True when a curve fits materially better than a line.

    R² cannot see curvature on a short series: it measures how *big* the
    residuals are, and a deliberately quadratic five-point series still scores
    0.963, which passes for a good fit almost anywhere.

    The obvious alternative — compare the residuals at the ends against those in
    the middle, relative to their own spread — is self-defeating. Curvature
    inflates the residual spread, which raises the very threshold it has to
    clear, so the stronger the curvature the harder it is to detect.

    Fitting a parabola and asking whether it explains significantly more escapes
    that trap, because the noise estimate then comes from the quadratic fit,
    where the curvature has already been absorbed rather than being counted as
    noise. This is the standard extra-sum-of-squares F test.
    """
    n = int(delta_t.size)
    dof = n - 3  # three parameters in a parabola
    if n < _MIN_POINTS_FOR_CURVATURE or dof <= 0:
        return False

    ss_linear = float(np.sum(linear_residuals**2))
    if ss_linear <= 0:
        return False  # the line already fits exactly; nothing to improve on

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            coeffs = np.polyfit(delta_t, delta_v, 2)
    except (np.linalg.LinAlgError, ValueError):
        return False  # ill-conditioned: report nothing rather than guess

    ss_quadratic = float(np.sum((delta_v - np.polyval(coeffs, delta_t)) ** 2))
    if ss_quadratic <= 0:
        return True  # a parabola fits exactly where the line did not

    f_statistic = (ss_linear - ss_quadratic) / (ss_quadratic / dof)
    return f_statistic > _CURVATURE_F


def _clean(value: float) -> float | None:
    """Round for the record, or None when the number does not exist.

    NaN is not JSON-safe and, more importantly, reads as a value. `None` says
    the quantity was not determined, which is the honest report for a standard
    error with no degrees of freedom behind it.
    """
    return round(value, 6) if math.isfinite(value) else None


def _warn(field: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        field=field,
        severity=Severity.WARNING,
        message=message,
        detected_at=utc_now(),
    )


def _error(message: str) -> AnalyzerOutput:
    return AnalyzerOutput(
        outputs={},
        derived_arrays={},
        issues=(
            ValidationIssue(
                field="analyze",
                severity=Severity.ERROR,
                message=message,
                detected_at=utc_now(),
            ),
        ),
    )
