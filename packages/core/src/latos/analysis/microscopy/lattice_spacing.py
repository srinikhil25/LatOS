"""`MicroscopyLatticeAnalyzer` — lattice spacing from one TEM frame.

Why this exists
---------------
`calibration.py` recovers a pixel size from the burned-in info bar and
`lattice.py` measures a fringe spacing by windowed FFT. Both are tested, and
neither had ever run: no analyzer claimed a microscopy technique, so the
registry never dispatched to them and nothing in the dataset carried a
TEM-derived length. This module is the missing wiring, not new science.

The consequence of that gap was concrete. TEM is the only technique covering
all nine conditions of the MXene dataset, so it was the only route to a
cross-technique check on the basal spacing that XRD reports from the (002)
line — and that check could not be made.

What it produces
----------------
One spacing per frame, pooled over the windows that yielded a detection, with
the FFT resolution floor as its uncertainty. Pooling *across* frames into a
sample-level estimate is `lattice.aggregate_frames`, deliberately not done
here: an analyzer sees one measurement, and one frame largely re-measures one
flake, so a group estimate belongs a level up.

What it refuses
---------------
A frame whose pixel size cannot be recovered yields no length at all, and says
so as a WARNING rather than returning a number of pixels dressed up as
nanometres. That covers three real cases in this dataset: frames saved without
an info bar, frames whose field-of-view strip matches no template, and frames
whose strip reads "2 m" — what the instrument writes when it failed to record
the magnification, which `parse_length` refuses by design.

Where the templates come from
-----------------------------
Value strips are matched against a labelled set a human built once per
instrument (`StripTemplates`). It is dataset-local, because the labels were
read off that dataset's own images, so the analyzer looks for it beside the
data: `templates_path` if given, else the nearest `*_templates.npz` in an
ancestor directory of the frame. Without one, nothing is calibrated and every
frame is refused — loudly, which is the point.

The scale bar
-------------
`nm_per_px` assumes the printed field of view spans the image area's width, and
the rule drawn in the bar is what tests that. The rule's length is reported in
both pixels and nanometres so the assumption can be checked, but no verdict is
issued: the rule's own printed legend lives in a different cell of the info bar
and the template set carries labels only for the field of view, so there is
nothing here to compare it against automatically. Reporting the number and
leaving the judgement to a reader beats inventing a tolerance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, ClassVar

import numpy as np

from latos.analysis.base_analyzer import AnalyzerInputs, AnalyzerOutput, BaseAnalyzer
from latos.analysis.microscopy.calibration import (
    Calibration,
    StripTemplates,
    decode_field_of_view,
    measure_scale_bar,
    split_info_bar,
)
from latos.analysis.microscopy.lattice import (
    DEFAULT_D_WINDOW_NM,
    DEFAULT_MIN_REPEATS,
    DEFAULT_SEARCH_WINDOW_NM,
    LatticePeak,
    scan_frame,
)
from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now

__all__ = ["MicroscopyLatticeAnalyzer"]

# Suffixes this analyzer will open. `.bmp` is excluded on purpose: those
# exports carry no info bar, and `microscopy_bmp` already refuses to borrow a
# calibration from a sibling.
_IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".jpg", ".jpeg"})

_TEMPLATE_GLOB = "*_templates.npz"

# How far up from the frame to look for the dataset's template set.
_TEMPLATE_SEARCH_LEVELS = 8

_NDIM_2D = 2


def _issue(field: str, severity: Severity, message: str) -> ValidationIssue:
    return ValidationIssue(
        field=field,
        severity=severity,
        message=message,
        detected_at=utc_now(),
    )


def _refused(message: str, severity: Severity = Severity.WARNING) -> AnalyzerOutput:
    """No length from this frame, and the reason why.

    A refusal is not an error in the dataset: most frames in a TEM session are
    low-magnification survey shots with no resolved fringes. It is only an
    ERROR when the analyzer could not even try.
    """
    return AnalyzerOutput(outputs={}, issues=(_issue("lattice_d_nm", severity, message),))


@lru_cache(maxsize=8)
def _load_templates(path: str) -> StripTemplates | None:
    """Template sets are small and shared by hundreds of frames, so cache them."""
    try:
        return StripTemplates.load(path)
    except (OSError, ValueError, KeyError):
        return None


def _find_templates(
    image_path: Path,
    given: str | None,
) -> tuple[StripTemplates | None, str | None]:
    """The labelled strip set for this frame, and where it came from."""
    if given:
        return _load_templates(given), given
    for parent in list(image_path.parents)[:_TEMPLATE_SEARCH_LEVELS]:
        for candidate in sorted(parent.glob(_TEMPLATE_GLOB)):
            loaded = _load_templates(str(candidate))
            if loaded is not None:
                return loaded, str(candidate)
    return None, None


def _load_grayscale(path: Path) -> np.ndarray | None:
    """One frame as a 2-D array, or None if it will not open.

    Colour frames are reduced to luminance rather than refused: the info bar and
    the fringes are both monochrome, and an RGB save is a file-format accident.
    """
    try:
        if path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile  # noqa: PLC0415

            array = np.asarray(tifffile.imread(path))
        else:
            from PIL import Image  # noqa: PLC0415

            with Image.open(path) as img:
                array = np.asarray(img.convert("L"))
    except Exception:  # any decode failure is one refusal, not a crash
        return None
    if array.ndim > _NDIM_2D:
        array = array[..., :3].mean(axis=-1)
    return array if array.ndim == _NDIM_2D else None


def _first_image(measurement: Any) -> Path | None:
    for ref in measurement.files:
        path = Path(ref.path)
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            return path
    return None


def _scale_bar(image: np.ndarray, nm_per_px: float) -> dict[str, Any]:
    """The drawn rule, in pixels and in nanometres at this pixel size.

    Reported so a reader can check the calibration convention against the
    rule's printed legend. No automated verdict: the legend sits in a cell the
    template set carries no labels for, so there is nothing to compare with.
    """
    rule_px = measure_scale_bar(image)
    if rule_px is None:
        return {}
    return {
        "scale_bar_px": int(rule_px),
        "scale_bar_nm": round(float(rule_px) * nm_per_px, 3),
    }


def _open_calibrated(
    measurement: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Calibration, str | None] | AnalyzerOutput:
    """Open the frame and recover its pixel size, or say why neither happened.

    Split out from `analyze` because every step here is a way the frame can be
    unusable before any measurement is attempted, and each needs its own
    message. Returns `(image, image_area, calibration, templates_source)` on
    success, and a finished refusal on failure.
    """
    path = _first_image(measurement)
    if path is None:
        return _refused("No image file on this measurement.", Severity.ERROR)

    templates, source = _find_templates(path, params.get("templates_path"))
    if templates is None:
        return _refused(
            "No strip templates found for this instrument, so no pixel size can "
            "be read from the info bar. Point `templates_path` at a set built "
            f"for it, or place one matching {_TEMPLATE_GLOB} beside the data.",
            Severity.ERROR,
        )

    image = _load_grayscale(path)
    if image is None:
        return _refused(f"Could not read {path.name} as a 2-D image.", Severity.ERROR)

    calibration = decode_field_of_view(image, templates)
    if not calibration.ok:
        return _refused(_why_uncalibrated(image, calibration))

    parts = split_info_bar(image)
    return image, (image if parts is None else parts[0]), calibration, source


def _resolvable_window(
    *,
    nm_per_px: float,
    area_px: int,
    tile_size: int,
    d_window: tuple[float, float],
    min_repeats: float,
    min_px_per_period: float,
) -> tuple[tuple[float, float] | None, ValidationIssue | None]:
    """The spacings this frame can actually measure, and what it had to give up.

    Two limits bracket the window, and they bite at opposite magnifications.

    Coarse end — a spacing has to be sampled. The smallest spacing *sought*
    sets this, not the one found: gating on the result would be circular, since
    the answer would decide whether the question was allowed. Below the floor
    the FFT still returns something, but what it returns tracks the pixel size
    rather than the sample — a 2.3 um field in this dataset reported a 2.28 nm
    "spacing" from a period of two pixels, which is aliasing, not a lattice.

    Fine end — a spacing has to repeat. A window spanning S nm holds at most
    S/`min_repeats` of the largest measurable spacing, so at high magnification
    the top of the requested range is unreachable however strong that
    reflection is. That is physical, not a parameter choice: at a 14.4 nm field
    even the whole 2048-px frame gives 5.1 repeats of 2.80 nm. The window is
    narrowed to what the frame supports and the caller is told, rather than the
    range being reported as covered.
    """
    lo, hi = float(d_window[0]), float(d_window[1])
    px_per_period = lo / nm_per_px
    if px_per_period < min_px_per_period:
        return None, _issue(
            "lattice_d_nm",
            Severity.WARNING,
            f"Pixel size {nm_per_px:.4f} nm/px samples the smallest requested "
            f"spacing ({lo:.2f} nm) over {px_per_period:.1f} px, below the "
            f"{min_px_per_period:g} px floor. At this magnification the FFT "
            f"reports its own resolution limit rather than the sample.",
        )

    span_nm = min(int(tile_size), int(area_px)) * nm_per_px
    supportable = span_nm / min_repeats
    if hi <= supportable:
        return (lo, hi), None
    if supportable <= lo:
        return None, _issue(
            "lattice_d_nm",
            Severity.WARNING,
            f"A {span_nm:.1f} nm view cannot hold {min_repeats:g} repeats of even "
            f"the smallest requested spacing ({lo:.2f} nm), so nothing in the "
            f"window is measurable on this frame.",
        )
    return (lo, supportable), _issue(
        "d_window_nm",
        Severity.INFO,
        f"Window narrowed to {lo:.2f}-{supportable:.2f} nm: a {span_nm:.1f} nm view "
        f"holds only {min_repeats:g} repeats of {supportable:.2f} nm, so spacings "
        f"above that are not measurable on this frame however strong they are.",
    )


def _summarise(peaks: tuple[LatticePeak, ...]) -> dict[str, Any]:
    """Frame-level numbers from the windows that yielded a detection.

    The median over windows, not the mean: a frame often contains one well
    resolved flake and several marginal ones, and the mean follows the
    marginal tail.
    """
    d_values = [p.d_nm for p in peaks]
    centre = float(median(d_values))
    return {
        "lattice_d_nm": round(centre, 4),
        "lattice_d_err_nm": round(float(median([p.d_err_nm for p in peaks])), 4),
        "lattice_d_spread_nm": round(float(np.median(np.abs(np.asarray(d_values) - centre))), 4),
        "n_windows_detected": len(peaks),
        "window_d_nm": [round(v, 4) for v in d_values],
        "window_angle_deg": [round(p.angle_deg, 1) for p in peaks],
        "window_contrast": [round(p.contrast, 1) for p in peaks],
        "window_n_repeats": [round(p.n_repeats, 1) for p in peaks],
        "max_orders_observed": max(p.n_orders for p in peaks),
    }


class MicroscopyLatticeAnalyzer(BaseAnalyzer):
    """Fringe spacing from a calibrated high-resolution TEM frame."""

    name: ClassVar[str] = "microscopy-lattice"
    version: ClassVar[str] = "0.2.0"
    accepts_techniques: ClassVar[tuple[Technique, ...]] = (
        Technique.TEM,
        Technique.STEM,
    )
    default_params: ClassVar[dict[str, Any]] = {
        # Labelled value strips for this instrument. None searches for the
        # dataset's own `*_templates.npz` beside the frame.
        "templates_path": None,
        # Window the FFT is scanned over. 1024 px at these magnifications is a
        # few tens of nanometres — large enough for the repeats the resolution
        # floor needs, small enough that one flake dominates the window.
        "tile_size": 1024,
        # Accepted spacings, nm. The default brackets the basal spacings of
        # MXene and its MAX precursors.
        "d_window_nm": list(DEFAULT_D_WINDOW_NM),
        # Where the fundamental is searched for before harmonics are resolved.
        "search_window_nm": list(DEFAULT_SEARCH_WINDOW_NM),
        # Fewer repeats than this inside a window and the FFT cannot place the
        # peak well enough for the spacing to mean anything.
        "min_repeats": DEFAULT_MIN_REPEATS,
        # Pixels the smallest sought spacing must span, or the frame is refused.
        # On the MXene dataset the magnifications are discrete enough that
        # anything from 15 to 50 selects the same frames; 25 sits mid-range, so
        # the selection does not hinge on the exact value.
        "min_px_per_period": 25.0,
    }

    def accepts(self, measurement: Any) -> bool:
        """Accept a microscopy measurement carrying an openable image file.

        Cheap by contract: suffix only. Whether the frame has a readable info
        bar, and whether it resolves fringes at all, are questions for
        `analyze` — both need the pixels.
        """
        return _first_image(measurement) is not None

    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        """Calibrate the frame, then measure its fringe spacing."""
        params = inputs.params
        opened = _open_calibrated(inputs.measurement, params)
        if isinstance(opened, AnalyzerOutput):
            return opened

        image, area, calibration, source = opened
        nm_per_px = float(calibration.nm_per_px)
        tile_size = int(params.get("tile_size", 1024))
        requested = tuple(params.get("d_window_nm", DEFAULT_D_WINDOW_NM))
        min_repeats = float(params.get("min_repeats", DEFAULT_MIN_REPEATS))

        window, window_issue = _resolvable_window(
            nm_per_px=nm_per_px,
            area_px=min(area.shape),
            tile_size=tile_size,
            d_window=requested,
            min_repeats=min_repeats,
            min_px_per_period=float(params.get("min_px_per_period", 25.0)),
        )
        if window is None:
            return AnalyzerOutput(outputs={}, issues=(window_issue,))

        peaks = scan_frame(
            area,
            nm_per_px,
            tile_size=tile_size,
            d_window_nm=window,
            search_window_nm=tuple(params.get("search_window_nm", DEFAULT_SEARCH_WINDOW_NM)),
            min_repeats=min_repeats,
        )

        context: dict[str, Any] = {
            "nm_per_px": round(nm_per_px, 6),
            "field_of_view_nm": round(float(calibration.field_of_view_nm or 0.0), 3),
            "field_of_view_label": calibration.label,
            "image_area_px": int(calibration.image_width),
            "templates_source": source,
            # What was actually searched, which is not always what was asked
            # for. A caller pooling frames needs to know the range differed.
            "d_window_nm_used": [round(window[0], 3), round(window[1], 3)],
            "px_per_smallest_period": round(window[0] / nm_per_px, 1),
            **_scale_bar(image, nm_per_px),
        }
        issues = () if window_issue is None else (window_issue,)
        if not peaks:
            return AnalyzerOutput(
                outputs={"n_windows_detected": 0, **context},
                issues=(
                    *issues,
                    _issue(
                        "lattice_d_nm",
                        Severity.INFO,
                        "Calibrated, but no window resolved lattice fringes inside "
                        "the accepted spacing range. Expected for a survey frame.",
                    ),
                ),
            )
        return AnalyzerOutput(outputs={**_summarise(peaks), **context}, issues=issues)


def _why_uncalibrated(image: np.ndarray, calibration: Calibration) -> str:
    """Name the specific reason, because the three causes need different fixes."""
    if split_info_bar(image) is None:
        return (
            "Frame saved without its info bar, so it carries no pixel size and no "
            "length measured on it would mean anything."
        )
    if calibration.label is None:
        return (
            "The field-of-view strip matches no template for this bar width. Label "
            "it once and add it to the template set."
        )
    return (
        f"The info bar reads a field of view of {calibration.label!r}, which is not "
        f"a possible one — the instrument writes this when it failed to record the "
        f"magnification. Refused rather than used."
    )
