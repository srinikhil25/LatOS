"""Pixel-size calibration from a burned-in microscope info bar.

Many microscopes write their settings into a strip along the bottom of the
saved image rather than into file metadata. On the JEOL exports this module was
built for, the JPEG carries no EXIF at all and the APP1 segment is zero-filled,
so the burned-in strip is the *only* record of the field of view. Without it
every spacing measured from the image is a number of pixels, not a length.

Why template matching rather than OCR
------------------------------------
The strip is a fixed bitmap overlay: the same glyphs land on the same pixels in
every image the instrument writes at a given size. That makes exact matching
both simpler and more reliable than a general OCR engine, needs no extra
dependency, and — unlike OCR — fails loudly instead of silently returning a
misread digit. A misread field of view is the worst possible failure here,
because it rescales every downstream result by a plausible-looking factor.

Whole value strings are matched rather than individual characters. A dataset
typically uses only a handful of magnifications, so there are far fewer
distinct strings than glyphs, each one is read once by a human, and the
character-segmentation problem disappears entirely — no deciding whether a
period is a period or a smudge, and no glyphs merging at small sizes.

Calibration convention
----------------------
The reported field width spans the **image area**, not the file, so:

    nm_per_px = field_of_view_nm / image_area_width_px

`measure_scale_bar` exists to check exactly that. The drawn scale bar's length
in pixels times `nm_per_px` should reproduce its printed legend; if it does
not, the convention is wrong for that instrument and every derived length is
wrong with it.

Scope
-----
The cell boundaries are per-instrument, expressed in a reference bar width and
scaled to whatever the image actually is. `JEOL_2100F` is supplied; another
microscope needs its own `InfoBarLayout`, obtained by looking at one image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "JEOL_2100F",
    "Calibration",
    "InfoBarLayout",
    "StripTemplates",
    "decode_field_of_view",
    "harvest_strips",
    "measure_scale_bar",
    "nm_per_pixel",
    "parse_length",
    "split_info_bar",
    "value_strip",
]

# Multipliers onto nanometres for the units these bars print.
_UNITS: dict[str, float] = {"pm": 1e-3, "nm": 1.0, "um": 1e3, "µm": 1e3, "mm": 1e6}

# Ink is this much darker than the strip background. The background is not
# always white - it tracks the image tone - so an absolute threshold fails.
_INK_DELTA = 40

# A field of view wider than a millimetre is not a microscope image; these
# exports write one when the instrument failed to record the magnification.
_MAX_FIELD_NM = 1e6

# Strips whose widths differ by more than this are never compared.
_WIDTH_TOLERANCE_PX = 3

# A scale-bar rule must be at least this long and this solid, so a row of
# legend text is never mistaken for the rule.
_MIN_RULE_PX = 20
_MIN_RULE_SOLIDITY = 0.7

_NDIM_2D = 2

# A printed length is exactly "<magnitude> <unit>".
_LENGTH_PARTS = 2


@dataclass(frozen=True, slots=True)
class InfoBarLayout:
    """Where the cells sit within one instrument's info bar.

    Attributes:
        reference_width: Bar width the `cells` offsets were measured at.
        cells: ``{name: (x0, x1)}`` in reference coordinates.
        inset: Pixels trimmed from each cell edge, to drop the drawn cell
            borders. Without this every column reads as ink and no glyph can
            be segmented.
    """

    reference_width: int
    cells: dict[str, tuple[int, int]]
    inset: int = 3

    def cell_bounds(self, name: str, bar_width: int) -> tuple[int, int]:
        """Cell bounds in the coordinates of a bar `bar_width` px wide."""
        if name not in self.cells:
            raise KeyError(f"unknown cell {name!r}; layout has {sorted(self.cells)}")
        scale = bar_width / self.reference_width
        x0, x1 = self.cells[name]
        pad = max(2, round(self.inset * scale))
        return int(x0 * scale) + pad, int(x1 * scale) - pad


# Measured from the JEM-2100F exports: icon, microscope, accelerating voltage,
# field horizontal width, magnification, then the drawn scale bar.
JEOL_2100F = InfoBarLayout(
    reference_width=2048,
    cells={
        "microscope": (64, 378),
        "voltage": (378, 600),
        "field_of_view": (600, 981),
        "magnification": (981, 1166),
        "scale_bar": (1166, 2047),
    },
)


@dataclass(frozen=True, slots=True)
class Calibration:
    """Result of reading one image's info bar."""

    field_of_view_nm: float | None
    nm_per_px: float | None
    label: str | None
    image_width: int

    @property
    def ok(self) -> bool:
        """True when a usable pixel size was recovered."""
        return self.nm_per_px is not None


def split_info_bar(image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split a frame into ``(image_area, info_bar)``.

    The image area is assumed square, with the bar occupying the extra rows
    beneath it — the layout these instruments write. Returns None when the
    frame is square, meaning it was saved without a bar.

    Raises:
        ValueError: If `image` is not 2-D.
    """
    image = np.asarray(image)
    if image.ndim != _NDIM_2D:
        raise ValueError(f"image must be 2-D grayscale, got shape {image.shape}")
    height, width = image.shape
    if height <= width:
        return None
    return image[:width, :], image[width:, :]


def _ink_mask(bar: np.ndarray, inset: int) -> np.ndarray:
    threshold = int(np.median(bar)) - _INK_DELTA
    mask = bar < threshold
    if inset:
        mask[:inset, :] = False
        mask[-inset:, :] = False
    return mask


def _bands(mask: np.ndarray, *, axis: int, min_size: int) -> list[tuple[int, int]]:
    """Contiguous runs along one axis that contain ink."""
    present = mask.any(axis=axis)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, on in enumerate(np.append(present, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_size:
                runs.append((start, i))
            start = None
    return runs


def value_strip(
    image: np.ndarray,
    cell: str,
    layout: InfoBarLayout = JEOL_2100F,
) -> np.ndarray | None:
    """Tight-cropped boolean bitmap of a cell's value row.

    Each cell holds a label above a value. The value is the lower of the two
    ink bands, which is found rather than assumed so the same code works at
    both bar sizes an instrument writes.

    Returns None when the frame has no info bar or the cell is blank.
    """
    parts = split_info_bar(image)
    if parts is None:
        return None
    _area, bar = parts
    x0, x1 = layout.cell_bounds(cell, bar.shape[1])
    if x1 <= x0:
        return None
    scale = bar.shape[1] / layout.reference_width
    inset = max(2, round(layout.inset * scale))
    mask = _ink_mask(bar[:, x0:x1], inset)
    rows = _bands(mask, axis=1, min_size=3)
    if not rows:
        return None
    r0, r1 = rows[-1]
    band = mask[r0:r1, :]
    cols = np.where(band.any(axis=0))[0]
    if not len(cols):
        return None
    return band[:, cols.min() : cols.max() + 1]


def _pad_to(bitmap: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[: bitmap.shape[0], : bitmap.shape[1]] = bitmap[: shape[0], : shape[1]]
    return out


def _mismatch(a: np.ndarray, b: np.ndarray) -> int:
    shape = (max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1]))
    return int((_pad_to(a, shape) != _pad_to(b, shape)).sum())


def _tolerance(bitmap: np.ndarray) -> float:
    """Pixels allowed to differ: a floor, plus a little slack for JPEG noise."""
    return max(8.0, 0.02 * bitmap.size)


@dataclass
class StripTemplates:
    """Labelled value strips for one instrument, keyed by bar width.

    Templates are per bar width and never rescaled between widths. Resampling a
    1-bit glyph blurs it enough that distinct characters stop separating
    cleanly, so each size an instrument writes gets its own set.
    """

    by_width: dict[int, list[tuple[np.ndarray, str]]] = field(default_factory=dict)

    def add(self, bar_width: int, bitmap: np.ndarray, label: str) -> None:
        """Record `bitmap` as meaning `label` for bars of `bar_width` px."""
        self.by_width.setdefault(int(bar_width), []).append((np.asarray(bitmap, bool), label))

    def match(self, bar_width: int, bitmap: np.ndarray) -> str | None:
        """Label for `bitmap`, or None if nothing matches within tolerance."""
        bitmap = np.asarray(bitmap, bool)
        best: tuple[float, str] | None = None
        for template, label in self.by_width.get(int(bar_width), ()):
            if abs(template.shape[1] - bitmap.shape[1]) > _WIDTH_TOLERANCE_PX:
                continue
            score = _mismatch(template, bitmap)
            if score <= _tolerance(bitmap) and (best is None or score < best[0]):
                best = (score, label)
        return None if best is None else best[1]

    def save(self, path: str | Path) -> None:
        """Write the template set to a compressed ``.npz`` file."""
        payload: dict[str, np.ndarray] = {}
        labels: list[str] = []
        for width, entries in sorted(self.by_width.items()):
            for i, (bitmap, label) in enumerate(entries):
                payload[f"{width}:{i}"] = bitmap
                labels.append(f"{width}:{i}={label}")
        np.savez_compressed(Path(path), _labels=np.array(labels), **payload)

    @classmethod
    def load(cls, path: str | Path) -> StripTemplates:
        """Read a template set written by `save`."""
        data = np.load(Path(path), allow_pickle=False)
        mapping = dict(entry.split("=", 1) for entry in data["_labels"].tolist())
        out = cls()
        for key, label in mapping.items():
            width = int(key.split(":", 1)[0])
            out.add(width, data[key], label)
        return out


def parse_length(text: str) -> float | None:
    """Convert a printed length such as ``"21.7 nm"`` to nanometres.

    Returns None for anything unparseable, including the impossible readings
    these exports occasionally write (a field of view of ``"2 m"`` appears when
    the instrument fails to record the magnification). Silently accepting one
    would rescale a whole condition's results.
    """
    parts = text.split()
    if len(parts) != _LENGTH_PARTS:
        return None
    value, unit = parts
    if unit not in _UNITS:
        return None
    try:
        magnitude = float(value)
    except ValueError:
        return None
    if magnitude <= 0:
        return None
    nanometres = magnitude * _UNITS[unit]
    return None if nanometres > _MAX_FIELD_NM else nanometres


def nm_per_pixel(field_of_view_nm: float, image_width_px: int) -> float:
    """Pixel size, from a field of view spanning the image area's width.

    Raises:
        ValueError: If either argument is not positive.
    """
    if field_of_view_nm <= 0 or image_width_px <= 0:
        raise ValueError(
            f"field of view and width must be positive, "
            f"got {field_of_view_nm!r}, {image_width_px!r}",
        )
    return field_of_view_nm / image_width_px


def decode_field_of_view(
    image: np.ndarray,
    templates: StripTemplates,
    layout: InfoBarLayout = JEOL_2100F,
) -> Calibration:
    """Read the field of view from an image's info bar and derive the scale.

    Never raises on an unreadable bar: an undecodable frame comes back with
    `nm_per_px` None so the caller can exclude it, which is the only safe
    handling for a frame whose scale is unknown.
    """
    image = np.asarray(image)
    parts = split_info_bar(image)
    if parts is None:
        width = image.shape[1] if image.ndim == _NDIM_2D else 0
        return Calibration(None, None, None, width)
    area, bar = parts
    width = area.shape[1]
    strip = value_strip(image, "field_of_view", layout)
    if strip is None:
        return Calibration(None, None, None, width)
    label = templates.match(bar.shape[1], strip)
    if label is None:
        return Calibration(None, None, None, width)
    field_nm = parse_length(label)
    if field_nm is None:
        return Calibration(None, None, label, width)
    return Calibration(field_nm, nm_per_pixel(field_nm, width), label, width)


def harvest_strips(
    strips: list[tuple[int, np.ndarray]],
) -> list[tuple[int, np.ndarray, int]]:
    """Group identical value strips so each distinct one is labelled once.

    Args:
        strips: ``(bar_width, bitmap)`` pairs, typically one per image.

    Returns:
        ``(bar_width, representative_bitmap, count)`` sorted by descending
        count. A dataset with many images usually collapses to a handful of
        entries, which is the point: a human reads those few and every image is
        decoded from them.
    """
    groups: list[list] = []
    for width, raw_bitmap in strips:
        bitmap = np.asarray(raw_bitmap, bool)
        for group in groups:
            if group[0] != width:
                continue
            if abs(group[1].shape[1] - bitmap.shape[1]) > _WIDTH_TOLERANCE_PX:
                continue
            if _mismatch(group[1], bitmap) <= _tolerance(bitmap):
                group[2] += 1
                break
        else:
            groups.append([width, bitmap, 1])
    groups.sort(key=lambda g: -g[2])
    return [(g[0], g[1], g[2]) for g in groups]


def measure_scale_bar(
    image: np.ndarray,
    layout: InfoBarLayout = JEOL_2100F,
) -> int | None:
    """Length in pixels of the drawn scale bar, for checking the calibration.

    The rule is found as the widest nearly-solid horizontal run in the scale
    bar cell's lower band. Returns None when no such run exists.

    Multiply the result by `nm_per_px` and compare against the bar's printed
    legend: agreement confirms that the field of view spans the image width,
    which is the assumption every derived length rests on.
    """
    parts = split_info_bar(image)
    if parts is None:
        return None
    _area, bar = parts
    x0, x1 = layout.cell_bounds("scale_bar", bar.shape[1])
    if x1 <= x0:
        return None
    scale = bar.shape[1] / layout.reference_width
    inset = max(2, round(layout.inset * scale))
    mask = _ink_mask(bar[:, x0:x1], inset)
    # min_size 1: a scale bar may legitimately be drawn as a hairline. Stray
    # single-pixel rows cannot survive the solidity and length checks below,
    # so admitting them here costs nothing.
    rows = _bands(mask, axis=1, min_size=1)
    if not rows:
        return None
    r0, r1 = rows[-1]
    band = mask[r0:r1, :]
    best = 0
    for row in band:
        on = np.where(row)[0]
        if len(on) < _MIN_RULE_PX:
            continue
        span = int(on.max() - on.min() + 1)
        if row[on.min() : on.max() + 1].mean() > _MIN_RULE_SOLIDITY and span > best:
            best = span
    return best or None
