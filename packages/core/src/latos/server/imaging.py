"""Render microscopy images (TIF/…) to PNG for the desktop viewer.

Microscopy files come in awkward flavors: JPEG-compressed TIFs (which
`tifffile` can't decode without `imagecodecs`), 16-bit grayscale, float
data, multi-page stacks. Pillow — already a Latos dependency — reads
all the common cases directly, so we lean on it and only hand-normalize
the high-bit-depth modes that don't map to a displayable 8-bit image.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["render_to_png"]

# PIL modes that carry more than 8 bits per channel and therefore need
# min-max normalization before they can be shown as a normal image.
_HIGH_BIT_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N", "F"})

# Modes that are already directly encodable as PNG without conversion.
_DISPLAY_MODES = frozenset({"L", "LA", "RGB", "RGBA", "P"})


def render_to_png(path: Path) -> bytes:
    """Read an image file and return PNG-encoded bytes.

    Raises whatever Pillow raises for an unreadable/unsupported file;
    the caller (the image endpoint) turns that into an HTTP error.
    """
    with Image.open(path) as image:
        image.load()
        prepared = _to_displayable(image)
        buffer = io.BytesIO()
        prepared.save(buffer, format="PNG")
        return buffer.getvalue()


def _to_displayable(image: Image.Image) -> Image.Image:
    """Coerce any PIL image into an 8-bit, PNG-encodable mode."""
    if image.mode in _HIGH_BIT_MODES:
        return _normalize_to_8bit(image)
    if image.mode in _DISPLAY_MODES:
        return image
    # Anything exotic (CMYK, YCbCr, …) → straight RGB conversion.
    return image.convert("RGB")


def _normalize_to_8bit(image: Image.Image) -> Image.Image:
    """Min-max stretch a high-bit-depth image to 8-bit grayscale.

    A flat image (all pixels equal) maps to all-zero rather than
    dividing by zero.
    """
    data = np.asarray(image).astype(np.float64)
    low = float(data.min())
    high = float(data.max())
    scaled = (data - low) / (high - low) * 255.0 if high > low else np.zeros_like(data)
    return Image.fromarray(scaled.astype(np.uint8), mode="L")
