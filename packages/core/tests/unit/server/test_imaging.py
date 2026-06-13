"""Unit tests for `latos.server.imaging.render_to_png`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from latos.server.imaging import render_to_png

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _save_tif(path: Path, array: np.ndarray) -> Path:
    # Pillow infers the mode from the numpy dtype (uint8 -> "L",
    # uint16 -> "I;16", 3-channel -> "RGB"); the `mode=` arg is
    # deprecated in Pillow 13.
    Image.fromarray(array).save(path, format="TIFF")
    return path


def test_renders_8bit_grayscale(tmp_path: Path):
    array = (np.arange(64 * 64, dtype=np.uint8) % 255).reshape(64, 64)
    png = render_to_png(_save_tif(tmp_path / "g8.tif", array))
    assert png.startswith(_PNG_MAGIC)


def test_renders_rgb(tmp_path: Path):
    array = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    png = render_to_png(_save_tif(tmp_path / "rgb.tif", array))
    assert png.startswith(_PNG_MAGIC)


def test_normalizes_16bit_grayscale(tmp_path: Path):
    # A 16-bit ramp well outside 0-255 must be stretched, not clipped.
    array = (np.linspace(1000, 60000, 48 * 48, dtype=np.uint16)).reshape(48, 48)
    path = _save_tif(tmp_path / "g16.tif", array)
    png = render_to_png(path)
    assert png.startswith(_PNG_MAGIC)
    # Reload the PNG: the stretch should span the full 8-bit range.
    import io

    out = np.asarray(Image.open(io.BytesIO(png)))
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


def test_flat_image_does_not_divide_by_zero(tmp_path: Path):
    array = np.full((16, 16), 4242, dtype=np.uint16)
    path = _save_tif(tmp_path / "flat.tif", array)
    png = render_to_png(path)
    assert png.startswith(_PNG_MAGIC)
