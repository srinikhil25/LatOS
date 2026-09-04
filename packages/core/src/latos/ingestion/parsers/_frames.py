"""Shared geometry for microscopy frame parsers.

The JEM-2100F, and most microscope capture software, writes a square image area
with an info bar in extra rows beneath it — the bar holding the microscope, the
accelerating voltage, the field of view and the drawn scale. So a frame taller
than it is wide carries a scale and a square one does not, which is exactly the
test `latos.analysis.microscopy.calibration.split_info_bar` applies before
decoding. Applied here it needs the image dimensions rather than the pixels,
which is what lets a 1.3 GB folder of frames be triaged from their headers.

Shared by the JPEG and TIFF parsers because it is the same instrument writing
the same layout in two containers. The BMP parser deliberately does NOT use it:
its trailing rows are an 8-px element caption, not a calibration bar, and
reusing this test there would manufacture a pixel size that does not exist.
"""

from __future__ import annotations

from typing import Any

from latos.core.enums import Severity
from latos.core.models import ValidationIssue, utc_now

__all__ = ["info_bar_geometry", "no_info_bar_issue"]


def info_bar_geometry(width: int, height: int) -> dict[str, Any]:
    """Whether an info bar is present, and how tall it is."""
    present = height > width
    return {
        "info_bar_present": present,
        "info_bar_height_px": int(height - width) if present else 0,
        "image_area_px": int(width) if present else int(min(width, height)),
    }


def no_info_bar_issue() -> ValidationIssue:
    """The warning for a frame saved without its bar.

    A WARNING rather than an INFO because it is not a remark about the file, it
    is a limit on what the file can ever support: with no bar there is no pixel
    size, so no length measured on the frame means anything, and a lattice
    spacing derived from it would be a number with no units behind it.
    """
    return ValidationIssue(
        field="info_bar",
        severity=Severity.WARNING,
        message=(
            "Frame is square, so it was saved without the instrument "
            "info bar. No pixel size can be recovered from it and no "
            "length measured on it is meaningful."
        ),
        detected_at=utc_now(),
    )
