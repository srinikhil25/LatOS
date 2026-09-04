"""Microscopy parser for `.jpg`/`.jpeg` frames (metadata-only).

File format
-----------
JPEG is what the JEM-2100F and most microscope capture software write when
the operator saves a frame rather than exporting a TIFF. It carries none of
TIFF's tag structure, so almost everything about the acquisition lives in the
**info bar** the instrument renders into the image itself — microscope,
accelerating voltage, field of view, magnification, and the drawn scale bar.

Why this parser exists
----------------------
Without it every microscope JPEG is unclassified and never reaches the
analysis layer. On the MXene dataset that was 609 of 860 files: the entire
TEM arm, including the frames whose defects were already known. Latos owns a
working info-bar decoder in `latos.analysis.microscopy.calibration`; it simply
had nothing to decode, because ingestion dropped the files first.

What is recorded, and what is deliberately not
----------------------------------------------
Metadata only. Pixels are not loaded — the same contract as
`MicroscopyTifParser`, and for the same reason: 609 frames is 1.3 GB, and the
one question worth answering at ingest is answerable from the image header.

That question is whether the frame **has an info bar at all**. These exports
write a square image area with the bar in extra rows beneath it, so a frame
taller than it is wide carries a bar and a square one does not — exactly the
test `split_info_bar` applies, and it needs the dimensions rather than the
pixels. A frame with no bar has no recoverable scale: no length can ever be
derived from it, and saying so here is what stops it being averaged in later.

Field-of-view decoding stays in the analysis layer. It needs per-instrument
glyph templates, and a parser has no business hunting for a template file on
disk. `decode_field_of_view` already refuses the impossible readings these
exports occasionally write (a field of view of 2 metres, written when the
instrument failed to record the magnification), so nothing is lost by leaving
that where it is — it just becomes reachable.

Technique inference: JPEG says nothing about modality, so this defaults to
`Technique.SEM` and the orchestrator's folder-aware refinement promotes it to
TEM/STEM where the folders say so. Same contract as the TIFF parser.

Validation policy: see `xrd_rigaku_txt.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from PIL import ExifTags, Image, UnidentifiedImageError

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData
from latos.ingestion.parsers._frames import info_bar_geometry, no_info_bar_issue

__all__ = ["MicroscopyJpegParser"]

# Every JPEG opens with the SOI marker followed by the first segment marker.
_JPEG_MAGIC = b"\xff\xd8\xff"

# EXIF tags worth keeping when a capture program wrote any. Microscope exports
# usually write none of these, which is itself worth recording.
_EXIF_TAGS_OF_INTEREST = frozenset(
    {"Make", "Model", "Software", "DateTime", "DateTimeOriginal", "Artist", "ImageDescription"}
)

# EXIF timestamps are "YYYY:MM:DD HH:MM:SS" — colons in the date, per the spec.
_EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


class MicroscopyJpegParser(BaseParser):
    """Parser for microscopy `.jpg`/`.jpeg` frames (metadata-only)."""

    name: ClassVar[str] = "microscopy-jpeg"
    version: ClassVar[str] = "1.0.0"
    # Default to SEM; the orchestrator refines from TEM/ STEM/ folder names.
    technique: ClassVar[Technique] = Technique.SEM
    supported_extensions: ClassVar[tuple[str, ...]] = (".jpg", ".jpeg")

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """Confidence 1.0 if the file opens with the JPEG SOI marker."""
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("rb") as fh:
                head = fh.read(3)
        except OSError:
            return 0.0
        return 1.0 if head.startswith(_JPEG_MAGIC) else 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Read dimensions, EXIF and info-bar presence. Pixels NOT loaded."""
        issues: list[ValidationIssue] = []
        metadata: dict[str, Any] = {}
        measured_at: datetime | None = None

        try:
            # Pillow reads the header lazily, so this costs no pixel decode.
            with Image.open(path) as img:
                width, height = img.size
                metadata["image_width"] = int(width)
                metadata["image_height"] = int(height)
                metadata["mode"] = str(img.mode)
                exif = _extract_exif(img)
            metadata.update(exif)
            measured_at = _parse_exif_datetime(
                exif.get("DateTimeOriginal") or exif.get("DateTime"), issues
            )
        except (OSError, UnidentifiedImageError) as exc:
            issues.append(
                ValidationIssue(
                    field="file",
                    severity=Severity.ERROR,
                    message=f"Could not read JPEG: {exc}",
                    detected_at=utc_now(),
                ),
            )
            return self._empty_result(issues)

        metadata.update(info_bar_geometry(metadata["image_width"], metadata["image_height"]))
        if not metadata["info_bar_present"]:
            issues.append(no_info_bar_issue())

        return ParsedData(
            technique=self.technique,
            arrays={},  # Metadata-only, as for TIFF.
            metadata=metadata,
            instrument=_build_instrument_name(metadata),
            measured_at=measured_at,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )

    def _empty_result(self, issues: list[ValidationIssue]) -> ParsedData:
        """Minimal ParsedData for a frame that could not be opened."""
        return ParsedData(
            technique=self.technique,
            arrays={},
            metadata={},
            instrument="Microscopy (.jpg)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )


# ─── Module-level helpers ───────────────────────────────────────────
def _extract_exif(img: Image.Image) -> dict[str, Any]:
    """EXIF values we care about, as plain strings. Absent EXIF is normal."""
    try:
        raw = img.getexif()
    except (OSError, AttributeError):
        return {}
    out: dict[str, Any] = {}
    for tag_id, value in raw.items():
        name = ExifTags.TAGS.get(tag_id)
        if name in _EXIF_TAGS_OF_INTEREST and isinstance(value, str) and value.strip():
            out[name] = value.strip()
    return out


def _build_instrument_name(metadata: dict[str, Any]) -> str:
    """Best available instrument label from EXIF, else a format-only fallback."""
    make = str(metadata.get("Make", "")).strip()
    model = str(metadata.get("Model", "")).strip()
    if make and model:
        return model if model.startswith(make) else f"{make} {model}"
    return make or model or str(metadata.get("Software", "")).strip() or "Microscopy (.jpg)"


def _parse_exif_datetime(value: Any, issues: list[ValidationIssue]) -> datetime | None:
    """Parse an EXIF timestamp; warn (never raise) on a malformed one."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip(), _EXIF_DATETIME_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        issues.append(
            ValidationIssue(
                field="datetime",
                severity=Severity.WARNING,
                message=f"Unparseable EXIF timestamp {value!r}; acquisition time not recorded.",
                detected_at=utc_now(),
            ),
        )
        return None
