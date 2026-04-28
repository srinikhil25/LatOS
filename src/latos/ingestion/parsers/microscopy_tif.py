"""Microscopy parser for `.tif`/`.tiff` files (metadata-only).

File format
-----------
TIFF (`.tif`, `.tiff`) — the universal image format used by SEM, TEM,
HR-FE-SEM, optical microscopes, and many other imaging instruments.
Files often carry rich metadata in TIFF tags (Make, Model, DateTime,
Artist, scale-bar info encoded in ImageDescription, etc.).

Stage 1C parses **metadata only**. Pixel arrays are deliberately NOT
loaded into `ParsedData.arrays` — image content handling is deferred to
Stage 5, when the VLM (Qwen3-VL via Ollama) needs them. Until then, the
TIFF file remains on disk and is referenced via `FileRef.path`; only its
metadata flows through SQLite.

This produces a ParsedData with `arrays={}` (matching the `HallXlsParser`
single-temperature pattern). No Parquet file is written — `ArrayStore`
already handles empty-arrays gracefully.

Technique inference: TIFF tags don't reliably distinguish SEM/TEM/STEM
without instrument-specific decoding. We default to `Technique.SEM` and
emit a warning so Stage 2 (sample resolution) can refine via folder
context. (Folder names like `TEM/`, `STEM/`, `SEM/` are user-provided
and Stage 2's job to interpret.)

Validation policy: see `xrd_rigaku_txt.py` — same contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import tifffile

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["MicroscopyTifParser"]

# TIFF "magic bytes" — first 4 bytes identify the byte order + format
# version. Both little- and big-endian variants exist in the wild.
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")

# TIFF DateTime tag format per the spec: "YYYY:MM:DD HH:MM:SS" — note
# the colons in the date portion (not dashes).
_TIFF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"

# Tags we extract into metadata. Tag numbers are TIFF baseline + Exif.
_TAGS_OF_INTEREST: tuple[tuple[int, str], ...] = (
    (256, "image_width"),
    (257, "image_height"),
    (258, "bits_per_sample"),
    (259, "compression"),
    (262, "photometric"),
    (271, "make"),
    (272, "model"),
    (274, "orientation"),
    (277, "samples_per_pixel"),
    (282, "x_resolution"),
    (283, "y_resolution"),
    (296, "resolution_unit"),
    (305, "software"),
    (306, "datetime"),
    (315, "artist"),
    (270, "image_description"),
)


class MicroscopyTifParser(BaseParser):
    """Parser for microscopy `.tif`/`.tiff` files (metadata-only)."""

    name: ClassVar[str] = "microscopy-tif"
    version: ClassVar[str] = "1.0.0"
    # Default to SEM; orchestrator/Stage 2 can override based on context.
    technique: ClassVar[Technique] = Technique.SEM
    supported_extensions: ClassVar[tuple[str, ...]] = (".tif", ".tiff")

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """Confidence 1.0 if file starts with TIFF magic bytes."""
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("rb") as fh:
                head = fh.read(4)
        except OSError:
            return 0.0
        return 1.0 if any(head.startswith(magic) for magic in _TIFF_MAGIC) else 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a `.tif` file's metadata into a `ParsedData`. Pixels NOT loaded."""
        issues: list[ValidationIssue] = []
        metadata: dict[str, Any] = {}
        instrument: str | None = None
        measured_at: datetime | None = None

        try:
            with tifffile.TiffFile(path) as tf:
                if not tf.pages:
                    issues.append(
                        ValidationIssue(
                            field="data",
                            severity=Severity.ERROR,
                            message="TIFF has no pages.",
                            detected_at=utc_now(),
                        ),
                    )
                    return self._empty_result(issues)
                page = tf.pages[0]
                metadata = _extract_metadata_from_page(page)
                instrument = _build_instrument_name(metadata)
                measured_at = _parse_tiff_datetime(metadata.get("datetime"), issues)
                # Always include shape + dtype for downstream code.
                metadata["shape"] = list(page.shape)
                metadata["dtype"] = str(page.dtype)
                metadata["n_pages"] = len(tf.pages)
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    field="file",
                    severity=Severity.ERROR,
                    message=f"Could not read TIFF: {exc}",
                    detected_at=utc_now(),
                ),
            )
            return self._empty_result(issues)
        except tifffile.TiffFileError as exc:
            issues.append(
                ValidationIssue(
                    field="tiff",
                    severity=Severity.ERROR,
                    message=f"Invalid TIFF: {exc}",
                    detected_at=utc_now(),
                ),
            )
            return self._empty_result(issues)

        # Technique inference disclaimer. Folder context is the only
        # reliable signal for SEM/TEM/STEM — Stage 2 handles that.
        issues.append(
            ValidationIssue(
                field="technique",
                severity=Severity.INFO,
                message=(
                    "Technique defaulted to SEM; refine with folder context "
                    "(TEM/, STEM/, SEM/, ...) in Stage 2 sample resolution."
                ),
                detected_at=utc_now(),
            ),
        )

        return ParsedData(
            technique=self.technique,
            arrays={},  # Metadata-only; pixel content deferred to Stage 5.
            metadata=metadata,
            instrument=instrument,
            measured_at=measured_at,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )

    def _empty_result(self, issues: list[ValidationIssue]) -> ParsedData:
        """Build a minimal ParsedData when parsing failed early."""
        return ParsedData(
            technique=self.technique,
            arrays={},
            metadata={},
            instrument=None,
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )


# ─── Module-level helpers ───────────────────────────────────────────
def _extract_metadata_from_page(page: Any) -> dict[str, Any]:
    """Pluck JSON-safe values out of the TIFF tags we care about."""
    metadata: dict[str, Any] = {}
    for tag_number, output_name in _TAGS_OF_INTEREST:
        tag = page.tags.get(tag_number)
        if tag is None:
            continue
        coerced = _coerce_tiff_value(tag.value)
        if coerced is not None:
            metadata[output_name] = coerced
    return metadata


def _coerce_tiff_value(value: Any) -> Any:
    """Coerce a TIFF tag value into a JSON-safe scalar/list, or None.

    TIFF tags can be strings, ints, floats, tuples (for rationals), or
    enum-like objects. We render enums via their string repr (e.g.
    `<COMPRESSION.JPEG: 7>` → `"COMPRESSION.JPEG"`).
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        # TIFF rationals come as 2-element tuples (numerator, denominator).
        # Just JSON-stringify the components.
        coerced_items = [_coerce_tiff_value(v) for v in value]
        return [v for v in coerced_items if v is not None]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return None
    # Enums (tifffile.COMPRESSION etc.) — use repr form.
    s = str(value)
    return s if "object at 0x" not in s else None


def _build_instrument_name(metadata: dict[str, Any]) -> str | None:
    """Combine Make + Model into a single instrument identifier, if available."""
    make = metadata.get("make")
    model = metadata.get("model")
    if make and model:
        return f"{make} {model}"
    return make or model or None


def _parse_tiff_datetime(
    text: Any,
    issues: list[ValidationIssue],
) -> datetime | None:
    """Parse a TIFF DateTime tag (`YYYY:MM:DD HH:MM:SS`) into a tz-aware datetime."""
    if text is None:
        return None
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.strptime(text.strip(), _TIFF_DATETIME_FORMAT)
    except ValueError:
        issues.append(
            ValidationIssue(
                field="datetime",
                severity=Severity.WARNING,
                message=f"Could not parse TIFF DateTime tag: {text!r}",
                detected_at=utc_now(),
            ),
        )
        return None
    # TIFF DateTime is the local time at the camera/instrument with no
    # timezone — same caveat as Bruker SPX. Attach UTC + warn.
    issues.append(
        ValidationIssue(
            field="measured_at",
            severity=Severity.WARNING,
            message=(
                "TIFF DateTime tag has no timezone; attached UTC for consistency. "
                "Treat measured_at as approximate — it is local time at the instrument."
            ),
            detected_at=utc_now(),
        ),
    )
    return dt.replace(tzinfo=UTC)
