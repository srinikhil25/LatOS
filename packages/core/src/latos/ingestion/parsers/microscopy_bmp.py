"""Parser for `.bmp` frames — chiefly JEOL EDS elemental maps (metadata-only).

File format
-----------
BMP is what JEOL's Analysis Station writes for the *images* of an EDS session:
one map per element per view, plus a bright-field reference, sitting beside the
`.emsa` spectra of the same acquisition. The filename is the only place the
element is recorded:

    View000 Ti K.bmp     titanium, K line, view 000
    View000 Mo L.bmp     molybdenum, L line
    View000 BF.bmp       bright-field reference for that view

Reading that convention is the difference between "19 unreadable images" and
"three views of Ti3AlC2 mapped for Al, C and Ti". Nothing else in the file
carries it — BMP has no tag structure at all.

Why the JPEG parser's info-bar test is NOT reused
-------------------------------------------------
`MicroscopyJpegParser` treats a frame taller than it is wide as carrying the
instrument's info bar, which is true of the JEM-2100F's JPEG exports. It is
**not** true here, and copying it would manufacture a calibration that does not
exist. Measured on this dataset:

- the 267x275 maps carry an 8-row white strip holding the element caption;
- a 512x568 grayscale frame carries a 56-row dark strip.

Neither is proportioned like `JEOL_2100F`, whose cells are measured against a
2048-px reference bar. So the trailing strip is recorded as an observation and
nothing is claimed about pixel size. A BMP here is qualitative by default.

What is recorded
----------------
Metadata only, as for TIFF and JPEG: dimensions, colour mode, the trailing
strip, and — when the filename says so — the view, element and X-ray line.

Technique: `EDS` when the name identifies an element map or a bright-field
reference, because that is what the file is however it was acquired. Otherwise
`SEM`, the placeholder the orchestrator's folder-aware refinement corrects.

Validation policy: see `xrd_rigaku_txt.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image, UnidentifiedImageError

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["MicroscopyBmpParser"]

# BMP files open with the "BM" signature.
_BMP_MAGIC = b"BM"

# `View000 Ti K` / `View000 BF` — the Analysis Station export convention.
_MAP_NAME_RE = re.compile(r"^view\s*(?P<view>\d+)\s+(?P<rest>.+?)\s*$", re.IGNORECASE)
# An X-ray line: a shell, optionally a sub-line and an index (K, Ka, Ka1, Lb2).
_XRAY_LINE_RE = re.compile(r"^(?P<shell>[KLM])(?P<sub>[ab])?(?P<index>[12])?$")
# The bright-field reference image shares the view's framing but maps nothing.
_BRIGHT_FIELD = "bf"

# Checked against the real symbols so an arbitrary filename cannot be read as
# chemistry. Written out in full rather than derived, because a parser should not
# import a materials library to answer whether two letters name an element.
# fmt: off
_ELEMENT_SYMBOLS = frozenset([
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
])
# fmt: on

# Confidence when the filename identifies the file as an EDS map or reference,
# versus a bare BMP that is merely sitting in a project folder. The registry
# threshold is 0.5, so a plain BMP is still claimed — but the gap records that
# one of these two answers rests on much more evidence than the other.
_CONFIDENCE_NAMED = 1.0
_CONFIDENCE_PLAIN = 0.6


class MicroscopyBmpParser(BaseParser):
    """Parser for `.bmp` EDS maps and microscopy frames (metadata-only)."""

    name: ClassVar[str] = "microscopy-bmp"
    version: ClassVar[str] = "1.0.0"
    # Placeholder for an unnamed BMP; an identified map returns EDS from `parse`.
    technique: ClassVar[Technique] = Technique.SEM
    supported_extensions: ClassVar[tuple[str, ...]] = (".bmp",)

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """Higher confidence when the filename names an element map."""
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("rb") as fh:
                head = fh.read(2)
        except OSError:
            return 0.0
        if not head.startswith(_BMP_MAGIC):
            return 0.0
        return _CONFIDENCE_NAMED if _parse_map_name(path.stem) else _CONFIDENCE_PLAIN

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Read dimensions, mode and the filename's element. Pixels NOT loaded."""
        issues: list[ValidationIssue] = []
        metadata: dict[str, Any] = {}

        try:
            with Image.open(path) as img:
                width, height = img.size
                metadata["image_width"] = int(width)
                metadata["image_height"] = int(height)
                metadata["mode"] = str(img.mode)
        except (OSError, UnidentifiedImageError) as exc:
            issues.append(
                ValidationIssue(
                    field="file",
                    severity=Severity.ERROR,
                    message=f"Could not read BMP: {exc}",
                    detected_at=utc_now(),
                ),
            )
            return self._empty_result(issues)

        # Recorded, never interpreted — see the module docstring on why this is
        # not the JPEG parser's info bar.
        metadata["trailing_strip_px"] = max(0, metadata["image_height"] - metadata["image_width"])

        named = _parse_map_name(path.stem)
        technique = self.technique
        if named is not None:
            metadata.update(named)
            technique = Technique.EDS
            if named.get("image_kind") == "element_map":
                issues.append(
                    ValidationIssue(
                        field="eds_map",
                        severity=Severity.INFO,
                        message=(
                            f"Elemental map for {named['element']} "
                            f"{named['xray_line']}. A map shows where a line's counts "
                            f"fall, not how much of the element is present; "
                            f"quantification comes from the spectra of the same "
                            f"acquisition, not from this image."
                        ),
                        detected_at=utc_now(),
                    ),
                )

        return ParsedData(
            technique=technique,
            arrays={},  # Metadata-only, as for TIFF and JPEG.
            metadata=metadata,
            instrument=None,
            measured_at=None,
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
            instrument="Microscopy (.bmp)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )


# ─── Module-level helpers ───────────────────────────────────────────
def _parse_map_name(stem: str) -> dict[str, Any] | None:
    """Decode `View000 Ti K` / `View000 BF`, or None if the name is not one.

    Deliberately strict. The element is checked against the real symbols and the
    line against the shells that exist, so an arbitrary two-word filename cannot
    be read as chemistry — inventing an element from a filename is exactly the
    kind of confident wrong answer this tool exists to avoid.
    """
    match = _MAP_NAME_RE.match(stem.strip())
    if match is None:
        return None
    view = f"View{int(match.group('view')):03d}"
    rest = match.group("rest").split()

    if len(rest) == 1 and rest[0].lower() == _BRIGHT_FIELD:
        return {"view": view, "image_kind": "bright_field"}

    line_parts = 2
    if len(rest) != line_parts:
        return None
    element, line = rest[0], rest[1]
    if element not in _ELEMENT_SYMBOLS or not _XRAY_LINE_RE.match(line):
        return None
    return {
        "view": view,
        "image_kind": "element_map",
        "element": element,
        "xray_line": line,
    }
