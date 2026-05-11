"""XRD parser for PANalytical Empyrean `.xrdml` (XML) exports.

File format
-----------
XML with namespace `http://www.xrdml.com/XRDMeasurement/1.7` (or similar
version). Structure relevant to us::

    <xrdMeasurements>
      <sample>
        <id>...</id>
      </sample>
      <xrdMeasurement>
        <usedWavelength>
          <kAlpha1 unit="Angstrom">1.5405980</kAlpha1>
          ...
        </usedWavelength>
        <incidentBeamPath>
          <xRayTube>
            <tension unit="kV">40.0</tension>
            <current unit="mA">15.0</current>
            <anodeMaterial>Cu</anodeMaterial>
          </xRayTube>
        </incidentBeamPath>
        <scan>
          <header>
            <startTimeStamp>2024-06-21T11:00:13+05:30</startTimeStamp>
          </header>
          <dataPoints>
            <positions axis="2Theta" unit="deg">
              <startPosition>5.01...</startPosition>
              <endPosition>100.00...</endPosition>
            </positions>
            <commonCountingTime unit="seconds">58.395</commonCountingTime>
            <intensities unit="counts">2493 2336 2284 ...</intensities>
          </dataPoints>
        </scan>
      </xrdMeasurement>
    </xrdMeasurements>

The 2θ array is *not* stored explicitly — only the start/end positions and
the count of intensity values are. We synthesize 2θ via `np.linspace(start,
end, n_points)`.

Validation policy: see `xrd_rigaku_txt.py` — same contract. Never raise,
emit `ValidationIssue`s for missing/malformed fields.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from xml.etree import ElementTree as ET

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["PanalyticalXrdmlParser"]


# How many bytes to read from the file head for can_parse sniffing.
# We just need to see the root element + its xmlns.
_SNIFF_BYTES = 2048

# Signatures for can_parse. Both must appear in the head for a confident match.
_XRDML_SIGNATURES = (b"xrdMeasurements", b"xrdml.com")


def _local(tag: str) -> str:
    """Strip XML namespace from a tag name. `{ns}foo` → `foo`."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_first(root: ET.Element, local_name: str) -> ET.Element | None:
    """Find the first descendant whose local name (namespace-stripped) matches.

    PANalytical's XRDML uses a versioned namespace
    (`http://www.xrdml.com/XRDMeasurement/1.7`, `1.5`, etc.). Hardcoding the
    URI would make the parser brittle across instrument firmwares. Instead
    we walk the tree comparing local names — slightly slower, much more
    forgiving.
    """
    for el in root.iter():
        if _local(el.tag) == local_name:
            return el
    return None


def _find_text(root: ET.Element, local_name: str) -> str | None:
    """Return the text content of the first matching element, or None."""
    el = _find_first(root, local_name)
    if el is None or el.text is None:
        return None
    return el.text.strip() or None


def _find_text_in(parent: ET.Element, local_name: str) -> str | None:
    """Like `_find_text` but scoped to descendants of `parent`."""
    for el in parent.iter():
        if _local(el.tag) == local_name and el.text is not None:
            text = el.text.strip()
            if text:
                return text
    return None


def _to_float(value: str | None) -> float | None:
    """Coerce optional string to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class PanalyticalXrdmlParser(BaseParser):
    """Parser for PANalytical Empyrean `.xrdml` XML files."""

    name: ClassVar[str] = "panalytical-xrdml"
    version: ClassVar[str] = "1.0.1"
    technique: ClassVar[Technique] = Technique.XRD
    supported_extensions: ClassVar[tuple[str, ...]] = (".xrdml",)

    # ─── can_parse ───────────────────────────────────────────────────
    def can_parse(self, path: Path) -> float:
        """Return 1.0 for true XRDML files, 0.0 otherwise.

        The signature pair (`xrdMeasurements` root + `xrdml.com` namespace)
        is sufficiently specific that a partial-match tier isn't useful —
        any XML file that has those strings IS an XRDML.
        """
        if not self._extension_matches(path):
            return 0.0
        try:
            with path.open("rb") as fh:
                head = fh.read(_SNIFF_BYTES)
        except OSError:
            return 0.0
        if all(sig in head for sig in _XRDML_SIGNATURES):
            return 1.0
        return 0.0

    # ─── parse ───────────────────────────────────────────────────────
    def parse(self, path: Path) -> ParsedData:
        """Parse a `.xrdml` file into a `ParsedData`."""
        issues: list[ValidationIssue] = []

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except ET.ParseError as exc:
            issues.append(
                ValidationIssue(
                    field="xml",
                    severity=Severity.ERROR,
                    message=f"Malformed XML: {exc}",
                    detected_at=utc_now(),
                ),
            )
            return self._empty_result(issues)
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    field="file",
                    severity=Severity.ERROR,
                    message=f"Could not read file: {exc}",
                    detected_at=utc_now(),
                ),
            )
            return self._empty_result(issues)

        two_theta, intensity = self._extract_arrays(root, issues)
        metadata = self._extract_metadata(root, len(two_theta))
        instrument = self._extract_instrument(root)
        measured_at = self._extract_measured_at(root, issues)

        if not two_theta:
            issues.append(
                ValidationIssue(
                    field="data",
                    severity=Severity.ERROR,
                    message="No <intensities> element found or empty.",
                    detected_at=utc_now(),
                ),
            )

        arrays: dict[str, np.ndarray] = (
            {
                "two_theta": np.asarray(two_theta, dtype=np.float64),
                "intensity": np.asarray(intensity, dtype=np.float64),
            }
            if two_theta
            else {}
        )

        return ParsedData(
            technique=self.technique,
            arrays=arrays,
            metadata=metadata,
            instrument=instrument,
            measured_at=measured_at,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )

    # ─── Internals ───────────────────────────────────────────────────
    def _empty_result(self, issues: list[ValidationIssue]) -> ParsedData:
        """Build a minimal ParsedData when parsing failed early."""
        return ParsedData(
            technique=self.technique,
            arrays={},
            metadata={},
            # Fall back to the format label when the XML didn't yield a
            # specific instrument name (the success path may override).
            instrument="PANalytical (.xrdml)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )

    @staticmethod
    def _extract_arrays(
        root: ET.Element,
        issues: list[ValidationIssue],
    ) -> tuple[list[float], list[float]]:
        """Pull `<intensities>` and `<positions axis="2Theta">` from the tree.

        Returns (two_theta_list, intensity_list) — empty lists if anything
        is missing or unparseable.
        """
        intensity_el = _find_first(root, "intensities")
        if intensity_el is None or intensity_el.text is None:
            return [], []

        try:
            intensities = [float(x) for x in intensity_el.text.split()]
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    field="intensities",
                    severity=Severity.ERROR,
                    message=f"Could not parse <intensities> as floats: {exc}",
                    detected_at=utc_now(),
                ),
            )
            return [], []

        if not intensities:
            return [], []

        # Find the 2θ positions element. There may be multiple <positions>
        # blocks (one per axis: 2Theta, Omega, etc.); we pick the one with
        # axis="2Theta", falling back to the first if none is labeled.
        start, end = None, None
        for pos in root.iter():
            if _local(pos.tag) == "positions" and pos.get("axis") == "2Theta":
                start = _to_float(_find_text_in(pos, "startPosition"))
                end = _to_float(_find_text_in(pos, "endPosition"))
                break

        if start is None or end is None:
            issues.append(
                ValidationIssue(
                    field="positions",
                    severity=Severity.WARNING,
                    message="Missing 2Theta start/end positions; using indices as x-axis.",
                    detected_at=utc_now(),
                ),
            )
            two_theta = list(range(len(intensities)))
        else:
            # XRDML stores only start, end, and the count of intensity values.
            # 2θ for each point is computed via linspace.
            two_theta = list(np.linspace(start, end, num=len(intensities)))

        return [float(t) for t in two_theta], intensities

    @staticmethod
    def _extract_metadata(root: ET.Element, n_points: int) -> dict[str, Any]:
        """Pluck the headers we care about into a JSON-safe metadata dict."""
        sample_id = _find_text(root, "id")

        # Wavelengths.
        ka1 = ka2 = kbeta = ratio = None
        for el in root.iter():
            if _local(el.tag) == "kAlpha1":
                ka1 = _to_float(el.text)
            elif _local(el.tag) == "kAlpha2":
                ka2 = _to_float(el.text)
            elif _local(el.tag) == "kBeta":
                kbeta = _to_float(el.text)
            elif _local(el.tag) == "ratioKAlpha2KAlpha1":
                ratio = _to_float(el.text)

        # X-ray tube info.
        tube = _find_first(root, "xRayTube")
        if tube is not None:
            tube_name = tube.get("name")
            tension = _to_float(_find_text_in(tube, "tension"))
            current = _to_float(_find_text_in(tube, "current"))
            anode = _find_text_in(tube, "anodeMaterial")
        else:
            tube_name = tension = current = anode = None

        # Counting + scan parameters.
        common_time = _to_float(_find_text(root, "commonCountingTime"))

        # Scan range from the 2Theta positions.
        scan_start = scan_end = None
        for pos in root.iter():
            if _local(pos.tag) == "positions" and pos.get("axis") == "2Theta":
                scan_start = _to_float(_find_text_in(pos, "startPosition"))
                scan_end = _to_float(_find_text_in(pos, "endPosition"))
                break

        # scan element attributes.
        scan_el = _find_first(root, "scan")
        scan_mode = scan_el.get("mode") if scan_el is not None else None
        scan_axis = scan_el.get("scanAxis") if scan_el is not None else None

        return {
            "sample_id": sample_id,
            "wavelength_ka1": ka1,
            "wavelength_ka2": ka2,
            "wavelength_kbeta": kbeta,
            "ratio_ka2_ka1": ratio,
            "tube_name": tube_name,
            "tube_tension_kv": tension,
            "tube_current_ma": current,
            "anode_material": anode,
            "common_counting_time_s": common_time,
            "scan_start_deg": scan_start,
            "scan_finish_deg": scan_end,
            "scan_mode": scan_mode,
            "scan_axis": scan_axis,
            "n_points": n_points,
        }

    @staticmethod
    def _extract_instrument(root: ET.Element) -> str | None:
        """Best-effort instrument string from the X-ray tube name."""
        tube = _find_first(root, "xRayTube")
        if tube is not None:
            name = tube.get("name")
            if name:
                return name
        return None

    @staticmethod
    def _extract_measured_at(
        root: ET.Element,
        issues: list[ValidationIssue],
    ) -> datetime | None:
        """Parse `<startTimeStamp>` as an aware datetime, if present."""
        text = _find_text(root, "startTimeStamp")
        if text is None:
            return None
        try:
            # Format example: "2024-06-21T11:00:13+05:30" — fromisoformat
            # in 3.11+ handles the timezone offset directly.
            dt = datetime.fromisoformat(text)
        except ValueError:
            issues.append(
                ValidationIssue(
                    field="startTimeStamp",
                    severity=Severity.WARNING,
                    message=f"Could not parse <startTimeStamp>: {text!r}",
                    detected_at=utc_now(),
                ),
            )
            return None
        if dt.tzinfo is None:
            # Local-time stamps are technically allowed but tell us nothing.
            # ParsedData rejects naive datetimes, so skip rather than fail.
            issues.append(
                ValidationIssue(
                    field="startTimeStamp",
                    severity=Severity.WARNING,
                    message="<startTimeStamp> lacks timezone info; ignored.",
                    detected_at=utc_now(),
                ),
            )
            return None
        return dt
