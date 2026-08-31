"""XPS/ESCA parser for multi-region text exports (several regions in one file).

File format
-----------
Regions are concatenated, each introduced by a ``Dataset`` line and carrying its
own acquisition settings before a four-column block::

    Dataset wide:13(1-MX-NO-50)
    Dwell Time (s)	0.100000
    Number of sweeps	1
    Kinetic Energy(eV)	Binding Energy(eV)	Intensity(cps)	Transmission Value
    53.600000	1200.000000	98540.000000	1.000000
    ...
    Dataset Ti 2p:14(1-MX-NO-50)
    Dwell Time (s)	0.100000
    Number of sweeps	25
    ...

Why the sweep count is the whole point
--------------------------------------
**Each region is acquired with its own number of sweeps**, and the exported
intensity is a raw total, not a rate. In a typical survey the core lines that
matter get 25 sweeps while the strong lines get 15, so comparing raw counts
between two regions overstates the first by a factor of 1.67 before any
chemistry is involved. Nothing in the file warns about this and the numbers
look perfectly ordinary.

So ``intensity_per_sweep`` is published alongside the raw counts, already
divided by (sweeps x dwell). It is the array to use for anything that compares
regions - composition, ratios, elemental quantification. The raw ``intensity_cps``
is kept for provenance.

Storage layout
--------------
Regions have different lengths, and `ParsedData` requires all arrays to be
co-indexed. They are therefore concatenated end to end with a ``region_index``
array marking which region each point belongs to, and `metadata["regions"]`
carrying the name, row span and acquisition settings of each. Nothing is lost
and the result still writes as one flat table.

Validation policy
-----------------
Never raises. Problems surface as `ValidationIssue`s:

* no regions found                    -> ERROR   (not a multi-region export)
* a region has no data rows           -> WARNING (named, and dropped)
* sweep or dwell missing for a region -> WARNING (per-sweep intensity unavailable)
* sweep counts differ between regions -> INFO    (always worth stating; see above)
* binding energy non-monotonic        -> WARNING (within a named region)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.models import ValidationIssue, utc_now
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = ["MultiRegionXpsTxtParser"]

# "Dataset Ti 2p:14(1-MX-NO-50)" -> region, scan number, sample id.
_DATASET_RE = re.compile(r"^Dataset\s+(.+?):(\d+)\((.*)\)\s*$")
_DWELL_RE = re.compile(r"^Dwell\s+Time.*?\t([\d.eE+-]+)\s*$", re.IGNORECASE)
_SWEEPS_RE = re.compile(r"^Number\s+of\s+sweeps\s*\t([\d.eE+-]+)\s*$", re.IGNORECASE)

# The four-column block: kinetic energy, binding energy, intensity, transmission.
_EXPECTED_COLUMNS = 4

_SNIFF_BYTES = 512


def _issue(message: str, severity: Severity, field: str = "xps") -> ValidationIssue:
    return ValidationIssue(field=field, message=message, severity=severity, detected_at=utc_now())


def _slug(name: str) -> str:
    """`Ti 2p` -> `ti_2p`, for use in an array or metadata key."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


class MultiRegionXpsTxtParser(BaseParser):
    """Multi-region XPS/ESCA text export, one file holding every region."""

    name: ClassVar[str] = "xps-multiregion-txt"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.XPS
    supported_extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def can_parse(self, path: Path) -> float:
        """Confidence from the `Dataset <region>:<n>(<sample>)` marker."""
        try:
            head = path.read_bytes()[:_SNIFF_BYTES]
        except OSError:
            return 0.0
        text = head.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if _DATASET_RE.match(line.rstrip("\r")):
                return 1.0
            if line.strip():
                # The marker must open the file; anything else first means
                # this is some other format that happens to mention Dataset.
                break
        return 0.0

    def parse(self, path: Path) -> ParsedData:
        """Read every region into one concatenated, region-indexed table."""
        issues: list[ValidationIssue] = []
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            return self._empty(f"Could not read the file: {exc}")

        regions = self._split_regions(lines)
        if not regions:
            return self._empty(
                "No 'Dataset' markers — this is not a multi-region XPS export.",
            )

        kinetic: list[np.ndarray] = []
        binding: list[np.ndarray] = []
        intensity: list[np.ndarray] = []
        transmission: list[np.ndarray] = []
        per_sweep: list[np.ndarray] = []
        index: list[np.ndarray] = []
        described: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        cursor = 0

        for order, region in enumerate(regions):
            name, sample, dwell, sweeps, rows = region
            if sample:
                sample_ids.add(sample)
            if not rows:
                issues.append(
                    _issue(
                        f"Region {name!r} contains no data rows and was dropped.", Severity.WARNING
                    )
                )
                continue
            block = np.asarray(rows, dtype=float)
            ke, be, cps, trans = block[:, 0], block[:, 1], block[:, 2], block[:, 3]

            # Per-sweep rate: the only intensity safe to compare across regions.
            scale = (sweeps or 0.0) * (dwell or 0.0)
            if scale > 0:
                rate = cps / scale
            else:
                rate = np.full_like(cps, np.nan)
                issues.append(
                    _issue(
                        f"Region {name!r} gives no sweep count or dwell time, so its "
                        "per-sweep intensity cannot be computed and it must not be "
                        "compared against other regions.",
                        Severity.WARNING,
                    )
                )

            if be.size > 1 and not (np.all(np.diff(be) > 0) or np.all(np.diff(be) < 0)):
                issues.append(
                    _issue(
                        f"Binding energy is not monotonic within region {name!r}.", Severity.WARNING
                    )
                )

            kinetic.append(ke)
            binding.append(be)
            intensity.append(cps)
            transmission.append(trans)
            per_sweep.append(rate)
            index.append(np.full(be.size, float(order)))
            described.append(
                {
                    "name": name,
                    "key": _slug(name),
                    "index": order,
                    "n_points": int(be.size),
                    "dwell_s": dwell,
                    "sweeps": sweeps,
                    "binding_energy_min_ev": float(be.min()),
                    "binding_energy_max_ev": float(be.max()),
                    "row_start": cursor,
                    "row_end": cursor + int(be.size),
                }
            )
            cursor += int(be.size)

        if not described:
            return self._empty("Every region was empty.", extra_issues=issues)

        sweep_counts = {d["sweeps"] for d in described if d["sweeps"]}
        if len(sweep_counts) > 1:
            issues.append(
                _issue(
                    f"Regions were acquired with different sweep counts "
                    f"({sorted(sweep_counts)}). Raw intensities are therefore NOT "
                    "comparable between regions — use `intensity_per_sweep` for any "
                    "composition or ratio.",
                    Severity.INFO,
                )
            )

        metadata: dict[str, Any] = {
            "n_regions": len(described),
            "regions": described,
            "region_names": [d["name"] for d in described],
            "n_points": cursor,
        }
        if len(sample_ids) == 1:
            metadata["sample_id"] = next(iter(sample_ids))
        elif len(sample_ids) > 1:
            metadata["sample_id"] = sorted(sample_ids)
            issues.append(
                _issue(
                    f"The file mixes {len(sample_ids)} sample ids ({sorted(sample_ids)}); "
                    "it may be two exports concatenated.",
                    Severity.WARNING,
                )
            )

        return ParsedData(
            technique=Technique.XPS,
            arrays={
                "kinetic_energy_ev": np.concatenate(kinetic),
                "binding_energy_ev": np.concatenate(binding),
                "intensity_cps": np.concatenate(intensity),
                "intensity_per_sweep": np.concatenate(per_sweep),
                "transmission": np.concatenate(transmission),
                "region_index": np.concatenate(index),
            },
            metadata=metadata,
            instrument="XPS/ESCA (multi-region text export)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )

    @staticmethod
    def _split_regions(
        lines: list[str],
    ) -> list[tuple[str, str, float | None, float | None, list[list[float]]]]:
        """Break the file into ``(name, sample, dwell, sweeps, rows)`` per region."""
        out: list[tuple[str, str, float | None, float | None, list[list[float]]]] = []
        name: str | None = None
        sample = ""
        dwell: float | None = None
        sweeps: float | None = None
        rows: list[list[float]] = []

        def flush() -> None:
            if name is not None:
                out.append((name, sample, dwell, sweeps, rows))

        for raw in lines:
            line = raw.rstrip("\r")
            header = _DATASET_RE.match(line)
            if header:
                flush()
                name, sample = header.group(1).strip(), header.group(3).strip()
                dwell = sweeps = None
                rows = []
                continue
            if name is None:
                continue
            dwell_match = _DWELL_RE.match(line)
            if dwell_match:
                dwell = float(dwell_match.group(1))
                continue
            sweeps_match = _SWEEPS_RE.match(line)
            if sweeps_match:
                sweeps = float(sweeps_match.group(1))
                continue
            parts = line.split("\t")
            if len(parts) != _EXPECTED_COLUMNS:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue  # the column-caption row lands here
        flush()
        return out

    def _empty(
        self,
        message: str,
        *,
        extra_issues: list[ValidationIssue] | None = None,
    ) -> ParsedData:
        issues = list(extra_issues or [])
        issues.append(_issue(message, Severity.ERROR))
        return ParsedData(
            technique=Technique.XPS,
            arrays={},
            metadata={},
            instrument="XPS/ESCA (multi-region text export)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )
