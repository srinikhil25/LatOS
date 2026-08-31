"""Thermoelectric parser for Quantum Design PPMS Thermal Transport Option exports.

File format
-----------
Tab-separated ASCII. Line 1 is the sample geometry the operator typed into
MultiVu; line 2 is the column header; the rest is data::

    thickness: 0.250000, width: 5.000000, length: 10.000000
    time(s)	T(K)	B(T)	Position	rho(micr	S(microV	kappa(W/	dT(K)	...
    3.08000000E+2	3.01029602E+2	6.01766351E-6	...

Column names are TRUNCATED by the export (``rho(micr``, ``kappa(W/``,
``PF(mW/mK``), so columns are matched by prefix rather than exact name.

Two layouts exist. The single-channel layout above, and a two-channel layout
used when a second sample is mounted, whose columns are suffixed per channel
(``R1_Ohm``/``R2_Ohm``, ``S1``/``S2``, ``dT1``/``dT2``). Both are read here;
the second channel is reported as empty when it carries no data, which is the
common case and worth surfacing rather than silently ignoring.

Why the geometry line matters more than it looks
------------------------------------------------
If the operator does not type the sample dimensions, MultiVu writes
``1.000000`` for all three and computes transport quantities against that
dummy geometry. The failure is silent and the numbers look plausible. It
splits the output in two:

* ``rho`` and ``PF`` scale with (width x thickness) / length, so they are wrong
  by a **known constant** and can be rescaled once the true dimensions are
  supplied. `geometry_correction` returns that constant.
* ``kappa`` scales as length / (width x thickness), so the PRODUCT ``rho.kappa``
  is geometry-free — and therefore so are ``ZT`` and ``L/L0``. No geometry entry,
  right or wrong, can repair them. If those are bad, the measurement is bad.

Seebeck needs no geometry at all: it is a voltage over a temperature difference.
It is routinely the only quantity worth keeping from a run like this.

Validation policy
-----------------
Never raises. Problems surface as `ValidationIssue`s:

* no data rows                     -> ERROR   (aborted run)
* geometry left at the 1/1/1 default -> WARNING (rho, PF need rescaling)
* heater current never applied     -> WARNING (no thermoelectric measurement;
                                     any Seebeck column is an offset artefact)
* thermal conductivity negative    -> WARNING (thermal channel failed; kappa,
                                     ZT and L/L0 are unusable)
* Lorenz ratio far from unity      -> WARNING (rho and kappa are inconsistent)
* resistivity negative             -> WARNING (that channel failed)
* second channel present but empty -> INFO    (a sample was mounted and never
                                     wired; its data was not acquired)
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

__all__ = ["PpmsTtoParser", "geometry_correction"]

# The geometry line MultiVu writes first, in millimetres.
_GEOMETRY_RE = re.compile(
    r"thickness:\s*([\d.eE+-]+)\s*,\s*width:\s*([\d.eE+-]+)\s*,\s*length:\s*([\d.eE+-]+)",
)

# Published array name -> the header prefixes that can carry it, most specific
# first. Prefixes rather than exact names because the export truncates its own
# headers (``rho(micr``, ``kappa(W/``). Two prefix families appear: the
# single-channel TTO layout (``T(K)``, ``S(microV``) and the two-channel layout
# (``TR_K``, ``S1``, ``HeaterI_``), so both are listed against the same array.
_COLUMNS: dict[str, tuple[str, ...]] = {
    "temperature_k": ("T(K", "TS1_K", "TR_K"),
    "seebeck_microv_per_k": ("S(micro", "S1"),
    "resistivity_microohm_cm": ("rho(", "rho1_mic"),
    "resistance_ohm": ("R1_Ohm",),
    "thermal_conductivity_w_per_mk": ("kappa(W", "kappa"),
    "zt": ("ZT",),
    "power_factor_mw_per_mk2": ("PF(", "PF1"),
    "lorenz_ratio": ("L/L0",),
    "delta_t_k": ("dT(K", "dT1"),
    "heater_current_ma": ("heaterI", "HeaterI_"),
    "heat_current_w": ("Jq(W",),
    "time_s": ("time(s", "Time_s"),
}

# Two-channel layout: channel 2 columns, checked for emptiness.
_CHANNEL2 = ("TS2_K", "R2_Ohm", "S2", "dT2")

# MultiVu's placeholder when no geometry was entered (mm).
_DEFAULT_DIMENSION_MM = 1.0

# A Lorenz ratio outside this band means rho and kappa disagree badly.
_LORENZ_LO, _LORENZ_HI = 0.3, 5.0

# Bytes read when sniffing the format.
_SNIFF_BYTES = 2048

# A channel this close to zero across every point was never energised or wired.
_ZERO_TOL = 1e-12

# A data row needs at least a time and a temperature to be worth keeping.
_MIN_DATA_COLUMNS = 2


def geometry_correction(
    thickness_mm: float,
    width_mm: float,
    length_mm: float,
    *,
    entered_thickness_mm: float = 1.0,
    entered_width_mm: float = 1.0,
    entered_length_mm: float = 1.0,
) -> float:
    """Factor to rescale a resistivity computed against the wrong geometry.

    Multiply ``rho`` by this; divide ``PF`` by it. ``kappa`` takes the
    reciprocal. ``ZT`` and ``L/L0`` need no correction and cannot receive one,
    because the geometry cancels in the ``rho.kappa`` product they are built on.

    Raises:
        ValueError: If any dimension is not positive.
    """
    dims = (
        thickness_mm,
        width_mm,
        length_mm,
        entered_thickness_mm,
        entered_width_mm,
        entered_length_mm,
    )
    if any(d <= 0 for d in dims):
        raise ValueError(f"all dimensions must be positive, got {dims!r}")
    true_ratio = (width_mm * thickness_mm) / length_mm
    entered_ratio = (entered_width_mm * entered_thickness_mm) / entered_length_mm
    return true_ratio / entered_ratio


def _issue(message: str, severity: Severity, field: str = "ppms_tto") -> ValidationIssue:
    return ValidationIssue(field=field, message=message, severity=severity, detected_at=utc_now())


class PpmsTtoParser(BaseParser):
    """Quantum Design PPMS Thermal Transport Option tab-separated export."""

    name: ClassVar[str] = "ppms-tto-txt"
    version: ClassVar[str] = "1.0.0"
    technique: ClassVar[Technique] = Technique.THERMOELECTRIC
    supported_extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def can_parse(self, path: Path) -> float:
        """Confidence from the geometry line and the characteristic columns."""
        try:
            head = path.read_bytes()[:_SNIFF_BYTES]
        except OSError:
            return 0.0
        text = head.decode("utf-8", errors="replace")
        has_geometry = bool(_GEOMETRY_RE.search(text))
        # The columns are the reliable signal, not the geometry line: the
        # PROCESSED export omits that line and starts straight at the header,
        # while only its `_raw` sibling carries it. Requiring geometry here
        # would reject exactly the files most callers hand over.
        has_columns = ("S(micro" in text or "\tS1\t" in text) and (
            "kappa" in text or "Jq(" in text or "TS1_K" in text
        )
        if has_columns:
            return 1.0 if has_geometry else 0.9
        return 0.6 if has_geometry else 0.0

    def parse(self, path: Path) -> ParsedData:  # noqa: PLR0912, PLR0915
        """Read one TTO export into arrays, metadata and issues."""
        issues: list[ValidationIssue] = []
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {}

        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            return self._empty(f"Could not read the file: {exc}")

        geometry_line = lines[0] if lines else ""
        inline_match = _GEOMETRY_RE.search(geometry_line)
        # Where the header sits depends on whether THIS file opens with a
        # geometry line - not on whether geometry was found anywhere. Deriving
        # it from the sibling lookup below would skip the real header row.
        header_index = 1 if inline_match else 0
        match = inline_match
        if match is None:
            # The processed export starts at the column header. Its geometry
            # lives in the `_raw` sibling written by the same run, so read it
            # from there rather than reporting the dimensions as unknown.
            sibling = path.with_name(f"{path.stem}_raw{path.suffix}")
            if sibling.exists():
                try:
                    first = sibling.read_text(errors="replace").splitlines()[:1]
                except OSError:
                    first = []
                if first:
                    from_raw = _GEOMETRY_RE.search(first[0])
                    if from_raw:
                        match = from_raw
                        metadata["geometry_source"] = sibling.name
        if match:
            thickness, width, length = (float(g) for g in match.groups())
            metadata |= {
                "entered_thickness_mm": thickness,
                "entered_width_mm": width,
                "entered_length_mm": length,
            }
            if all(abs(d - _DEFAULT_DIMENSION_MM) < _ZERO_TOL for d in (thickness, width, length)):
                metadata["geometry_entered"] = False
                issues.append(
                    _issue(
                        "Sample geometry was left at the 1/1/1 default, so resistivity and "
                        "power factor are wrong by a constant factor. Rescale them with "
                        "geometry_correction() once the real dimensions are known. ZT and "
                        "L/L0 are geometry-free and CANNOT be repaired this way.",
                        Severity.WARNING,
                    )
                )
            else:
                metadata["geometry_entered"] = True

        if len(lines) <= header_index:
            return self._empty("File contains no column header.")
        header = [h.strip() for h in lines[header_index].split("\t")]

        rows: list[list[float]] = []
        bad_rows = 0
        for line in lines[header_index + 1 :]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < _MIN_DATA_COLUMNS:
                bad_rows += 1
                continue
            try:
                rows.append([float(p) if p.strip() else np.nan for p in parts])
            except ValueError:
                bad_rows += 1
        if bad_rows:
            issues.append(
                _issue(f"{bad_rows} line(s) could not be read as numbers.", Severity.WARNING)
            )
        if not rows:
            return self._empty(
                "No data rows — the run was aborted before any point was recorded.",
                metadata=metadata,
                extra_issues=issues,
            )

        width_cols = max(len(r) for r in rows)
        table = np.full((len(rows), width_cols), np.nan)
        for i, r in enumerate(rows):
            table[i, : len(r)] = r

        for array_name, prefixes in _COLUMNS.items():
            # Prefixes are ordered most specific first, and the search is
            # prefix-major: a later, looser prefix must not win over an earlier
            # exact one just because its column appears sooner in the header.
            for prefix in prefixes:
                hit = next(
                    (
                        j
                        for j, column in enumerate(header)
                        if column.startswith(prefix) and j < width_cols
                    ),
                    None,
                )
                if hit is not None:
                    arrays[array_name] = table[:, hit]
                    break

        # Two-channel layout: report a mounted-but-unwired second sample.
        present = [c for c in _CHANNEL2 if c in header]
        if present:
            empty = [
                c
                for c in present
                if np.all(np.abs(np.nan_to_num(table[:, header.index(c)])) < _ZERO_TOL)
            ]
            metadata["second_channel_columns"] = present
            metadata["second_channel_empty"] = len(empty) == len(present)
            if len(empty) == len(present):
                issues.append(
                    _issue(
                        "The export has a second measurement channel and every one of its "
                        "columns is zero. A second sample was mounted but never wired; its "
                        "data was not acquired and does not exist elsewhere.",
                        Severity.INFO,
                    )
                )

        metadata["n_points"] = int(table.shape[0])
        issues.extend(self._physics_checks(arrays))

        if "seebeck_microv_per_k" in arrays:
            metadata["seebeck_median_microv_per_k"] = float(
                np.nanmedian(arrays["seebeck_microv_per_k"])
            )
        if "temperature_k" in arrays:
            t = arrays["temperature_k"]
            metadata |= {
                "temperature_min_k": float(np.nanmin(t)),
                "temperature_max_k": float(np.nanmax(t)),
            }

        features: dict[str, float] = {}
        if "seebeck_microv_per_k" in arrays:
            features["seebeck_median_microv_per_k"] = float(
                np.nanmedian(arrays["seebeck_microv_per_k"])
            )

        return ParsedData(
            technique=Technique.THERMOELECTRIC,
            arrays=arrays,
            metadata=metadata,
            instrument="Quantum Design PPMS (Thermal Transport Option)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
            features=features,
        )

    @staticmethod
    def _physics_checks(arrays: dict[str, np.ndarray]) -> list[ValidationIssue]:
        """Flag the failure modes this instrument produces silently."""
        out: list[ValidationIssue] = []

        heater = arrays.get("heater_current_ma")
        if heater is not None and np.all(np.abs(np.nan_to_num(heater)) < _ZERO_TOL):
            out.append(
                _issue(
                    "Heater current is zero for every point: no thermal gradient was ever "
                    "applied. This is a resistance-versus-temperature sweep, not a "
                    "thermoelectric measurement, and any Seebeck column is an offset "
                    "artefact rather than a measured value.",
                    Severity.WARNING,
                )
            )

        kappa = arrays.get("thermal_conductivity_w_per_mk")
        if kappa is not None:
            finite = kappa[np.isfinite(kappa)]
            if finite.size and np.all(finite < 0):
                out.append(
                    _issue(
                        "Thermal conductivity is negative at every point — the thermal "
                        "channel failed. kappa, ZT and the Lorenz ratio are unusable; "
                        "Seebeck is unaffected because it needs no heat-flow number.",
                        Severity.WARNING,
                    )
                )

        lorenz = arrays.get("lorenz_ratio")
        if lorenz is not None:
            finite = lorenz[np.isfinite(lorenz)]
            if finite.size:
                median = float(np.median(np.abs(finite)))
                if not (_LORENZ_LO <= median <= _LORENZ_HI):
                    out.append(
                        _issue(
                            f"Lorenz ratio L/L0 has median magnitude {median:.2e}, far from "
                            "the expected ~1. Resistivity and thermal conductivity are "
                            "mutually inconsistent. Note this is geometry-free, so entering "
                            "the sample dimensions will not change it.",
                            Severity.WARNING,
                        )
                    )

        rho = arrays.get("resistivity_microohm_cm")
        if rho is not None:
            finite = rho[np.isfinite(rho)]
            if finite.size and np.any(finite < 0):
                out.append(
                    _issue(
                        f"Resistivity is negative for {int(np.sum(finite < 0))} of "
                        f"{finite.size} points — that channel failed for this run.",
                        Severity.WARNING,
                    )
                )
        return out

    def _empty(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        extra_issues: list[ValidationIssue] | None = None,
    ) -> ParsedData:
        issues = list(extra_issues or [])
        issues.append(_issue(message, Severity.ERROR))
        return ParsedData(
            technique=Technique.THERMOELECTRIC,
            arrays={},
            metadata=metadata or {},
            instrument="Quantum Design PPMS (Thermal Transport Option)",
            measured_at=None,
            issues=tuple(issues),
            parser_name=self.name,
            parser_version=self.version,
        )
