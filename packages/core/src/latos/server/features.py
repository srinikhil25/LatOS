"""Cross-sample feature extraction — the foundation of Stage 6.

Runs the analysis layer across every sample and assembles one *sample ×
property* table: crystallite size (XRD), band gap (UV-DRS), peak zT / Seebeck
/ thermal conductivity (derived transport), and the Hall carrier metrics —
each cell carrying where it came from and whether it is trustworthy.

Both the correlation engine and the reporting/export layer read from this
table, so it is deliberately technique-agnostic: a property appears for a
sample only if that sample actually has the data, and nothing is hard-coded
to one material system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from latos.analysis.base_analyzer import AnalyzerInputs
from latos.analysis.hall.metrics import cross_config_reliability
from latos.analysis.registry import default_registry
from latos.analysis.transport import TransportError
from latos.core.enums import Technique
from latos.server.optimization_data import (
    _HALL_CARRIER_FEATURES,
    _hall_features,
    feature_value,
    peak_target,
)
from latos.server.transport_data import sample_zt

if TYPE_CHECKING:
    from latos.core.models import Project, Sample
    from latos.ingestion.array_store import ArrayStore

__all__ = ["FeatureCell", "FeatureRow", "FeatureTable", "extract_features"]

# Scalar analyzer outputs promoted to cross-sample features:
# analyzer output key -> (feature name, unit).
_ANALYZER_FEATURE_KEYS: dict[str, tuple[str, str]] = {
    "mean_crystallite_size_nm": ("crystallite_size_nm", "nm"),
    "band_gap_ev": ("band_gap_ev", "eV"),
}

# Only these techniques' analyzers yield the features above — running every
# analyzer on every measurement (incl. XRD peak-fitting on dozens of TEM/XPS
# files) is needlessly slow, so we restrict to the producing techniques.
_ANALYZER_TECHNIQUES: frozenset[Technique] = frozenset({Technique.XRD, Technique.UV_DRS})

# Speed overrides for the batch feature run. The cross-sample crystallite
# size is a *summary*: fitting the strongest dozen reflections is
# representative and ~10x faster than the full-scan peak fit (which is a
# per-sample deep-dive on the Fit screen). Applied uniformly, so the value
# stays comparable across samples.
_ANALYZER_PARAM_OVERRIDES: dict[str, dict[str, object]] = {
    "xrd-peak-fit": {"max_peaks": 12},
}

# Peak-of-a-measured-array transport features: name -> unit.
_PEAK_FEATURES: dict[str, str] = {
    "seebeck_uv_k": "µV/K",
    "thermal_conductivity": "W/m·K",
}

# Hall-measured features (unreliable when the cross-configuration check fails):
# name -> unit.
_HALL_FEATURES: dict[str, str] = {
    "carrier_concentration_cm3": "cm⁻³",
    "mobility_cm2_vs": "cm²/V·s",
    "conductivity_s_cm": "S/cm",
}


@dataclass(frozen=True)
class FeatureCell:
    """One property's value for one sample, with provenance."""

    value: float
    unit: str
    source: str  # human-readable origin, e.g. "xrd · CS-1.xrdml" or "derived"
    reliable: bool = True  # False when the source data failed a quality check


@dataclass(frozen=True)
class FeatureRow:
    """All extracted properties for a single sample."""

    sample_id: str
    sample_name: str
    features: dict[str, FeatureCell] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureTable:
    """The sample × property matrix. `properties` is the sorted column union."""

    properties: list[str]
    rows: list[FeatureRow]


def _analyzer_features(sample: Sample, store: ArrayStore) -> dict[str, FeatureCell]:
    """Scalar features from running the analyzers over a sample's measurements."""
    out: dict[str, FeatureCell] = {}
    registry = default_registry()
    for measurement in sample.measurements:
        if measurement.technique not in _ANALYZER_TECHNIQUES:
            continue
        arrays = store.load(measurement.id)
        if not arrays:
            continue
        for analyzer in registry.find_for(measurement):
            try:
                result = analyzer.analyze(
                    AnalyzerInputs(
                        measurement=measurement,
                        arrays=arrays,
                        params=analyzer.merge_params(_ANALYZER_PARAM_OVERRIDES.get(analyzer.name)),
                    )
                )
            except Exception:  # one bad analyzer must not sink the whole table
                continue
            fname_src = measurement.files[0].path.name if measurement.files else "?"
            source = f"{measurement.technique.value} · {fname_src}"
            for key, (fname, unit) in _ANALYZER_FEATURE_KEYS.items():
                value = result.outputs.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    out[fname] = FeatureCell(float(value), unit, source)
    return out


def _transport_features(sample: Sample, store: ArrayStore) -> dict[str, FeatureCell]:
    """Derived-transport features: peak zT, peak Seebeck, peak thermal κ."""
    out: dict[str, FeatureCell] = {}
    try:
        zt = sample_zt(sample, store.load)  # one derivation, reused below
    except TransportError:
        zt = None
    if zt is not None:
        wf = any("Wiedemann" in w for w in zt.warnings)
        out["peak_zt"] = FeatureCell(zt.peak_zt, "—", "derived (R&S+LFA)", reliable=not wf)
    for name, unit in _PEAK_FEATURES.items():
        value = peak_target(sample, store, name)
        if value is not None:
            out[f"peak_{name}"] = FeatureCell(value, unit, "derived")
    return out


def _hall_feature_cells(sample: Sample) -> dict[str, FeatureCell]:
    """Hall carrier metrics, flagged unreliable when the cross-config check fails."""
    features = _hall_features(sample)
    reliable = True
    if features is not None:
        level, _ = cross_config_reliability(features)
        reliable = level != "unreliable"
    out: dict[str, FeatureCell] = {}
    for name, unit in _HALL_FEATURES.items():
        value = feature_value(sample, name)
        if value is not None:
            flag = reliable if name in _HALL_CARRIER_FEATURES else True
            out[name] = FeatureCell(value, unit, "hall", reliable=flag)
    return out


def extract_features(project: Project, store: ArrayStore) -> FeatureTable:
    """Assemble the sample × property table for `project`.

    Each sample contributes only the properties it actually has data for;
    every cell records its source and a reliability flag (e.g. a
    Wiedemann-Franz-violating zT, or Hall metrics from a failed
    cross-configuration check, are marked unreliable).
    """
    rows: list[FeatureRow] = []
    names: set[str] = set()
    for sample in project.samples:
        cells: dict[str, FeatureCell] = {}
        cells.update(_analyzer_features(sample, store))
        cells.update(_transport_features(sample, store))
        cells.update(_hall_feature_cells(sample))
        names.update(cells)
        rows.append(FeatureRow(sample.id, sample.canonical_name, cells))
    return FeatureTable(properties=sorted(names), rows=rows)
