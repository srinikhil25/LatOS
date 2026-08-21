"""`AnalyzerRegistry` — holds analyzers and finds which apply to a measurement.

Mirrors `latos.ingestion.registry.ParserRegistry` in shape, with two
behavioural differences:

1. Selection is a **set** operation, not a confidence-pick. Multiple
   analyzers can run on the same measurement (e.g. for a UV-DRS scan
   you might want both `uvdrs-tauc` for the band gap and a future
   `uvdrs-absorbance` for derived absorbance arrays). `find_for()`
   returns every analyzer that accepts the measurement.

2. Fast filtering happens in two phases: the `accepts_techniques`
   class attribute lets the registry skip analyzers that don't even
   handle this technique, before any per-measurement `accepts()` call.

Why a registry vs. a flat list at the call site
-----------------------------------------------
Same reasoning as the parser registry: tests can build a one-analyzer
registry to verify dispatch in isolation; the UI's "Run analysis"
action queries by `Measurement` without knowing about concrete classes.
"""

from __future__ import annotations

from collections.abc import Iterable

from latos.analysis.base_analyzer import BaseAnalyzer
from latos.core.models import Measurement

__all__ = [
    "AnalyzerRegistry",
    "default_registry",
]


class AnalyzerRegistry:
    """Holds a list of `BaseAnalyzer`s and finds those applicable to a measurement."""

    def __init__(self, analyzers: Iterable[BaseAnalyzer] = ()) -> None:
        """Build a registry, optionally pre-populated with `analyzers`.

        Args:
            analyzers: Initial analyzers to register. Insertion order is
                preserved and returned from `find_for()` — gives callers
                a stable iteration order (e.g. for UI display).
        """
        self._analyzers: list[BaseAnalyzer] = []
        for a in analyzers:
            self.register(a)

    # ─── Registration ────────────────────────────────────────────────
    def register(self, analyzer: BaseAnalyzer) -> None:
        """Add an analyzer. Re-registering the same `name` raises `ValueError`.

        We disallow duplicate registration for the same reason
        `ParserRegistry` does: it's almost always a bug (the same
        analyzer registered from two import paths).
        """
        for existing in self._analyzers:
            if existing.name == analyzer.name:
                raise ValueError(
                    f"Analyzer with name {analyzer.name!r} is already registered.",
                )
        self._analyzers.append(analyzer)

    @property
    def analyzers(self) -> tuple[BaseAnalyzer, ...]:
        """Tuple of registered analyzers in registration order."""
        return tuple(self._analyzers)

    def __len__(self) -> int:
        return len(self._analyzers)

    def __contains__(self, name: object) -> bool:
        """Membership by analyzer `name`."""
        if not isinstance(name, str):
            return False
        return any(a.name == name for a in self._analyzers)

    def get(self, name: str) -> BaseAnalyzer | None:
        """Return the analyzer with this `name`, or None.

        Used by the service when re-running a specific analyzer by name
        (e.g. from the UI's "rerun" button), and by tests.
        """
        for a in self._analyzers:
            if a.name == name:
                return a
        return None

    # ─── Dispatch ────────────────────────────────────────────────────
    def find_for(self, measurement: Measurement) -> tuple[BaseAnalyzer, ...]:
        """Return every analyzer that accepts `measurement`, in registration order.

        Two-phase filter: skip on `accepts_techniques` first (cheap
        enum check), then ask `accepts(measurement)` (analyzer-specific
        gating that may inspect file count, array shapes, etc.).

        A buggy `accepts()` that raises is treated as "doesn't accept" —
        same defensive policy as `ParserRegistry.find_parser`. We'd rather
        omit one analyzer than crash dispatch.
        """
        matches: list[BaseAnalyzer] = []
        for analyzer in self._analyzers:
            if measurement.technique not in analyzer.accepts_techniques:
                continue
            try:
                ok = analyzer.accepts(measurement)
            except Exception:
                continue
            if ok:
                matches.append(analyzer)
        return tuple(matches)


def default_registry() -> AnalyzerRegistry:
    """Build an `AnalyzerRegistry` populated with every production analyzer.

    Order matters only for UI display — every applicable analyzer runs
    independently, so registration order doesn't change correctness.

    Currently registers:
    - `UvDrsTaucAnalyzer`: band gap from UV-DRS Kubelka-Munk + Tauc plot.
    - `XrdPeakFitAnalyzer`: SNIP-baselined pseudo-Voigt peak fit for XRD,
      with Bragg d-spacings and Scherrer crystallite sizes.
    - `EdsCompositionAnalyzer`: element ID + semi-quantitative composition.
    - `HallMetricsAnalyzer`: carrier type / concentration / mobility with
      a σ = q·n·μ internal-consistency check.
    - `XpsRegionsAnalyzer`: apex peak binding energies per region, with the
      C 1s charge-reference offset.
    - `TransportSummaryAnalyzer`: per-measurement R&S (Seebeck sign, power
      factor) and LFA (κ range) summaries; sample-level zT stays in
      `transport_data.sample_zt`.
    - `ThermovoltageSlopeAnalyzer`: ionic Seebeck coefficient as the slope of
      ΔV against ΔT, separating it from the electrode-polarisation intercept
      that a single-point reading silently absorbs.

    Two analyzers now claim `Technique.THERMOELECTRIC`, which is what the
    registry's set semantics are for: a resistivity-and-Seebeck run has no
    ΔV-versus-ΔT series and a thermovoltage run has no κ, so each declines the
    other's data in `analyze` and reports why.

    Microscopy (particle-size from TEM/SEM images) is deliberately absent —
    it needs the Stage 5 vision work, not a numeric kernel.
    """
    # Local imports keep the module light when only the registry types
    # are needed (e.g. by tests that build their own one-analyzer registry).
    from latos.analysis.eds.composition import EdsCompositionAnalyzer  # noqa: PLC0415
    from latos.analysis.hall.metrics import HallMetricsAnalyzer  # noqa: PLC0415
    from latos.analysis.thermovoltage.slope import ThermovoltageSlopeAnalyzer  # noqa: PLC0415
    from latos.analysis.transport.summary import TransportSummaryAnalyzer  # noqa: PLC0415
    from latos.analysis.uv_drs.tauc import UvDrsTaucAnalyzer  # noqa: PLC0415
    from latos.analysis.xps.regions import XpsRegionsAnalyzer  # noqa: PLC0415
    from latos.analysis.xrd.peak_fit import XrdPeakFitAnalyzer  # noqa: PLC0415

    return AnalyzerRegistry(
        [
            UvDrsTaucAnalyzer(),
            XrdPeakFitAnalyzer(),
            EdsCompositionAnalyzer(),
            HallMetricsAnalyzer(),
            XpsRegionsAnalyzer(),
            TransportSummaryAnalyzer(),
            ThermovoltageSlopeAnalyzer(),
        ],
    )
