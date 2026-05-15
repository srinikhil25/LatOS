"""`BaseAnalyzer` — the contract every Latos analyzer must satisfy.

A "parser" turns a file on disk into a `Measurement`. An "analyzer"
turns a `Measurement` into an `AnalysisResult` — a derived scientific
number (band gap, peak positions, zT-at-temperature, ...).

Design mirrors `BaseParser` for consistency:
- Analyzers declare their `name`, `version`, and the techniques they
  accept as **class attributes**, validated at import time via
  `__init_subclass__`.
- `accepts(measurement)` is **cheap** — it answers "could this analyzer
  produce a result from this measurement?" without doing any heavy
  numerical work. Used by `AnalyzerRegistry` to filter the candidate
  set before any expensive call.
- `analyze(inputs)` does the actual computation and must **never
  raise** on bad data. Problems surface as `ValidationIssue`s on the
  returned `AnalyzerOutput`.

Caching:
- The same `(measurement_id, analyzer_name, analyzer_version,
  params_fingerprint)` tuple should always produce the same outputs.
  Bumping `version` invalidates the cache the same way bumping a
  parser's version does.

This module defines the contract types only. `AnalysisService`
orchestrates runs and `AnalyzerRegistry` dispatches.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from latos.core.enums import Technique
from latos.core.models import Measurement, ValidationIssue

__all__ = [
    "AnalyzerInputs",
    "AnalyzerOutput",
    "BaseAnalyzer",
]


# Same shape patterns we already enforce on parsers.
_ANALYZER_NAME_RE = re.compile(r"^[a-z][a-z0-9\-]*[a-z0-9]$")
_ANALYZER_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True, slots=True)
class AnalyzerInputs:
    """Everything an analyzer needs to do its job.

    Bundled into a single frozen object so the analyzer signature stays
    stable as more inputs are added in later stages (e.g. companion
    measurements for cross-technique analyzers).

    Attributes:
        measurement: The Measurement being analyzed. Read-only — the
            analyzer must not mutate it.
        arrays: Numeric arrays loaded from `ArrayStore` for this
            measurement. The same `{name: ndarray}` shape a parser
            originally produced. Empty dict allowed (and expected) for
            metadata-only techniques like microscopy.
        params: Analyzer-specific parameters chosen by the caller
            (band-gap type, peak-finder prominence, fit range, ...).
            Must be JSON-safe so they round-trip through the cache key.
    """

    measurement: Measurement
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnalyzerOutput:
    """What an analyzer returns from `analyze()`.

    Maps 1:1 onto the persistent `AnalysisResult` (the service fills in
    the id, measurement_id, analyzer_name, analyzer_version, and
    computed_at).

    Attributes:
        outputs: Scalar / list / dict-of-scalars JSON-safe payload of
            derived values. The headline numbers (band_gap_ev,
            r_squared, peak_centers_2theta, ...). Must be JSON-safe.
        derived_arrays: Optional `{name: ndarray}` of derived 1-D
            curves (a fit line, a Tauc-transformed spectrum). All
            arrays must be 1-D and co-indexed, same contract as
            `ParsedData.arrays`. Empty dict when the analyzer is
            scalar-only.
        issues: ValidationIssues describing problems detected during
            analysis. Empty tuple when everything went cleanly.
    """

    outputs: dict[str, Any] = field(default_factory=dict)
    derived_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)


class BaseAnalyzer(ABC):
    """Abstract contract for a Latos analyzer.

    Concrete subclasses MUST set these class attributes:
        name: lowercase kebab-case identifier (e.g. "uvdrs-tauc").
        version: semver string. Bump on any behavioural change.
        accepts_techniques: tuple of `Technique` enum values this
            analyzer can run on. Empty tuple is rejected — every
            analyzer is rooted in at least one technique.
        default_params: JSON-safe dict of default parameter values.
            The UI surfaces these as form defaults; callers can
            override individually.

    Concrete subclasses MUST implement:
        accepts(measurement) -> bool   — additional per-measurement gating
        analyze(inputs) -> AnalyzerOutput
    """

    name: ClassVar[str] = ""
    version: ClassVar[str] = ""
    accepts_techniques: ClassVar[tuple[Technique, ...]] = ()
    default_params: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate analyzer metadata as soon as the subclass is defined."""
        super().__init_subclass__(**kwargs)

        # Skip validation for intermediate abstract classes. Same trick
        # as BaseParser: ABCMeta hasn't populated __abstractmethods__ yet
        # at this point, so we look for the `__isabstractmethod__`
        # marker on the methods themselves.
        for method_name in ("accepts", "analyze"):
            method = getattr(cls, method_name, None)
            if method is None or getattr(method, "__isabstractmethod__", False):
                return

        if not isinstance(cls.name, str) or not _ANALYZER_NAME_RE.match(cls.name):
            raise TypeError(
                f"{cls.__name__}.name must be lowercase kebab-case "
                f"(e.g. 'uvdrs-tauc'), got {cls.name!r}",
            )
        if not isinstance(cls.version, str) or not _ANALYZER_VERSION_RE.match(cls.version):
            raise TypeError(
                f"{cls.__name__}.version must be semver MAJOR.MINOR.PATCH, "
                f"got {cls.version!r}",
            )
        if not isinstance(cls.accepts_techniques, tuple) or not cls.accepts_techniques:
            raise TypeError(
                f"{cls.__name__}.accepts_techniques must be a non-empty tuple of Technique",
            )
        for t in cls.accepts_techniques:
            if not isinstance(t, Technique):
                raise TypeError(
                    f"{cls.__name__}.accepts_techniques entries must be Technique, got {t!r}",
                )
        if not isinstance(cls.default_params, dict):
            raise TypeError(
                f"{cls.__name__}.default_params must be a dict, "
                f"got {type(cls.default_params).__name__}",
            )

    # ─── Abstract API ────────────────────────────────────────────────
    @abstractmethod
    def accepts(self, measurement: Measurement) -> bool:
        """True if this analyzer can produce a result from `measurement`.

        The registry pre-filters by `accepts_techniques` before calling
        this — so implementations only need to check finer-grained
        constraints (e.g. "I need at least 50 wavelength points",
        "I require both reflectance and absorbance columns").

        Must NOT raise: return False on any unexpected condition.
        """

    @abstractmethod
    def analyze(self, inputs: AnalyzerInputs) -> AnalyzerOutput:
        """Compute a derived result from `inputs`.

        Must NOT raise on malformed input. Surface problems as
        `ValidationIssue`s on the returned `AnalyzerOutput.issues`.
        Returning an `AnalyzerOutput` with empty `outputs` and at least
        one error-severity issue is the standard "I tried but couldn't
        produce a usable result" pattern.

        The analyzer should NOT persist anything itself — `AnalysisService`
        handles array storage and SQL persistence.
        """

    # ─── Helpers ─────────────────────────────────────────────────────
    def merge_params(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        """Return `default_params` merged with caller `overrides`.

        Shallow merge — sufficient for the parameter shapes we use
        today (flat dicts of scalars). If we ever grow nested-dict
        parameters, switch this to a deep-merge utility.
        """
        merged = dict(self.default_params)
        if overrides:
            merged.update(overrides)
        return merged

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} version={self.version!r}>"
