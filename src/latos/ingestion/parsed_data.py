"""`ParsedData` — the uniform output shape every parser returns.

Every parser, regardless of technique or file format, returns this dataclass.
Differences between techniques live in:
- `arrays`: named numeric arrays (e.g. {"two_theta", "intensity"} for XRD,
  {"binding_energy", "intensity"} for XPS, {"temperature", "seebeck"} for TE).
- `metadata`: parser-specific scalar metadata, JSON-safe (so it can round-trip
  through SQLite JSON columns and Parquet metadata sidecars).

Validation policy: parsers never crash on malformed input. They parse what
they can and emit `ValidationIssue`s describing problems. The orchestrator
decides what to do with files that produced errors vs. warnings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from latos.core.enums import Severity, Technique
from latos.core.exceptions import ValidationError
from latos.core.models import ValidationIssue

__all__ = ["ParsedData"]


# Semver-shaped parser version: MAJOR.MINOR.PATCH, digits only.
# Strict enough to catch typos (`"1.0"`, `"v1.0.0"`); lenient enough that
# parser authors don't have to think about pre-release/build metadata.
_PARSER_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Parser names are short kebab-case identifiers used as cache keys and
# in log lines: e.g. "rigaku-xrd-txt", "casaxps-csv", "shimadzu-uvdrs-txt".
_PARSER_NAME_RE = re.compile(r"^[a-z][a-z0-9\-]*[a-z0-9]$")


# Whitelist of types allowed inside `metadata`. Anything not in this set
# (including numpy scalars, sets, custom objects) must be converted to a
# stdlib type by the parser BEFORE building `ParsedData`. Strict input
# guarantees we can serialize metadata to SQLite JSON columns and Parquet
# sidecars without surprises.
def _is_json_safe(value: Any) -> bool:
    """Return True if `value` is built from None, bool, int, float, str, list, dict only."""
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list | tuple):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    return False


@dataclass(frozen=True, slots=True)
class ParsedData:
    """Result of running a parser on a single file.

    Attributes:
        technique: Which `Technique` enum value this measurement represents.
        arrays: Mapping from array name to 1-D or 2-D numpy array. Empty
            dict allowed (e.g. .tif metadata-only parser in Stage 1C).
        metadata: JSON-safe dict of scalar metadata. No numpy scalars,
            no custom objects — convert before constructing.
        instrument: Free-text instrument identifier from the file headers
            (e.g. "Rigaku Ultima IV"). None if not detectable.
        measured_at: When the measurement was acquired, if the file
            records it. Timezone-aware required when present.
        issues: Validation problems detected during parsing. Empty tuple
            when the file parsed cleanly. Parsers never raise — they
            emit issues.
        parser_name: Identifier of the parser that produced this result.
            Lowercase kebab-case, e.g. "rigaku-xrd-txt".
        parser_version: Semver string. The orchestrator uses this with
            `(file_hash, parser_version)` as a re-parse cache key.
    """

    technique: Technique
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    instrument: str | None
    measured_at: datetime | None
    issues: tuple[ValidationIssue, ...]
    parser_name: str
    parser_version: str

    def __post_init__(self) -> None:
        """Validate every invariant; raise `ValidationError` on any violation."""
        self._check_technique()
        self._check_arrays()
        self._check_metadata()
        self._check_instrument()
        self._check_measured_at()
        self._check_issues()
        self._check_parser_identity()

    # ─── Field-level validators ──────────────────────────────────────
    def _check_technique(self) -> None:
        if not isinstance(self.technique, Technique):
            raise ValidationError(
                f"technique must be a Technique enum, got {type(self.technique).__name__}",
            )

    def _check_arrays(self) -> None:
        if not isinstance(self.arrays, dict):
            raise ValidationError("arrays must be a dict[str, np.ndarray]")
        for key, arr in self.arrays.items():
            if not isinstance(key, str) or not key:
                raise ValidationError(f"arrays keys must be non-empty strings, got {key!r}")
            if not isinstance(arr, np.ndarray):
                raise ValidationError(
                    f"arrays[{key!r}] must be np.ndarray, got {type(arr).__name__}",
                )
            if arr.ndim not in (1, 2):
                raise ValidationError(
                    f"arrays[{key!r}] must be 1-D or 2-D, got ndim={arr.ndim}",
                )

    def _check_metadata(self) -> None:
        if not isinstance(self.metadata, dict):
            raise ValidationError("metadata must be a dict")
        if not _is_json_safe(self.metadata):
            raise ValidationError(
                "metadata must be JSON-safe (None, bool, int, float, str, list, dict only)",
            )

    def _check_instrument(self) -> None:
        if self.instrument is not None and not isinstance(self.instrument, str):
            raise ValidationError(
                f"instrument must be str or None, got {type(self.instrument).__name__}",
            )

    def _check_measured_at(self) -> None:
        if self.measured_at is None:
            return
        if not isinstance(self.measured_at, datetime):
            raise ValidationError(
                f"measured_at must be datetime or None, got {type(self.measured_at).__name__}",
            )
        if self.measured_at.tzinfo is None:
            raise ValidationError("measured_at must be timezone-aware")

    def _check_issues(self) -> None:
        if not isinstance(self.issues, tuple):
            raise ValidationError("issues must be a tuple of ValidationIssue")
        for i, issue in enumerate(self.issues):
            if not isinstance(issue, ValidationIssue):
                raise ValidationError(
                    f"issues[{i}] must be ValidationIssue, got {type(issue).__name__}",
                )

    def _check_parser_identity(self) -> None:
        if not isinstance(self.parser_name, str) or not _PARSER_NAME_RE.match(self.parser_name):
            raise ValidationError(
                f"parser_name must be lowercase kebab-case (e.g. 'rigaku-xrd-txt'), "
                f"got {self.parser_name!r}",
            )
        if not isinstance(self.parser_version, str) or not _PARSER_VERSION_RE.match(
            self.parser_version,
        ):
            raise ValidationError(
                f"parser_version must be semver MAJOR.MINOR.PATCH, got {self.parser_version!r}",
            )

    @property
    def has_errors(self) -> bool:
        """True if any issue has Severity.ERROR."""
        return any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """True if any issue is at Severity.WARNING."""
        return any(i.severity is Severity.WARNING for i in self.issues)

    def array_names(self) -> tuple[str, ...]:
        """Sorted tuple of array names — convenient for snapshot tests."""
        return tuple(sorted(self.arrays.keys()))
