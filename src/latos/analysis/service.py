"""`AnalysisService` — orchestrates analyzer runs and persistence.

Layered above `BaseAnalyzer` and `AnalyzerRegistry`. Responsibilities:

1. Load the measurement's arrays from `ArrayStore`.
2. Hand them, plus caller-chosen parameters, to the analyzer.
3. Persist the `AnalyzerOutput` as an `AnalysisResult` via the
   repository — including writing derived arrays (if any) to a
   separate Parquet sidecar.
4. Cache by `(measurement_id, analyzer_name, analyzer_version,
   params_fingerprint)`: re-running the same analyzer with the same
   params reuses the stored result instead of re-computing. Bumping
   the analyzer's `version` invalidates entries the same way bumping
   a parser version invalidates the parse cache.

Derived-array filenames look like `<measurement_id>.<analyzer_name>.<short_id>.parquet`
under the same `.latos/arrays/` directory as parsed arrays. Same
folder keeps backup/sync rules simple; the `.<analyzer_name>.` segment
makes them visually distinct from parser output.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from latos.analysis.base_analyzer import (
    AnalyzerInputs,
    AnalyzerOutput,
    BaseAnalyzer,
)
from latos.core.exceptions import AnalysisError
from latos.core.models import (
    AnalysisResult,
    Measurement,
    Project,
    Sample,
    ValidationIssue,
    new_id,
    utc_now,
)
from latos.ingestion.array_store import ArrayStore
from latos.persistence.repository import ProjectRepository

__all__ = [
    "AnalysisService",
    "AnalysisRunOutcome",
]


@dataclass(frozen=True, slots=True)
class AnalysisRunOutcome:
    """Result of a single `AnalysisService.run()` call.

    Returned for both fresh runs and cache hits so callers can show
    the user what happened.

    Attributes:
        result: The persisted `AnalysisResult` (whether freshly
            computed or loaded from cache).
        from_cache: True when an existing `AnalysisResult` matched
            and was reused without re-running the analyzer.
    """

    result: AnalysisResult
    from_cache: bool


class AnalysisService:
    """Run analyzers on measurements and persist the results.

    One service instance is bound to one project (one DB + one
    ArrayStore). The UI layer constructs it once per opened project
    and reuses it across all analysis actions.

    The service does NOT own the analyzer registry — callers pass an
    analyzer in directly. This keeps the service free of registry-
    discovery logic and lets tests inject a single analyzer.
    """

    def __init__(
        self,
        *,
        repository: ProjectRepository,
        array_store: ArrayStore,
    ) -> None:
        self._repo = repository
        self._arrays = array_store

    # ─── Public API ──────────────────────────────────────────────────
    def run(
        self,
        analyzer: BaseAnalyzer,
        measurement: Measurement,
        *,
        params: dict[str, Any] | None = None,
        force: bool = False,
    ) -> AnalysisRunOutcome:
        """Run `analyzer` on `measurement` and persist the result.

        Args:
            analyzer: The analyzer to execute. Must accept this
                measurement (checked here — same `AnalyzerRegistry`
                gating logic). Mismatch raises `AnalysisError`.
            measurement: The Measurement to analyze.
            params: Overrides for the analyzer's `default_params`.
                Shallow-merged with defaults before being handed to
                the analyzer. Must be JSON-safe (it forms part of the
                cache key).
            force: If True, re-run even when a cached match exists.
                Used by the UI's "Re-run analysis" button when the
                user wants a fresh computation regardless of cache.

        Returns:
            `AnalysisRunOutcome` carrying the persisted
            `AnalysisResult` and a `from_cache` flag.

        Raises:
            AnalysisError: If the analyzer does not accept the
                measurement, or if persisting the result fails.

        Errors detected *inside* `analyzer.analyze()` do NOT raise —
        they surface as `ValidationIssue`s on the returned result, the
        same contract `BaseParser.parse` uses. This keeps the UI free
        of try/except scaffolding for routine analysis failures.
        """
        if measurement.technique not in analyzer.accepts_techniques:
            raise AnalysisError(
                f"Analyzer {analyzer.name!r} does not accept technique "
                f"{measurement.technique.value!r}",
            )
        if not analyzer.accepts(measurement):
            raise AnalysisError(
                f"Analyzer {analyzer.name!r} rejected measurement {measurement.id!r}",
            )

        merged_params = analyzer.merge_params(params)
        params_fp = _fingerprint(merged_params)

        # Cache lookup: same analyzer, same version, same params → reuse.
        # We don't compare `outputs` content; if `force=True` was not set
        # and the key matches, the prior compute is the source of truth.
        if not force:
            cached = self._find_cached(
                measurement,
                analyzer_name=analyzer.name,
                analyzer_version=analyzer.version,
                params_fp=params_fp,
            )
            if cached is not None:
                return AnalysisRunOutcome(result=cached, from_cache=True)

        # Cache miss (or force=True): load arrays and run.
        arrays = self._arrays.load(measurement.id)
        inputs = AnalyzerInputs(
            measurement=measurement,
            arrays=arrays,
            params=merged_params,
        )
        output = analyzer.analyze(inputs)
        _validate_output(output, analyzer_name=analyzer.name)

        # Persist derived arrays first — if the Parquet write fails,
        # we don't want a half-finished AnalysisResult row pointing
        # at a missing file.
        result_id = new_id()
        derived_path: Path | None = None
        if output.derived_arrays:
            derived_path = self._write_derived_arrays(
                measurement_id=measurement.id,
                analyzer_name=analyzer.name,
                result_id=result_id,
                arrays=output.derived_arrays,
            )

        result = AnalysisResult(
            id=result_id,
            measurement_id=measurement.id,
            analyzer_name=analyzer.name,
            analyzer_version=analyzer.version,
            params=merged_params,
            outputs=dict(output.outputs),
            derived_arrays_path=derived_path,
            issues=output.issues,
            computed_at=utc_now(),
        )
        self._persist(measurement, result, replace_key=(analyzer.name, params_fp))
        return AnalysisRunOutcome(result=result, from_cache=False)

    # ─── Persistence helpers ─────────────────────────────────────────
    def _find_cached(
        self,
        measurement: Measurement,
        *,
        analyzer_name: str,
        analyzer_version: str,
        params_fp: str,
    ) -> AnalysisResult | None:
        """Return the cached AnalysisResult for this (analyzer, params), or None.

        Matches on analyzer_name + analyzer_version + params fingerprint.
        If the analyzer's version has been bumped, the old result is
        not a cache hit — it's stale data the user can still see in
        the UI but a fresh run produces a new row.
        """
        for r in measurement.analysis_results:
            if r.analyzer_name != analyzer_name:
                continue
            if r.analyzer_version != analyzer_version:
                continue
            if _fingerprint(r.params) != params_fp:
                continue
            return r
        return None

    def _persist(
        self,
        measurement: Measurement,
        result: AnalysisResult,
        *,
        replace_key: tuple[str, str],
    ) -> None:
        """Save `result` against the project, replacing any prior same-key entry.

        Re-running with `force=True` (or after an analyzer-version
        bump that invalidates the cache) should not pile up an
        ever-growing list of analysis_results for one measurement.
        We replace any prior result that shares (analyzer_name,
        params_fingerprint) for the same measurement.

        `replace_key` is `(analyzer_name, params_fp)`. Version is
        deliberately NOT part of the key — when an analyzer version is
        bumped, the user's "latest" view for the same params should
        show the new result, with the old one displaced rather than
        accumulated.
        """
        # Load the full project, splice in the change, save it back.
        # `save()` is a full upsert (Stage 1B contract); we trade write
        # efficiency for correctness simplicity.
        project = self._repo.load_first()
        if project is None:
            raise AnalysisError("No project in this database to persist analysis into")

        new_samples: list[Sample] = []
        for sample in project.samples:
            new_measurements: list[Measurement] = []
            for m in sample.measurements:
                if m.id != measurement.id:
                    new_measurements.append(m)
                    continue
                # Found the target measurement: drop any prior result
                # with the same (analyzer_name, params_fp) and append the new one.
                target_name, target_fp = replace_key
                kept = tuple(
                    r
                    for r in m.analysis_results
                    if not (
                        r.analyzer_name == target_name
                        and _fingerprint(r.params) == target_fp
                    )
                )
                new_measurements.append(
                    Measurement(
                        id=m.id,
                        sample_id=m.sample_id,
                        technique=m.technique,
                        instrument=m.instrument,
                        measured_at=m.measured_at,
                        parsed_at=m.parsed_at,
                        parser_version=m.parser_version,
                        files=m.files,
                        issues=m.issues,
                        parsed_data_path=m.parsed_data_path,
                        analysis_results=(*kept, result),
                    ),
                )
            new_samples.append(
                Sample(
                    id=sample.id,
                    project_id=sample.project_id,
                    canonical_name=sample.canonical_name,
                    aliases=sample.aliases,
                    measurements=tuple(new_measurements),
                ),
            )

        updated = Project(
            id=project.id,
            name=project.name,
            root_path=project.root_path,
            created_at=project.created_at,
            schema_version=project.schema_version,
            samples=tuple(new_samples),
            unassigned_files=project.unassigned_files,
        )
        self._repo.save(updated)

    def _write_derived_arrays(
        self,
        *,
        measurement_id: str,
        analyzer_name: str,
        result_id: str,
        arrays: dict[str, np.ndarray],
    ) -> Path:
        """Atomically write derived arrays as a Parquet sidecar.

        Lives in the same `arrays_dir` as parser output but with a
        compound filename so the two never collide:
        `<measurement_id>.<analyzer_name>.<result_short>.parquet`.
        `result_short` is the first 8 chars of the result id, enough
        to disambiguate within a measurement without bloating filenames.
        """
        # Same atomicity dance as ArrayStore.write: .tmp → os.replace.
        # We don't go through ArrayStore.write here because that's keyed
        # on `measurement_id` alone and would clobber the parsed arrays.
        directory = self._arrays.directory
        target = directory / f"{measurement_id}.{analyzer_name}.{result_id[:8]}.parquet"
        tmp = target.with_suffix(target.suffix + ".tmp")
        table = pa.table({name: pa.array(arr) for name, arr in arrays.items()})
        try:
            pq.write_table(table, tmp)  # type: ignore[no-untyped-call]
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, target)
        return target


# ─── Module helpers ─────────────────────────────────────────────────
def _fingerprint(params: dict[str, Any]) -> str:
    """Stable hex fingerprint of a JSON-safe params dict.

    Uses canonical JSON (sorted keys, no whitespace) so two
    semantically-equal dicts always hash the same. SHA-256 truncated
    to 16 chars — that's 64 bits of entropy, plenty for collision
    avoidance within a single measurement's analysis history.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _validate_output(output: AnalyzerOutput, *, analyzer_name: str) -> None:
    """Enforce the contract on an AnalyzerOutput before we persist it.

    The analyzer itself is "trusted code" in the sense that we don't
    re-validate every field, but a few invariants are easy to check
    and catch wrong return shapes early:

    - `outputs` must be JSON-safe (it goes into a JSON column).
    - `derived_arrays` must all be 1-D numpy arrays of equal length
      (same as `ParsedData`).
    - `issues` must be a tuple of `ValidationIssue` (frozen contract).
    """
    if not isinstance(output, AnalyzerOutput):
        raise AnalysisError(
            f"Analyzer {analyzer_name!r} returned {type(output).__name__}, "
            "expected AnalyzerOutput",
        )
    if not isinstance(output.outputs, dict):
        raise AnalysisError(
            f"Analyzer {analyzer_name!r} returned non-dict `outputs`",
        )
    lengths: set[int] = set()
    for name, arr in output.derived_arrays.items():
        if not isinstance(name, str) or not name:
            raise AnalysisError(
                f"Analyzer {analyzer_name!r} derived_arrays key invalid: {name!r}",
            )
        if not isinstance(arr, np.ndarray) or arr.ndim != 1:
            raise AnalysisError(
                f"Analyzer {analyzer_name!r} derived_arrays[{name!r}] must be 1-D ndarray",
            )
        lengths.add(arr.shape[0])
    if len(lengths) > 1:
        raise AnalysisError(
            f"Analyzer {analyzer_name!r} derived_arrays have mismatched lengths: "
            f"{sorted(lengths)}",
        )
    if not isinstance(output.issues, tuple) or any(
        not isinstance(i, ValidationIssue) for i in output.issues
    ):
        raise AnalysisError(
            f"Analyzer {analyzer_name!r} issues must be a tuple of ValidationIssue",
        )
