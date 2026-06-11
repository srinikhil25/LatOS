"""End-to-end labeling pipeline: `Project` → `SampleCluster`s.

The orchestrator's job is to parse files and persist them. The
labeling pipeline's job is to *re-cluster* the result with the
Stage 2 hint+similarity machinery so cosmetic name differences (a
researcher writing "CS Pure" in the XRD folder and "CS (Pure)" in
the XPS folder) collapse into one logical sample.

Why a post-process, not a rewrite of the orchestrator
-----------------------------------------------------
The orchestrator already produces an `IngestionResult` with one
`Sample` per name-as-spelled. Folding the labeling pipeline directly
into ingestion would mean running the parser → cluster → persist
loop in a different order, which is a much bigger change. The
post-process approach lets us:

- Keep ingestion deterministic (same file → same `Sample` regardless
  of clustering threshold).
- Re-run clustering with different thresholds without re-parsing
  every file (fast iteration in the UI).
- Feed Stage 2D's confirmation page from the same data structure
  whether it comes from a fresh ingestion or a cached re-open.

What the pipeline does NOT do
-----------------------------
- It does not re-parse files. Metadata-based hints (`metadata_sample_name`,
  etc.) need to be wired through the orchestrator's `ParsedData` →
  `Measurement` mapping to be available here. For the current
  iteration we use path-segment + filename hints only, which already
  cover the headline Dhivya regression cases.
- It does not persist the clustered samples back to the DB. The
  `ClusterDecisions` JSON owns the user's edits; the cluster output
  itself is recomputed on every project open.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from latos.ingestion.labeling.cluster import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SampleCluster,
    cluster_samples,
)
from latos.ingestion.labeling.hints import SampleHints, extract_hints

if TYPE_CHECKING:
    from pathlib import Path

    from latos.core.models import Project

__all__ = ["cluster_project", "hints_for_project"]


def cluster_project(
    project: Project,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[SampleCluster, ...]:
    """Run the full Stage 2 pipeline against an ingested project.

    Args:
        project: The persisted `Project` with samples + measurements.
            Files come from `sample.measurements[].files[].path`.
        similarity_threshold: Forwarded to `cluster_samples`. Lower
            it to merge more aggressively (good when the data is
            very clean), raise it to be more conservative (good
            when sample names overlap a lot).

    Returns:
        Tuple of `SampleCluster`, sorted by canonical name.
        Empty tuple when the project has no files (e.g. fresh
        project where ingestion is still running). The orchestrator
        guarantees `project.samples` is well-formed; we don't
        defensive-check here.
    """
    hints = hints_for_project(project)
    return cluster_samples(hints, similarity_threshold=similarity_threshold)


def hints_for_project(project: Project) -> tuple[SampleHints, ...]:
    """Extract per-file hints for every file in `project`.

    Each unique file path produces exactly one `SampleHints` — a file
    that's referenced by multiple measurements (rare, but possible
    if the orchestrator ever shares files across techniques) is
    deduplicated by path so the cluster phase doesn't double-vote.

    The project root is forwarded to `extract_hints` so the path-
    segment walker stops at the project root rather than collecting
    `D:`, `data_raw`, etc. as candidate sample names.
    """
    seen: set[Path] = set()
    out: list[SampleHints] = []
    for path in _iter_unique_file_paths(project):
        if path in seen:
            continue
        seen.add(path)
        out.append(extract_hints(path, root=project.root_path))
    return tuple(out)


def _iter_unique_file_paths(project: Project) -> Iterable[Path]:
    """Yield every file path referenced by any measurement of any sample."""
    for sample in project.samples:
        for measurement in sample.measurements:
            for fileref in measurement.files:
                yield fileref.path
