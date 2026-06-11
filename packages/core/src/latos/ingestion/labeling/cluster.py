"""Fuzzy clustering of sample-name hints into coherent groups.

Pipeline (Stage 2A → 2B → 2C):

    [files] ── extract_hints() ──▶ [SampleHints, per file]
                                      │
                       normalize()    │
                                      ▼
                               [normalized strings]
                                      │
                     similarity scoring (rapidfuzz)
                                      │
                                      ▼
                          [similarity graph (networkx)]
                                      │
                              connected components
                                      │
                                      ▼
                             [SampleCluster, ...]

Each `SampleCluster` carries:
- A `canonical` display name (the most-likely-correct raw form).
- All observed raw aliases that fell into the cluster.
- The set of file paths belonging to the cluster (assigned by
  confidence-weighted vote — each file's strongest hints decide which
  cluster gets it).

Why this layered shape
----------------------
The orchestrator (Stage 1D) currently uses one folder-name heuristic
to pick a sample for each file. That works on tidy data and breaks the
moment a researcher writes `CS Pure` in the XRD folder and
`CS (Pure)` in the XPS folder. Stage 2 replaces that single-source
heuristic with a multi-source vote: every file casts confidence-
weighted votes for whichever sample cluster its hints fall into,
and the cluster system picks the winner. Same files, same data, but
the labels merge correctly.

Similarity scoring
------------------
We combine three rapidfuzz metrics with `max(...)` rather than
averaging — each metric catches a different kind of variation, and
we want a high score on *any* of them to count as "these are the
same sample":

| Metric            | Catches                                     |
|-------------------|---------------------------------------------|
| Levenshtein ratio | One-character typos, separator differences  |
| Token-sort ratio  | Word reordering ("MX-001 batch" vs "batch MX-001") |
| Jaro-Winkler      | Common-prefix similarity (good on short ids)|

Threshold default of 0.85 is conservative for the cosmetic-difference
class of bug we're fixing. The user can override per-call when
clustering is too eager (raise the threshold) or not eager enough
(lower it — but watch for over-merging).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from latos.ingestion.labeling.normalize import normalize

if TYPE_CHECKING:
    from latos.ingestion.labeling.hints import SampleHints

__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "SampleCluster",
    "cluster_samples",
    "pick_canonical",
    "similarity",
]


# Default edge threshold for the similarity graph. Tuned conservatively
# for Stage 2's primary failure mode (cosmetic-difference splits like
# `CS Pure` / `CS (Pure)`). Hand-validated against the Dhivya dataset:
# 0.85 merges the known-equivalent variants without false positives
# between distinct samples (CS-1 / CS-3 stay separate, CS / CSCBI
# stay separate).
DEFAULT_SIMILARITY_THRESHOLD: float = 0.85


@dataclass(frozen=True, slots=True)
class SampleCluster:
    """A fuzzy cluster of sample-name aliases for one logical sample.

    Attributes:
        canonical: The display-friendly name picked by `pick_canonical`.
            Always one of the strings in `aliases`.
        aliases: Every raw (non-normalized) form the cluster absorbed,
            sorted alphabetically for stable test output. The canonical
            is included.
        file_paths: Files that voted into this cluster, sorted by path
            for stable output.
        normalized_forms: The unique normalized strings that the
            similarity graph found connected. Useful for diagnostics
            ("which two strings did the cluster phase decide were the
            same sample?") and for downstream tools that want to
            re-cluster with a different threshold.
    """

    canonical: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    file_paths: tuple[Path, ...] = field(default_factory=tuple)
    normalized_forms: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def cluster_samples(
    hints_per_file: Sequence[SampleHints],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[SampleCluster, ...]:
    """Cluster every file in `hints_per_file` into `SampleCluster`s.

    Args:
        hints_per_file: One `SampleHints` per file, as produced by
            `extract_hints`. Files that produced zero hints (e.g. an
            extension-less file at the root) are still kept; they end
            up in their own one-file cluster keyed by their filename
            or, if even that's missing, their `Path` string.
        similarity_threshold: Edge threshold in [0, 1]. Strings whose
            similarity meets or exceeds this value get an edge in the
            graph, and edges connect into clusters via connected
            components. Default `DEFAULT_SIMILARITY_THRESHOLD` (0.85).

    Returns:
        Tuple of `SampleCluster`, sorted by canonical name.
    """
    if similarity_threshold < 0.0 or similarity_threshold > 1.0:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")

    norm_to_raws, file_votes = _gather_votes(hints_per_file)

    if not norm_to_raws:
        # Edge case: no file produced any usable hint. Fall back to
        # one cluster per file keyed by stem or path string.
        return tuple(_fallback_clusters(hints_per_file))

    # Build the similarity graph: nodes are normalized strings, edges
    # are scored similarity >= threshold.
    graph = _build_similarity_graph(
        list(norm_to_raws.keys()),
        threshold=similarity_threshold,
    )

    # Connected components → clusters of normalized strings.
    components: list[set[str]] = [set(c) for c in nx.connected_components(graph)]
    norm_to_component = {norm: idx for idx, comp in enumerate(components) for norm in comp}

    file_to_component = _assign_files_to_components(file_votes, norm_to_component)

    # Files with hints that all normalized to nothing get their own
    # one-file fallback clusters, returned alongside the main ones.
    files_in_components = set(file_to_component.keys())
    leftover = [h for h in hints_per_file if h.file_path not in files_in_components]

    clusters = _materialize_clusters(components, norm_to_raws, file_to_component)
    clusters.extend(_fallback_clusters(leftover))
    return tuple(sorted(clusters, key=lambda c: c.canonical))


def similarity(a: str, b: str) -> float:
    """Combined string-similarity score in [0, 1].

    Returns the maximum of three rapidfuzz metrics — Levenshtein ratio,
    token-sort ratio, and Jaro-Winkler — so a high score on *any* of
    them counts as a strong match. Each metric catches a different
    failure mode; using `max` rather than averaging keeps us from
    diluting a strong signal in one with a weak score in the others.

    Both inputs are passed through `normalize()` first, so callers
    don't need to pre-normalize. Empty inputs return 0.0.
    """
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    # `fuzz.ratio` and `fuzz.token_sort_ratio` return 0..100; scale to
    # 0..1. JaroWinkler returns 0..1 already.
    lev = fuzz.ratio(na, nb) / 100.0
    tok = fuzz.token_sort_ratio(na, nb) / 100.0
    jaro = JaroWinkler.normalized_similarity(na, nb)
    return max(lev, tok, jaro)


def pick_canonical(aliases: Iterable[str]) -> str:
    """Choose the canonical display name from a set of aliases.

    Rules, applied in order:
    1. Shortest length wins (after stripping leading/trailing whitespace).
    2. Among equally-short, alphabetically first wins.
    3. Empty input raises `ValueError`.

    "Most common form" from the original Stage 2C design is
    deliberately omitted: alias frequency in `SampleCluster.aliases`
    is post-dedup, so it's always 1 per alias and a count tiebreak
    would be a no-op. If we later track raw-form occurrence counts
    per file, this can be revisited.
    """
    cleaned = sorted(
        (a.strip() for a in aliases if a and a.strip()),
        key=lambda s: (len(s), s),
    )
    if not cleaned:
        raise ValueError("pick_canonical received an empty alias set")
    return cleaned[0]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _gather_votes(
    hints_per_file: Sequence[SampleHints],
) -> tuple[defaultdict[str, set[str]], defaultdict[Path, dict[str, float]]]:
    """Collect (normalized -> raw aliases) and (file -> vote totals).

    Vote weight is the per-hint confidence. Multiple hints from the
    same file that normalize to the same string add their weights -
    so a file with three weak path-segment hints all pointing at
    "Aa" outvotes a single rival metadata hint with a higher
    individual weight.

    Float weights mean we can't lean on `Counter` (which is int-typed
    under mypy strict); a plain `defaultdict[str, float]` does the
    same job for our needs.
    """
    norm_to_raws: defaultdict[str, set[str]] = defaultdict(set)
    file_votes: defaultdict[Path, dict[str, float]] = defaultdict(dict)
    for hints in hints_per_file:
        for _tag, raw, conf in hints.candidates():
            norm = normalize(raw)
            if not norm:
                continue
            norm_to_raws[norm].add(raw)
            votes = file_votes[hints.file_path]
            votes[norm] = votes.get(norm, 0.0) + conf
    return norm_to_raws, file_votes


def _assign_files_to_components(
    file_votes: dict[Path, dict[str, float]],
    norm_to_component: dict[str, int],
) -> dict[Path, int]:
    """Pick the highest-vote component for each file.

    Ties (extremely rare in practice - confidences are floats summed
    across multiple hints) break by component index, since
    `max(...)` is stable on equal keys and our component indices
    increase with iteration order.
    """
    file_to_component: dict[Path, int] = {}
    for path, votes in file_votes.items():
        totals: dict[int, float] = {}
        for norm, weight in votes.items():
            comp_idx = norm_to_component.get(norm)
            if comp_idx is not None:
                totals[comp_idx] = totals.get(comp_idx, 0.0) + weight
        if totals:
            file_to_component[path] = max(totals, key=lambda i: (totals[i], -i))
    return file_to_component


def _materialize_clusters(
    components: list[set[str]],
    norm_to_raws: dict[str, set[str]],
    file_to_component: dict[Path, int],
) -> list[SampleCluster]:
    """Build a `SampleCluster` per non-empty component.

    Empty-file components are dropped on purpose: they exist when a
    generic path segment ("XRD", "data") got picked up by the hint
    extractor with zero confidence and no file's strongest hint
    landed there. Aliases are never lost - if a file genuinely
    belonged to that name, it would have a non-zero vote and the
    component would have at least one file.
    """
    clusters: list[SampleCluster] = []
    for idx, comp in enumerate(components):
        cluster_files = sorted(p for p, c_idx in file_to_component.items() if c_idx == idx)
        if not cluster_files:
            continue
        aliases: set[str] = set()
        for norm in comp:
            aliases.update(norm_to_raws[norm])
        clusters.append(
            SampleCluster(
                canonical=pick_canonical(aliases),
                aliases=tuple(sorted(aliases)),
                file_paths=tuple(cluster_files),
                normalized_forms=tuple(sorted(comp)),
            )
        )
    return clusters


def _build_similarity_graph(normalized_strings: Sequence[str], *, threshold: float) -> nx.Graph:
    """Build the similarity graph for a set of pre-normalized strings.

    All-pairs comparison. With the kind of input volumes Stage 1's
    Dhivya dataset produces (~50-200 unique normalized strings), this
    is O(n^2) on a few thousand operations - well below where we'd
    need rapidfuzz's `process.cdist` blocking for performance.
    """
    graph = nx.Graph()
    graph.add_nodes_from(normalized_strings)

    n = len(normalized_strings)
    for i in range(n):
        a = normalized_strings[i]
        for j in range(i + 1, n):
            b = normalized_strings[j]
            score = _normalized_similarity(a, b)
            if score >= threshold:
                graph.add_edge(a, b, weight=score)
    return graph


def _normalized_similarity(a: str, b: str) -> float:
    """`similarity()` skipping the `normalize()` step on already-normalized input."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    lev = fuzz.ratio(a, b) / 100.0
    tok = fuzz.token_sort_ratio(a, b) / 100.0
    jaro = JaroWinkler.normalized_similarity(a, b)
    return max(lev, tok, jaro)


def _fallback_clusters(
    hints_per_file: Sequence[SampleHints],
) -> list[SampleCluster]:
    """One single-file cluster per leftover file with no usable hints."""
    out: list[SampleCluster] = []
    for hints in hints_per_file:
        # Best-effort canonical name: filename stem if any, else the
        # path string. We never raise here — every file should land in
        # *some* cluster so the orchestrator can always produce a
        # FileOutcome.
        if hints.from_filename:
            label = hints.from_filename
        elif hints.from_path_segments:
            label = hints.from_path_segments[0]
        else:
            label = str(hints.file_path)
        out.append(
            SampleCluster(
                canonical=label,
                aliases=(label,),
                file_paths=(hints.file_path,),
                normalized_forms=(normalize(label) or label,),
            )
        )
    return out
