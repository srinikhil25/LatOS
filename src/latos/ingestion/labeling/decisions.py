"""User decisions on top of an auto-clustering result.

Stage 2C produces clusters from heuristics. Stage 2D lets the user
*edit* those clusters — rename a sample, merge two clusters that the
algorithm split, or split one that it over-merged. Those edits are
the part of the labeling output the user owns; we persist them
separately from the project's database so re-running ingestion
doesn't trample them.

Storage shape
-------------
A single JSON file at ``<project_root>/.latos/cluster_decisions.json``::

    {
      "renames": {
        "auto_canonical_a": "User-Chosen Name",
        ...
      },
      "merges": [
        ["auto_canonical_x", "auto_canonical_y", "auto_canonical_z"],
        ...
      ],
      "splits": {
        "auto_canonical_q": {
          "C:/path/to/file_a.csv": "MX-1",
          "C:/path/to/file_b.csv": "MX-2",
          ...
        }
      }
    }

Three kinds of edit are supported:

- **Rename**: change the canonical display name of one cluster. Keyed
  by the *auto* canonical (what Stage 2C produced), so re-ingesting
  the same data produces the same key and the rename re-applies. The
  alias set isn't touched.
- **Merge**: collapse N clusters into one. The first canonical in the
  list is the surviving canonical (renames are applied AFTER merges,
  so the user can also rename the merged result).
- **Split**: pull individual files out of a cluster into a different
  cluster. Keyed by the source cluster's auto canonical, then by
  file path → target canonical. The target canonical may be a brand-
  new sample name — apply() will create a single-file cluster for it.

Why JSON, not the database
--------------------------
The DB stores the *result* of clustering (samples + measurements).
Decisions are user intent that survives across ingestion runs and
must be portable when sharing a project folder. JSON in `.latos/`
is the same pattern used for `recent.json` and Stage 1E.3's progress
data — easy to inspect with a text editor, easy to back up.

Decisions are append-only on the user's side: each edit replaces the
prior entry under the same key, but the file as a whole is rewritten
atomically on every save. This gives us ACID-on-rename semantics
without a transaction layer.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from latos.ingestion.labeling.cluster import SampleCluster, pick_canonical

# Smallest merge that does any work — single-name "groups" are no-ops.
_MIN_MERGE_GROUP_SIZE = 2

__all__ = [
    "ClusterDecisions",
    "DECISIONS_FILENAME",
    "apply_decisions",
    "load_decisions",
    "save_decisions",
]


# Filename inside `<project>/.latos/` where decisions live. Mirroring
# the pattern of `data.db` and `arrays/` — every Latos-owned artifact
# stays in `.latos/` so a project folder is one rmdir away from clean.
DECISIONS_FILENAME = "cluster_decisions.json"


@dataclass(frozen=True, slots=True)
class ClusterDecisions:
    """Edits the user has applied on top of Stage 2C's auto-clustering.

    Treated as immutable for safety — every edit operation returns a
    new instance via `with_*()` helpers. That keeps history, undo, and
    test setup straightforward (no shared mutable state).

    Attributes:
        renames: Auto canonical → user canonical. Empty when no
            renames were made. Order is insertion order so JSON
            round-trips deterministically.
        merges: Tuple of merge groups. Each group is a tuple of
            canonical names, with the first canonical surviving.
            Groups are independent; a name appearing in two groups
            is a corruption signal that `apply_decisions` flags
            during application.
        splits: Auto canonical → {file path string → target canonical}.
            Pulls individual files out of one cluster into another (or
            into a new single-file cluster). Path keys are the string
            form of the file's `Path` so JSON can persist them.
    """

    renames: dict[str, str] = field(default_factory=dict)
    merges: tuple[tuple[str, ...], ...] = ()
    splits: dict[str, dict[str, str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Pure helpers — return new instances rather than mutating self.
    # ------------------------------------------------------------------

    def with_rename(self, auto_canonical: str, new_canonical: str) -> ClusterDecisions:
        """Return a copy with `auto_canonical` renamed to `new_canonical`.

        An empty `new_canonical` (after strip) clears the rename
        instead of recording a blank name.
        """
        new_canonical = new_canonical.strip()
        renames = dict(self.renames)
        if not new_canonical or new_canonical == auto_canonical:
            renames.pop(auto_canonical, None)
        else:
            renames[auto_canonical] = new_canonical
        return ClusterDecisions(renames=renames, merges=self.merges, splits=self.splits)

    def with_merge(self, canonicals: Iterable[str]) -> ClusterDecisions:
        """Return a copy with the given canonicals merged into one group.

        Order matters: the first item in `canonicals` becomes the
        surviving canonical. Names that are already in another merge
        group are pulled out of that group and added to the new one.
        Single-item groups are dropped — a merge of one is a no-op.
        """
        seen: list[str] = []
        for raw in canonicals:
            stripped = raw.strip()
            if stripped and stripped not in seen:
                seen.append(stripped)
        if len(seen) < _MIN_MERGE_GROUP_SIZE:
            return self
        # Drop any prior groups that mention these names.
        kept = tuple(g for g in self.merges if not any(c in g for c in seen))
        new_groups = (*kept, tuple(seen))
        return ClusterDecisions(renames=self.renames, merges=new_groups, splits=self.splits)

    def with_split(
        self,
        auto_canonical: str,
        file_assignments: dict[Path | str, str],
    ) -> ClusterDecisions:
        """Return a copy with files reassigned out of `auto_canonical`.

        `file_assignments` maps each affected file to its new cluster
        canonical (which may be brand-new). Empty mapping clears any
        prior split for `auto_canonical`.
        """
        splits = dict(self.splits)
        if not file_assignments:
            splits.pop(auto_canonical, None)
        else:
            normalised = {str(p): target.strip() for p, target in file_assignments.items()}
            normalised = {p: t for p, t in normalised.items() if t}
            if normalised:
                splits[auto_canonical] = normalised
            else:
                splits.pop(auto_canonical, None)
        return ClusterDecisions(renames=self.renames, merges=self.merges, splits=splits)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_decisions(project_root: Path) -> ClusterDecisions:
    """Load decisions from `<project_root>/.latos/<DECISIONS_FILENAME>`.

    Missing file → empty `ClusterDecisions`. Malformed file raises
    `ValueError` so a corrupt decisions file is caught loudly rather
    than silently dropping the user's edits.
    """
    path = _decisions_path(project_root)
    if not path.exists():
        return ClusterDecisions()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt cluster decisions file at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Cluster decisions root must be an object, got {type(raw).__name__}")

    renames_raw = raw.get("renames", {})
    if not isinstance(renames_raw, dict):
        raise ValueError("`renames` must be an object")
    renames = {str(k): str(v) for k, v in renames_raw.items()}

    merges_raw = raw.get("merges", [])
    if not isinstance(merges_raw, list):
        raise ValueError("`merges` must be a list")
    merges = tuple(tuple(str(c) for c in group) for group in merges_raw if isinstance(group, list))

    splits_raw = raw.get("splits", {})
    if not isinstance(splits_raw, dict):
        raise ValueError("`splits` must be an object")
    splits: dict[str, dict[str, str]] = {}
    for k, mapping in splits_raw.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"`splits[{k!r}]` must be an object")
        splits[str(k)] = {str(p): str(t) for p, t in mapping.items()}

    return ClusterDecisions(renames=renames, merges=merges, splits=splits)


def save_decisions(project_root: Path, decisions: ClusterDecisions) -> Path:
    """Atomically save `decisions` to `<project_root>/.latos/<DECISIONS_FILENAME>`.

    Returns the written path. Atomic via tmp-file + os.replace so a
    crash mid-write can't truncate the existing decisions to half a
    JSON document.
    """
    path = _decisions_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "renames": decisions.renames,
        "merges": [list(group) for group in decisions.merges],
        "splits": decisions.splits,
    }

    # tempfile.NamedTemporaryFile + os.replace is the cross-platform
    # atomic-rename idiom. We set delete=False so we can close the
    # handle before replace (Windows otherwise refuses the rename).
    fd, tmp_path = tempfile.mkstemp(
        prefix=".cluster_decisions.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the partial tmp file; we don't want a
        # stray .tmp polluting `.latos/` on the next save.
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return path


def _decisions_path(project_root: Path) -> Path:
    return project_root / ".latos" / DECISIONS_FILENAME


# ---------------------------------------------------------------------------
# Apply decisions to a clustering result
# ---------------------------------------------------------------------------


def apply_decisions(
    clusters: Iterable[SampleCluster],
    decisions: ClusterDecisions,
) -> tuple[SampleCluster, ...]:
    """Apply renames + merges + splits to `clusters`, in that order.

    Order matters: splits run first (so a file pulled out of cluster
    A into "MX-7" is no longer in A when A gets merged), then merges
    (so a renamed merged-canonical lookup uses the *post-merge*
    surviving name as its rename key), then renames last.

    Returns a fresh tuple of `SampleCluster`, sorted by canonical.
    The input clusters are not mutated.
    """
    by_canonical: dict[str, SampleCluster] = {c.canonical: c for c in clusters}
    by_canonical = _apply_splits(by_canonical, decisions.splits)
    by_canonical = _apply_merges(by_canonical, decisions.merges)
    by_canonical = _apply_renames(by_canonical, decisions.renames)
    return tuple(sorted(by_canonical.values(), key=lambda c: c.canonical))


def _apply_splits(
    by_canonical: dict[str, SampleCluster],
    splits: dict[str, dict[str, str]],
) -> dict[str, SampleCluster]:
    """Pull files out of source clusters into target clusters.

    A target that doesn't yet exist is created as a new cluster
    holding only the split-out files. The source cluster keeps the
    files that weren't reassigned. A source that has *all* its files
    pulled out vanishes.
    """
    out = dict(by_canonical)
    for source, mapping in splits.items():
        src = out.get(source)
        if src is None:
            continue
        # Group reassignments by target canonical.
        per_target: dict[str, list[Path]] = {}
        kept_files: list[Path] = []
        for fp in src.file_paths:
            target = mapping.get(str(fp))
            if target:
                per_target.setdefault(target, []).append(fp)
            else:
                kept_files.append(fp)

        if not per_target:
            continue

        # Update the source cluster — drop files (or remove entirely if empty).
        if kept_files:
            out[source] = SampleCluster(
                canonical=src.canonical,
                aliases=src.aliases,
                file_paths=tuple(sorted(kept_files)),
                normalized_forms=src.normalized_forms,
            )
        else:
            del out[source]

        # Create / extend each target cluster.
        for target, files in per_target.items():
            existing = out.get(target)
            if existing is None:
                out[target] = SampleCluster(
                    canonical=target,
                    aliases=(target,),
                    file_paths=tuple(sorted(files)),
                    normalized_forms=(),
                )
            else:
                out[target] = SampleCluster(
                    canonical=existing.canonical,
                    aliases=existing.aliases,
                    file_paths=tuple(sorted([*existing.file_paths, *files])),
                    normalized_forms=existing.normalized_forms,
                )
    return out


def _apply_merges(
    by_canonical: dict[str, SampleCluster],
    merges: tuple[tuple[str, ...], ...],
) -> dict[str, SampleCluster]:
    """Collapse each merge group into a single cluster keyed by group[0]."""
    out = dict(by_canonical)
    for group in merges:
        survivors = [c for c in (out.get(name) for name in group) if c is not None]
        if len(survivors) < _MIN_MERGE_GROUP_SIZE:
            # A merge group whose names don't all exist (e.g. one was
            # already split out / removed) is a no-op rather than an
            # error — the user's intent ("these are the same sample")
            # still holds for whatever's left.
            continue
        survivor_canonical = group[0]
        merged_aliases: set[str] = set()
        merged_files: list[Path] = []
        merged_norms: set[str] = set()
        for c in survivors:
            merged_aliases.update(c.aliases)
            merged_files.extend(c.file_paths)
            merged_norms.update(c.normalized_forms)
        # Drop the absorbed clusters.
        for name in group:
            out.pop(name, None)
        canonical = (
            survivor_canonical
            if survivor_canonical in merged_aliases
            else (pick_canonical(merged_aliases) if merged_aliases else survivor_canonical)
        )
        out[canonical] = SampleCluster(
            canonical=canonical,
            aliases=tuple(sorted(merged_aliases)),
            file_paths=tuple(sorted(merged_files)),
            normalized_forms=tuple(sorted(merged_norms)),
        )
    return out


def _apply_renames(
    by_canonical: dict[str, SampleCluster],
    renames: dict[str, str],
) -> dict[str, SampleCluster]:
    """Rewrite each cluster's canonical to its renamed form, if any."""
    if not renames:
        return by_canonical
    out: dict[str, SampleCluster] = {}
    for canonical, cluster in by_canonical.items():
        new_name = renames.get(canonical, canonical)
        if new_name == canonical:
            out[canonical] = cluster
        else:
            new_aliases = tuple(sorted({*cluster.aliases, new_name}))
            out[new_name] = SampleCluster(
                canonical=new_name,
                aliases=new_aliases,
                file_paths=cluster.file_paths,
                normalized_forms=cluster.normalized_forms,
            )
    return out
