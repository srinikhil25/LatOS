"""Pure project-editing transforms for the human-verification gate.

Each function takes a `Project` and returns a *new* `Project` (the
domain model is frozen). The four structural edits — rename a sample,
re-type a measurement, merge samples, split measurements into a new
sample — all reset the project to `NEEDS_REVIEW`: any change to the
categorization invalidates a prior confirmation. `confirm` / `reopen`
flip the status explicitly.

Keeping these as pure functions (no DB, no I/O) makes them trivially
testable; `ServerState.apply_edit` is what persists the result.
"""

from __future__ import annotations

from dataclasses import replace

from latos.core.enums import ReviewStatus, Technique
from latos.core.models import Measurement, Project, Sample, new_id, utc_now

__all__ = [
    "EditError",
    "confirm",
    "merge_samples",
    "move_measurements_to_new_sample",
    "move_measurements_to_sample",
    "remove_measurements",
    "rename_sample",
    "reopen",
    "set_measurement_technique",
]


class EditError(ValueError):
    """A project edit could not be applied (bad id, empty result, …)."""


def _needs_review(project: Project, samples: tuple[Sample, ...]) -> Project:
    """Rebuild `project` with new samples, forced back to NEEDS_REVIEW."""
    return replace(
        project,
        samples=samples,
        review_status=ReviewStatus.NEEDS_REVIEW,
        confirmed_at=None,
    )


def _find_sample(project: Project, sample_id: str) -> Sample:
    for sample in project.samples:
        if sample.id == sample_id:
            return sample
    raise EditError(f"Unknown sample: {sample_id}")


# ─── Status transitions ──────────────────────────────────────────────
def confirm(project: Project) -> Project:
    """Mark the project CONFIRMED so downstream phases may run."""
    return replace(project, review_status=ReviewStatus.CONFIRMED, confirmed_at=utc_now())


def reopen(project: Project) -> Project:
    """Send a confirmed project back to NEEDS_REVIEW for further edits."""
    return replace(project, review_status=ReviewStatus.NEEDS_REVIEW, confirmed_at=None)


# ─── Structural edits ────────────────────────────────────────────────
def rename_sample(project: Project, sample_id: str, new_name: str) -> Project:
    """Rename one sample's canonical name."""
    if not new_name.strip():
        raise EditError("Sample name cannot be empty")
    _find_sample(project, sample_id)  # existence check
    samples = tuple(
        replace(s, canonical_name=new_name.strip()) if s.id == sample_id else s
        for s in project.samples
    )
    return _needs_review(project, samples)


def set_measurement_technique(
    project: Project,
    measurement_id: str,
    technique: Technique,
) -> Project:
    """Override the technique of a single measurement (e.g. TEM → STEM)."""
    found = False
    new_samples: list[Sample] = []
    for sample in project.samples:
        measurements = []
        for m in sample.measurements:
            if m.id == measurement_id:
                measurements.append(replace(m, technique=technique))
                found = True
            else:
                measurements.append(m)
        new_samples.append(replace(sample, measurements=tuple(measurements)))
    if not found:
        raise EditError(f"Unknown measurement: {measurement_id}")
    return _needs_review(project, tuple(new_samples))


def merge_samples(project: Project, source_ids: list[str], target_id: str) -> Project:
    """Move every measurement from `source_ids` into `target_id`.

    The target keeps its name; each source's name + aliases are added to
    the target's aliases so the merge is traceable. Sources are dropped.
    """
    if not source_ids:
        raise EditError("No source samples to merge")
    if target_id in source_ids:
        raise EditError("Target sample cannot also be a source")
    target = _find_sample(project, target_id)
    sources = [_find_sample(project, sid) for sid in source_ids]

    moved = [replace(m, sample_id=target_id) for src in sources for m in src.measurements]
    new_aliases = set(target.aliases)
    for src in sources:
        new_aliases.add(src.canonical_name)
        new_aliases.update(src.aliases)

    drop = set(source_ids)
    new_samples: list[Sample] = []
    for sample in project.samples:
        if sample.id in drop:
            continue
        if sample.id == target_id:
            new_samples.append(
                replace(
                    sample,
                    aliases=tuple(sorted(new_aliases)),
                    measurements=(*sample.measurements, *moved),
                ),
            )
        else:
            new_samples.append(sample)
    return _needs_review(project, tuple(new_samples))


def move_measurements_to_new_sample(
    project: Project,
    measurement_ids: list[str],
    new_name: str,
) -> Project:
    """Pull the given measurements out into a brand-new sample.

    The fix for over-merged identity: select the measurements that don't
    belong and split them into their own sample. Source samples that
    become empty are dropped.
    """
    if not measurement_ids:
        raise EditError("No measurements selected")
    if not new_name.strip():
        raise EditError("New sample name cannot be empty")

    wanted = set(measurement_ids)
    new_sample_id = new_id()
    moved: list[Measurement] = []
    new_samples: list[Sample] = []

    for sample in project.samples:
        kept = []
        for m in sample.measurements:
            if m.id in wanted:
                moved.append(replace(m, sample_id=new_sample_id))
            else:
                kept.append(m)
        if kept:
            new_samples.append(replace(sample, measurements=tuple(kept)))
        # else: sample became empty → dropped

    if len(moved) != len(wanted):
        found_ids = {m.id for m in moved}
        missing = wanted - found_ids
        raise EditError(f"Unknown measurement(s): {sorted(missing)}")

    new_samples.append(
        Sample(
            id=new_sample_id,
            project_id=project.id,
            canonical_name=new_name.strip(),
            aliases=(),
            measurements=tuple(moved),
        ),
    )
    return _needs_review(project, tuple(new_samples))


def move_measurements_to_sample(
    project: Project,
    measurement_ids: list[str],
    target_sample_id: str,
) -> Project:
    """Reassign the given measurements to an existing sample.

    The right-click "Move to…" action. Source samples emptied by the
    move are dropped.
    """
    if not measurement_ids:
        raise EditError("No measurements selected")
    _find_sample(project, target_sample_id)  # existence check

    wanted = set(measurement_ids)
    moved: list[Measurement] = []
    remaining: dict[str, list[Measurement]] = {}

    for sample in project.samples:
        kept = []
        for m in sample.measurements:
            if m.id in wanted:
                moved.append(replace(m, sample_id=target_sample_id))
            else:
                kept.append(m)
        remaining[sample.id] = kept

    if len(moved) != len(wanted):
        missing = wanted - {m.id for m in moved}
        raise EditError(f"Unknown measurement(s): {sorted(missing)}")

    new_samples: list[Sample] = []
    for sample in project.samples:
        kept = remaining[sample.id]
        if sample.id == target_sample_id:
            new_samples.append(replace(sample, measurements=(*kept, *moved)))
        elif kept:
            new_samples.append(replace(sample, measurements=tuple(kept)))
        # else: a non-target sample emptied by the move → dropped
    return _needs_review(project, tuple(new_samples))


def remove_measurements(project: Project, measurement_ids: list[str]) -> Project:
    """Drop measurements from the project (soft delete).

    The raw files on disk are NOT touched — this only excludes them from
    the Latos project and downstream analysis. A re-ingest restores them.
    Samples emptied by the removal are dropped.
    """
    if not measurement_ids:
        raise EditError("No measurements selected")
    wanted = set(measurement_ids)
    removed = 0
    new_samples: list[Sample] = []

    for sample in project.samples:
        kept = [m for m in sample.measurements if m.id not in wanted]
        removed += len(sample.measurements) - len(kept)
        if kept:
            new_samples.append(replace(sample, measurements=tuple(kept)))
        # else: sample fully removed → dropped

    if removed != len(wanted):
        raise EditError("One or more measurements not found")
    return _needs_review(project, tuple(new_samples))
