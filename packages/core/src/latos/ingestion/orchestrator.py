"""`Orchestrator` — turn a folder of instrument files into a saved `Project`.

This is the integration layer where every layer below it finally connects:

    Crawler (1D.1)         walks the folder, hashes, classifies
        ↓
    BaseParser (1C.x)      parses each classified file → ParsedData
        ↓
    Domain models (1A)     ParsedData → Sample / Measurement / FileRef
        ↓
    ArrayStore (1C.2)      writes Parquet array files atomically
    Repository (1B)        saves SQL rows
        ↓
    Project (returned to caller)

The orchestrator is the *only* place in the codebase where all of those
modules show up in the same file. Everything else stays one-layer-narrow.

Sample-grouping heuristic (Stage 1 only)
----------------------------------------
A file's sample is the name of its immediate parent folder, *unless*
that folder name is generic (e.g. "XRD", "data", "raw"). In that case,
we walk up the tree looking for a non-generic ancestor; if none is
found within three levels, we fall back to the file's own stem.

This is deliberately dumb. Stage 2's smart-labeling layer (mechanical
heuristics + AI/VLM) replaces it entirely. For Stage 1, dumb-but-stable
beats clever-but-surprising: the user can fix wrong groupings in the
review UI (Stage 1E), and re-running ingestion preserves their decisions
because we key on `sha256`, not on inferred sample names.

Idempotence
-----------
Re-ingesting the same folder is fast and produces no duplicates: each
file's `sha256` is checked against the existing project's `FileRow`s
before we parse. A file we've seen with the same hash *and* the same
parser version skips parsing entirely.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from latos.core.enums import FileRole, ReviewStatus, Severity, Technique
from latos.core.models import (
    FileRef,
    Measurement,
    Project,
    Sample,
    ValidationIssue,
    new_id,
    utc_now,
)
from latos.ingestion.array_store import ArrayStore
from latos.ingestion.base_parser import BaseParser
from latos.ingestion.crawler import (
    CrawlEntry,
    ProgressCallback,
    crawl,
)
from latos.ingestion.labeling.normalize import normalize
from latos.ingestion.parsed_data import ParsedData
from latos.ingestion.registry import ParserRegistry
from latos.persistence.db import (
    ensure_project_dirs,
    project_arrays_dir,
)
from latos.persistence.repository import ProjectRepository
from latos.persistence.schema import LATEST_SCHEMA_VERSION

__all__ = [
    "FileOutcome",
    "IngestionResult",
    "Orchestrator",
    "Outcome",
]


# Folder names too generic to identify a sample. The grouping heuristic
# walks up the tree past these. All comparisons are case- and
# whitespace-folded (`_normalize_folder`).
_GENERIC_FOLDER_NAMES: frozenset[str] = frozenset(
    {
        # Technique labels
        "xrd",
        "xps",
        "uvdrs",
        "uv-drs",
        "uv_drs",
        "hall",
        "hall measurement",
        "hall measurements",
        "thermoelectric",
        "thermoelectric properties",
        "eds",
        "edx",
        "eds-edx",
        "tem",
        "sem",
        "stem",
        "fe-sem",
        "fe sem",
        "hr-fe-sem",
        "hr fe sem",
        "microscopy",
        "raman",
        # Generic data folders
        "data",
        "raw",
        "raw data",
        "rawdata",
        "results",
        "characterization",
        "analysis",
        "files",
        "samples",
    },
)

# How many levels up from a generic parent we'll search before giving up
# and falling back to the filename stem. Three matches the typical
# `RAW DATA / TECHNIQUE / SAMPLE / file` depth seen in the predecessor data.
_MAX_PARENT_LOOKUP_DEPTH = 3


class Outcome(StrEnum):
    """Result of ingesting one file."""

    PARSED = "parsed"  # parser ran, measurement saved
    PARSED_WITH_ISSUES = "parsed_with_issues"  # parser ran but emitted ERROR-severity issues
    PARSE_FAILED = "parse_failed"  # parser raised; file dropped
    SKIPPED_UNCLASSIFIED = "skipped_unclassified"  # below-threshold; not parsed
    SKIPPED_HASH_FAILED = "skipped_hash_failed"  # crawler couldn't hash; not parsed
    SKIPPED_CACHED = "skipped_cached"  # already in DB with matching parser_version


@dataclass(frozen=True, slots=True)
class FileOutcome:
    """Per-file record of what happened during ingestion."""

    path: Path
    relative_path: Path
    sha256: str | None
    outcome: Outcome
    sample_name: str | None
    parser_name: str | None
    measurement_id: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Complete result of one `Orchestrator.ingest()` call.

    The `Project` is the persisted aggregate that the UI / CLI consumes.
    `outcomes` is the per-file ledger — useful for showing the user a
    "what happened" table after ingestion.
    """

    project: Project
    outcomes: tuple[FileOutcome, ...]

    @property
    def parsed_count(self) -> int:
        """Count of files that were parsed successfully (no errors)."""
        return sum(1 for o in self.outcomes if o.outcome == Outcome.PARSED)

    @property
    def failed_count(self) -> int:
        """Count of files where parsing raised an exception."""
        return sum(1 for o in self.outcomes if o.outcome == Outcome.PARSE_FAILED)

    @property
    def unclassified_count(self) -> int:
        """Count of files no parser claimed at >= threshold confidence."""
        return sum(1 for o in self.outcomes if o.outcome == Outcome.SKIPPED_UNCLASSIFIED)

    @property
    def cached_count(self) -> int:
        """Count of files that hit the parse cache (sha256 + parser_version match)."""
        return sum(1 for o in self.outcomes if o.outcome == Outcome.SKIPPED_CACHED)


# Factories let tests inject in-memory engines / tmp_path stores. Production
# uses the defaults defined in `_default_repo_factory` / `_default_store_factory`.
#
# The repo factory returns a *context manager* so the engine can be disposed
# when ingestion finishes — SQLite leaks file handles otherwise, and on
# Windows that prevents `tmp_path` cleanup. The array store doesn't hold
# resources, so its factory stays a plain callable.
RepositoryFactory = Callable[[Path], AbstractContextManager[ProjectRepository]]
ArrayStoreFactory = Callable[[Path], ArrayStore]


@dataclass
class Orchestrator:
    """End-to-end ingestion: folder → CrawlReport → parsing → persisted Project.

    Construct with a `ParserRegistry` (typically `default_registry()`)
    and optionally inject custom repository / array-store factories
    for testing. Each call to `.ingest(root)` is independent.
    """

    registry: ParserRegistry
    repo_factory: RepositoryFactory = field(default=None)  # type: ignore[assignment]
    array_store_factory: ArrayStoreFactory = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Defer the default factories' import until we need them — avoids
        # an import cycle (persistence → ingestion → persistence).
        if self.repo_factory is None:
            self.repo_factory = _default_repo_factory
        if self.array_store_factory is None:
            self.array_store_factory = _default_store_factory

    # ─── Public API ──────────────────────────────────────────────────
    def ingest(
        self,
        root: Path,
        *,
        project_name: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Ingest `root` end-to-end. Returns the persisted Project + outcomes.

        Args:
            root: Folder of raw instrument data. `<root>/.latos/` is
                created if missing.
            project_name: Display name for the project. Defaults to the
                root folder's basename.
            on_progress: Forwarded to the crawler. Stage 1E uses this
                to drive a progress bar.

        Returns:
            `IngestionResult` with the saved `Project` and a per-file ledger.
        """
        root = Path(root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Ingestion root is not a directory: {root}")

        ensure_project_dirs(root)
        store = self.array_store_factory(root)

        # `with` ensures SQLite engine disposal even if ingestion raises —
        # leaked engines hold file handles that block tmp dir cleanup on
        # Windows and prevent the user from deleting/moving the project
        # folder while Latos is still "open" on it.
        with self.repo_factory(root) as repo:
            existing = self._load_existing_project(repo)
            project_id = existing.id if existing is not None else new_id()
            created_at = existing.created_at if existing is not None else utc_now()
            display_name = (
                project_name
                if project_name is not None
                else (existing.name if existing is not None else root.name)
            )

            # Index existing files by sha256 → (measurement_id, parser_version)
            # for cache hits during this ingestion run.
            cached_files = _index_existing_files(existing) if existing else {}

            # Crawl.
            report = crawl(root, self.registry, on_progress=on_progress)

            # Build samples/measurements from the report.
            samples_by_name: dict[str, _SampleAccumulator] = {}
            outcomes: list[FileOutcome] = []
            unassigned: list[FileRef] = []

            if existing is not None:
                # Seed accumulators with prior samples so re-ingesting preserves them.
                for s in existing.samples:
                    key = _canonical_key(s.canonical_name)
                    target = samples_by_name.get(key)
                    if target is None:
                        samples_by_name[key] = _SampleAccumulator(
                            id=s.id,
                            canonical_name=s.canonical_name,
                            aliases=set(s.aliases),
                            measurements=list(s.measurements),
                        )
                    else:
                        # Two samples from a prior, pre-canonicalization
                        # ingest collide on the canonical key (e.g.
                        # "CS Pure" and "CS (Pure)"). Heal the split by
                        # folding this one into the first; rewrite the
                        # moved measurements' sample_id so the rebuilt
                        # Sample stays internally consistent.
                        target.aliases.add(s.canonical_name)
                        target.aliases.update(s.aliases)
                        target.measurements.extend(
                            replace(m, sample_id=target.id) for m in s.measurements
                        )
                unassigned = list(existing.unassigned_files)

            current_run_ids: set[str] = set()
            for entry in report.entries:
                # A single file can produce multiple measurements (e.g. a
                # multi-sheet workbook → one measurement per sheet); the
                # orchestrator iterates the list and records one
                # FileOutcome per produced (or skipped) measurement.
                entry_outcomes = self._handle_entry(
                    entry=entry,
                    root=root,
                    project_id=project_id,
                    store=store,
                    samples_by_name=samples_by_name,
                    cached_files=cached_files,
                )
                outcomes.extend(entry_outcomes)
                for o in entry_outcomes:
                    if o.outcome in (Outcome.PARSED, Outcome.PARSED_WITH_ISSUES) and (
                        o.measurement_id is not None
                    ):
                        current_run_ids.add(o.measurement_id)

            # Dedupe: drop OLD measurements superseded by current-run
            # ones with the same sha256. Sibling measurements from
            # this run (multi-sheet workbook → one per sheet) all stay.
            _dedupe_measurements_by_sha256(samples_by_name, current_run_ids=current_run_ids)

            # Preserve a prior confirmation across re-ingest: re-opening a
            # confirmed project shouldn't silently demote it. The edit
            # endpoints are what flip a project back to NEEDS_REVIEW when
            # the user actually changes its categorization.
            review_status = (
                existing.review_status if existing is not None else ReviewStatus.NEEDS_REVIEW
            )
            confirmed_at = existing.confirmed_at if existing is not None else None

            project = Project(
                id=project_id,
                name=display_name,
                root_path=root,
                created_at=created_at,
                schema_version=LATEST_SCHEMA_VERSION,
                samples=tuple(acc.build(project_id) for acc in samples_by_name.values()),
                unassigned_files=tuple(unassigned),
                review_status=review_status,
                confirmed_at=confirmed_at,
            )

            repo.save(project)
            return IngestionResult(project=project, outcomes=tuple(outcomes))

    # ─── Internals ───────────────────────────────────────────────────
    @staticmethod
    def _load_existing_project(repo: ProjectRepository) -> Project | None:
        """Load the existing project for this repo, or None on empty DB."""
        try:
            return repo.load_first()
        except Exception:
            return None

    def _handle_entry(  # noqa: PLR0911
        self,
        *,
        entry: CrawlEntry,
        root: Path,
        project_id: str,
        store: ArrayStore,
        samples_by_name: dict[str, _SampleAccumulator],
        cached_files: dict[str, tuple[str, str]],
    ) -> list[FileOutcome]:
        """Process one crawl entry.

        Returns a list because a single file can produce multiple
        measurements (multi-sheet workbooks → one measurement per
        sheet). Updates `samples_by_name` in place.
        """
        # Hash failure → can't dedup or parse safely.
        if entry.error is not None or entry.sha256 is None:
            return [
                FileOutcome(
                    path=entry.path,
                    relative_path=entry.relative_path,
                    sha256=entry.sha256,
                    outcome=Outcome.SKIPPED_HASH_FAILED,
                    sample_name=None,
                    parser_name=None,
                    measurement_id=None,
                    error=entry.error,
                ),
            ]

        # Below confidence threshold → unknown technique. Not parsed.
        if entry.classified_parser is None:
            return [
                FileOutcome(
                    path=entry.path,
                    relative_path=entry.relative_path,
                    sha256=entry.sha256,
                    outcome=Outcome.SKIPPED_UNCLASSIFIED,
                    sample_name=None,
                    parser_name=entry.best_match_parser,  # diagnostic
                    measurement_id=None,
                    error=None,
                ),
            ]

        # Find the parser instance for this classification.
        match = self.registry.find_parser(entry.path)
        if match is None:
            # Defensive: classified_parser was set, but find_parser now
            # disagrees. Treat as unclassified.
            return [
                FileOutcome(
                    path=entry.path,
                    relative_path=entry.relative_path,
                    sha256=entry.sha256,
                    outcome=Outcome.SKIPPED_UNCLASSIFIED,
                    sample_name=None,
                    parser_name=entry.classified_parser,
                    measurement_id=None,
                    error=None,
                ),
            ]
        parser = match.parser

        # Cache hit: same sha256 already stored with the same parser_version → skip.
        cached = cached_files.get(entry.sha256)
        if cached is not None and cached[1] == parser.version:
            return [
                FileOutcome(
                    path=entry.path,
                    relative_path=entry.relative_path,
                    sha256=entry.sha256,
                    outcome=Outcome.SKIPPED_CACHED,
                    sample_name=None,
                    parser_name=parser.name,
                    measurement_id=cached[0],
                    error=None,
                ),
            ]

        # Parse — `parse_all` may return multiple ParsedData (one per
        # workbook sheet, etc.) or just one for the typical case.
        try:
            parsed_list = parser.parse_all(entry.path)
        except Exception as exc:
            return [
                FileOutcome(
                    path=entry.path,
                    relative_path=entry.relative_path,
                    sha256=entry.sha256,
                    outcome=Outcome.PARSE_FAILED,
                    sample_name=None,
                    parser_name=parser.name,
                    measurement_id=None,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            ]

        if not parsed_list:
            # Parser returned zero results — treat as a parse failure.
            return [
                FileOutcome(
                    path=entry.path,
                    relative_path=entry.relative_path,
                    sha256=entry.sha256,
                    outcome=Outcome.PARSE_FAILED,
                    sample_name=None,
                    parser_name=parser.name,
                    measurement_id=None,
                    error="parse_all returned no ParsedData entries",
                ),
            ]

        # Pre-compute the path-based fallback sample name once per file —
        # all sheets share it when the parser doesn't supply a sheet name.
        path_sample_name, path_generic_warning = _infer_sample_name(entry.path, root)
        # Folder-aware technique refinement for parsers that default to
        # a placeholder (microscopy → SEM by default; the folder tells
        # us whether it's actually TEM, STEM, FE-SEM, etc.).
        refined_technique = _refine_technique_from_folders(entry.path, root)

        file_ref = FileRef(
            path=entry.path,
            sha256=entry.sha256,
            size_bytes=entry.size_bytes,
            role=FileRole.RAW,
            scanned_at=entry.mtime,
        )

        outcomes: list[FileOutcome] = []
        for parsed in parsed_list:
            outcomes.append(
                self._persist_one_parsed(
                    parsed=parsed,
                    entry=entry,
                    parser=parser,
                    project_id=project_id,
                    store=store,
                    samples_by_name=samples_by_name,
                    file_ref=file_ref,
                    path_sample_name=path_sample_name,
                    path_generic_warning=path_generic_warning,
                    refined_technique=refined_technique,
                ),
            )
        return outcomes

    def _persist_one_parsed(
        self,
        *,
        parsed: ParsedData,
        entry: CrawlEntry,
        parser: BaseParser,
        project_id: str,
        store: ArrayStore,
        samples_by_name: dict[str, _SampleAccumulator],
        file_ref: FileRef,
        path_sample_name: str,
        path_generic_warning: ValidationIssue | None,
        refined_technique: Technique | None,
    ) -> FileOutcome:
        """Persist one `ParsedData` as a `Measurement` and return its outcome."""
        # Sheet name (when the parser provides one) is a stronger
        # sample-name signal than the file's parent folder: a workbook
        # like `zT calculation.xlsx` with sheets ["CS", "CSCBI-1", ...]
        # tells us each sheet is a distinct sample. Stage 2's labeling
        # pipeline still gets to merge cosmetically-different names
        # across files.
        meta = parsed.metadata or {}
        # A parser may know the sample name better than the folder does —
        # e.g. files grouped in technique folders (`LFA/CS LFA.xlsx`) where
        # the sample is in the filename, not the parent folder. An explicit
        # `sample_name` wins; then a worksheet name (multi-sample workbook);
        # then the path-based fallback.
        explicit_name = meta.get("sample_name")
        sheet_name = meta.get("sheet_name")
        if isinstance(explicit_name, str) and explicit_name.strip():
            sample_name = explicit_name.strip()
            generic_warning = None
        elif isinstance(sheet_name, str) and sheet_name.strip():
            sample_name = sheet_name.strip()
            generic_warning = None  # sheet name isn't a fallback
        else:
            sample_name = path_sample_name
            generic_warning = path_generic_warning

        measurement_id = new_id()
        parsed_data_path = store.write(measurement_id, parsed)

        issues = list(parsed.issues)
        if generic_warning is not None:
            issues.append(generic_warning)

        # Apply technique refinement only when the parser defaulted to
        # a placeholder (microscopy → SEM). Other parsers know their
        # technique with certainty and shouldn't be overridden.
        technique = (
            refined_technique
            if refined_technique is not None and parsed.technique is Technique.SEM
            else parsed.technique
        )

        accumulator = _get_or_create_accumulator(samples_by_name, sample_name)
        measurement = Measurement(
            id=measurement_id,
            sample_id=accumulator.id,
            technique=technique,
            instrument=parsed.instrument,
            measured_at=parsed.measured_at,
            parsed_at=utc_now(),
            parser_version=parser.version,
            files=(file_ref,),
            issues=tuple(issues),
            parsed_data_path=parsed_data_path,
            features=dict(parsed.features),
        )

        accumulator.measurements.append(measurement)

        outcome_kind = (
            Outcome.PARSED_WITH_ISSUES
            if any(i.severity is Severity.ERROR for i in measurement.issues)
            else Outcome.PARSED
        )
        return FileOutcome(
            path=entry.path,
            relative_path=entry.relative_path,
            sha256=entry.sha256,
            outcome=outcome_kind,
            sample_name=sample_name,
            parser_name=parser.name,
            measurement_id=measurement_id,
            error=None,
        )


# ─── Sample inference ──────────────────────────────────────────────────
def _normalize_folder(name: str) -> str:
    """Lowercase + strip + collapse whitespace, for generic-name matching."""
    return " ".join(name.lower().strip().split())


def _is_generic(name: str) -> bool:
    """True if `name` is a generic technique/data folder label."""
    return _normalize_folder(name) in _GENERIC_FOLDER_NAMES


def _canonical_key(name: str) -> str:
    """Identity key for sample grouping: collapse cosmetic spelling variants.

    Two names that differ only in case, separators, or a leading
    "Sample:"/"Specimen" prefix — e.g. ``CS (Pure)`` vs ``CS Pure`` vs
    ``cs_pure`` — share one key and therefore one `Sample`. Digits are
    preserved, so a doping series (``CS-1`` vs ``CS-3``) stays distinct:
    this merges the cosmetic duplicates *without* the aggressive
    fuzzy merging that over-grouped doped samples (BUG #16).

    The researcher confirms or overrides every grouping in the review
    gate, so this is the safe "identical-after-cleaning" tier only —
    anything requiring a similarity judgement is left to the human.

    Falls back to a stripped-lowercased form when normalization empties
    the string (a name that is all separators) so every sample still
    gets a stable, non-empty key.
    """
    return normalize(name) or name.strip().lower()


# Folder-name → technique map for microscopy. Keys are case-insensitive
# normalized folder names (whitespace-collapsed via `_normalize_folder`).
# Microscopy parsers default to SEM because TIFF tags rarely identify
# the actual modality; the user's folder structure usually does.
_MICROSCOPY_FOLDER_TECHNIQUES: dict[str, Technique] = {
    "sem": Technique.SEM,
    "fe-sem": Technique.SEM,
    "fe sem": Technique.SEM,
    "hr-fe-sem": Technique.SEM,
    "hr fe sem": Technique.SEM,
    "tem": Technique.TEM,
    "hr-tem": Technique.TEM,
    "hr tem": Technique.TEM,
    "stem": Technique.STEM,
}

# When a path contains more than one microscopy keyword (real datasets
# nest, e.g. `TEM/STEM/<sample>/TEM/CS/img.tif`), the most *specific*
# modality wins rather than the nearest folder. STEM ⊂ TEM-family, so a
# `STEM` anywhere in the path means STEM even if a `TEM` folder sits
# closer to the file; SEM is the least specific fallback.
_TECHNIQUE_SPECIFICITY: tuple[Technique, ...] = (
    Technique.STEM,
    Technique.TEM,
    Technique.SEM,
)


def _refine_technique_from_folders(path: Path, root: Path) -> Technique | None:
    """Return a more specific microscopy technique from parent folder names.

    The microscopy TIFF parser defaults to `Technique.SEM` because TIFF
    metadata rarely identifies the modality (SEM vs TEM vs STEM). Most
    researchers organise images into folders like `TEM/`, `HR-FE-SEM/`,
    or `STEM/`.

    We collect *every* microscopy folder between the file and the project
    root (not just the nearest), then resolve by specificity — so a
    `STEM` image filed under a top-level `TEM/` umbrella with a nested
    `.../TEM/CS/` leaf still classifies as STEM, not TEM. Walking to the
    root (rather than a fixed depth) is what lets us see the outer
    keyword; it's bounded and only runs for already-claimed microscopy
    files.

    Returns `None` when no parent folder matches our microscopy
    keywords — the caller leaves the parser's default in place.
    """
    matches: set[Technique] = set()
    p = path.parent
    while p not in (root, p.parent):
        match = _MICROSCOPY_FOLDER_TECHNIQUES.get(_normalize_folder(p.name))
        if match is not None:
            matches.add(match)
        p = p.parent
    for technique in _TECHNIQUE_SPECIFICITY:
        if technique in matches:
            return technique
    return None


def _infer_sample_name(path: Path, root: Path) -> tuple[str, ValidationIssue | None]:
    """Pick a sample name for a file, with a warning if the inference was a fallback.

    Walks up to `_MAX_PARENT_LOOKUP_DEPTH` levels of parent folders
    looking for a non-generic name. If all are generic, falls back to
    the file's own stem and emits a WARNING-level `ValidationIssue`
    so the UI can flag it for user review.
    """
    parents = []
    p = path.parent
    for _ in range(_MAX_PARENT_LOOKUP_DEPTH):
        # Stop walking once we hit the project root — names beyond it
        # are user-system folders we have no business inferring from.
        # `p.parent == p` catches the filesystem-root sentinel.
        if p in (root, p.parent):
            break
        parents.append(p.name)
        if not _is_generic(p.name):
            return p.name, None
        p = p.parent

    # All parents (up to depth) were generic — fall back to filename stem.
    fallback = path.stem
    issue = ValidationIssue(
        field="sample_name",
        severity=Severity.WARNING,
        message=(
            f"Sample name inferred from filename stem ({fallback!r}); "
            f"all parent folders ({parents!r}) are generic. Please review."
        ),
        detected_at=utc_now(),
    )
    return fallback, issue


# ─── Sample accumulator ────────────────────────────────────────────────
@dataclass
class _SampleAccumulator:
    """Mutable scratchpad while building a Sample's measurement list.

    We can't mutate frozen `Sample` objects, so we accumulate measurements
    in a list and call `build()` at the end to produce the immutable
    domain object.
    """

    id: str
    canonical_name: str
    aliases: set[str] = field(default_factory=set)
    measurements: list[Measurement] = field(default_factory=list)

    def build(self, project_id: str) -> Sample:
        """Materialize the accumulator into a frozen `Sample`."""
        return Sample(
            id=self.id,
            project_id=project_id,
            canonical_name=self.canonical_name,
            aliases=tuple(sorted(self.aliases)),
            measurements=tuple(self.measurements),
        )


def _get_or_create_accumulator(
    samples_by_name: dict[str, _SampleAccumulator],
    sample_name: str,
) -> _SampleAccumulator:
    """Look up or create the `_SampleAccumulator` for a raw sample name.

    The dict key is `_canonical_key(sample_name)`, so cosmetically
    different spellings of the same sample (``CS Pure`` and
    ``CS (Pure)``) resolve to one accumulator and their measurements
    aggregate under a single `Sample`.

    The first spelling seen becomes the display `canonical_name` (the
    user can rename it in review); every other spelling that maps to
    the same key is recorded as an alias so the original wording is
    never lost.
    """
    key = _canonical_key(sample_name)
    acc = samples_by_name.get(key)
    if acc is None:
        acc = _SampleAccumulator(id=new_id(), canonical_name=sample_name)
        samples_by_name[key] = acc
    elif sample_name != acc.canonical_name:
        acc.aliases.add(sample_name)
    return acc


def _dedupe_measurements_by_sha256(
    samples_by_name: dict[str, _SampleAccumulator],
    *,
    current_run_ids: set[str],
) -> None:
    """Drop OLD measurements superseded by current-run ones with the same sha256.

    A single file can legitimately appear in multiple measurements -
    a multi-sheet workbook produces one measurement per sheet, all
    referencing the same sha256. Stage 1F dropped the UNIQUE(sha256)
    constraint to allow this.

    The dedup invariant is narrower than it used to be: for any sha256
    that the current run produced at least one measurement for, drop
    every prior measurement seeded from the existing project (those
    were created from an older parser version or before the
    multi-sheet expansion). Files where the current run produced no
    measurements (cache hit) keep their seeded measurement untouched.

    Mutates accumulators in place; samples may end up empty (kept
    as-is - the user's review UI decides what to do).
    """
    # Find which sha256s the current run produced at least one measurement for.
    current_shas: set[str] = set()
    for acc in samples_by_name.values():
        for m in acc.measurements:
            if m.id in current_run_ids:
                for f in m.files:
                    current_shas.add(f.sha256)

    # For each sha256 in current_shas, drop measurements NOT in current_run_ids
    # (those are the stale seeded ones). All current-run siblings stay.
    for acc in samples_by_name.values():
        kept: list[Measurement] = []
        for m in acc.measurements:
            file_shas = {f.sha256 for f in m.files}
            if m.id in current_run_ids:
                kept.append(m)
            elif file_shas & current_shas:
                # Seeded measurement whose file was re-parsed this
                # run — drop it.
                continue
            else:
                kept.append(m)
        acc.measurements = kept


# ─── Cache helpers ────────────────────────────────────────────────────
def _index_existing_files(project: Project) -> dict[str, tuple[str, str]]:
    """Map sha256 → (measurement_id, parser_version) from an existing project.

    Used to short-circuit re-parsing on idempotent re-ingestions: if a
    file's hash is already present *and* the parser version matches, we
    skip it entirely.
    """
    out: dict[str, tuple[str, str]] = {}
    for sample in project.samples:
        for m in sample.measurements:
            for f in m.files:
                out[f.sha256] = (m.id, m.parser_version)
    return out


# ─── Default factories ────────────────────────────────────────────────
@contextmanager
def _default_repo_factory(root: Path) -> Iterator[ProjectRepository]:
    """Build a `ProjectRepository` for `root` and dispose the engine on exit.

    SQLAlchemy engines hold file handles via SQLite's WAL pragma; not
    disposing them blocks Windows from cleaning up tmp dirs and prevents
    the user from moving/deleting the project folder while Latos is open.
    """
    # Local imports to avoid pulling SQLAlchemy in unless this code is hit.
    from latos.persistence.db import (  # noqa: PLC0415 — see comment
        create_project_engine,
        init_schema,
        make_session_factory,
    )

    engine = create_project_engine(root)
    init_schema(engine)
    try:
        yield ProjectRepository(make_session_factory(engine))
    finally:
        engine.dispose()


def _default_store_factory(root: Path) -> ArrayStore:
    """Build an `ArrayStore` for the given project root."""
    return ArrayStore(project_arrays_dir(root))
