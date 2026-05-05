"""`crawl()` — folder walker that hashes and classifies every file.

Pure-data layer between the filesystem and the orchestrator (Stage 1D.2).
The crawler:

1. Walks `root` recursively, skipping system files and Latos-internal
   directories.
2. For each surviving file, computes the SHA-256 (cached by mtime/size).
3. Asks the `ParserRegistry` which parser would handle the file.
4. Records everything in a `CrawlReport` of frozen `CrawlEntry`s.

The crawler does NOT call `parser.parse()` — only `can_parse()`. Actual
parsing happens in the orchestrator. This separation keeps the crawler
fast (sniffing reads only file headers) and side-effect-free.

Why two passes
--------------
We walk the tree twice. The first pass collects every path that survives
the skip filter so we know the total file count up-front. The second
pass does the expensive work (hash + classify) and reports progress
against that known total. The first pass costs only `os.scandir()` calls
— microseconds per file — so the cost is negligible compared to the
benefit of accurate progress reporting in the UI.

What's deferred
---------------
- Concurrency / parallel hashing: single-threaded is plenty for the
  ~1k-file folders we target. Profile in 1D.3 if it becomes a problem.
- Symlink following: not followed; loops are too easy to create
  accidentally and the predictability of "what you see in the report
  is what you have on disk" is worth more than the flexibility.
- On-disk hash cache: the in-memory `HashCache` survives one crawl run.
  Cross-run persistence is an orchestrator-level concern (Stage 1D.2+).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from latos.ingestion.hashing import HashCache, hash_file
from latos.ingestion.registry import MIN_CONFIDENCE, ParserRegistry

__all__ = [
    "CrawlEntry",
    "CrawlReport",
    "ProgressCallback",
    "crawl",
]


# ─── Skip rules ─────────────────────────────────────────────────────
# Directory names skipped during the walk. `.latos` is critical — it
# holds Latos's own SQLite DB and Parquet files, and recursing into it
# would cause the crawler to ingest its own outputs.
_DEFAULT_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".latos",
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        "node_modules",
        ".idea",
        ".vscode",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "venv",
        ".venv",
        "env",
        ".env",
    },
)

# Specific filenames always skipped. Operating-system metadata, Office
# lockfiles, etc. — never user data.
_DEFAULT_SKIP_FILES: frozenset[str] = frozenset(
    {
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        "ehthumbs.db",
    },
)

# Filename glob patterns (matched via Path.match). Excel auto-creates
# `~$workbook.xlsx` lockfiles when a file is open; ingesting one would
# fail and clutter the report.
_DEFAULT_SKIP_PATTERNS: tuple[str, ...] = (
    "~$*",  # MS Office lockfiles (Word, Excel, PowerPoint)
    "*.tmp",
    "*.swp",  # vim swap files
)


# ─── Types ──────────────────────────────────────────────────────────
ProgressCallback = Callable[[int, int, Path], None]
"""Signature: `on_progress(index, total, current_path)` called before each file.

`index` is 0-based and runs from 0 to `total-1`. The callback may
not raise; the crawler doesn't catch exceptions from it.
"""


@dataclass(frozen=True, slots=True)
class CrawlEntry:
    """One file's record in a `CrawlReport`.

    The crawler emits one of these per surviving file (those passing
    skip filters). Files that errored during hashing or classification
    are STILL recorded — `error` carries the human-readable reason and
    the orchestrator decides whether to surface it or skip silently.

    Attributes:
        path: Absolute path to the file.
        relative_path: Path relative to the crawl root (for display).
        size_bytes: From the file's stat at the time of crawling.
        mtime: Last-modified time, tz-aware UTC.
        sha256: 64-char hex SHA-256, or None if hashing failed.
        classified_parser: `parser.name` if confidence >= threshold, else None.
            Files where this is None are "unknown technique" and won't
            be parsed by the orchestrator.
        best_match_parser: `parser.name` of the highest-scoring parser,
            even when below threshold. Useful for diagnostics — lets a
            researcher see "best guess was rigaku-xrd-asc at 0.4" rather
            than just "no match".
        confidence: Highest score returned by any parser's can_parse(),
            or 0.0 if no parser claimed the file.
        error: Human-readable error string if hashing failed or sniffing
            raised. None on success. The entry is still emitted so the
            UI can show "this file had a problem" rather than silently
            dropping it.
    """

    path: Path
    relative_path: Path
    size_bytes: int
    mtime: datetime
    sha256: str | None
    classified_parser: str | None
    best_match_parser: str | None
    confidence: float
    error: str | None


@dataclass(frozen=True, slots=True)
class CrawlReport:
    """The complete output of one `crawl()` call.

    `entries` is in walk order (typically alphabetical per directory,
    then by directory). Use the `classified` / `unclassified` / `errored`
    properties for filtered views — they're computed lazily on access.
    """

    root: Path
    entries: tuple[CrawlEntry, ...]

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def classified(self) -> tuple[CrawlEntry, ...]:
        """Entries with a parser confident enough to handle them."""
        return tuple(e for e in self.entries if e.classified_parser is not None)

    @property
    def unclassified(self) -> tuple[CrawlEntry, ...]:
        """Entries no parser claimed at >= threshold confidence."""
        return tuple(e for e in self.entries if e.classified_parser is None and e.error is None)

    @property
    def errored(self) -> tuple[CrawlEntry, ...]:
        """Entries where hashing or classification failed."""
        return tuple(e for e in self.entries if e.error is not None)


# ─── Public API ─────────────────────────────────────────────────────
def crawl(
    root: Path,
    registry: ParserRegistry,
    *,
    hash_cache: HashCache | None = None,
    confidence_threshold: float = MIN_CONFIDENCE,
    skip_dirs: frozenset[str] = _DEFAULT_SKIP_DIRS,
    skip_files: frozenset[str] = _DEFAULT_SKIP_FILES,
    on_progress: ProgressCallback | None = None,
) -> CrawlReport:
    """Walk `root`, hash + classify every file, return a `CrawlReport`.

    Args:
        root: Folder to walk. Must exist and be a directory.
        registry: ParserRegistry to use for classification.
        hash_cache: Optional cache; reused across calls in the same run
            so unchanged files aren't re-hashed.
        confidence_threshold: Minimum `can_parse()` score for a file to
            be marked `classified`. Defaults to `MIN_CONFIDENCE`.
        skip_dirs: Directory names skipped during the walk.
        skip_files: Filenames skipped (exact match).
        on_progress: Optional callback fired before each file is processed.

    Returns:
        A `CrawlReport` whose `entries` are in walk order.

    Raises:
        FileNotFoundError: if `root` doesn't exist.
        NotADirectoryError: if `root` is not a directory.
    """
    if not root.exists():
        raise FileNotFoundError(f"Crawl root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Crawl root is not a directory: {root}")

    # Pass 1: enumerate every surviving path so we know the total upfront.
    paths = list(_walk(root, skip_dirs=skip_dirs, skip_files=skip_files))
    total = len(paths)

    # Pass 2: hash + classify each file.
    entries: list[CrawlEntry] = []
    for i, path in enumerate(paths):
        if on_progress is not None:
            on_progress(i, total, path)
        entry = _process_file(
            path=path,
            root=root,
            registry=registry,
            hash_cache=hash_cache,
            threshold=confidence_threshold,
        )
        entries.append(entry)

    return CrawlReport(root=root, entries=tuple(entries))


# ─── Internals ──────────────────────────────────────────────────────
def _walk(
    root: Path,
    *,
    skip_dirs: frozenset[str],
    skip_files: frozenset[str],
) -> Iterator[Path]:
    """Yield every file under `root` that survives the skip filters.

    Uses `os.walk(followlinks=False)` so symlinks are NOT traversed.
    Hidden files/directories (those whose name starts with `.`) and
    anything in `skip_dirs` / `skip_files` are filtered out before the
    descent continues.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Filter dirnames IN PLACE — os.walk uses the live list to decide
        # which directories to descend into next.
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for fname in filenames:
            if fname in skip_files:
                continue
            if fname.startswith("."):
                continue
            if any(_match_pattern(fname, pat) for pat in _DEFAULT_SKIP_PATTERNS):
                continue
            yield Path(dirpath) / fname


def _match_pattern(name: str, pattern: str) -> bool:
    """Lightweight glob match supporting only `*` (matches any chars).

    We deliberately avoid `fnmatch` here because the patterns we use
    (`~$*`, `*.tmp`, `*.swp`) don't need full POSIX semantics, and the
    minimal version is faster on the per-file hot path.
    """
    if "*" not in pattern:
        return name == pattern
    if pattern.startswith("*") and not pattern[1:].startswith("*"):
        return name.endswith(pattern[1:])
    if pattern.endswith("*") and not pattern[:-1].endswith("*"):
        return name.startswith(pattern[:-1])
    # Pattern with `*` in the middle — fall back to fnmatch.
    import fnmatch  # noqa: PLC0415 — cheap-import, only on rare patterns

    return fnmatch.fnmatchcase(name, pattern)


def _process_file(
    *,
    path: Path,
    root: Path,
    registry: ParserRegistry,
    hash_cache: HashCache | None,
    threshold: float,
) -> CrawlEntry:
    """Hash + classify one file, building its `CrawlEntry`.

    Errors at any step are captured into `entry.error`; the entry is
    always returned (never raised). This is what lets the orchestrator
    show partial results in the UI even when some files in the folder
    are unreadable.
    """
    # Stat once — used for size, mtime, AND error detection.
    try:
        stat = path.stat()
    except OSError as exc:
        return CrawlEntry(
            path=path,
            relative_path=_safe_relative(path, root),
            size_bytes=0,
            mtime=datetime.now(UTC),
            sha256=None,
            classified_parser=None,
            best_match_parser=None,
            confidence=0.0,
            error=f"stat failed: {exc}",
        )

    size_bytes = stat.st_size
    # `fromtimestamp(..., tz=UTC)` keeps us tz-aware end-to-end.
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

    # Hash — may fail on permission errors or disappearing files.
    sha256: str | None = None
    error: str | None = None
    try:
        sha256 = hash_file(path, cache=hash_cache)
    except OSError as exc:
        error = f"hash failed: {exc}"

    # Classify. We ask the registry with min_confidence=0.0 so we get the
    # best match even when below threshold (for diagnostics), then split
    # into `classified_parser` (>= threshold) vs. `best_match_parser`
    # (any non-zero score).
    classified_parser: str | None = None
    best_match_parser: str | None = None
    confidence = 0.0
    if error is None:
        match = registry.find_parser(path, min_confidence=0.0)
        if match is not None:
            best_match_parser = match.parser.name
            confidence = match.confidence
            if confidence >= threshold:
                classified_parser = match.parser.name

    return CrawlEntry(
        path=path,
        relative_path=_safe_relative(path, root),
        size_bytes=size_bytes,
        mtime=mtime,
        sha256=sha256,
        classified_parser=classified_parser,
        best_match_parser=best_match_parser,
        confidence=confidence,
        error=error,
    )


def _safe_relative(path: Path, root: Path) -> Path:
    """Path.relative_to but never raises — falls back to the absolute path.

    Defensive: a symlink that resolved outside `root` would normally
    raise `ValueError` from `relative_to`. We don't follow symlinks, but
    the cost of being safe here is one branch.
    """
    try:
        return path.relative_to(root)
    except ValueError:
        return path
