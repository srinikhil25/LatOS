"""`SampleHints` — every plausible sample-name candidate for one file.

Stage 2A's job is purely *gathering*: for a file, collect every string
that might be the sample name and tag each with a per-source confidence.
Normalization (2B) and clustering (2C) consume this output; they don't
re-read the file or re-parse anything.

Sources we mine (in descending baseline reliability):

| Source key             | Weight | Where it comes from                       |
|------------------------|--------|-------------------------------------------|
| `metadata_sample_name` | 1.00   | Parser-extracted "SampleName" header      |
| `metadata_sample_id`   | 1.00   | Parser-extracted XML `<id>` / equivalent  |
| `metadata_sample`      | 0.95   | Parser-extracted generic "sample" key     |
| `metadata_title`       | 0.95   | EMSA `#TITLE`, generic title fields       |
| `metadata_specimen`    | 0.95   | "specimen" header variant                 |
| `metadata_name`        | 0.85   | Generic "name" key (least specific)       |
| `filename_stem`        | 0.70   | File stem with extension stripped         |
| `excel_sheet`          | 0.65   | Sheet name (currently empty — gap noted)  |
| `path_segment_d0`      | 0.60   | Immediate parent folder (non-generic)     |
| `path_segment_dN`      | 0.30-0.50 | Higher parents (decay 0.1 per level)   |
| `path_segment_generic` | 0.20   | Folder matched our generic-name allowlist |

Confidence is informational — Stage 2C uses it to weight similarity
edges, not to filter hints. We surface every candidate; the cluster
phase decides which to trust.

Why not re-read the file
------------------------
The orchestrator already runs every parser and persists `parsed.metadata`.
Re-opening files at the labeling stage would double the IO cost on the
critical path. Stage 2A reads only `parsed_data.metadata` and the
file's `Path` — strictly cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latos.ingestion.parsed_data import ParsedData

__all__ = [
    "SampleHints",
    "extract_hints",
]


# Metadata keys we accept as sample-name candidates, in priority order.
# A parser may surface any subset; we look up each key in turn and emit
# at most one hint per key. Values are casefolded only for filename
# normalization later — we keep the raw string here so the UI can
# preserve user-facing capitalization.
#
# Keep the (key, source-tag, weight) triples co-located so adding a new
# parser-side metadata convention is a one-line edit.
_METADATA_SOURCES: tuple[tuple[str, str, float], ...] = (
    ("sample_name", "metadata_sample_name", 1.00),
    ("sample_id", "metadata_sample_id", 1.00),
    ("sample", "metadata_sample", 0.95),
    ("title", "metadata_title", 0.95),
    ("specimen", "metadata_specimen", 0.95),
    ("name", "metadata_name", 0.85),
)

# Generic folder names that *aren't* a sample identifier. Mirror of the
# orchestrator's set so the two stay in sync until Stage 2 fully
# replaces the orchestrator-side inference.
_GENERIC_FOLDER_NAMES: frozenset[str] = frozenset(
    {
        "data",
        "raw",
        "raw data",
        "rawdata",
        "characterization",
        "characterizations",
        "results",
        "output",
        "outputs",
        "samples",
        "measurements",
        "scans",
        "images",
        "image",
        "files",
        # Technique-name folders: the file is *of* this technique, not
        # named *by* this technique.
        "xrd",
        "x-ray",
        "xrd data",
        "xrd patterns",
        "xps",
        "xps data",
        "uv-drs",
        "uv drs",
        "uvdrs",
        "uv-vis",
        "hall",
        "hall data",
        "hall measurement",
        "hall measurements",
        "thermoelectric",
        "te",
        "zt calculation",
        "zt",
        "eds",
        "edx",
        "tem",
        "sem",
        "stem",
        "raman",
        "microscopy",
    }
)

# Path-segment confidence by walk depth. Index 0 is the file's immediate
# parent. We decay by 0.1 per level out to 4 levels; deeper segments
# share the floor weight. Generic segments collapse to a single low
# weight regardless of depth.
_PATH_DEPTH_WEIGHTS: tuple[float, ...] = (0.60, 0.50, 0.40, 0.30)
_PATH_FLOOR_WEIGHT = 0.30
_PATH_GENERIC_WEIGHT = 0.20

# Cap on how many parent segments we walk. Beyond this we leave the
# rest of the filesystem out of consideration even if the project root
# is unknown.
_MAX_PATH_DEPTH = 6

# Filename suffix patterns we strip when emitting a filename-stem hint
# so things like `MX-001_run5` and `MX-001 (1)` don't proliferate as
# distinct candidates. The cluster phase still gets the *original* stem
# alongside the cleaned one — neither is canonical until clustering says
# so. These are conservative: they only strip trailing tokens that are
# plainly not part of a sample name.
_FILENAME_TRAILING_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Run-with-keyword first (`_run5`, `-scan42`, `-run-7`). Has to run
    # before the bare-digit pattern below — otherwise the digit-only
    # rule eats the trailing `-7`, leaving an unanchorable `-run`.
    re.compile(r"[ _\-]+(?:run|scan|rep|trial)[ _\-]?\d+\s*$", re.IGNORECASE),
    # Trailing duplicate marker like `Foo (1)`.
    re.compile(r"\s*\(\d+\)\s*$"),
    # Bare numeric suffix preceded by SPACE or UNDERSCORE only — not
    # by a dash. `CS-1` is a sample identifier; we must not mangle it
    # into `CS`. `CS-1_5` is sample `CS-1` plus run 5; we strip `_5`.
    re.compile(r"[ _]+\d{1,3}\s*$"),
)


@dataclass(frozen=True, slots=True)
class SampleHints:
    """Every plausible sample-name candidate for a single file.

    Attributes:
        file_path: Absolute path the hints were derived from.
        from_path_segments: Parent folder names from immediate-parent
            outward, capped at `_MAX_PATH_DEPTH`. Empty if the file is
            at the root.
        from_filename: File stem with extension stripped. May be `None`
            if the file has no stem.
        from_filename_cleaned: `from_filename` with trailing run/index
            suffixes (`_5`, `(1)`, `_run3`) stripped. `None` if cleaning
            would produce an empty string.
        from_file_metadata: `{source_tag: value}` strings extracted from
            `parsed_data.metadata`. Empty dict if no parser data given
            or no recognizable keys present.
        from_file_content: First-line "Sample: X" extracted from the
            raw file. `None` until a future iteration wires in
            content-side hints (kept as a slot so `cluster.py` doesn't
            need to grow another attribute when we ship it).
        from_excel_sheet: Sheet name a workbook parser actually parsed.
            `None` until the Excel parsers expose this in metadata.
        confidence_per_source: Each source key (above) → confidence in
            [0, 1]. Iteration order matches the order hints were added.
    """

    file_path: Path
    from_path_segments: tuple[str, ...] = ()
    from_filename: str | None = None
    from_filename_cleaned: str | None = None
    from_file_metadata: dict[str, str] = field(default_factory=dict)
    from_file_content: str | None = None
    from_excel_sheet: str | None = None
    confidence_per_source: dict[str, float] = field(default_factory=dict)

    def candidates(self) -> tuple[tuple[str, str, float], ...]:
        """Flatten every hint into (source_tag, value, confidence) tuples.

        Convenience for the cluster phase, which doesn't care which
        attribute a candidate came from — only the source tag (so it
        can apply chemistry-aware boosters per source) and the weight.
        """
        out: list[tuple[str, str, float]] = []

        for tag, value in self.from_file_metadata.items():
            conf = self.confidence_per_source.get(tag, 0.0)
            out.append((tag, value, conf))

        if self.from_filename is not None:
            out.append(
                (
                    "filename_stem",
                    self.from_filename,
                    self.confidence_per_source.get("filename_stem", 0.0),
                )
            )
        if (
            self.from_filename_cleaned is not None
            and self.from_filename_cleaned != self.from_filename
        ):
            out.append(
                (
                    "filename_stem_cleaned",
                    self.from_filename_cleaned,
                    self.confidence_per_source.get("filename_stem_cleaned", 0.0),
                )
            )

        if self.from_excel_sheet is not None:
            out.append(
                (
                    "excel_sheet",
                    self.from_excel_sheet,
                    self.confidence_per_source.get("excel_sheet", 0.0),
                )
            )

        if self.from_file_content is not None:
            out.append(
                (
                    "file_content",
                    self.from_file_content,
                    self.confidence_per_source.get("file_content", 0.0),
                )
            )

        for depth, segment in enumerate(self.from_path_segments):
            tag = "path_segment_generic" if _is_generic(segment) else f"path_segment_d{depth}"
            conf = self.confidence_per_source.get(tag, 0.0)
            out.append((tag, segment, conf))

        return tuple(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_hints(
    file_path: Path,
    *,
    parsed_data: ParsedData | None = None,
    root: Path | None = None,
) -> SampleHints:
    """Return a `SampleHints` for `file_path`.

    Args:
        file_path: Absolute path to the file. Required for path-segment
            and filename hints.
        parsed_data: The `ParsedData` returned by the file's parser, if
            available. When `None`, only path/filename hints are emitted
            (useful for files that failed to parse but still benefit
            from being clustered against parsed siblings).
        root: Project root. When set, path-segment walking stops at the
            root rather than the filesystem root, so `D:/projects/MyMX/`
            doesn't contribute `projects` and `MyMX` as sample hints.

    Returns:
        A fully populated `SampleHints`. Callers should treat each hint
        attribute as advisory; the cluster phase decides what wins.
    """
    confidences: dict[str, float] = {}

    # ─── Filename stem(s) ──────────────────────────────────────────────
    filename: str | None = file_path.stem or None
    filename_cleaned: str | None = None
    if filename:
        confidences["filename_stem"] = 0.70
        cleaned = _clean_filename(filename)
        if cleaned and cleaned != filename:
            filename_cleaned = cleaned
            confidences["filename_stem_cleaned"] = 0.70

    # ─── Path segments ─────────────────────────────────────────────────
    segments = _walk_parents(file_path, root)
    for depth, segment in enumerate(segments):
        if _is_generic(segment):
            confidences["path_segment_generic"] = _PATH_GENERIC_WEIGHT
        else:
            tag = f"path_segment_d{depth}"
            confidences[tag] = _path_weight_for_depth(depth)

    # ─── Metadata ──────────────────────────────────────────────────────
    metadata_hints: dict[str, str] = {}
    if parsed_data is not None:
        for raw_key, source_tag, weight in _METADATA_SOURCES:
            value = parsed_data.metadata.get(raw_key)
            if isinstance(value, str) and value.strip():
                metadata_hints[source_tag] = value.strip()
                confidences[source_tag] = weight

    return SampleHints(
        file_path=file_path,
        from_path_segments=tuple(segments),
        from_filename=filename,
        from_filename_cleaned=filename_cleaned,
        from_file_metadata=metadata_hints,
        from_file_content=None,
        from_excel_sheet=None,
        confidence_per_source=confidences,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _walk_parents(file_path: Path, root: Path | None) -> list[str]:
    """Collect parent folder names from immediate parent outward.

    Stops at `root` (exclusive) when given, or `_MAX_PATH_DEPTH` levels
    out otherwise. The filesystem root (`p.parent == p`) always stops
    the walk so we never emit `C:` / `/` as a hint.

    We compare the walk path to `root` *as given* — no `resolve()`. On
    Windows, resolving an unrooted-looking path like `/proj/CS-1` adds
    a drive letter (`D:/proj/CS-1`) that the walk doesn't, so the
    equality check would never fire. Tests rely on cheap string-level
    comparison, and the orchestrator only ever passes absolute paths.
    """
    out: list[str] = []
    p = file_path.parent

    for _ in range(_MAX_PATH_DEPTH):
        if p == p.parent:
            # Hit the filesystem root — `Path('/').parent == Path('/')`
            break
        if root is not None and p == root:
            break
        if p.name:
            out.append(p.name)
        p = p.parent

    return out


def _path_weight_for_depth(depth: int) -> float:
    """Weight for a non-generic path segment at given depth (0 = parent)."""
    if depth < len(_PATH_DEPTH_WEIGHTS):
        return _PATH_DEPTH_WEIGHTS[depth]
    return _PATH_FLOOR_WEIGHT


def _is_generic(name: str) -> bool:
    """True if `name` matches our generic folder/technique allowlist."""
    return name.strip().lower() in _GENERIC_FOLDER_NAMES


def _clean_filename(stem: str) -> str:
    """Strip trailing run/scan/index suffixes from a filename stem."""
    cleaned = stem
    # Apply each pattern in turn; one pass is fine because the patterns
    # only match trailing tokens.
    for pattern in _FILENAME_TRAILING_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()
