"""Manual end-to-end ingestion smoke for any folder.

A small CLI wrapper around `Orchestrator.ingest()` that produces a
human-readable summary, so a user can sanity-check what Latos would do
with their data BEFORE Stage 1E ships the proper UI.

Usage::

    python scripts/ingest.py <folder>
    python scripts/ingest.py D:/Materials-Informatics/data_raw/dhivya_data
    python scripts/ingest.py <folder> --name "MyProject"

The script:

1. Copies the folder to a tmp project root by default (so `.latos/` lives
   in tmp, not next to your real data). Pass `--in-place` to instead
   write `.latos/` directly under `<folder>`.
2. Runs the full ingestion pipeline (crawler + parsers + persistence).
3. Prints a one-page summary: timings, per-outcome counts, per-technique
   breakdown, per-sample listing, and the first N unclassified files.

Exit code is 0 unless ingestion raises (which it shouldn't — parser
crashes are caught and reported as `PARSE_FAILED` outcomes).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from latos.ingestion.orchestrator import IngestionResult, Orchestrator, Outcome
from latos.ingestion.registry import default_registry

# Cap how many unclassified files we list inline. Anything more clutters
# the summary; users who want the full list can iterate `result.outcomes`
# in their own code.
_MAX_UNCLASSIFIED_TO_SHOW = 15

# Force UTF-8 stdout on Windows so box-drawing characters and Unicode
# arrows in summaries don't crash on cp1252 consoles.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def _format_summary(result: IngestionResult, elapsed: float) -> str:
    """Build the human-readable summary block from an `IngestionResult`."""
    lines: list[str] = []
    bar = "─" * 70

    lines.append(bar)
    lines.append(f"  Latos ingestion: {result.project.name}")
    lines.append(f"  Root: {result.project.root_path}")
    lines.append(bar)

    # Timings + counts.
    lines.append(f"\n  Total files seen:      {len(result.outcomes)}")
    lines.append(f"  Wall-clock time:       {elapsed:.2f}s")
    lines.append("")

    counts = Counter(o.outcome for o in result.outcomes)
    lines.append("  Outcomes:")
    for outcome in Outcome:
        n = counts.get(outcome, 0)
        if n > 0:
            lines.append(f"    {outcome.value:<24} {n}")

    # Per-technique breakdown.
    technique_counts = Counter(
        m.technique.value for s in result.project.samples for m in s.measurements
    )
    if technique_counts:
        lines.append("")
        lines.append("  Measurements by technique:")
        for tech, n in sorted(technique_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {tech:<24} {n}")

    # Sample list.
    lines.append("")
    lines.append(f"  Samples ({len(result.project.samples)}):")
    for sample in sorted(result.project.samples, key=lambda s: s.canonical_name):
        techs = Counter(m.technique.value for m in sample.measurements)
        tech_str = ", ".join(f"{t}={n}" for t, n in sorted(techs.items()))
        lines.append(
            f"    {sample.canonical_name!r:<35} "
            f"{len(sample.measurements):>2} measurement(s)  [{tech_str}]"
        )

    # Unclassified files (first N — anything more clutters the summary).
    unclassified = [o for o in result.outcomes if o.outcome == Outcome.SKIPPED_UNCLASSIFIED]
    if unclassified:
        lines.append("")
        lines.append(
            f"  Unclassified files ({len(unclassified)} "
            f"— first {_MAX_UNCLASSIFIED_TO_SHOW} shown):",
        )
        for o in unclassified[:_MAX_UNCLASSIFIED_TO_SHOW]:
            best = f" (best-guess: {o.parser_name})" if o.parser_name else ""
            lines.append(f"    {o.relative_path}{best}")
        if len(unclassified) > _MAX_UNCLASSIFIED_TO_SHOW:
            lines.append(f"    ... and {len(unclassified) - _MAX_UNCLASSIFIED_TO_SHOW} more")

    # Failures.
    failed = [o for o in result.outcomes if o.outcome == Outcome.PARSE_FAILED]
    if failed:
        lines.append("")
        lines.append(f"  PARSE FAILURES ({len(failed)}):")
        for o in failed:
            lines.append(f"    {o.relative_path}")
            lines.append(f"      {o.error}")

    lines.append("")
    lines.append(bar)
    return "\n".join(lines)


def _run(folder: Path, project_name: str | None, in_place: bool) -> int:
    """Execute one ingestion run and print its summary."""
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory.", file=sys.stderr)
        return 1

    # If --in-place, ingest the user's actual folder. Otherwise copy it
    # to a temp location so we don't drop a `.latos/` folder next to
    # their data unless they asked for it.
    if in_place:
        return _ingest_at(folder, project_name)

    with tempfile.TemporaryDirectory(prefix="latos_ingest_") as tmp:
        proj = Path(tmp) / folder.name
        print(f"Copying {folder} → {proj} ...")
        t0 = time.perf_counter()
        shutil.copytree(folder, proj)
        print(f"  copy took {time.perf_counter() - t0:.1f}s\n")
        return _ingest_at(proj, project_name)


def _ingest_at(root: Path, project_name: str | None) -> int:
    """Run the orchestrator on `root` and print the summary."""
    orchestrator = Orchestrator(registry=default_registry())

    t0 = time.perf_counter()
    try:
        result = orchestrator.ingest(root, project_name=project_name)
    except Exception as exc:
        print(f"\nIngestion raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - t0

    print(_format_summary(result, elapsed))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a folder of instrument files and print a summary.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder of raw instrument data to ingest.",
    )
    parser.add_argument(
        "--name",
        dest="project_name",
        default=None,
        help="Display name for the project (defaults to the folder's basename).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Write `.latos/` directly under <folder>. Default is to copy the "
            "folder to a temp directory and ingest there, leaving your data "
            "untouched."
        ),
    )
    args = parser.parse_args()
    return _run(args.folder, args.project_name, args.in_place)


if __name__ == "__main__":
    sys.exit(main())
