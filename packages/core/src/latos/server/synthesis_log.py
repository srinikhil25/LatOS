"""Synthesis-log ingestion — the researcher's recipe, next to the raw files.

Instrument exports never record how a sample was *made*; that knowledge
lives in the researcher's lab notebook. The synthesis log closes the gap:
a small CSV dropped anywhere in the project folder, one row per sample,
one column per synthesis variable::

    sample, doping_pct, anneal_temp_c
    CS,     0,          450
    CS-1,   1,          450
    CS-3,   3,          450

At the end of every ingestion the log is applied to the project's
synthesis parameters (the optimization input space):

* Rows are matched to samples by **normalized name** — the same
  normalization the sample-labeling pipeline uses — so ``CS-1`` in the
  log matches a sample identified as ``cs_1`` or an alias.
* The log **wins** for the variables it names (it is the declared source
  of truth); variables entered only in the UI are preserved.
* Problems (unmatched rows, non-numeric cells) never abort the ingest —
  they are reported in the returned `LogReport` and logged.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from latos.ingestion.labeling.normalize import normalize
from latos.server import synthesis_store

if TYPE_CHECKING:
    from latos.core.models import Project

__all__ = ["LogReport", "apply_log", "find_log", "parse_log"]

_LOG = logging.getLogger("latos.synthesis_log")

# Recognized log filenames (case-insensitive), anywhere in the project
# tree outside `.latos/`.
_LOG_FILENAMES = frozenset({"synthesis.csv", "synthesis_log.csv"})


@dataclass(frozen=True, slots=True)
class LogReport:
    """What applying a synthesis log actually did.

    Attributes:
        path: The log file that was applied.
        variables: Variable names the log declared (header row).
        applied: Number of (sample, variable) values written.
        matched_samples: Sample names from the log that matched a project sample.
        unmatched_rows: Log row names with no matching sample — typos or
            samples whose files weren't in the folder.
        problems: Cell-level issues (non-numeric values), human-readable.
    """

    path: Path
    variables: tuple[str, ...]
    applied: int
    matched_samples: tuple[str, ...]
    unmatched_rows: tuple[str, ...]
    problems: tuple[str, ...]


def find_log(root: Path) -> Path | None:
    """The synthesis log in `root`'s tree, or None.

    Shallowest match wins (a log at the project root beats one buried in
    a technique subfolder); `.latos/` is never searched.
    """
    candidates = [
        p
        for p in root.rglob("*.csv")
        if p.name.lower() in _LOG_FILENAMES and ".latos" not in p.parts
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (len(p.parts), str(p).lower()))


def parse_log(path: Path) -> tuple[dict[str, dict[str, float]], tuple[str, ...], tuple[str, ...]]:
    """Parse a synthesis log into ``{row_name: {variable: value}}``.

    Returns (rows, variables, problems). Empty cells are simply "not
    given"; non-numeric cells are reported and skipped, never guessed.
    """
    min_columns = 2  # 'sample' plus at least one variable
    rows: dict[str, dict[str, float]] = {}
    problems: list[str] = []
    # utf-8-sig: Excel loves to prepend a BOM to CSV exports.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header or len(header) < min_columns:
            return {}, (), (f"{path.name}: header needs 'sample' plus at least one variable",)
        variables = [h.strip() for h in header[1:]]
        for lineno, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                continue
            name = row[0].strip()
            values: dict[str, float] = {}
            for var, cell in zip(variables, row[1:], strict=False):
                text = cell.strip()
                if not text:
                    continue
                try:
                    values[var] = float(text)
                except ValueError:
                    problems.append(
                        f"{path.name} line {lineno}: {var} = {text!r} is not a number; skipped"
                    )
            if values:
                rows[name] = values
    return rows, tuple(variables), tuple(problems)


def apply_log(root: Path, project: Project) -> LogReport | None:
    """Find, parse and apply the synthesis log for `project`. None if absent.

    Never raises on log content: problems are collected in the report and
    logged as warnings. The merge policy is per-variable: the log wins for
    the variables it names; anything else already stored is preserved.
    """
    path = find_log(root)
    if path is None:
        return None
    rows, variables, problems = parse_log(path)

    # Normalized name (canonical + aliases) -> sample id.
    name_to_id: dict[str, str] = {}
    for sample in project.samples:
        for candidate in (sample.canonical_name, *sample.aliases):
            key = normalize(candidate) or candidate.strip().lower()
            name_to_id.setdefault(key, sample.id)

    params = synthesis_store.load_params(root)
    applied = 0
    matched: list[str] = []
    unmatched: list[str] = []
    for row_name, values in rows.items():
        key = normalize(row_name) or row_name.strip().lower()
        sample_id = name_to_id.get(key)
        if sample_id is None:
            unmatched.append(row_name)
            continue
        matched.append(row_name)
        merged = {**params.get(sample_id, {}), **values}
        params[sample_id] = merged
        applied += len(values)
    synthesis_store.save_params(root, params)

    report = LogReport(
        path=path,
        variables=variables,
        applied=applied,
        matched_samples=tuple(matched),
        unmatched_rows=tuple(unmatched),
        problems=tuple(problems),
    )
    if unmatched:
        _LOG.warning(
            "synthesis log %s: %d row(s) matched no sample: %s",
            path.name, len(unmatched), ", ".join(unmatched),
        )
    for problem in problems:
        _LOG.warning("synthesis log: %s", problem)
    _LOG.info(
        "synthesis log %s applied: %d value(s) across %d sample(s)",
        path.name, applied, len(matched),
    )
    return report
