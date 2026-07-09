"""Flag samples that probably aren't real samples — for human review.

Two cheap, transparent detectors (no ML, no guessing intent):

- ``mixed_samples`` — the sample's *files* are named after OTHER samples
  in the project. A folder ``Divyamahalakshmi_07042025`` holding
  ``CS-1.xls``, ``CS-3.xls``, ``CS-5.xls`` is almost certainly a
  measurement-session bucket whose files belong to the real CS-1 / CS-3
  / CS-5 samples. This is strong, content-based evidence — it matches
  filenames against the project's own sample names, it doesn't guess.

- ``non_sample_name`` — the sample's *name* looks like something other
  than a specimen: a date (``...07042025``) or a generic catch-all
  folder word (``Images``, ``Data``, ``Results``).

Both are flags only. Latos never reassigns, splits, or deletes
automatically — the reviewer fixes them with the existing rename /
move / split tools. Researcher-prefixed duplicates (``Dr.MN-...``) are
deliberately left to the merge-suggestions pass so a sample is never
flagged twice for the same underlying issue.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from latos.ingestion.labeling.normalize import normalize

if TYPE_CHECKING:
    from latos.core.models import Sample

__all__ = ["SampleAnomaly", "flag_anomalies"]

# A run of 6+ digits reads as a date/batch stamp, not a sample number
# (``CS-1`` has one digit; ``...07042025`` has eight).
_DATE_RUN_RE = re.compile(r"\d{6,}")

# Catch-all folder names that are never a specimen. Stored in normalized
# form (separators stripped) to match `normalize()` output.
_GENERIC_NONSAMPLE: frozenset[str] = frozenset(
    {
        "images",
        "image",
        "data",
        "rawdata",
        "results",
        "output",
        "outputs",
        "misc",
        "miscellaneous",
        "other",
        "others",
        "untitled",
        "newfolder",
        "test",
        "temp",
        "tmp",
    }
)

# A sample's files must point at at least this many *other* samples
# before we call it a mixed session bucket — one match could be a naming
# coincidence; two or more is a pattern.
_MIN_MIXED_TARGETS = 2

# How many related sample names to show inline before adding an ellipsis.
_MIXED_PREVIEW_LIMIT = 3


@dataclass(frozen=True, slots=True)
class SampleAnomaly:
    """One flagged sample needing human attention.

    Attributes:
        sample_id: The flagged sample.
        sample_name: Its display name.
        kind: ``"mixed_samples"`` or ``"non_sample_name"``.
        message: Plain-language explanation for the reviewer.
        related: For ``mixed_samples``, the other sample names this
            folder's files appear to belong to. Empty otherwise.
    """

    sample_id: str
    sample_name: str
    kind: str
    message: str
    related: tuple[str, ...] = ()


def _key(name: str) -> str:
    return normalize(name) or name.strip().lower()


def _file_labels(sample: Sample) -> set[str]:
    """Normalized stems of every file referenced by this sample."""
    labels: set[str] = set()
    for m in sample.measurements:
        for f in m.files:
            stem = Path(f.path).stem
            key = _key(stem)
            if key:
                labels.add(key)
    return labels


def _is_non_sample_name(name: str) -> bool:
    key = _key(name)
    return key in _GENERIC_NONSAMPLE or bool(_DATE_RUN_RE.search(key))


def flag_anomalies(samples: Iterable[Sample]) -> list[SampleAnomaly]:
    """Return one anomaly per suspicious sample (mixed wins over name).

    A sample flagged ``mixed_samples`` is not also flagged for its name,
    so the reviewer sees a single, most-actionable reason per sample.
    """
    sample_list = list(samples)
    key_to_name = {_key(s.canonical_name): s.canonical_name for s in sample_list}

    out: list[SampleAnomaly] = []
    for s in sample_list:
        own_key = _key(s.canonical_name)
        matched = sorted(
            {
                key_to_name[label]
                for label in _file_labels(s)
                if label in key_to_name and label != own_key
            }
        )
        if len(matched) >= _MIN_MIXED_TARGETS:
            preview = ", ".join(matched[:_MIXED_PREVIEW_LIMIT]) + (
                "…" if len(matched) > _MIXED_PREVIEW_LIMIT else ""
            )
            out.append(
                SampleAnomaly(
                    sample_id=s.id,
                    sample_name=s.canonical_name,
                    kind="mixed_samples",
                    message=(
                        f"This folder's files are named after other samples "
                        f"({preview}). It looks like a measurement session — "
                        f"consider moving each file to its real sample."
                    ),
                    related=tuple(matched),
                )
            )
            continue

        if _is_non_sample_name(s.canonical_name):
            out.append(
                SampleAnomaly(
                    sample_id=s.id,
                    sample_name=s.canonical_name,
                    kind="non_sample_name",
                    message=(
                        "This name looks like a folder or date rather than a "
                        "sample. Rename it, or move its files to the right sample."
                    ),
                )
            )

    return out
