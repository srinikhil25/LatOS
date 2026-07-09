"""Merge *suggestions* for the review gate — suggest-only, never automatic.

Increment 1 (the orchestrator's canonical key) already collapses sample
names that are *identical after cleaning* (``CS Pure`` ≡ ``CS (Pure)``).
This module handles the next tier: names that are merely *similar* and
therefore require a human judgement —

- a researcher prefix:  ``Dr.MN-dhivya-cscbi1``  vs  ``CSCBI-1``
- a typo:               ``Dr.MN-dhivya-cskbi3``  vs  ``CSCBI-3``
- word reordering / extra labels.

We never merge these automatically. We surface a ranked list of candidate
pairs; the reviewer confirms or dismisses each one. Two guards keep the
list trustworthy:

- **Digit guard** — two names whose digit runs differ (``CS-1`` vs
  ``CS-3``, ``CSCBI-1`` vs ``CSCBI-3``) are *never* suggested. Digits
  carry doping / series meaning and are the single most reliable way to
  tell real samples apart, so a digit mismatch vetoes the pair outright.
  This is what stops the over-merge that BUG #16 warned about.
- **Length guard** — names shorter than `MIN_LEN` normalized characters
  (``CS``) don't anchor suggestions; they match too much to be useful.

Similarity combines four rapidfuzz metrics via ``max`` so a strong score
on *any* one counts: Levenshtein ratio (typos), token-sort (reordering),
Jaro-Winkler (shared prefixes), and partial-ratio (substring containment,
which is what catches a researcher prefix wrapping a real sample name).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from latos.ingestion.labeling.normalize import normalize

__all__ = ["MergeSuggestion", "SampleLike", "suggest_merges"]

# Minimum combined score (0–100) for a pair to be surfaced at all.
SUGGEST_THRESHOLD = 82.0
# At/above this score a suggestion is "high" confidence; below it "medium".
HIGH_CONFIDENCE = 92.0
# Normalized names shorter than this don't anchor suggestions on their own.
MIN_LEN = 3
# Absolute floor: a 1-character name never anchors, even via containment.
_MIN_ANCHOR_LEN = 2

_DIGIT_RUN_RE = re.compile(r"\d+")


class SampleLike(Protocol):
    """The shape `suggest_merges` needs from a sample (id + display name).

    Declared as read-only properties so a frozen dataclass (the real
    `Sample`) matches structurally.
    """

    @property
    def id(self) -> str:
        """Stable unique identifier."""

    @property
    def canonical_name(self) -> str:
        """Human-readable display name."""


@dataclass(frozen=True, slots=True)
class MergeSuggestion:
    """A proposed merge of sample ``source`` into sample ``target``.

    ``target`` is the cleaner spelling (shorter name) to keep; ``source``
    is the one that would fold into it. The reviewer makes the call.
    """

    target_id: str
    target_name: str
    source_id: str
    source_name: str
    score: float  # 0–100
    confidence: str  # "high" | "medium"
    reason: str


def _digit_runs(s: str) -> tuple[str, ...]:
    """Digit groups in order, e.g. ``"cscbi1"`` → ``("1",)``."""
    return tuple(_DIGIT_RUN_RE.findall(s))


def _pair_score(na: str, nb: str) -> float:
    """Combined similarity in 0–100 for two already-normalized names."""
    return max(
        fuzz.ratio(na, nb),
        fuzz.token_sort_ratio(na, nb),
        JaroWinkler.normalized_similarity(na, nb) * 100.0,
        fuzz.partial_ratio(na, nb),  # substring containment → prefix/suffix noise
    )


def _reason(na: str, nb: str, score: float) -> str:
    if na in nb or nb in na:
        return "one name contains the other (likely a researcher prefix or label)"
    return f"names are {score:.0f}% similar"


def suggest_merges(samples: Iterable[SampleLike]) -> list[MergeSuggestion]:
    """Return ranked merge suggestions for a project's samples.

    Only pairs that pass the digit + length guards and meet
    `SUGGEST_THRESHOLD` are returned, highest score first. The result is
    advisory: nothing is merged here.
    """
    items = [(s.id, s.canonical_name, normalize(s.canonical_name)) for s in samples]
    out: list[MergeSuggestion] = []

    for i in range(len(items)):
        ai, aname, na = items[i]
        for j in range(i + 1, len(items)):
            bj, bname, nb = items[j]
            if na == nb:
                continue  # identical-after-cleaning is already merged at ingestion
            if _digit_runs(na) != _digit_runs(nb):
                continue  # digit mismatch vetoes — doping/series discriminator

            # Length guard: short names match too much to anchor a suggestion
            # — UNLESS one fully contains the other (e.g. "CS" ⊂ "CS Pure"),
            # which is high-precision even for a 2-char name. Never below
            # _MIN_ANCHOR_LEN, and a clean substring with no digits is a
            # strong "same undoped sample" signal.
            contained = na in nb or nb in na
            if min(len(na), len(nb)) < _MIN_ANCHOR_LEN:
                continue
            if (len(na) < MIN_LEN or len(nb) < MIN_LEN) and not contained:
                continue

            score = _pair_score(na, nb)
            if score < SUGGEST_THRESHOLD:
                continue

            # Keep the cleaner (shorter, then alphabetical) name as the target.
            if (len(aname), aname) <= (len(bname), bname):
                target, source = (ai, aname), (bj, bname)
            else:
                target, source = (bj, bname), (ai, aname)

            out.append(
                MergeSuggestion(
                    target_id=target[0],
                    target_name=target[1],
                    source_id=source[0],
                    source_name=source[1],
                    score=round(score, 1),
                    confidence="high" if score >= HIGH_CONFIDENCE else "medium",
                    reason=_reason(na, nb, score),
                )
            )

    out.sort(key=lambda s: s.score, reverse=True)
    return out
