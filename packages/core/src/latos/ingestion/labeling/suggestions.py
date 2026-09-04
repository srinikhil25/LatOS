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

# Word guard (below). `normalize` strips every separator, so
# "MX_Ti3C2Tx_Air_50" and "MX_Ti3C2Tx_Ar_50" arrive as
# "mxti3c2txair50" / "mxti3c2txar50" — a one-character difference in a
# twenty-character string, which scores 99%. The words have to be read off the
# ORIGINAL name, before normalization destroys them.
_WORD_RE = re.compile(r"[A-Za-z]+")

# A word that names a measurement TECHNIQUE describes how a specimen was looked
# at, not which specimen it is, so a pair differing only in these is still the
# same sample and stays mergeable. Includes the lab's own abbreviations, which
# the `Technique` enum does not carry: ESCA for XPS, RS for Raman, EDX for EDS.
_TECHNIQUE_WORDS = frozenset(
    {
        "xrd",
        "xps",
        "esca",
        "raman",
        "rs",
        "eds",
        "edx",
        "tem",
        "sem",
        "stem",
        "hall",
        "ppms",
        "uv",
        "drs",
        "image",
        "images",
        "map",
        "mapping",
        "deg",
        "spectrum",
        "spectra",
        "scan",
    }
)

# How many distinct samples a word must appear in before it counts as this
# project's controlled vocabulary rather than a one-off typo. Two is enough:
# a misspelling is written once, a factor level is written every time it is used.
_VOCAB_MIN_SAMPLES = 2


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


def _words(name: str) -> frozenset[str]:
    """Lower-cased alphabetic words of an ORIGINAL (un-normalized) name.

    ``"MX_Ti3C2Tx_Air_50_CAF_ESCA"`` → ``{mx, ti, c, tx, air, caf, esca}``.
    """
    return frozenset(w.lower() for w in _WORD_RE.findall(name))


def _substituted_words(
    a: str, b: str, vocabulary: dict[str, int]
) -> tuple[frozenset[str], frozenset[str]]:
    """Non-technique words each name has that the other lacks.

    Both sides non-empty means a word was SUBSTITUTED rather than merely added,
    which is the difference between "Air 50" vs "Ar 50" (two conditions) and
    "Dr.MN-dhivya-cscbi1" vs "CSCBI-1" (a prefix on one side only).
    """
    wa, wb = _words(a), _words(b)
    return (
        frozenset(w for w in wa - wb if w not in _TECHNIQUE_WORDS),
        frozenset(w for w in wb - wa if w not in _TECHNIQUE_WORDS),
    )


def _word_veto(a: str, b: str, vocabulary: dict[str, int]) -> bool:
    """True when the pair differs by an established word on BOTH sides.

    The alphabetic counterpart of the digit guard. A word appearing across
    several samples is this project's vocabulary — a gas, a substrate, a
    condition — and swapping one for another names a different specimen. A word
    appearing once is a typo, which is exactly what a suggestion should catch,
    so those pairs are still offered.
    """
    only_a, only_b = _substituted_words(a, b, vocabulary)
    if not (only_a and only_b):
        return False  # pure addition (a prefix or a label), not a substitution
    return all(vocabulary.get(w, 0) >= _VOCAB_MIN_SAMPLES for w in only_a | only_b)


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

    # How many samples each word appears in, which is what separates this
    # project's vocabulary from a one-off misspelling. Built once over the
    # original names, since `normalize` removes the separators words need.
    vocabulary: dict[str, int] = {}
    for _, name, _ in items:
        for word in _words(name):
            vocabulary[word] = vocabulary.get(word, 0) + 1

    for i in range(len(items)):
        ai, aname, na = items[i]
        for j in range(i + 1, len(items)):
            bj, bname, nb = items[j]
            if na == nb:
                continue  # identical-after-cleaning is already merged at ingestion
            if _digit_runs(na) != _digit_runs(nb):
                continue  # digit mismatch vetoes — doping/series discriminator
            if _word_veto(aname, bname, vocabulary):
                continue  # a condition word was substituted — a different sample

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
