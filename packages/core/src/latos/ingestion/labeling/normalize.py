"""Aggressive sample-name normalization.

Stage 2B turns every string a hint extractor produced into a single,
case-stable, separator-free, Unicode-canonical form. The output is what
the cluster phase (2C) compares — we want the trivial differences
("CS Pure" vs "cs_pure" vs "CS-Pure") to *disappear* at this stage so
the similarity scoring can focus on the differences that actually
matter (e.g. `CS-1` vs `CS-2`).

The normalization pipeline
--------------------------
1.  `str.strip()` — drop leading/trailing whitespace, including any
    accidentally-pasted tabs or newlines.
2.  `unicodedata.normalize("NFKC", ...)` — collapse compatibility-
    equivalent forms. Notable consequences for materials science:
    - Subscript digits (`Cs₃Bi₂I₉`) decompose to regular digits.
    - Full-width digits / letters (Excel imports from Japanese
      keyboards, e.g. U+FF23 U+FF33 U+FF11) become ASCII.
    - Combining diacritics fold into precomposed forms.
3.  `str.lower()` — casefold via straight `.lower()` (sufficient for the
    Latin-letter sample names we see in practice; full `casefold()` is
    overkill and slightly slower).
4.  Leading-prefix strip — the user-facing label often reads
    `Sample: MX-1` or `specimen_001`. We drop the prefix so the
    remainder lines up with bare-name variants.
5.  Separator scrub — remove every space / hyphen / underscore /
    parenthesis / dot / slash / colon. The cluster phase sees a
    digits-and-letters-only string. Digits are preserved on purpose:
    they carry chemistry (`Cs3Bi2I9`) and sample numbering (`MX-001`).

Why this aggressive
-------------------
The scariest false-positive in materials labeling isn't merging two
similar names — it's *failing* to merge two identical samples written
with cosmetic differences. A research dataset that splits `CS-1` and
`CS_1` into two samples produces wrong cross-correlation reports and
an entire stack of derived analyses lined up against the wrong sample.
We bias toward over-merging at this stage and let 2C's similarity graph
re-discriminate when the underlying names actually differ.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "normalize",
    "tokens",
]


# Prefixes the user might prepend to a sample name when filling in
# instrument software ("Sample: MX-1") or labeling spreadsheet rows
# ("Specimen 7"). These are dropped *after* lowercasing so we don't
# need a case-insensitive flag, and *before* separator scrubbing so
# the trailing separator after the prefix word can be matched
# explicitly.
#
# Quantified `(?:...)+` so consecutive prefix words collapse in one
# pass — without this, `normalize("samplesample1")` produces `sample1`
# on the first call (one prefix matched, `1` left) and then `1` on
# the second call, breaking idempotency. The outer `+` lets us strip
# any number of `sample`/`specimen`-style words back-to-back, with
# optional separators between them.
_SAMPLE_PREFIX_RE = re.compile(
    r"^(?:(?:sample|specimen|spec|sampleid|specimenid)[\s\-_:.]*)+",
    re.IGNORECASE,
)

# Internal separators we strip outright. Includes whitespace
# (`\s` covers ASCII space, tab, NBSP after NFKC), the usual
# punctuation (-, _, ., /, :), parentheses, and brackets. Anything
# left after this is letters + digits.
_SEPARATOR_RE = re.compile(r"[\s\-_()./:\[\]\\]+")


def normalize(s: str) -> str:
    """Aggressive normalization: trim, NFKC, lowercase, strip prefix, drop seps.

    Idempotent: `normalize(normalize(x)) == normalize(x)` for every
    string. The empty string normalizes to itself; an all-separator
    string normalizes to the empty string.

    Args:
        s: Any string. Non-`str` inputs raise `TypeError` (let bad
            callers fail loudly rather than silently coerce).

    Returns:
        The normalized form. Always lowercase, ASCII-letters-and-digits-
        plus-anything-NFKC-couldn't-fold, and free of separators.
    """
    if not isinstance(s, str):
        raise TypeError(f"normalize() expected str, got {type(s).__name__}")

    s = s.strip()
    if not s:
        return ""

    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _SAMPLE_PREFIX_RE.sub("", s)
    s = _SEPARATOR_RE.sub("", s)
    # Final NFKC pass: `str.lower()` can decompose certain characters
    # (e.g. Turkish capital `İ` U+0130 → `i̇`, Latin `Á` lowercased
    # under some sequences leaves `a` + combining acute). Without this
    # second fold, `normalize(normalize(x))` differs from `normalize(x)`
    # because the second call recomposes what the first left decomposed
    # — breaking idempotency. Hypothesis caught this; the second pass
    # is the one-line fix.
    return unicodedata.normalize("NFKC", s)


def tokens(s: str) -> tuple[str, ...]:
    """Split `s` into normalized tokens for token-sort similarity (2C).

    The split is on **whitespace only** (after NFKC + lowercase + leading
    "sample"/"specimen" prefix removal). Within-token separators
    (`MX-001`, `cs_pure`) survive the split as one piece and are then
    collapsed by `normalize()` — so `MX-001 batch 3` produces three
    tokens (`mx001`, `batch`, `3`), not four. This matches what the
    cluster phase wants from token-sort similarity: stable groupings
    even when researchers swap word order.

    Returns an empty tuple for empty / all-separator input.
    """
    if not isinstance(s, str):
        raise TypeError(f"tokens() expected str, got {type(s).__name__}")

    s = unicodedata.normalize("NFKC", s.strip()).lower()
    if not s:
        return ()
    # Strip the leading "Sample:"/"Specimen "/etc. prefix run *before*
    # splitting so a colon or dash glued to the prefix word doesn't
    # leak into the first real token.
    s = _SAMPLE_PREFIX_RE.sub("", s)
    if not s:
        return ()
    pieces = s.split()  # whitespace-only split — see docstring
    return tuple(p for p in (normalize(p) for p in pieces) if p)
