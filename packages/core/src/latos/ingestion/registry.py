"""`ParserRegistry` — confidence-pick dispatcher for `BaseParser`s.

Given a file path, the registry asks every registered parser for a
`can_parse()` confidence score and returns the one with the highest
score above a threshold. If no parser claims the file with sufficient
confidence, the file is treated as "unknown technique" and flagged for
the user (Stage 1D's orchestrator does the flagging — the registry just
returns `None`).

Why a registry vs. a flat list at the call site
-----------------------------------------------
- Tests can build a registry with one or two parsers and verify dispatch
  in isolation without the whole zoo.
- Stage 4+ may want technique-scoped registries (e.g. an XRD-only
  registry for a "re-classify XRD files" workflow).
- Adding a new parser becomes one line in `default_registry()` instead
  of a code change at every dispatch site.

Confidence threshold
--------------------
`MIN_CONFIDENCE = 0.5` per the architecture decision in Stage 1C
planning. A confident-but-not-certain parser (e.g. extension match
without header verification) scores 0.5-0.7. A definitive match scores
1.0. Anything below 0.5 is "this isn't your file" and dispatch returns
`None`.

Tie-breaking: when multiple parsers tie on the highest score, the one
registered FIRST wins. This is deterministic and lets users adjust by
re-ordering registration calls if a tie ever causes a misclassification.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from latos.ingestion.base_parser import BaseParser
from latos.ingestion.parsed_data import ParsedData

__all__ = [
    "MIN_CONFIDENCE",
    "ParserMatch",
    "ParserRegistry",
    "default_registry",
]

# Files scoring below this on every registered parser are considered
# unrecognized. 0.5 splits "I'm pretty sure" (>= 0.5) from "wild guess"
# (< 0.5). Tunable per-call via `find_parser(min_confidence=...)`.
MIN_CONFIDENCE = 0.5


@dataclass(frozen=True, slots=True)
class ParserMatch:
    """A parser that claims to handle a file, with its confidence score.

    Returned by `ParserRegistry.find_parser`. Callers usually don't need
    to look at this directly — `ParserRegistry.parse()` is the typical
    entry point — but it's exposed for tests and diagnostics.
    """

    parser: BaseParser
    confidence: float


class ParserRegistry:
    """Holds a list of `BaseParser`s and dispatches by confidence-pick."""

    def __init__(self, parsers: Iterable[BaseParser] = ()) -> None:
        """Build a registry, optionally pre-populated with `parsers`.

        Args:
            parsers: Initial parsers to register. Order is preserved and
                used for tie-breaking. Subsequent calls to `register()`
                append to the same ordered list.
        """
        # Preserve insertion order. Tuple of unique-by-name parsers — a
        # collision means someone registered the same parser twice, which
        # would skew confidence-picking.
        self._parsers: list[BaseParser] = []
        for p in parsers:
            self.register(p)

    # ─── Registration ────────────────────────────────────────────────
    def register(self, parser: BaseParser) -> None:
        """Add a parser. Re-registering the same `name` raises `ValueError`.

        We disallow re-registration because the typical mistake is to
        register the *same* parser instance twice from different code
        paths — silently allowing it would inflate `len(registry)` and
        slow `find_parser()` proportionally.
        """
        for existing in self._parsers:
            if existing.name == parser.name:
                raise ValueError(
                    f"Parser with name {parser.name!r} is already registered.",
                )
        self._parsers.append(parser)

    @property
    def parsers(self) -> tuple[BaseParser, ...]:
        """Tuple of registered parsers in registration order."""
        return tuple(self._parsers)

    def __len__(self) -> int:
        return len(self._parsers)

    def __contains__(self, name: object) -> bool:
        """Membership by parser `name` — `"rigaku-xrd-txt" in registry`."""
        if not isinstance(name, str):
            return False
        return any(p.name == name for p in self._parsers)

    # ─── Dispatch ────────────────────────────────────────────────────
    def find_parser(
        self,
        path: Path,
        *,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> ParserMatch | None:
        """Return the highest-confidence parser for `path`, or None.

        Asks every registered parser for `can_parse(path)`. Skips parsers
        that raise (defensive — `BaseParser.can_parse` contract says they
        shouldn't raise, but registry stays robust if one does). Returns
        the first parser with the maximum score IF that score is at
        least `min_confidence`.

        Args:
            path: File to dispatch.
            min_confidence: Minimum acceptable confidence. Defaults to
                `MIN_CONFIDENCE` (0.5). Pass 0.0 to accept any non-zero
                score (useful for diagnostics; not for production
                dispatch).

        Returns:
            `ParserMatch` with the winning parser and score, or None if
            no parser scored at or above `min_confidence`.
        """
        best: ParserMatch | None = None
        for parser in self._parsers:
            try:
                score = parser.can_parse(path)
            except Exception:
                # Defensive: BaseParser's contract says can_parse never
                # raises, but a buggy third-party parser shouldn't take
                # down dispatch. Skip and continue.
                continue
            if score < min_confidence:
                continue
            if best is None or score > best.confidence:
                best = ParserMatch(parser=parser, confidence=score)
        return best

    def parse(
        self,
        path: Path,
        *,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> ParsedData | None:
        """Find the right parser and run it. Returns None if no parser claimed the file.

        Convenience over `find_parser(path).parser.parse(path)` — the
        common case. Note: the parser may still return a `ParsedData`
        with errors; the orchestrator (Stage 1D) decides how to handle
        those. Returning a `ParsedData` here only means "a parser
        handled this file"; it does NOT mean "the parse succeeded."
        """
        match = self.find_parser(path, min_confidence=min_confidence)
        if match is None:
            return None
        return match.parser.parse(path)


def default_registry() -> ParserRegistry:
    """Build a `ParserRegistry` populated with every Stage 1C parser.

    Order matters for tie-breaking. We register more-specific parsers
    first (those with strict signature checks like `xrdml`) and more-
    permissive ones last (those whose extensions are ambiguous, like
    `.csv`). This way, on the rare tied score, the more specific
    parser wins.

    Stage 1D's orchestrator builds one of these once per project scan
    and reuses it across all files in the folder.
    """
    # Local import to avoid a heavy ingestion-time import cost when the
    # registry isn't being used. The parsers themselves do not import
    # the registry, so this stays acyclic.
    from latos.ingestion.parsers import (  # noqa: PLC0415 — see comment above
        BrukerSpxParser,
        CasaXpsCsvParser,
        HallXlsParser,
        MicroscopyTifParser,
        PanalyticalXrdmlParser,
        RigakuXrdAscParser,
        RigakuXrdTxtParser,
        ThermoelectricXlsxParser,
        UvDrsXlsxParser,
    )

    return ParserRegistry(
        [
            # Specific signatures first — these never produce false positives.
            PanalyticalXrdmlParser(),
            BrukerSpxParser(),
            MicroscopyTifParser(),
            HallXlsParser(),
            # Extension-keyed but with content sniffs — moderately specific.
            ThermoelectricXlsxParser(),
            UvDrsXlsxParser(),
            RigakuXrdTxtParser(),
            RigakuXrdAscParser(),
            # CasaXPS is keyed only on .csv extension + structure; least
            # specific, registered last so it can't beat anything else.
            CasaXpsCsvParser(),
        ],
    )
