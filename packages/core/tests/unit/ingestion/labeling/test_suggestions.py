"""Tests for `latos.ingestion.labeling.suggestions.suggest_merges`."""

from __future__ import annotations

from dataclasses import dataclass

from latos.ingestion.labeling.suggestions import MergeSuggestion, suggest_merges


@dataclass(frozen=True)
class _Sample:
    id: str
    canonical_name: str


def _samples(*names: str) -> list[_Sample]:
    return [_Sample(id=f"id{i}", canonical_name=n) for i, n in enumerate(names)]


def _pairs(suggestions: list[MergeSuggestion]) -> set[frozenset[str]]:
    return {frozenset({s.target_name, s.source_name}) for s in suggestions}


class TestDigitGuard:
    def test_doping_series_never_suggested(self):
        # The single most important guard: a doping series must never be
        # proposed for merging, however similar the prefixes are.
        out = suggest_merges(_samples("CS-1", "CS-3", "CS-5"))
        assert out == []

    def test_cscbi_series_never_suggested(self):
        out = suggest_merges(_samples("CSCBI-1", "CSCBI-3", "CSCBI 5"))
        assert out == []

    def test_pure_vs_numbered_not_suggested(self):
        # "CS (Pure)" (no digit) vs "CS-1" (digit 1) — different samples.
        out = suggest_merges(_samples("CS (Pure)", "CS-1"))
        assert out == []


class TestPrefixNoise:
    def test_researcher_prefix_suggested(self):
        out = suggest_merges(_samples("CSCBI-1", "Dr.MN-dhivya-cscbi1"))
        assert len(out) == 1
        s = out[0]
        # Cleaner (shorter) name is the merge target.
        assert s.target_name == "CSCBI-1"
        assert s.source_name == "Dr.MN-dhivya-cscbi1"
        assert s.confidence == "high"  # exact substring containment

    def test_typo_inside_prefix_suggested_as_medium(self):
        # "cskbi3" (typo) wrapped in a researcher prefix vs "CSCBI-3".
        out = suggest_merges(_samples("CSCBI-3", "Dr.MN-dhivya-cskbi3"))
        assert len(out) == 1
        assert _pairs(out) == {frozenset({"CSCBI-3", "Dr.MN-dhivya-cskbi3"})}


class TestNoFalsePositives:
    def test_distinct_materials_not_suggested(self):
        out = suggest_merges(_samples("CS (Pure)", "cUsE3", "Images"))
        assert out == []

    def test_distinct_short_name_not_suggested(self):
        # "CS" vs "cUsE3" — not a substring, different digits → no suggestion.
        out = suggest_merges(_samples("CS", "cUsE3"))
        assert out == []


class TestShortNameContainment:
    def test_cs_pure_suggested(self):
        # "CS" ⊂ "CS (Pure)" — same undoped sample, two spellings.
        out = suggest_merges(_samples("CS", "CS (Pure)"))
        assert len(out) == 1
        assert _pairs(out) == {frozenset({"CS", "CS (Pure)"})}
        # The shorter, cleaner "CS" is the merge target.
        assert out[0].target_name == "CS"

    def test_cs_not_merged_into_doping_series(self):
        # "CS" must NOT be suggested against CS-1/CS-3 (digit guard).
        out = suggest_merges(_samples("CS", "CS-1", "CS-3", "CS (Pure)"))
        pairs = _pairs(out)
        assert frozenset({"CS", "CS (Pure)"}) in pairs
        assert frozenset({"CS", "CS-1"}) not in pairs
        assert frozenset({"CS", "CS-3"}) not in pairs

    def test_empty_and_single(self):
        assert suggest_merges([]) == []
        assert suggest_merges(_samples("OnlyOne")) == []


class TestRanking:
    def test_sorted_by_score_descending(self):
        out = suggest_merges(
            _samples("CSCBI-1", "Dr.MN-dhivya-cscbi1", "CSCBI-3", "Dr.MN-dhivya-cskbi3")
        )
        scores = [s.score for s in out]
        assert scores == sorted(scores, reverse=True)
        # The exact-substring pair outranks the typo pair.
        assert out[0].score >= out[-1].score
